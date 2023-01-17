import tensorflow as tf

import pandas as pd

import numpy as np

import sys

import time

from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering

import textwrap

import re

import attr

import abc

import string

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from IPython.display import HTML

from os import listdir

from os.path import isfile, join



import warnings  

warnings.filterwarnings('ignore')

MAX_ARTICLES = 1000

base_dir = '/kaggle/input'

data_dir = base_dir + '/covid-19-articles'

data_path = data_dir + '/covid19.csv'

model_path = base_dir + '/biobert-qa/biobert_squad2_cased'

df=pd.read_csv(data_path)

class ResearchQA(object):

    def __init__(self, data_path, model_path):

        print('Loading data from', data_path)

        self.df = pd.read_csv(data_path)

        print('Initializing model from', model_path)

        self.model = TFAutoModelForQuestionAnswering.from_pretrained(model_path, from_pt=True)

        tf.saved_model.save(self.model, '/kaggle/output')

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.retrievers = {}

        self.build_retrievers()

        self.main_question_dict = dict()

        

    

    def build_retrievers(self):

        df = self.df

        abstracts = df[df.abstract.notna()].abstract

        self.retrievers['abstract'] = TFIDFRetrieval(abstracts)

        body_text = df[df.body_text.notna()].body_text

        self.retrievers['body_text'] = TFIDFRetrieval(body_text)



    def retrieve_candidates(self, section_path, question, top_n):

        candidates = self.retrievers[section_path[0]].retrieve(question, top_n)

        return self.df.loc[candidates.index]

          

        

    def get_answers(self, question, section='abstract', keyword=None, max_articles=1000, batch_size=4):

        df = self.df

        answers = []

        section_path = section.split('/')



        if keyword:

            candidates = df[df[section_path[0]].str.contains(keyword, na=False, case=False)]

        else:

            candidates = self.retrieve_candidates(section_path, question, top_n=max_articles) #get top N candidate articles based on similarity score

        if max_articles:

            candidates = candidates.head(max_articles)



        text_list = []

        indices = []

        for idx, row in candidates.iterrows():

            if section_path[0] == 'body_text':

                text = self.get_body_section(row.body_text, section_path[1])

            else:

                text = row[section]

            if text and isinstance(text, str):

                text_list.append(text)

                indices.append(idx)



        num_batches = len(text_list) // batch_size

        all_answers = []

        for i in range(num_batches):

            batch = text_list[i * batch_size:(i+1) * batch_size]

            answers = self.get_answers_from_text_list(question, batch)

            all_answers.extend(answers)



        last_batch = text_list[batch_size * num_batches:]

        if last_batch:

            all_answers.extend(self.get_answers_from_text_list(question, last_batch))



        columns = ['doi', 'authors', 'journal', 'publish_time', 'title', 'cohort_size']

        processed_answers = []

        for i, a in enumerate(all_answers):

            if a:

                row = candidates.loc[indices[i]]

                new_row = [a.text, a.start_score, a.end_score, a.input_text]

                new_row.extend(row[columns].values)

                processed_answers.append(new_row)

        answer_df = pd.DataFrame(processed_answers, columns=(['answer', 'start_score',

                                                 'end_score', 'context'] + columns))

        return answer_df.sort_values(['start_score', 'end_score'], ascending=False)



    def get_body_section(self, body_text, section_name):

      sections = body_text.split('<SECTION>\n')

      for section in sections:

        lines = section.split('\n')

        if len(lines) > 1:

          if section_name.lower() in lines[0].lower():

            return section



    def get_answers_from_text_list(self, question, text_list, max_tokens=512):

      tokenizer = self.tokenizer

      model = self.model

      inputs = tokenizer.batch_encode_plus(

          [(question, text) for text in text_list], add_special_tokens=True, return_tensors='tf',

          max_length=max_tokens, truncation_strategy='only_second', pad_to_max_length=True)

      input_ids = inputs['input_ids'].numpy()

      answer_start_scores, answer_end_scores = model(inputs)

      answer_start = tf.argmax(

          answer_start_scores, axis=1

      ).numpy()  # Get the most likely beginning of each answer with the argmax of the score

      answer_end = (

          tf.argmax(answer_end_scores, axis=1) + 1

      ).numpy()  # Get the most likely end of each answer with the argmax of the score



      answers = []

      for i, text in enumerate(text_list):

        input_text = tokenizer.decode(input_ids[i, :], clean_up_tokenization_spaces=True)

        input_text = input_text.split('[SEP] ', 2)[1]

        answer = tokenizer.decode(

            input_ids[i, answer_start[i]:answer_end[i]], clean_up_tokenization_spaces=True)

        score_start = answer_start_scores.numpy()[i][answer_start[i]]

        score_end = answer_end_scores.numpy()[i][answer_end[i]-1]

        if answer and not '[CLS]' in answer:

          answers.append(Answer(answer, score_start, score_end, input_text))

        else:

          answers.append(None)

      return answers

    



class Retrieval(abc.ABC):

  """Base class for retrieval methods."""



  def __init__(self, docs, keys=None):

    """

    Args:

      docs: a pd.Series of strings. The text to retrieve.

      keys: a pd.Series. Keys (e.g. ID, title) associated with each document.

    """

    self._docs = docs.copy()

    if keys is not None:

      self._docs.index = keys

    self._model = None

    self._doc_vecs = None



  def _top_documents(self, q_vec, top_n=10):

    similarity = cosine_similarity(self._doc_vecs, q_vec)

    rankings = np.argsort(np.squeeze(similarity))[::-1]

    ranked_indices = self._docs.index[rankings]

    return self._docs[ranked_indices][:top_n]



  @abc.abstractmethod

  def retrieve(self, query, top_n=10):

    pass



class TFIDFRetrieval(Retrieval):

  """Retrieve documents based on cosine similarity of TF-IDF vectors with query."""



  def __init__(self, docs, keys=None):

    """

    Args:

      docs: a list or pd.Series of strings. The text to retrieve.

      keys: a list or pd.Series. Keys (e.g. ID, title) associated with each document.

    """

    super(TFIDFRetrieval, self).__init__(docs, keys)

    self._model = TfidfVectorizer()

    self._doc_vecs = self._model.fit_transform(docs)



  def retrieve(self, query, top_n=10):

    q_vec = self._model.transform([query])

    return self._top_documents(q_vec, top_n)



@attr.s

class Answer(object):

    text = attr.ib()

    start_score = attr.ib()

    end_score = attr.ib()

    input_text = attr.ib()

    

style = '''

<style>

.hilight {

  background-color:#cceeff;

}

a {

  color: #000 !important;

  text-decoration: underline;

}

.question {

  font-size: 20px;

  font-style: italic;

  margin: 10px 0;

}

.info {

  padding: 10px 0;

}

table.dataframe {

  max-height: 450px;

  text-align: left;

}

.meta {

  margin-top: 10px;

}

.journal {

  color: green;

}

.footer {

  position: absolute;

  bottom: 20px;

  left: 20px;

}

</style>

'''



def format_context(row):

  text = row.context

  answer = row.answer

  highlight_start = text.find(answer)



  def find_context_start(text):

    idx = len(text) - 1

    while idx >= 2:

      if text[idx].isupper() and re.match(r'\W ', text[idx - 2:idx]):

        return idx

      idx -= 1

    return 0 

  context_start = find_context_start(text[:highlight_start])

  highlight_end = highlight_start + len(answer)

  context_html = (text[context_start:highlight_start] + '<span class=hilight>' + 

                  text[highlight_start:highlight_end] + '</span>' + 

                  text[highlight_end:highlight_end + 1 + text[highlight_end:].find('. ')])

  context_html += f'<br><br>score: {row.start_score:.2f}'

  return context_html





def format_author(authors):

  if not authors or not isinstance(authors, str):

    return 'Unknown Authors'

  name = authors.split(';')[0]

  name = name.split(',')[0]

  return name + ' et al'



def format_info(row):

  meta = []

  authors = format_author(row.authors) 

  if authors:

    meta.append(authors)

  meta.append(row.publish_time)

  meta = ', '.join(meta)

 

  html = f'''\

  <a class="title" target=_blank href="http://doi.org/{row.doi}">{row.title}</a>\

  <div class="meta">{meta}</div>\

  '''



  journal = row.journal

  if journal and isinstance(journal, str):

    html += f'<div class="journal">{journal}</div>'



  return html



def render_results(main_question, answers):

  id = main_question[:20].replace(' ', '_')

  html = f'<h1 id="{id}" style="font-size:20px;">{main_question}</h1>'

  for q, a in answers.items():

    # TODO: skipping empty answers. Maybe we should show

    # top retrieved docs.

    if a.empty:

      continue

    # clean up question

    if '?' in q:

        q = q.split('?')[0] + '?'

    html += f'<div class=question>{q}</div>' + format_answers(a)

  display(HTML(style + html))



def format_answers(a):

    a = a.sort_values('start_score', ascending=False)

    a.drop_duplicates('doi', inplace=True)

    out = []

    for i, row in a.iterrows():

        if row.start_score < 0:

            continue

        info = format_info(row)

        context = format_context(row)



        cohort = ''

        if not np.isnan(row.cohort_size):

            cohort = int(row.cohort_size)

        temp=df[df['doi']==row.doi]

        text = temp['body_text']

        summ=summarizer(str(text), max_length=1000,   min_length=30)

        out.append([context, info,summ])

    out = pd.DataFrame(out, columns=['answer', 'article','summ'])

    return out.to_html(escape=False, index=False)



def render_answers(a):

    display(HTML(style + format_answers(a)))
from transformers import pipeline

summarizer = pipeline('summarization')
model1 = TFAutoModelForQuestionAnswering.from_pretrained(model_path, from_pt=True)

tf.saved_model.save(model1, '/kaggle/working/')
qa = ResearchQA(data_path, model_path)
answers = qa.get_answers('What drugs are effective?',max_articles=5)

render_answers(answers)
answers = qa.get_answers('What kind of cytokines play a major role in host response?',max_articles=5)

render_answers(answers)