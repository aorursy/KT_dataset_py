# task 1: What do we know about COVID-19 risk factors?

# Task details: What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?

# https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=558



questions = [{'question':"Is smoking a risk factor?",'keyword':None},

             {'question':"Is a pre-existing pulmonary disease a risk factor?",'keyword':None},

             {'question':"Do co-existing conditions make the virus more transmissible?",'keyword':None},

             {'question':"Is being a pregnant woman a risk factor?",'keyword':'pregnant'},

             {'question':"Is being a neonate a risk factor?",'keyword':'neonate'},

             {'question':"Are there differences in risk factors associated to socio-economic factors?",'keyword':None},

             {'question':"How does the transmission happen?",'keyword':'transmission'},

             {'question':"What is the reproductive rate?",'keyword':None},

             {'question':"What is the incubation period?",'keyword':None},

             {'question':"What are the modes of transmission?",'keyword':None},

             {'question':"What are the enviromental factors?",'keyword':None},

             {'question':"How long is the serial interval?",'keyword':None},

             {'question':"What is the severity of disease among high risk groups and patients?",'keyword':None},

             {'question':"What is the risk of death among high risk groups and patients?",'keyword':None},

             {'question':"What is the susceptibility of populations?",'keyword':None},

             {'question':"What are the public health mitigation measures that could be effective for control?",'keyword':None}]
!pip install transformers

import torch

import pandas as pd

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer

#device_available = torch.cuda.is_available()

device_available = False

from IPython.core.display import display, HTML

import seaborn as sns

import matplotlib.pyplot as plt





# Use plot styling from seaborn.

sns.set(style='darkgrid')



# Increase the plot size and font size.

#sns.set(font_scale=1.5)

plt.rcParams["figure.figsize"] = (20,8)



model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

if device_available:

    model.cuda()



tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# this is just to load the files needed instead of running the model each time.



import pickle



def load_or_run_answer_question_dict(question,keyword):

    pickle_name = question.replace(' ','_').replace('?','_')

    path_to_file = F"/kaggle/input/kaggle/{pickle_name}.pickle"

    print(path_to_file)

    try:

      df = pickle.load(open(path_to_file, "rb"))

    except (OSError, IOError) as e:

        df = answer_question_dict(question, keyword)

        pickle.dump(df, open(path_to_file, "wb"))

    return df

# print(os.listdir("../input"))

# print(os.listdir("../input/datacompetition"))
import textwrap



def get_dataset(csv_path):

    corpus = []

    csv_df = pd.read_csv(csv_path).dropna(subset=['authors', 'abstract']).drop_duplicates(subset='abstract')

    csv_df = csv_df[csv_df['abstract']!='Unknown']

    for ix,row in csv_df.iterrows():

        if row['abstract'] and not pd.isna(row['abstract']):

            temp_dict = dict()

            temp_dict['abstract'] = row['abstract']

            temp_dict['title'] = row['title']

            temp_dict['authors'] = row['authors']

            temp_dict['url'] = row['doi']

            temp_dict['publish_time'] = row['publish_time']

            corpus.append(temp_dict)

    return corpus



wrapper = textwrap.TextWrapper(width=80) 



corpus = get_dataset('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-03-27/metadata.csv')
def answer_question_dict(question, keyword=None, show_visualization=False):



    '''

    Takes a `question` string and an `answer_text` string (which contains the

    answer), and identifies the words within the `answer_text` that are the

    answer. Prints them out.

    '''

    # select corpus

    answer_text = corpus



    # Initializing answers list

    answers = {}

    min_score = 0

    counter = 0 # for stopping iterations earlier

    

    for answer_option in answer_text:

      if keyword and keyword not in answer_option['abstract']:

        continue



      # ======== Tokenize ========

      # Apply the tokenizer to the input text, treating them as a text-pair.

      input_ids = tokenizer.encode(question, answer_option['abstract'],max_length=512)



      # Report how long the input sequence is.

      #print('Query has {:,} tokens.\n'.format(len(input_ids)))



      # ======== Set Segment IDs ========

      # Search the input_ids for the first instance of the `[SEP]` token.

      sep_index = input_ids.index(tokenizer.sep_token_id)



      # The number of segment A tokens includes the [SEP] token istelf.

      num_seg_a = sep_index + 1



      # The remainder are segment B.

      num_seg_b = len(input_ids) - num_seg_a



      # Construct the list of 0s and 1s.

      segment_ids = [0]*num_seg_a + [1]*num_seg_b

    



      # There should be a segment_id for every input token.

      assert len(segment_ids) == len(input_ids)

      



      # ======== Evaluate ========

      # Run our example question through the model.

        

      input_ids_tensor = torch.tensor([input_ids])

      segment_ids_tensor = torch.tensor([segment_ids])

      if device_available:

         input_ids_tensor = input_ids_tensor.to('cuda:0')

         segment_ids_tensor = segment_ids_tensor.to('cuda:0')



      start_scores, end_scores = model(input_ids_tensor, # The tokens representing our input text.

                                  token_type_ids=segment_ids_tensor) # The segment IDs to differentiate question from answer_text

    

      # only review answers with score above threshold

      score = round(torch.max(start_scores).item(), 3)



      if score>min_score and score>0:



        # ======== Reconstruct Answer ========

        

        # Find the tokens with the highest `start` and `end` scores.

        answer_start = torch.argmax(start_scores)

        answer_end = torch.argmax(end_scores)





        # Get the string versions of the input tokens.

        tokens = tokenizer.convert_ids_to_tokens(input_ids)



        # Start with the first token.

        answer = tokens[answer_start]



        # Select the remaining answer tokens and join them with whitespace.

        for i in range(answer_start + 1, answer_end + 1):

            

            # If it's a subword token, then recombine it with the previous token.

            if tokens[i][0:2] == '##':

                answer += tokens[i][2:]

            

            # Otherwise, add a space then the token.

            else:

                answer += ' ' + tokens[i]



        # ======== Add Answer to best answers list ========



        if len(answers)>4:

          min_score = min([d for d in answers.keys()])

          

        if len(answers)==10:

          answers.pop(min_score)

        answers[score] = [answer, score, '<a href="https://doi.org/'+str(answer_option['url'])+'" target="_blank">' + str(answer_option['title']) +'</a>', answer_option['abstract'], answer_option['publish_time']]

        

        visualization_start = max(answer_start-20,0)

        visualization_end = min((answer_end+1)+20,len(tokens))

        # Variables needed for graphs

        s_scores = start_scores.cpu().detach().numpy().flatten()

        e_scores = end_scores.cpu().detach().numpy().flatten()

        

        # We'll use the tokens as the x-axis labels. In order to do that, they all need

        # to be unique, so we'll add the token index to the end of each one.

        token_labels = []

        for (i, token) in enumerate(tokens):

            token_labels.append('{:} - {:>2}'.format(token, i))

        answers[score] = [answer, score, '<a href="https://doi.org/'+str(answer_option['url'])+'" target="_blank">' + str(answer_option['title']) +'</a>', answer_option['abstract'], answer_option['publish_time'], s_scores, e_scores, token_labels, visualization_start, visualization_end]

        

    # Return dataframe with relevant data

    df_columns = ['Answer', 'Confidence', 'Title', 'Abstract', 'Published', 's_scores', 'e_scores', 'token_labels', 'visualization_start', 'visualization_end']

    df = pd.DataFrame.from_dict(answers, orient='index',columns = df_columns)

    df.sort_values(by=['Confidence'], inplace=True, ascending=False)

    return df
for question in questions:

    print("======================")

    print(question)

    df = load_or_run_answer_question_dict(question['question'], question['keyword'])

    display(HTML(df[['Answer', 'Confidence', 'Title', 'Abstract', 'Published']].to_html(render_links=True, escape=False, index=False)))

    print("======================")
def start_word_plot(token_labels, s_scores):

  ax = sns.barplot(x=token_labels, y=s_scores, ci=None)

  ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

  ax.grid(True)

  plt.title('Start Word Scores [for first answer]')

  plt.show()



def end_word_plot(token_labels,e_scores):

  ax = sns.barplot(x=token_labels, y=e_scores, ci=None)

  ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

  ax.grid(True)

  plt.title('End Word Scores [for first answer]')

  plt.show()



df = load_or_run_answer_question_dict(questions[0]['question'], question['keyword'])

start_word_plot(df.iloc[0]['token_labels'],df.iloc[0]['s_scores'])

end_word_plot(df.iloc[0]['token_labels'],df.iloc[0]['e_scores'])