class ResearchPaperListAPI:

    

    def __init__(self):

        self.active = 1

        self.path = '/kaggle/input/CORD-19-research-challenge/2020-03-13/'

    

    def read_file(self, path_to_file):

        with open(path_to_file) as json_data:

            data = json.load(json_data)

        return data

    

    def read_all_file_names(self):

        l = os.listdir(self.path)

        file_list = []

        for l1 in l:

            if os.path.isdir(f"{self.path}{l1}"):

                for l2 in os.listdir(f"{self.path}{l1}"):

                    for l3 in os.listdir(f"{self.path}{l1}/{l2}"):

                        path = f"{self.path}{l1}/{l2}/{l3}"

                        file_list.append(path)

        return file_list

                                                    
import os

import json

from pandas.io.json import json_normalize

obj = ResearchPaperListAPI()

file_list = obj.read_all_file_names()

obj = ResearchPaperListAPI()

i = 0

limit = 50000



list_of_instances = []

for file in file_list:

    instance = []

    if i > limit:

        break

    i += 1

    

    # Read file content

    file_content = obj.read_file(file)

    

    # Paper id

    paper_id = file_content['paper_id']

    

    # Read Paper Title

    title = file_content['metadata']['title']

    #print (f"{i}: {title}")

    

    # Read author information

    authors = []

    for e in file_content['metadata']['authors']:

        authors.append({'first': e['first'], 'last': e['last']})

    #print (f"{i}: {authors}")

    

    # Read Abstract

    abstract = ''

    l = file_content['abstract']

    for subs in l:

        abstract += subs['text']



    # Read Paper text

    b_text = ''

    for section in file_content['body_text']:

        for dictt in section.values():

            if isinstance(dictt, str):

                b_text += dictt

    

    # Read Conclusion

    conclusion = file_content['body_text'][-1]['text']

    

    # Combine all in dictionary

    out_dict = {

        'paper_id': paper_id,

        'title': title,

        'authors': authors,

        'abstract': abstract,

        'body_text': b_text,

        'conclusion': conclusion

    }

    

    # Write output

    list_of_instances.append(out_dict)

    
len(list_of_instances)
import pandas as pd

df = pd.DataFrame(list_of_instances, columns=['paper_id', 'title', 'authors', 'abstract', 'body_text', 'conclusion'])
df.to_csv("research_paper_clean_data.csv")
df1 = pd.read_csv("research_paper_clean_data.csv")
df1.head()
import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud



def plot_wordcloud(wordcloud):

    plt.figure(figsize=(12,10))

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.show()
import nltk

from nltk.corpus import stopwords

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



abstracts = df1['abstract']



set(stopwords.words('english'))



stop_words = set(stopwords.words('english')) 



#def txt_preprocess(document):

#    sentences = nltk.sent_tokenize(document)

#    sentences = [nltk.word_tokenize(sent) for sent in sentences]

#    sentences = [nltk.pos_tag(sent) for sent in sentences]

#    return sentences



combined_abstract = ''

for abstract in abstracts:

    if isinstance(abstract, str):

        print (abstract)

        tokens = word_tokenize(abstract)

        filtered = []

        for token in tokens:

            if isinstance(token, str):

                if not token in stop_words:

                    filtered.append(token)

        combined_abstract += ' '.join(filtered)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(combined_abstract)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
abstracts = df1['title']



set(stopwords.words('english'))



stop_words = set(stopwords.words('english')) 



#def txt_preprocess(document):

#    sentences = nltk.sent_tokenize(document)

#    sentences = [nltk.word_tokenize(sent) for sent in sentences]

#    sentences = [nltk.pos_tag(sent) for sent in sentences]

#    return sentences



combined_abstract = ''

for abstract in abstracts:

    if isinstance(abstract, str):

        tokens = word_tokenize(abstract)

        filtered = []

        for token in tokens:

            if isinstance(token, str):

                if not token in stop_words:

                    filtered.append(token)

        combined_abstract += ' '.join(filtered)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(combined_abstract)

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
"""

abstracts = df1['body_text']



set(stopwords.words('english'))



stop_words = set(stopwords.words('english')) 



#def txt_preprocess(document):

#    sentences = nltk.sent_tokenize(document)

#    sentences = [nltk.word_tokenize(sent) for sent in sentences]

#    sentences = [nltk.pos_tag(sent) for sent in sentences]

#    return sentences



combined_abstract = ''

for abstract in abstracts:

    if isinstance(abstract, str):

        print (abstract)

        tokens = word_tokenize(abstract)

        filtered = []

        for token in tokens:

            if isinstance(token, str):

                if not token in stop_words:

                    filtered.append(token)

        combined_abstract += ' '.join(filtered)

"""
"""

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(combined_abstract)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()

"""