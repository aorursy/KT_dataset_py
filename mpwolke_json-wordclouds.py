# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def josn_DF():

    import json

    with open("../input/stanford-question-answering-dataset/train-v1.1.json") as input_file:    

        json_data = json.load(input_file)

        

    context = pd.DataFrame(columns=['id', 'context'])

    Question = pd.DataFrame(columns=['id', 'cID', 'Question'])

    Answer = pd.DataFrame(columns=['id','qID','cID','Answer'])  

    

    context_no_tracker = 0

    Question_no_tracker = 0

    Answer_no_tracker = 0

    json_layer1 = json_data['data']

    for i in tqdm(json_layer1):

        json_layer2 = i['paragraphs']

        for context_no_tracker,j in enumerate(json_layer2,context_no_tracker):

            context.loc[context_no_tracker] = [context_no_tracker,j['context']]

            for Question_no_tracker,k in enumerate(j['qas'],Question_no_tracker):

                Question.loc[Question_no_tracker] = [Question_no_tracker,context_no_tracker,k['question']]

                for  Answer_no_tracker,l in enumerate(k['answers'],Answer_no_tracker):

                    Answer.loc[Answer_no_tracker] = [Answer_no_tracker,Question_no_tracker,context_no_tracker,l['text']]

            

    return(context,Question,Answer)
context,Question,Answer = josn_DF()
context.head()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in context.context)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
Question.head()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in Question.Question)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()

Answer.head()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in Answer.Answer)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()