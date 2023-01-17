import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from collections import Counter
train=pd.read_csv("../input/tweet-sentiment-extraction/train.csv")

test=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
train["sentiment"]=train["sentiment"].map({"positive":1,"neutral":0,"negative":-1})

positive=train[train["sentiment"]==1]

neutral=train[train["sentiment"]==0]

negative=train[train["sentiment"]==-1]

positive["temp_list"]=positive["selected_text"].apply(lambda x:str(x).split())

top=Counter([item for sublist in positive["temp_list"] for item in sublist])

positive_table = pd.DataFrame(top.most_common(1000))

positive_table.columns = ["Common_words","count"]

positive_table.style.background_gradient(cmap="PuBu")
neutral["temp_list"]=neutral["selected_text"].apply(lambda x:str(x).split())

top=Counter([item for sublist in neutral["temp_list"] for item in sublist])

neutral_table = pd.DataFrame(top.most_common(1000))

neutral_table.columns = ["Common_words","count"]

neutral_table.style.background_gradient(cmap="OrRd")
negative["temp_list"]=negative["selected_text"].apply(lambda x:str(x).split())

top=Counter([item for sublist in negative["temp_list"] for item in sublist])

negative_table = pd.DataFrame(top.most_common(1000))

negative_table.columns = ["Common_words","count"]

negative_table.style.background_gradient(cmap="Wistia")
union_table=pd.concat([positive_table,neutral_table,negative_table],axis=0)

union_table
frequency=pd.DataFrame(columns=["Common_words","count"])

frequency["count"]=union_table["Common_words"].value_counts()

frequency["Common_words"]=frequency.index

frequency=frequency.reset_index(drop=True)

frequency
frequency_extraction=frequency[(frequency["count"]==3)|(frequency["count"]==2)]

frequency_extraction
words=[]

for row in range(len(frequency_extraction)):

    word=frequency_extraction.iloc[row,0]

    words.insert(row,word)

words
for number in range(len(frequency_extraction)):

    positive_table=positive_table[positive_table["Common_words"]!=words[number]]

positive_table
for number in range(len(frequency_extraction)):

    neutral_table=neutral_table[neutral_table["Common_words"]!=words[number]]

neutral_table
for number in range(len(frequency_extraction)):

    negative_table=negative_table[negative_table["Common_words"]!=words[number]]

negative_table
positive_words=[]

neutral_words=[]

negative_words=[]

for row in range(len(positive_table)):

    positive_word=positive_table.iloc[row,0]

    positive_words.insert(row,positive_word)

for row in range(len(neutral_table)):

    neutral_word=neutral_table.iloc[row,0]

    neutral_words.insert(row,neutral_word)    

for row in range(len(negative_table)):

    negative_word=negative_table.iloc[row,0]

    negative_words.insert(row,negative_word)
submission_words=[]

for row in range(len(test)):

    judgement=test.iloc[row,2]

    text_word=test.iloc[row,1]

    if judgement=="positive":

        for position in range(len(positive_words)):

            search=text_word.find(positive_words[position])

            if search!=-1:

                submission_words.insert(row,positive_words[position])

                break

        else:

            submission_words.insert(row,text_word)

    if judgement=="neutral":

        for position in range(len(neutral_words)):

            search=text_word.find(neutral_words[position])

            if search!=-1:

                submission_words.insert(row,neutral_words[position])

                break

        else:

            submission_words.insert(row,text_word)      

    if judgement=="negative":

        for position in range(len(negative_words)):

            search=text_word.find(negative_words[position])

            if search!=-1:

                submission_words.insert(row,negative_words[position])

                break

        else:

            submission_words.insert(row,text_word)  
submission=pd.DataFrame(pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv"))

submission["selected_text"]=list(map(str,submission_words))

submission.to_csv("submission.csv", index=False)