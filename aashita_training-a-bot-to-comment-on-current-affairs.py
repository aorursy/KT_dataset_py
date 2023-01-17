import pandas as pd
import markovify 
import spacy
import re

import warnings
warnings.filterwarnings('ignore')

from time import time
import gc
curr_dir = '../input/'
df1 = pd.read_csv(curr_dir + 'CommentsJan2017.csv')
df2 = pd.read_csv(curr_dir + 'CommentsFeb2017.csv')
df3 = pd.read_csv(curr_dir + 'CommentsMarch2017.csv')
df4 = pd.read_csv(curr_dir + 'CommentsApril2017.csv')
df5 = pd.read_csv(curr_dir + 'CommentsMay2017.csv')
df6 = pd.read_csv(curr_dir + 'CommentsJan2018.csv')
df7 = pd.read_csv(curr_dir + 'CommentsFeb2018.csv')
df8 = pd.read_csv(curr_dir + 'CommentsMarch2018.csv')
df9 = pd.read_csv(curr_dir + 'CommentsApril2018.csv')
comments = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9])
comments.drop_duplicates(subset='commentID', inplace=True)
comments.head(3)
comments.shape
comments.sectionName.value_counts()[:5]
def preprocess(comments):
    commentBody = comments.loc[comments.sectionName=='Politics', 'commentBody']
    commentBody = commentBody.str.replace("(<br/>)", "")
    commentBody = commentBody.str.replace('(<a).*(>).*(</a>)', '')
    commentBody = commentBody.str.replace('(&amp)', '')
    commentBody = commentBody.str.replace('(&gt)', '')
    commentBody = commentBody.str.replace('(&lt)', '')
    commentBody = commentBody.str.replace('(\xa0)', ' ')  
    return commentBody
commentBody = preprocess(comments)
commentBody.shape
del comments, df1, df2, df3, df4, df5, df6, df7, df8
gc.collect()
commentBody.sample().values[0]
start_time = time()
comments_generator = markovify.Text(commentBody, state_size = 5)
print("Run time for training the generator : {} seconds".format(round(time()-start_time, 2)))
# Print randomly-generated comments using the built model
def generate_comments(generator, number=10, short=False):
    count = 0
    while count < number:
        if short:
            comment = generator.make_short_sentence(140)
        else:
            comment = generator.make_sentence()
        if comment:
            count += 1
            print("Comment {}".format(count))
            print(comment)
            print()
    
generate_comments(comments_generator)
nlp = spacy.load("en")

class POSifiedText(markovify.Text):
    def word_split(self, sentence):
        return ["::".join((word.orth_, word.pos_)) for word in nlp(sentence)]

    def word_join(self, words):
        sentence = " ".join(word.split("::")[0] for word in words)
        return sentence
commentBody = preprocess(df9)
commentBody.shape
del comments_generator, df9
gc.collect()
# start_time = time()
# comments_generator_POSified = POSifiedText(commentBody, state_size = 2)
# print("Run time for training the generator : {} seconds".format(round(time()-start_time, 2)))
# generate_comments(comments_generator_POSified)