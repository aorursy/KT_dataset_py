import numpy as np
import pandas as pd
df_so = pd.read_csv('../input/posts2017to20181/RegexPostsTitle.csv', encoding='utf-8')

df_so['Id'] = df_so['Id'].apply(str)
df_so['Body'] = df_so['Body'].apply(str)
df_so['Title'] = df_so['Title'].apply(str)
df_comments = pd.read_csv('../input/posts2017to20181/RegexComments.csv', encoding='utf-8')

df_comments['c_Id'] = df_comments['c_Id'].apply(str)
df_comments['c_Text'] = df_comments['c_Text'].apply(str)
#question_comment = "Thanks for all your input.  I am looking for a regular expression to validate my user's password.  My users are allowed to enter a password between 8-15 characters long and it could be either alphamumeric or nonalphanumberic character.  Does that make more sense now?"
question_comment = "thanks :) yup working like gem :)&#xA;btw one question here. How to treat first character separately? I mean .. first character should be ALPHA and rest other chars can be alpha-numeric? I have found a work around by declaring two separated RegExps however."
import spacy
nlp = spacy.load('en_core_web_lg')
df = pd.DataFrame() #empty dataframe
#columns - comment, question, comment_created, question_created, similarity_score
id = 0
for comment in df_comments.itertuples(name='Pandas'):
    #print(getattr(row, "c_Id"), getattr(row, "c_Text"))
    ques = nlp(getattr(comment,"c_Text"))
    for question in df_so.itertuples(name='Pandas'):
        id += 1
        similarity_score = ques.similarity(nlp(getattr(question,"Title")))
        if (similarity_score >= 0.95):
            data = pd.DataFrame({
                'comment': getattr(comment,"c_Text"),
                'question' : getattr(question,"Title"),
                'comment_created' : getattr(comment,"c_CreationDate"),
                'question_created' : getattr(question,"CreationDate"),
                'similarity_score' : similarity_score
            }, index=[id])
            df = df.append(data)
        
df.to_csv('results.csv',encoding='utf-8')
df.head()