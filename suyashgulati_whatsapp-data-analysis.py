# import codecs
# import glob
# import logging
# import multiprocessing
# import os
# import pprint
# import re

# import nltk
# import sklearn.manifold
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import string
#df = pd.read_csv('C:/Users/SUYASH/PycharmProjects/misc/whatsapp/suyash_chat.txt', header=None,error_bad_lines=False,sep=']')
# # Removing [ from column 0
# df[["trash","date_time"]] = pd.DataFrame(df[0].str.split('[',1).tolist())
# df = df.dropna()
# # splitting name from column 1 to name
# df[["name","text"]] = pd.DataFrame(df[1].str.split(':',1).tolist(),index=df.index)
# df = df.dropna()
# df = df.drop([0,1,"trash"],axis = 1)
# df.head(2)
# # converting first column to datetime format
# df['date_time'] =  pd.to_datetime(df['date_time'],errors = "coerce",dayfirst = True)
# df.head(2)
# # stripping name and text column to remove whitespaces from start and end

# def strip(x):
#   stripped = x.strip()
#   return stripped

# df['name'] = df[["name"]].apply(lambda x: strip(*x), axis=1)
# df['text'] = df[["text"]].apply(lambda x: strip(*x), axis=1)
# df.name.unique()
# # merging known names together and unknown names to value 1 to remove later

# df.replace({'name' : { '\u202a+91\xa088265\xa089841\u202c' : "ME", '\u202a+91\xa084472\xa005046\u202c' : 1,
#                       '\u202a+91 91 65 020604\u202c' : 1,'Enzo' : 1, 'Shriram Yadav' : 1, 'MANDU GJU' : 1,
#                      'Kallu' : 1,'\u202a+91\xa096542\xa092717\u202c' : 1, 'The Billionaires Club 🤑' : 1, '\u200eSuyash changed the subject to “Our bitch' : 1,
#                      '\u202a+971\xa050\xa0132\xa07301\u202c' : 1,'the complete digital marketing course* - https' : 1, "archu" : 1}},inplace=True)
# df.name.unique()
# # removing names with value 1

# df = df[df["name"] != 1]
# df.head()
# cursing = ["bc" , "mc" , "gand", "chut","chod","lund","laude","rand","cunt","fuck","tutta","choot","bhosd","lavde","bsdk"]
# checking if text has bad words

# df['cursing'] = df["text"].str.lower().str.contains('|'.join(cursing))
# df.head()
# multimedia = ["omitted>" , "http" , "www."]
# # checking text have some link or is showing default text for media shared

# df['multi'] = df["text"].str.lower().str.contains('|'.join(multimedia))
# df.head()
#df.to_csv("drive/app/Misc/whatsapp/analysis/clean_text.csv",index=False,encoding='utf-8-sig')
from IPython.display import IFrame
IFrame('https://public.tableau.com/views/Thebillionairesclub/Story1?embed=y&:display_count=yes', width=1000, height=925)
