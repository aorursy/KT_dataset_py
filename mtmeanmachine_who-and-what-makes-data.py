#A Quick and Dirty Viz Notebook. More To Come#
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')
%matplotlib inline

response = pd.read_csv('../input/freeFormResponses.csv',low_memory=False)
schema = pd.read_csv('../input/SurveySchema.csv')
multi = pd.read_csv('../input/multipleChoiceResponses.csv',low_memory=False)

multi.columns = multi.loc[:0].values[0]
multi_new = multi.drop(0,0)

response.columns = response.loc[:0].values[0]
response = response.drop(0,0)

def split_cols(df,keyword):
    keyword_list = [x for x in df.columns if keyword in x]
    keyword_df = df[keyword_list]
    keyword_df.columns = [x.replace(' ','') for x in \
                          [x.split('-')[-1]for x in keyword_df.columns]]
    return keyword_df 
plt.figure()
pd.Series(multi_new['Which best describes \
your undergraduate major? - \
Selected Choice'].value_counts()/multi_new.shape[0]).plot(kind='barh',figsize=(6,8),grid=True)
plt.ylabel('A Look At Degrees')

plt.show()
langs = split_cols(multi_new,'languages')
frames = split_cols(multi_new,'frameworks')

frames.columns = [x if 'Learn' not in x else 'Scikit-Learn' for x in frames.columns]

fig,ax = plt.subplots(1,2,figsize=(16,6))

langs.count()[:-1].plot(kind='bar',ax=ax[1],title="What do they speak?")

frames.count()[:-1].plot(kind='bar',ax=ax[0],title="What do they use?")
plt.tight_layout()
plt.show()
multi_new['What is your gender? - Selected Choice'].value_counts()\
.plot(kind='pie',figsize=(7,7),title="What are our genders?")
plt.ylabel('')
plt.tight_layout()
plt.show()
multi_new['Select the title most similar \
to your current role (or most recent title if retired): - Selected Choice'].value_counts().plot(kind='barh',
figsize=(8,8),title="What did we do?")

plt.show()
dbs = split_cols(multi_new,'database')
dbs.count()[:-1].plot(kind="barh",figsize=(6,8),title="Where we store all that good data!")
plt.show()
dtypes = split_cols(multi_new,'interact')
dtypes.drop(dtypes.columns[-2],1,inplace=True)
dtypes.count()[:-2].plot(kind='Pie',figsize=(8,8),title="What we look at")
plt.ylabel("")
plt.tight_layout()
plt.show()