# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
#import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

# Venn diagram
from matplotlib_venn import venn2
import re
import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
eng_stopwords = stopwords.words('english')
import gc

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

train_data = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
train_host = train_data["host"].value_counts()
df = pd.DataFrame({'labels': train_host.index,
                   'values': train_host.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of hosts in Training data')
# top 20 hosts with more unique quesi
test_host = test_data["host"].value_counts()
df = pd.DataFrame({'labels': test_host.index,
                   'values': test_host.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of hosts in Testing data')
# We need to determine the length of the sentence to determine the model used.

#Using question_title + question_body + answer, about 75% of the samples meet the length limit of 512, and the other 25% need to be processed by truncate;


# distribution for quesiton title 


train_question_title=train_data['question_title'].str.len()
test_question_title=test_data['question_title'].str.len()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6))
sns.distplot(train_question_title,ax=ax1,color='blue')
sns.distplot(test_question_title,ax=ax2,color='green')
ax2.set_title('Distribution for Question Title in test data')
ax1.set_title('Distribution for Question Title in Training data')
plt.show()
train_data.columns.tolist()

train_question_body=train_data['question_body'].str.len()
test_question_body=test_data['question_body'].str.len()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6))
sns.distplot(train_question_body,ax=ax1,color='blue')
sns.distplot(test_question_body,ax=ax2,color='green')
ax2.set_title('Distribution for body in test data')
ax1.set_title('Distribution for body in Training data')
plt.show()
train_question_answer=train_data['answer'].str.len()
test_question_answer=test_data['answer'].str.len()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6))
sns.distplot(train_question_answer,ax=ax1,color='blue')
sns.distplot(test_question_answer,ax=ax2,color='green')
ax2.set_title('Distribution for Answers in test data')
ax1.set_title('Distribution for Answers in Training data')
plt.show()
train_question_title_body_answer=(train_data['question_title']+train_data['question_body']+train_data['answer']).str.len()
test_question_title_body_answer=(test_data['question_title']+test_data['question_body']+test_data['answer']).str.len()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,6))
sns.distplot(train_question_title_body_answer,ax=ax1,color='blue')
sns.distplot(test_question_title_body_answer,ax=ax2,color='green')
ax2.set_title('Distribution for title_body_answer in test data')
ax1.set_title('Distribution for title_body_answer in Training data')
plt.show()