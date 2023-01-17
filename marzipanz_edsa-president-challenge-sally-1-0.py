# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Dataframes etc
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib.colors import ListedColormap
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
sns.set_style('whitegrid')

#Machine learning:
from sklearn import preprocessing

## ML Cross validation and metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


## ML models 
from sklearn.linear_model import LogisticRegression

#Natural language processing
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
## to automate the NLP extraction
from sklearn.feature_extraction.text import CountVectorizer

df=pd.read_csv('../input/train.csv')
df.head()

df['president'].unique()
df.info()
#no nulls, nothing unexpected
df.describe()
df=df.drop(columns='speed length')
df['speech length']=df['text'].str.len()
df.head()
df_summary = pd.DataFrame(df.groupby('president')['speech length'].mean())
df_summary
df_summary=df_summary.reset_index()
df_summary = df_summary.sort_values(by='speech length')
sns.barplot(data = df_summary, x='president', y='speech length')


