import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import re
import nltk
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
pd.options.display.max_colwidth = 200
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.listdir('/kaggle/input/nlp-getting-started')
training_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
training_data.keyword = training_data.keyword.astype(str)

testing_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
testing_data.keyword = testing_data.keyword.astype(str)
# training_data.head()
# training_data.shape
# training_data.describe
# training_data.info()

# testing_data.head()
# testing_data.shape
# testing_data.describe
# testing_data.info()
# function to check missing values in the dataset

def check_missing_data(df):
    total = df.isnull().sum()
#     print(total)
    percentage = round(total / df.shape[0] *100)
#     print(percentage)
    missing_data = pd.concat([total, percentage], axis=1, keys= ['Total', 'Percent']).sort_values(by='Percent', ascending = False)
    missing_data = missing_data[missing_data['Total'] > 0]
    return missing_data

# check_missing_data(training_data)
# check_missing_data(testing_data)
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
ps = PorterStemmer()

# function to execute various pre-processing steps

def pre_process_data(row):
    
    # punctuation removal step
    row['text'] = re.sub(r'[^a-zA-Z\s]', '', row['text'], re.I|re.A)
    
    # lower casing step
    row['keyword'] = row['keyword'].lower()
    row['text'] = row['text'].lower()
    
    # tokenization step
    row['text'] = wpt.tokenize(row['text'])
    
    # stop word removal step
    row['text'] = [token for token in row['text'] if token not in stop_words]
    
    # stemming step
    row['text'] = [ps.stem(token) for token in row['text']]
    
    return row
training_data = training_data.apply(pre_process_data, axis=1)
testing_data = testing_data.apply(pre_process_data, axis=1)
