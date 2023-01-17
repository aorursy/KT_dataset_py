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
FILEPATH = '/kaggle/input/sms-spam-collection-dataset/spam.csv'
df = pd.read_csv(FILEPATH, encoding='iso-8859-1', engine = 'c') # engine 'c' used instead of 'python' for higher performance

df.head(10)
# delete unnecessary cols

cols = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']



df.drop(cols, axis = 1, inplace = True)
df.head()
# Title change v1 = result, v2 = input



df.rename(columns = {'v1' : 'Label','v2':'Message'},inplace=True)



# we can also use df.rename() option here
df.head()
# reorder options - must be applicable for all cols

df = df[['Message','Label']]

 
df.head()
# Buggy, please don't use

new_cols = ['my_input', 'my_result']



df1 = df.reindex(columns = new_cols)

# df1.to_csv(FILEPATH)

# df1.head()
# Rename cols by using .rename - can be used for selected cols



df.rename(columns = {'input' : 'my_new_input', 'result' : 'my_new_result'}, inplace = True)
df.head()
df.count()
# print first string



df.iloc[1][0]
df.iloc[2][0]
def find_message_length(msg):

    

    msg_words = msg.split(' ')

    

    msg_len = len(msg_words)

    

    return msg_len
print(find_message_length(df.iloc[0][0]))
# Create a new col called 'message_word_length' showing how many words in the message

df['input_words_count'] = df['my_new_input'].apply(find_message_length)

df.head()



# ref: https://rajacsp.github.io/mlnotes/python/data-wrangling/advanced-custom-lambda/
# show the unique labels



set(df['my_new_result'])
def find_length(msg):

    

    msg_len = len(msg)

    

    return msg_len
print(find_length(df.iloc[0][0]))
# Create a new col called 'message_word_length' showing how many words in the message

df['input_char_length'] = df['my_new_input'].apply(find_length)

df.head()
# History words count



import matplotlib.pyplot as plt



# to avoid popups use inline

%matplotlib inline 
# plt.hist(data['label'], bins=3, weights=np.ones(len(data['label'])) / len(data['label']))



import numpy as np



plt.hist(df['input_words_count'], bins = 100, weights = np.ones(len(df['input_words_count'])) / len(df['input_words_count']))



plt.xlabel('Word Length')

plt.ylabel('Group Count')

plt.title('Word Length Histogram')
# Find more than 80 words

df['input_words_count']
df_above_80 = df[df['input_words_count'] > 80]
df_above_80
import numpy as np



plt.hist(df['input_char_length'], bins = 100, weights = np.ones(len(df['input_char_length'])) / len(df['input_char_length']))



plt.xlabel('Char Length')

plt.ylabel('Group Count')

plt.title('Char Length Histogram')