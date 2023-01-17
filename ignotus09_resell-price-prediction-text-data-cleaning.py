# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import random

import string

from nltk.corpus import stopwords

import re

from stemming.porter2 import stem
# Reading a small sample of the entire training file, to quickly execute the data cleaning steps. We would later read the full data set when we do the training

# p = 0.01 # 1% of the data set

# df = pd.read_csv('../input/guess-my-price/train.tsv',delimiter = '\t',encoding = 'utf-8',skiprows = lambda i: i > 0 and random.random() > p)

df1 = pd.read_csv('../input/guess-my-price/train.tsv',delimiter = '\t',encoding = 'utf-8')
df1.head(5)
# Dropping random:

df1 = df1.drop(columns = ['random'])
# Missing summary function creates a summary of the number of missing values in all the columns of a dataframe

def missing_summary(df):

    columns = df.columns

    missing_dict = {}

    for cols in columns:

        missing_dict[cols] = df[cols].isna().sum()

    

    summ_missing = pd.DataFrame(list(missing_dict.items()), columns=['column_name', '#missing_values'])

    return summ_missing



summ_missing = missing_summary(df1)

summ_missing
# Dropping the rows where item_description is misssing from the data frame

df1.dropna(subset = ['item_description'],inplace = True)



# Imputing text "missing" in the category_name and brand_name column

df1.fillna("missing", inplace = True)



missing_summary(df1)
# selecting 10% of the data so that the trial cleaning steps are faster

df_samp = df1.sample(frac = 0.1)
# Selecting all the string columns from the sample data frame

df_samp_string = df_samp[['train_id','name','category_name','brand_name','item_description']]

df_samp_string.head(4)
# Removing emojis from the text columns



def deEmojify(inputString):

    return inputString.encode('ascii', 'ignore').decode('ascii')



df_samp_string['item_description_new'] = df_samp_string['item_description'].apply(deEmojify)

df_samp_string['category_name_new'] = df_samp_string['category_name'].apply(deEmojify)



df_samp_string.head(5)
string.punctuation
# Writing a function to replace punctuation and special characters and converting the string to lower character



def remove_punctuation(text):

    for punctuation in string.punctuation:

        text = text.replace(punctuation,'')

    return text



column =  df_samp_string.columns



for cols in column:

    if df_samp_string[cols].dtype == 'object':

        df_samp_string[cols] = df_samp_string[cols].apply(remove_punctuation).str.lower()

        

#Viewing the data frame        

df_samp_string.head(5)
# Removing stop words

stop = stopwords.words('english')

df_samp_string['item_description_new'] = df_samp_string['item_description'].str.split().apply(lambda x: [item for item in x if item not in stop])

df_samp_string['category_name_new'] = df_samp_string['category_name'].str.split().apply(lambda x: [item for item in x if item not in stop])



#Viewing the data frame        

df_samp_string.head(5)
# Code to remove all digits from a list of string 



def rem_num(list): 

	pattern = '[0-9]'

	list = [re.sub(pattern, '', i) for i in list] 

	return list



# Driver code 

df_samp_string['item_description_new'] = df_samp_string['item_description_new'].apply(rem_num)

df_samp_string['category_name_new'] = df_samp_string['category_name_new'].apply(rem_num)



#Viewing the data frame        

df_samp_string.head(5)
# Code to stem words

df_samp_string['item_description_new'] = df_samp_string['item_description_new'].apply(lambda words: [stem(w) for w in words])

df_samp_string['category_name_new'] = df_samp_string['category_name_new'].apply(lambda words: [stem(w) for w in words])



#Viewing the data frame        

df_samp_string.head(5)
# Combine the entries in item_description_new and category_name_new into one string

df_samp_string['item_description_clean'] = df_samp_string['item_description_new'].str.join(" ")

df_samp_string['category_name_clean'] = df_samp_string['category_name_new'].str.join(" ")



#Viewing the data frame

df_samp_string.head(5)