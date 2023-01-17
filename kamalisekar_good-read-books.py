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
#imporing libraries for analysing purpose

import pandas as pd

import numpy as np



#importing for visualization

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#reading the data using pandas

df= pd.read_csv("../input/goodreadsbooks/books.csv",error_bad_lines=False)
#viewing the sample of data to understand

df.head(2)
#chaning the index of data

df.set_index('title',inplace = True)
#to know the number of rows and column

df.shape
#checking the missing values

df.isnull().sum()
#quick summary of the dataset

df.describe(include='all')
#lets get types of attributes

df.dtypes
#let's change all column labels into string type

df.columns = list(map(str, df.columns))



# let's check the column labels types now

all(isinstance(column, str) for column in df.columns)

#arrange the data using text_reviews_count

df.sort_values(['text_reviews_count'],ascending =False, axis =0, inplace= True)

#assigning "text_reviews_count" in a new variable for easy visualizing

df_top10 = df['text_reviews_count'].head(10)

#visualize the highest text_reviews_count

df_top10.plot.barh(figsize=(10,6),color='steelblue',alpha= 1.0)



plt.xlabel('Text reviews count')

plt.ylabel('Title of book')

plt.title('Books with highest text reviews',fontsize= 16,alpha = 1.0)
#check the unique Languages

df['language_code'].unique()
#visualize the language frequency

plt.figure(figsize = (7,5))

sns.countplot(y='language_code',data= df,palette='Blues_d',alpha=0.3)
#column name has a empty space so let us remove it

df.rename(columns={'  num_pages':'num_pages'},inplace=True)
#visualize the relationship between rating and # of pages

sns.relplot(x='average_rating',y='num_pages',data=df)