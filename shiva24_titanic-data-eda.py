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
#---- Read all the required packages ---# 

import pandas as pd 

import numpy as np 

import seaborn as sns 

import matplotlib.pyplot as plt 

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
test_data.head()
train_data.dtypes
# --- let us check for null / nas in each column ---# 

train_data.notnull().agg(['sum', 'mean']).style
#--- now let us look at the distribution of Age variable to understand it better ---# 

sns.boxplot(train_data.Age)
train_data.Age.describe()
#--- distribution of survived variable in train data ---# 

train_data['Survived'].value_counts(normalize = True)
cols_to_check = ['Pclass', 'Sex', 'Cabin', 'Embarked'] 
fig, axes = plt.subplots(2, 2, figsize=(10, 10))



for i, ax in enumerate(fig.axes):

    if i < len(cols_to_check):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

        sns.countplot(x=train_data[cols_to_check[i]], alpha=0.7, data=train_data, ax=ax, hue = 'Survived')



fig.tight_layout()
train_data['age_bucket'] = pd.cut(train_data.Age, 10)
fig, ax = plt.subplots(figsize=(30, 10))

sns.countplot(x = train_data.age_bucket, data = train_data, hue = 'Survived', ax = ax)
sns.boxplot(x= 'Survived', y = 'Fare', data = train_data)
from scipy.stats import ttest_ind
t1 = train_data.loc[train_data['Survived'] == 1, 'Fare']

t2 = train_data.loc[train_data['Survived'] == 0 , 'Fare']
ttest_ind(t1, t2)
# -- Now let us play with the textual descriptions - that are names in the data ----# 

train_data.Name.tail()
import re 

import string

def clean_text(text):

    text = text.lower()

    chars = re.escape(string.punctuation)

    text = re.sub(r'['+chars+']', '',text)

    text = re.sub(r'[^\x00-\x7F]+','', text)

    #text = re.sub(r' ', '_', text)

    

    #text = text.replace("  ", " ") 

    

    

    return text
clean_text('Braund, Mr. Owen Harris')
train_data['name_cleaned'] = train_data['Name'].apply(lambda x : clean_text(x))
check_list= ['mr', 'mrs', 'ms', 'miss','mister', 'master' ]
def common_word(text1, check_list) : 

    temp = text1.split(" ")

    common_word = list(set(temp) & set(check_list))

    if len(common_word ) > 0 :

        return(' '.join(map(str, common_word)))

            

    else : 

        return('NA')

        
common_word('braund mr owen harris', check_list)
train_data['name_designation'] = train_data['name_cleaned'].apply(lambda x : common_word(x, check_list))
sns.countplot(x = 'name_designation', data = train_data, hue = 'Survived')