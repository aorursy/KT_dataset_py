import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

%matplotlib inline

import seaborn as sns 

import datetime

from scipy.sparse import csr_matrix, hstack

from scipy.stats import probplot

import re

from sklearn.preprocessing import LabelEncoder

from category_encoders import TargetEncoder

from sklearn.base import BaseEstimator, TransformerMixin



color = sns.color_palette()

sns.set_style("whitegrid")

sns.set_context("paper")

sns.palplot(color)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/adams-df/ADAMS_NLPtask_SS20/Train.csv')

test = pd.read_csv('../input/adams-df/ADAMS_NLPtask_SS20/Test.csv')
train.head()
test.head()
train.dtypes
test.dtypes
#uncomment to see, removed for better readability

#print(train.title[12], "///", test.Header[4])
#uncomment to see, removed for better readability

#print(train.text[9], "///", test.Text[4])#
#variables with int / float dtype are not exponentially big or small, their size can be reduced

num_vars = train.select_dtypes(include=np.float64).columns

train[num_vars] = train[num_vars].astype(np.float32)



num_vars = train.select_dtypes(include=np.int64).columns

train[num_vars] = train[num_vars].astype(np.int32)
#compare entries for duplicates,

#uncomment to see, removed for better readability



#train.text[279572]

#train.text[279573]
#compare entries for duplicates

print(

    train.title[279574],

    train.title[279575],

    train.scrappedDate[279572],

    train.scrappedDate[279576])
#drop duplicates

train = train.drop_duplicates(subset =['postId','title', 'url'], keep='last')

train.reset_index(drop=True, inplace = True)
#create year variable for easy comparison

train['createdDate'] = pd.to_datetime(train['createdDate'])

year = []

for num in train.createdDate:

    year.append(num.year)

    

train['year'] = year



train.year.value_counts()
train.language.value_counts()
#drop articles from before 2015, because they don't have enough entries

train = train[train.year >= 2017]



#drop languages except for english

train = train[train.language == 'en']



#reset indexing

train.reset_index(drop=True, inplace = True)
#make a new column for year in test_df, manual one-hot-encoding 

year_2017 = []

for num in test.PublicationDetails:

    if num.find('2017') >= 0 :

        year_2017.append(1)

    else: 

        year_2017.append(0)

        

year_2018 = []

for num in test.PublicationDetails:

    if num.find('2018') >= 0 :

        year_2018.append(1)

    else: 

        year_2018.append(0)

        

#add column to df

test['year_2017'] = year_2017

test['year_2018'] = year_2018



del year_2017

del year_2018

test.year_2017.value_counts()

test.year_2018.value_counts()
year_2018 = []

year_2017 = []

counter = 0



#encode train.year to the same format as test.year

for i in train['year']:

    if i == 2018:

        year_2018.append(1)

        year_2017.append(0)

    else:

        year_2018.append(0)

        year_2017.append(1)

    counter += 1

    

train['year_2017'] = year_2017

train['year_2018'] = year_2018
int_list = []

test['Responses'] = test['Responses'].astype(str)

for i in test['Responses']:

    if i != 'nan':

        int_list.append(re.findall(r'\d+', i))

    else:

        #int_list.append([np.nan])

        int_list.append([0])
#Thank you to 'Alex Martelli' [2]

flat_list = []

double = False

for sublist in int_list:

    double = False

    for item in sublist:

        if len(sublist)== 1:

            flat_list.append(item)

        else:

            if double == True:

                flat_list.append(item)

            else:

                double = True

            
test['Responses'] = flat_list

test['Responses'] = test['Responses'].astype(float)
#html cleaner still doesn't work 100%



def cleaning_func(text):

    #remove line breaks

    text = text.replace('\n', ' ').replace('\r', '')

    

    #remove urls

    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #[3]#



    #remove html tags

    cleanr = re.compile('<.*?>')

    text = re.sub(cleanr, '', text)

    return text
test['Text'] = test.Text.apply(cleaning_func) 
train['text'] = train.text.apply(cleaning_func) 
train.totalClapCount.describe()
train.boxplot(column = 'totalClapCount');

plt.title('Target variable')
sns.distplot(train['totalClapCount'], hist = False, kde = True, rug = True, norm_hist = True,

             color = 'darkblue', 

             kde_kws={'linewidth': 1},

             rug_kws={'color': 'black'})
#log transformed target variable

sns.distplot(np.log1p(train['totalClapCount']), hist = False, kde = True, rug = True, norm_hist = True,

             color = 'darkblue', 

             kde_kws={'linewidth': 2},

             rug_kws={'color': 'black'}) 
#map target variable to reduce noise 

myList = [0, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

clap = []

for i in train['totalClapCount']:

    if i < 2200:

        clap.append(min(myList, key=lambda x:abs(x-i)))

    else:

        clap.append(i)

        

train['clap'] = clap
train['clap_log'] = np.log1p(train['totalClapCount'])

train['recommends_log'] = np.log1p(train['recommends'])

train['wordCount_log'] = np.log1p(train['wordCount'])



clap_len_scatter = sns.scatterplot(x = "clap_log", y = "wordCount_log",          

                                      data = train)
clap_reco_scatter = sns.scatterplot(x = "clap_log", y = "recommends_log",          

                                      data = train)
train.author.value_counts()
test.info()
train['author'] = train['author'].astype('category')

test.rename(columns={'Author':'author'}, inplace=True)

test['author'] = test['author'].astype('category')
#te = TargetEncoder()

#X_target_encoded = te.fit(train['author'], train['totalClapCount'], handle_missing='return_nan', handle_unknown='return_nan')



#substitute unknown with mean

#test['author'] = X_target_encoded.transform(test['author'], y=None, override_return_df=False)



#substitute unknown with nAn

#test['author'] = X_target_encoded.transform(test['author'], y=None, override_return_df=False, )



#test['author'].value_counts()



#te = TargetEncoder()

#train['author'] = te.fit_transform(train['author'], train['totalClapCount'])
train.to_csv('train_processed.csv',index=False)
test.to_csv('test_processed.csv',index=False)