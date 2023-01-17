# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  ## plot library

%matplotlib inline

from sklearn import preprocessing  ## preprocessing form sklearn to deal with type object

import seaborn as sns  #import seaborn for correlation matrix

import xgboost as xgb #import xgboost to train missing values





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../Basic_income"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## Not simplified for/against

df=pd.read_csv("../input/basic_income_dataset_dalia.csv",encoding ='utf-8')

###Simplify data. Simplify column names

#rename the columns to be less wordy

df.rename(columns = {'question_bbi_2016wave4_basicincome_awareness':'awareness',

            'question_bbi_2016wave4_basicincome_vote':'vote',

            'question_bbi_2016wave4_basicincome_effect':'effect',

            'question_bbi_2016wave4_basicincome_argumentsfor':'arg_for',

            'question_bbi_2016wave4_basicincome_argumentsagainst':'arg_against',

                    'dem_full_time_job':'employed',

                     'dem_education_level':'education'



},

           inplace = True)

##simplify with only 3 categories

#df.vote.replace('I would probably vote for it','I would vote for it', inplace=True)



#df.vote.replace('I would probably vote against it','I would vote against it', inplace=True)
## Plotting basic income awareness wether employed

sub_df = df.groupby('employed')['awareness'].value_counts(normalize=True).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[1],cols[0],cols[2],cols[3]]]



sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income awareness wether employed')
## Plotting basic income awareness depending on education

sub_df = df.groupby('education')['awareness'].value_counts(normalize=True).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[1],cols[0],cols[2],cols[3]]]



sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income awareness depending on education')
## Plotting basic income awareness by education and job situation

sub_df = df.groupby(['education','employed'])['awareness'].value_counts(normalize=True).unstack()

##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[1],cols[0],cols[2],cols[3]]]

##reorder bars inplot to see less aware people first 

sub_df.sort_values(cols[1],inplace=True,ascending =False)



sub_df.plot(kind='bar',stacked=True,colormap='Spectral', title='Basic income awareness depending on education and employment')

sub_df.head()
## Plotting basic income awareness by education and job situation

sub_df = df.groupby(['education','employed'])['awareness'].value_counts(normalize=False).unstack()

##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[1],cols[0],cols[2],cols[3]]]

##reorder bars inplot to see less aware people first 

sub_df.sort_values(cols[1],inplace=True,ascending =False)



sub_df.plot(kind='bar',stacked=True,colormap='Spectral', title='Basic income awareness depending on education and employment')

sub_df.head()
## Plotting basic income vote by job situation

sub_df = df.groupby('employed')['vote'].value_counts(normalize=True).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote wether employed')

sub_df.head()
## Plotting basic income vote by education

sub_df = df.groupby('education')['vote'].value_counts(normalize=True).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see less aware people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education')

sub_df.head()
## Plotting basic income vote by education and job situation

sub_df = df.groupby(['education','employed'])['vote'].value_counts(normalize=True).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education and job situation')

sub_df.head()
## Plotting basic income vote by education and job situation

sub_df = df.groupby(['education','employed'])['vote'].value_counts(normalize=False).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education and job situation')

sub_df.head()
## Plotting basic income vote by education and job situation+ age group

sub_df = df[df['education']=='high'].groupby(['education','employed','age_group'])['vote'].value_counts(normalize=True).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[3],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote of high educated by age group')

sub_df.head()
## Plotting basic income vote in Czechia

sub_df = df[df['country_code']=='CZ'].groupby(['education','employed'])['vote'].value_counts(normalize=False).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education and job situation in Czechia')

sub_df.head()
## Plotting basic income vote in Greece

sub_df = df[df['country_code']=='GR'].groupby(['education','employed'])['vote'].value_counts(normalize=False).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education and job situation in Greece')

sub_df.head()
## Plotting basic income vote in Italy

sub_df = df[df['country_code']=='IT'].groupby(['education','employed'])['vote'].value_counts(normalize=False).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education and job situation in Italy')

sub_df.head()
## Plotting basic income vote in Austria

sub_df = df[df['country_code']=='AT'].groupby(['education','employed'])['vote'].value_counts(normalize=False).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education and job situation in Austria')

sub_df.head()
## Plotting basic income vote in Finland

sub_df = df[df['country_code']=='FI'].groupby(['education','employed'])['vote'].value_counts(normalize=False).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education and job situation in Finland')

sub_df.head()
## Plotting basic income vote in Denmark

sub_df = df[df['country_code']=='DK'].groupby(['education','employed'])['vote'].value_counts(normalize=False).unstack()



##reorder columns

cols = list(sub_df.columns.values)

sub_df = sub_df[[cols[0],cols[3],cols[1],cols[2],cols[4]]]



##reorder bars inplot to see non votiong people first 

sub_df.sort_values(cols[0],inplace=True,ascending =False)





sub_df.plot(kind='bar',stacked=True,colormap='Spectral',sort_columns=True, title='Basic income vote by education and job situation in Denmark')

sub_df.head()