# Import basic libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
# List files available
print(os.listdir("../input/"))
# Installing and loading the library
!pip install dabl

import dabl
titanic_df = pd.read_csv('../input/titanic/train.csv')

# A first look at data
titanic_df.shape
titanic_df.head()
titanic_df_clean = dabl.clean(titanic_df, verbose=1)

types = dabl.detect_types(titanic_df_clean)
print(types) 
dabl.plot(titanic_df, target_col="Survived")
ec = dabl.SimpleClassifier(random_state=0).fit(titanic_df, target_col="Survived") 
# installation and importing the library
!pip install missingno
import missingno as msno
# Let's check out the missing values first with the train.info() method
titanic_df.info()
msno.matrix(titanic_df)
msno.matrix(titanic_df.sample(50))
msno.bar(titanic_df)
msno.heatmap(titanic_df)
# installation and importing the library
!pip install emot
import emot
text = "The weather is ☁️, we might need to carry our ☂️ :("
emot.emoji(text)
emot.emoticons(text)
# installation and importing the library
!pip install flashtext
from flashtext import KeywordProcessor
twitter_df =  pd.read_csv('../input/nlp-getting-started/train.csv')
twitter_df.head()
corpus = ', '.join(twitter_df.text)
corpus[:1000]
# How many times does the word 'flood' appear in the corpus?
processor = KeywordProcessor()
processor.add_keyword('flood')
found = processor.extract_keywords(corpus)
print(len(found))
  
# Replacing all occurences of word 'forest fire'(case insensitive) with fire

processor = KeywordProcessor(case_sensitive = False)
processor.add_keyword('forest fire','fire')
found = processor.replace_keywords(corpus)
print(found[:100])
# installing and importing the library

!pip install pyflux

import pyflux as pf
maruti = pd.read_csv("../input/nifty50-stock-market-data/MARUTI.csv")
# Convert string to datetime64
maruti ['Date'] = maruti ['Date'].apply(pd.to_datetime)

maruti_df = maruti[['Date','VWAP']]

#Set Date column as the index column.
maruti_df.set_index('Date', inplace=True)
maruti_df.head()

plt.figure(figsize=(15, 5))
plt.ylabel("Volume Weighted Average Price'")
plt.plot(maruti_df)


my_model = pf.ARIMA(data=maruti_df, ar=4, ma=4, family=pf.Normal())
print(my_model.latent_variables)

result = my_model.fit("MLE")
result.summary()

my_model.plot_z(figsize=(15,5))
my_model.plot_fit(figsize=(15,10))
my_model.plot_predict_is(h=50, figsize=(15,5))
my_model.plot_predict(h=20,past_values=20,figsize=(15,5))
# installing and importing the library

#!pip install --upgrade bamboolib>=1.2.1
# Importing the necessary libraries 
#import bamboolib as bam
#bam.enable()


#Importing the training dataset
#df = pd.read_csv('../input/titanic/train.csv')
#df
# Installing the library
!pip install autoviz
# Instantiate the library
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
# Reading the dataset
house_price = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
house_price.head(3)
sep = '\,'
target = 'SalePrice'
datapath = '../input/house-prices-advanced-regression-techniques/'
filename = 'train.csv'
df = pd.read_csv(datapath+filename,sep=sep,index_col=None)
df = df.sample(frac=1.0,random_state=42)
print(df.shape)
df.head()

dft = AV.AutoViz(datapath+filename, sep=sep, depVar=target, dfte=None, header=0, verbose=0,
                            lowess=False,chart_format='svg',max_rows_analyzed=1500,max_cols_analyzed=30)
!pip install numerizer
from numerizer import numerize
numerize('forty two')

numerize('forty-two')
numerize('four hundred and sixty two')
numerize('twenty one thousand four hundred and seventy three')
numerize('one billion and one')
numerize('nine and three quarters')
numerize('platform nine and three quarters')
!pip install ppscore

import ppscore as pps

def heatmap(df):
    ax = sns.heatmap(df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    ax.set_title('PPS matrix')
    ax.set_xlabel('feature')
    ax.set_ylabel('target')
    return ax


def corr_heatmap(df):
    ax = sns.heatmap(df, vmin=-1, vmax=1, cmap="BrBG", linewidths=0.5, annot=True)
    ax.set_title('Correlation matrix')
    return ax
titanic_df_subset = titanic_df[["Survived", "Pclass", "Sex", "Age", "Ticket", "Fare", "Embarked"]]
pps.score(titanic_df_subset, "Sex", "Survived")

matrix = pps.matrix(titanic_df_subset)
heatmap(matrix)
# Correlation Matrix
f = plt.figure(figsize=(16,8))
f.add_subplot(1,2, 1)
corr_heatmap(titanic_df_subset.corr())

f.add_subplot(1,2, 2)
matrix = pps.matrix(titanic_df_subset)
heatmap(matrix)