import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# reading csv file
go = pd.read_csv('../input/googleplaystore.csv').dropna(how='all')

## Google play data frame informations
## How many Columns?
## What does each column has data type?
go.info()

## Omitting White space from object columns
## from left and right
def stripp():
    for i in range(len(go.columns)):
        if(type(go.columns[i])==go.dtypes[0]):
            go.ix[:,i] = go.ix[:,i].str.strip()
            
    print('Done stripping!')
    
stripp()
## Detecting Columns that contains Null values
go.isnull().any()
## how many null values does each column contains?
go.isnull().sum()
## 'Rating' Column has the most null values

## fill 'Rating' column using Linear interpolation
## and Round values to nearest one digit
go.Rating = np.round(go['Rating'].interpolate(method='linear'),1)

## Set Command and Conquer application type as Free
go.loc[go['Type'].isnull(),'Type'] = 'Free'
## Detecting Null values in 'Content Rating'
## Delete Unkonw details for application number 10472
go[go['Content Rating'].isnull()]
go.drop(10472,inplace=True)
## filling Current and android Version columns forward and backward
go.iloc[:,len(go.columns)-1].fillna(method='pad',inplace=True)
go.iloc[:,len(go.columns)-2].fillna(method='bfill',inplace=True)

## Fixing our Application title
go['App'] = go.App.str.title()

## Apply some Textual processing
## Casting into 'category'
def cat():
    go['Category'] = go['Category'].str.replace('_',' ')
    go['Category'] = go['Category'].str.title()
    go['Category'] = go['Category'].astype('category')
    print('Number of categories we have in Play Store: ',len(go['Category'].unique()),'\n')
    print('Each Category has the following number of Applictions:\n')
    print(go['Category'].value_counts())


cat()
## Plotting Play Store Rating Distribution
def rate_dist():
    print("The highest Rate was for an App called: '".title(),go['App'][go['Rating'].max()],"'")
    print('Category:',go['Category'][go['Rating'].max()])
    print('Size: ',go['Size'][go['Rating'].max()])
    fig = plt.figure(figsize=(20,10))
    plt.style.use('ggplot')
    plt.hist(go['Rating'],color='#0EC090')
    plt.title('Play Store App\'s')
    plt.xlabel('Rating')
    plt.ylabel('Appliactions')

rate_dist()
## casting 'Reviews' column to integers
go['Reviews'] = go['Reviews'].astype('int')


## extracting applications with fixed size and Varied ones
siz = go['Size'].where(go['Size'] == 'Varies with device','Fixed Size')
size_for_categ = pd.concat([go['Category'],siz],axis=1)
sizee = pd.crosstab(index=size_for_categ.iloc[:,0],columns=size_for_categ.iloc[:,1])
sizee

## Play Store application Size overview
## How many application varies with device?
## and what about others that came with fixed size?
## what is the percentage for each one?
def sizz():
    print('Fixed Size Applications in Play Store:',sizee.iloc[:,0].sum())
    print('Play Store Applications that Varies with device: ',sizee.iloc[:,1].sum())
    %matplotlib inline
    plt.figure(figsize=(10,10))
    plt.pie(sizee.sum(),colors=['#0EC090','#C00E39'],labels=['Fixed','Varies with device'],autopct='%1.2f%%')
    plt.title('Application Size')
    plt.figure(figsize=(20,25)) 
    sizee.iloc[:,0].plot.barh(width=0.9,color=sns.color_palette('Reds',len(sizee)))
    plt.style.use('ggplot')
    plt.xlabel('Applications')
    plt.title('Play Store Fixed Size App\'s')
    
    
    plt.figure(figsize=(20,25))
    sizee.iloc[:,1].plot.barh(width=0.9,color=sns.color_palette('Greens',len(sizee)))
    plt.xlabel('Applications')
    plt.title('Play Store Applications (Varies with device)')
    
sizz()
## How many Free aplication do we have in play store?
## what about paid ones
def typee():
    go['Type'] = go['Type'].astype('category')
    print('Play Store Application\'s Type:')
    print(go['Type'].value_counts())
    plt.figure(figsize=(10,10))
    go['Type'].value_counts().plot.bar(color=['#0EC090','#C00E39'])
    plt.xlabel('Type')
    plt.ylabel('Number of Applications')
      
typee()
## Google Content Rating
## Top 4 Audience
def aud():
    con = go['Content Rating'].value_counts()
    mask = go['Content Rating'].value_counts()> 4
    plt.figure(figsize=(15,10))
    plt.style.use('ggplot')
    con[mask].plot(kind='bar')

aud()
## Applications' Genres
## What is the avalilabel Genres?
## How many application does each genres have in play store?
## Top 5 Genres
def refine(string):
    if(string.__contains__(';')==True):
        i = string.find(';')
        s = string[0:i]
        return s
    else:
        return string
    
def gen():
    g = go['Genres'].apply(refine)
    print('Total Genres in Play Store: ',g.nunique())
    print('Top 5 Genres with the most Appliactions in Play Store:\n')
    print(g.value_counts().nlargest(5))
    
gen()