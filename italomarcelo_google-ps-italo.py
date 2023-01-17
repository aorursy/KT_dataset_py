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
dfa = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

dfu = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv')
def sepColumns(dataset):

    num = []

    cat = []

    for i in dataset.columns:

        if dataset[i].dtype == 'object':

            cat.append(i)

        else:

            num.append(i)

    return num, cat
num, cat = sepColumns(dfa)

num
cat
def edaFromData(dfA, allEDA=False, desc='Exploratory Data Analysis'):

    print(desc)

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nIs Null:\n{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if allEDA:  # here you put yours prefered analysis that detail more your dataset

        

        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Categoricals')

        print(dfA.describe(include=['object']))

        

        #print(df.loc[:, df.dtypes=='object'].columns)

        print(f'\nHead:\n{dfA.head()}')

        print(f'\nSamples:\n{dfA.sample(2)}')

        print(f'\nTail:\n{dfA.tail()}')
edaFromData(dfa[num])
dfa[num] = dfa[num].fillna(0)

edaFromData(dfa[num])
edaFromData(dfa[cat])
dfa['Android Ver'].sample(3)

dfa['Current Ver'].sample(3)
dfa['Content Rating'].sample(3)
dfa['Type'].sample(3)
dfa[cat] = dfa[cat].fillna('unknow')
edaFromData(dfa[cat])
dfa.sample(2)
dfa['App'] = dfa['App'].str.lower()
dfa.sample(2)
dfa.sample(2)
app = dfa.App.unique().tolist()

dfa['AppN'] = dfa['App'].apply(lambda x: app.index(x))

dfa.sample(2)
category = dfa.Category.unique().tolist()

category
dfa[dfa.Category == '1.9']

dfa
# row with wrong data

dfa = dfa.drop(index=10472)
dfa['CategoryN'] = dfa['Category'].apply(lambda x: category.index(x))

dfa.sample(2)
dfa['Size'].unique()
dfa['SizeN'] = dfa.Size

dfa['SizeN'] = dfa['SizeN'].apply(lambda x: x.replace('Varies with device', '0') )

dfa['SizeN'] = dfa['SizeN'].apply(lambda x: x.replace('M', '000000') )

dfa['SizeN'] = dfa['SizeN'].apply(lambda x: x.replace('k', '000') )

dfa['SizeN'] = dfa['SizeN'].apply(lambda x: x.replace('+', '') )

dfa['SizeN'] = dfa['SizeN'].apply(lambda x: x.replace(',', '') )

dfa['SizeN'] = dfa['SizeN'].apply(pd.to_numeric)
dfa.SizeN.sample(5)
dfa['InstallsN'] = dfa['Installs'].apply(lambda x: x.replace('+', '') )

dfa['InstallsN'] = dfa['InstallsN'].apply(lambda x: x.replace(',', '') )
dfa[['Installs', 'InstallsN']]
dfa['InstallsN'] = dfa['InstallsN'].astype(str).astype(int)
types = dfa.Type.unique().tolist()

types
dfa.Type = dfa['Type'].apply(lambda x: 'unknow' if x == '0' else x)
types = dfa.Type.unique().tolist()

types
dfa['TypeN'] = dfa['Type'].apply(lambda x: types.index(x))

dfa.sample(2)
dfa['Content Rating'].unique()
content = dfa['Content Rating'].unique().tolist()

dfa['ContentN'] = dfa['Content Rating'].apply(lambda x: content.index(x))

dfa.sample(2)
genres = dfa['Genres'].unique().tolist()

dfa['GenresN'] = dfa['Genres'].apply(lambda x: genres.index(x))

dfa.sample(2)
from datetime import datetime

# s = "November 19 2019, 12:00 AM"

# d = datetime.strptime(s, "%B %d %Y, %I:%M %p")

# print(d.isoformat())



def convertDate(s):

    try:

        d = datetime.strptime(s, "%B %d, %Y")

        d = d.isoformat()[:10]

    except:

        d = '2000-01-01'

    return d
convertDate('July 4, 2018')
dfa['Date'] = dfa['Last Updated'].apply(lambda x: convertDate(x))
dfa[['Date', 'Last Updated']].sample(2)
dfa[dfa.Date == '2000-01-01'][['Date', 'Last Updated']]
def splitDate(d, s='year'):

    try:

        if s == 'year':

            r = d.split('-')[0]

        elif s == 'month':

            r = d.split('-')[1]

        elif s == 'weekday':

            r = pd.Timestamp(d).weekday()

    except:

        r=''

    return r
# date = splitDate(dfa.Date.sample().values[0], 'month')

# date

dfa.Date.apply(lambda x: splitDate(x, 'month'))
dfa['Year'] = dfa['Date'].apply(lambda x: splitDate(x, 'year'))

dfa['Month'] = dfa['Date'].apply(lambda x: splitDate(x, 'month'))

dfa['WeekDay'] = dfa['Date'].apply(lambda x: splitDate(x, 'weekday'))
dfa['Year'] = dfa['Year'].apply(pd.to_numeric)

dfa['Month'] = dfa['Month'].apply(pd.to_numeric)

dfa['WeekDay'] = dfa['WeekDay'].apply(pd.to_numeric)
dfa.sample(2)
cat
dfa['Current Ver'].unique()
currentVer = dfa['Current Ver'].unique().tolist()

dfa['Current VerN'] = dfa['Current Ver'].apply(lambda x: currentVer.index(x))

dfa.sample(2)
androidVer = dfa['Android Ver'].unique().tolist()

dfa['Android VerN'] = dfa['Android Ver'].apply(lambda x: androidVer.index(x))

dfa.sample(2)
num, cat = sepColumns(dfa)

num
import seaborn as sns

import matplotlib.pyplot as plt

def snsBar(df,x,y,title='Comparing',figsize=(7,4), rotation=0):

    plt.figure(figsize=figsize)

    g=sns.barplot(x=x, y=y, data=df)

    ax=g

    ax.set_title(title)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

    for p in ax.patches:

                 ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),

                     ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),

                     textcoords='offset points')
cat3 = dfa[['InstallsN', 'Category']].groupby('Category').sum()

# cat3.CategoryN = cat3.CategoryN.apply(lambda x: round(x / 1000000, 2))

cat3 = cat3.sort_values(by='InstallsN', ascending=False).head(3)

cat3
snsBar(df=cat3, x=cat3.index, y='InstallsN', 

       title='Top 3 Categories - (bi installs)')
maxSize = dfa.SizeN.max()

f'maxSize is {maxSize}'
app3 = dfa[dfa.SizeN == dfa.SizeN.max()][['App', 'Size', 'SizeN']].sort_values(by='SizeN', ascending=False)

app3.SizeN = app3.SizeN.apply(lambda x: round(x / 1000000, 2))

app3
snsBar(df=app3, x='App', y='SizeN', 

       title='Apps Size- (MBytes)', figsize=(15,5), rotation=90)
appsOld = dfa[['App', 'Date']].sort_values(by='Date').head()
from datetime import date

def diffDates(y):

    y = y.split('-')

    x = date.today() - date(int(y[0]), int(y[1]), int(y[2]))

    return x
appsOld['days'] = appsOld.Date.apply(lambda x: diffDates(x).days)

appsOld
snsBar(df=appsOld, x='App', y='days', 

       title='Apps Olders', figsize=(7,5), rotation=75)
inst3 = dfa[['InstallsN','App']].groupby('App').sum().sort_values(

    by='InstallsN', ascending=False).head(3)

inst3
snsBar(df=inst3, x=inst3.index, y='InstallsN', 

       title='Top 3 Apps - (bi installs)')
years = dfa[(dfa.Year >=2016) & (dfa.Year <= 2018)][['InstallsN', 'Category']]

years = years.groupby('Category').sum()

years = years.sort_values(by='InstallsN', ascending=False).head(3)

years
snsBar(df=years, x=years.index, y='InstallsN', 

       title='Top 3 Apps - (16 to 18)')
best = dfa[dfa.Rating == dfa.Rating.max()][['App', 'Rating', 'InstallsN']].sort_values(by='Rating', ascending=False)

# bestr.SizeN = bestr.Rating.apply(lambda x: round(x / 1000000, 2))

best
besti = best[['App', 'InstallsN']].groupby('App').sum().sort_values(

    by='InstallsN', ascending=False).head(10)

besti
snsBar(df=besti, x=besti.index, y='InstallsN', 

       title='Best Rating', figsize=(15, 5), rotation=90)