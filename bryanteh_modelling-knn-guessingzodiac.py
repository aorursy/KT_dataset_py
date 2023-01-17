# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

profiles = pd.read_csv('/kaggle/input/profiles.csv')
pd.set_option('display.max_columns', None)

display(profiles.head(5))
print('Total rows:',profiles.shape[0])

print('Total columns:',profiles.shape[1])

# print('breakdown of column types:')

# print(profiles.dtypes.value_counts())

print(len(profiles.select_dtypes('int64').columns),'numerical columns(integer):',list(profiles.select_dtypes('int64').columns))

print(len(profiles.select_dtypes('float64').columns),'numerical columns(float):',list(profiles.select_dtypes('float64').columns))

print(len(profiles.select_dtypes('object').columns),'categorical columns:',list(profiles.select_dtypes('object').columns))
#inspect essay questions

meaning = dict({'essay0' : 'My self summary',

'essay1' : 'What I’m doing with my life',

'essay2' : 'I’m really good at',

'essay3' : 'The first thing people usually notice about me',

'essay4' : 'Favorite books, movies, show, music, and food',

'essay5' : 'The six things I could never do without',

'essay6' : 'I spend a lot of time thinking about',

'essay7' : 'On a typical Friday night I am',

'essay8' : 'The most private thing I am willing to admit',

'essay9' : 'You should message me if…'})



objects = profiles.select_dtypes('object').nunique().sort_values(ascending=False).reset_index()

summary = objects[objects['index'].str.contains('essay')]

summary['meaning'] = summary['index'].map(meaning)

summary[0] = summary[0]/len(profiles)

summary[['index','meaning',0]]
#inspect non-essay questions

objects = profiles.select_dtypes('object').nunique().sort_values(ascending=False).reset_index()

non_essay = [i for i in list(objects['index']) if not 'essay' in i]

profiles[non_essay].nunique()

# summary
#numeric data & clean it up

numeric = profiles._get_numeric_data()

numeric.describe() #found income has -1

numeric.income = numeric.income.replace({-1:np.nan})

profiles.income = profiles.income.replace({-1:np.nan})

numeric.height = numeric.height.replace({1:np.nan})

profiles.height = profiles.height.replace({1:np.nan})

numeric.describe() #found income has -1
#plot numeric data

plt.figure(figsize=[26,5])

k=1

for i in list(numeric.columns):

    plt.subplot(1,len(list(numeric.columns)),k)

    plt.hist(numeric[i], bins=40,label=i)

    plt.axvline(x=numeric[i].mean(),color='red',label='mean')

    plt.axvline(x=numeric[i].median(),color='green',label='median')

    plt.xlabel(i)

    plt.ylabel("Frequency")

    plt.legend()

    k+=1

plt.show()
profiles['sign_new'] = profiles.sign.str.split(' ').str.get(0)



sign_dict = dict(

{'leo':0,

'gemini':1,

'libra':2,

'cancer':3,

'virgo':4,

'taurus':5,

'scorpio':6,

'aries':7,

'pisces':8,

'sagittarius':9,

'aquarius':10,

'capricorn':11,})

profiles['sign_num'] = profiles['sign_new'].map(sign_dict)

profiles['sign_num'].value_counts()

x_col = ['sign_num','drinks','smokes','drugs'] + list(objects[objects['index'].str.contains('essay')]['index'])

zodiac = profiles[x_col]

for i in ['drinks','smokes','drugs']:

    print(i,zodiac[i].unique())

    

zodiac.drugs = zodiac.drugs.map({'never':0,'sometimes':1,'often':2})

zodiac.smokes = zodiac.smokes.map({'no':0,'when drinking':1,'sometimes':2,'yes':3,'trying to quit':4})

zodiac.drinks = zodiac.drinks.map({'not at all':0,'rarely':1,'socially':2,'often':3,'very often':4,'desperately':5})



display(zodiac[['drinks','smokes','drugs']].describe())

#plot numeric data

plt.figure(figsize=[26,5])

k=1

for i in ['drinks','smokes','drugs']:

    plt.subplot(1,3,k)

    plt.hist(zodiac[i],label=i)

    plt.axvline(x=zodiac[i].mean(),color='red',label='mean')

    plt.axvline(x=zodiac[i].median(),color='green',label='median')

    plt.xlabel(i)

    plt.ylabel("Frequency")

    plt.legend()

    k+=1

plt.show()
zodiac[list(objects[objects['index'].str.contains('essay')]['index'])]

for i in list(objects[objects['index'].str.contains('essay')]['index']):

    split = zodiac[i].str.split(' ')

    length = [len(i) if type(i)==list else 0 for i in split ]

    zodiac[i+'_length'] = length

length_col = [i for i in list(zodiac.columns) if 'length' in i]

zodiac[length_col].describe()
x = zodiac['sign_num']

y = zodiac['essay7_length']

plt.figure(figsize=[13,7])

sns.boxplot(x,y)
#use knn

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer



zodiac = zodiac.dropna()

y = zodiac.sign_num

x = zodiac.drop(columns=['sign_num']+list(objects[objects['index'].str.contains('essay')]['index']))



def missing_values_table(df):

    mis_val=df.isnull().sum()    

    mis_val_perc=100*df.isnull().sum()/len(df)

    mis_val_table=pd.concat([mis_val, mis_val_perc], axis=1) 

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    print ("Your selected data frame has " + str(df.shape[1]) + " columns.\n"+"There are " + str(mis_val_table_ren_columns.shape[0]) +

 " columns that have missing values.")

    return mis_val_table_ren_columns



miss = missing_values_table(x)

miss.head(5)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 6)



#impute x_train

# imputer = SimpleImputer(missing_values=np.nan, strategy = 'median')

# imputer.fit(x_train)

# x_train = imputer.transform( x_train )

# x_test = imputer.transform (x_test )



#scale 

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)



min_ = 1

max_ = 100

score = []

for i in range(min_,max_):

    classifier = KNeighborsClassifier(i)

    classifier.fit(x_train, y_train)

    score.append(classifier.score(x_test, y_test))

plt.plot(list(range(min_,max_)),score)

plt.show()