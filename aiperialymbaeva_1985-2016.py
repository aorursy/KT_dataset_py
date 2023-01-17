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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split

%matplotlib inline
suicide_df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
suicide_df.head()
p_null= (len(suicide_df) - suicide_df.count())*100.0/len(suicide_df)

p_null
train = suicide_df[['year','suicides_no','sex','population']]

suicide_df.isnull().any() 
train['sex'].fillna('female', inplace = True)

train.isnull().any()
train['sex'].interpolate(inplace = True)

train.isnull().any()
train['sex'].replace('female', 1, inplace = True)

train['sex'].replace('male', 0, inplace = True)

train.head()
sns.heatmap(train.corr(),cmap='coolwarm',annot=True)
min_max_scaler = preprocessing.MinMaxScaler()

scaled = min_max_scaler.fit_transform(train[['sex']])

train[['sex']] = pd.DataFrame(scaled)



train.head()
X_train, X_test, y_train, y_test = train_test_split(train[['year','suicides_no','population']], train['sex'], test_size = 0.3)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
drugTree = DecisionTreeClassifier(criterion="gini")

drugTree.fit(X_train,y_train)

predTree = drugTree.predict(X_test)



knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)



nbc = GaussianNB()

nbc.fit(X_train,y_train)

y_pred = nbc.predict(X_test)



logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)
from sklearn import metrics



print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

print(classification_report(y_test, predTree))

pd.DataFrame(

confusion_matrix(y_test, predTree),

columns=['Predicted No', 'Predicted Yes'],

index=['Actual No', 'Actual Yes']

)   
print("KNN's Accuracy: ", metrics.accuracy_score(y_test, pred))

print(classification_report(y_test, pred))

pd.DataFrame(

confusion_matrix(y_test, pred),

columns=['Predicted No', 'Predicted Yes'],

index=['Actual No', 'Actual Yes']

)  
print("NB's Accuracy: ", metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

pd.DataFrame(

confusion_matrix(y_test, y_pred),

columns=['Predicted No', 'Predicted Yes'],

index=['Actual No', 'Actual Yes']

)
print("LR's Accuracy: ", metrics.accuracy_score(y_test, predictions ))

print(classification_report(y_test, predictions))

pd.DataFrame(

confusion_matrix(y_test, predictions),

columns=['Predicted No', 'Predicted Yes'],

index=['Actual No', 'Actual Yes']

)
target=suicide_df['sex']

target_count = target.value_counts()

print('Class 0:', target_count[0])

print('Class 1:', target_count[1])

print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
suicide_df.info()

suicide_df.describe()
suicide_df.drop(['suicides/100k pop','HDI for year','gdp_per_capita ($)'],axis=1,inplace=True)

suicide_df.head()
suicide_df.groupby('age')['suicides_no'].agg('sum').plot(kind='bar',title='Статистика суицидов по возрастам')
suicide_df.groupby('sex')['suicides_no'].agg('sum').sort_values(ascending=False).plot(kind='bar',title='Статистика суицидов среди мужчин и женщин')
alpha =0.8   #

plt.figure(figsize=(15,20))

sns.countplot(y='country', data=suicide_df, alpha=alpha)

plt.title('Статистика суицидов по странам')

plt.show()
mask1=suicide_df['country-year']=='Kyrgyzstan2000'

mask2=suicide_df['country-year']=='Kyrgyzstan2001'

mask3=suicide_df['country-year']=='Kyrgyzstan2002'

mask4=suicide_df['country-year']=='Kyrgyzstan2003'

mask5=suicide_df['country-year']=='Kyrgyzstan2004'

mask6=suicide_df['country-year']=='Kyrgyzstan2005'

mask7=suicide_df['country-year']=='Kyrgyzstan2006'

mask8=suicide_df['country-year']=='Kyrgyzstan2007'

mask9=suicide_df['country-year']=='Kyrgyzstan2008'

mask10=suicide_df['country-year']=='Kyrgyzstan2009'

mask11=suicide_df['country-year']=='Kyrgyzstan2010'

mask12=suicide_df['country-year']=='Kyrgyzstan2011'

mask13=suicide_df['country-year']=='Kyrgyzstan2012'

mask14=suicide_df['country-year']=='Kyrgyzstan2013'

mask15=suicide_df['country-year']=='Kyrgyzstan2014'

mask16=suicide_df['country-year']=='Kyrgyzstan2015'

mask18=suicide_df['country-year']=='Kyrgyzstan1985'

mask19=suicide_df['country-year']=='Kyrgyzstan1986'

mask20=suicide_df['country-year']=='Kyrgyzstan1987'

mask21=suicide_df['country-year']=='Kyrgyzstan1988'

mask22=suicide_df['country-year']=='Kyrgyzstan1989'

mask23=suicide_df['country-year']=='Kyrgyzstan1990'

mask24=suicide_df['country-year']=='Kyrgyzstan1991'

mask25=suicide_df['country-year']=='Kyrgyzstan1992'

mask26=suicide_df['country-year']=='Kyrgyzstan1993'

mask27=suicide_df['country-year']=='Kyrgyzstan1994'

mask28=suicide_df['country-year']=='Kyrgyzstan1995'

mask29=suicide_df['country-year']=='Kyrgyzstan1996'

mask30=suicide_df['country-year']=='Kyrgyzstan1997'

mask31=suicide_df['country-year']=='Kyrgyzstan1998'

mask32=suicide_df['country-year']=='Kyrgyzstan1999'

suicide_kyrgyzstan=suicide_df[mask1|mask2|mask3|mask4|mask5|mask6|mask7|mask8|mask9|mask10|mask11|mask12|mask13|mask14|mask15|mask16

                         |mask18|mask19|mask20|mask21|mask22|mask23|mask24|mask25|mask26|mask27|mask28|mask29|mask30|mask31|mask32]

suicide_kyrgyzstan.info()

plt.figure(figsize=(15,6)) 

pld=suicide_kyrgyzstan.groupby('age')['suicides_no'].agg('sum').sort_values(ascending=False).plot(kind='bar',title='Статистика суицидов в Кыргызстане (по возрастам)')
plt.figure(figsize=(20,5)) 

suicide_kyrgyzstan.groupby('country-year')['suicides_no'].agg('sum').plot(kind='bar', title='Статистика суицидов в Кыргызстане(по годам)')
plt.figure(figsize=(15,5))

sns.barplot(x = 'sex', y = 'suicides_no', hue = 'age',data =suicide_kyrgyzstan)