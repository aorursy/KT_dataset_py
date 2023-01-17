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

        

        

        

import matplotlib.pyplot as plt



from seaborn import distplot

import plotly.express as px



import seaborn as sns

import plotly.graph_objects as go

import plotly.offline as py



import gc

import warnings

warnings.filterwarnings('ignore')





from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold,cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')



df.columns
df.head()
df.tail()
df.describe()
df.describe(include='O')
df.shape
df.info()
df.isnull().sum()
print('Age Distribution using Seaborm Distribution Plot')



distplot(df['Age'])

plt.show()



print('Skewness of the Age ',df['Age'].skew())

print('Kurtosis of the Age ',df['Age'].kurt())



print('\n Boxplot representation of the data \n')

sns.boxenplot(df['Age'])

plt.show()

plt.hist(df['Age'],bins=10,edgecolor='k')

plt.title('Age of Passenger ')

plt.xlabel('Age')

plt.ylabel('count')

plt.show()
df[df.Age<1]
df[df.Lastname=='ZELMIN']
family_member = df[df.duplicated(subset=['Lastname'])] #AHLSTROM is one of the name

#family_member.head()

family_member[family_member.Lastname =='AHLSTROM']
df_crew = df[df.Category=='C']

df_crew = df_crew[['Country','Category']].groupby('Country').count().sort_values(by='Category',ascending=False)

df_crew.head()

del(df_crew)

gc.collect()
#df_type = df.groupby('Survived').sum()

values = df.Survived.value_counts().values

labels = ['Death','Survived']



trace = go.Pie(values=values,labels=labels)

py.iplot([trace])
labels = df.Country.value_counts().index

values = df.Country.value_counts().values



fig = px.pie(df,labels=labels, values=values,title='Country wise Passenger data',names=labels,

            width=1000,height=800)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()

labels = ['Male','Female']

values = df.Sex.value_counts().values



fig = px.pie(df,labels=labels,values=values,names=labels)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
pd.DataFrame(df['Country'].unique())
from wordcloud import WordCloud





plt.figure(figsize=(12,8))

wordcloud = WordCloud(collocations=False

                     ).generate_from_text('*'.join(df.Country))



plt.imshow(wordcloud,interpolation='bilinear')

plt.show()
labels = ['Passenger','Crew']

values = df.Category.value_counts().values



fig = px.pie(df,labels=labels,values=values,names=labels)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
g = sns.factorplot('Survived','Country',data=df,

                   hue='Country',

                   size=13,

                   aspect=0.8,

                   palette='magma',

                   join=False

              )
grid = sns.FacetGrid(df, col='Survived', row='Category', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
plt.figure(figsize=(12,8))

sns.kdeplot(df.loc[df['Survived']==0,'Age'],label = 'Death')

plt.title('Death/Survival Plot based on age')



sns.kdeplot(df.loc[df['Survived']==1,'Age'],label ='Survived' )

plt.show()
#Creating a different dataset for Survivor and Non Survivors





df_survivors = df[df['Survived']==1]

df_death = df[df['Survived']==0]







labels = ['Male','Female']

values = df_survivors.Category.value_counts().values



fig = px.pie(df_survivors,labels=labels,values=values,names=labels,title='Male-Female Percentage from Survivors')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()



labels = ['Male','Female']

values = df_death.Category.value_counts().values

fig = px.pie(df_death,labels=labels,values=values,names=labels,title='Male-Female Percentage from Death')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
df[['Sex','Survived']].groupby('Sex').mean().sort_values(by='Survived',ascending=False)
print('Country with highest Survival Passenger and Crew')

plt_survivors = df_survivors.groupby('Country')['Country','Survived'].sum().sort_values(by='Survived',ascending=False)



plt_survivors.plot(kind='bar',color='magenta')

plt.show()



print('\n Top 5 Country with Highest number of Survivors \n',plt_survivors.head())



print('Country with highest Death Passenger and Crew')

plt_death = df_death.groupby('Country')['Survived'].count().sort_values(ascending=False)

plt_death.plot(kind='bar',color='red')

plt.show()



print('\n Top 5 Country with Highest number of Death \n',plt_death.head())



plt.figure(figsize=(12,8))

sns.countplot(x='Category',hue='Sex',data=df)

plt.title('Genderwise Survival')

plt.show()
age_data = df.loc[:,['Age','Survived']]

age_data['YEARS_AGE'] = age_data.loc[:,'Age']

age_data['AGE_BAND'] = pd.cut(age_data.loc[:,'YEARS_AGE'],bins = np.linspace(20,70,num=11))



age_data.head()
plt.figure(figsize=(10,5))

age_group = age_data.groupby('AGE_BAND').mean()

plt.bar(age_group.index.astype(str),age_group['Survived'])

plt.xticks(rotation=60)

plt.xlabel('Age Group(Bins)',fontsize=19)



plt.show()
df[['Category','Survived']].groupby(['Category']).mean().sort_values(by='Survived',ascending=False)
plt.figure(figsize=(12,8))

data = df.corr()

sns.heatmap(data,cmap='Reds',annot=True)

plt.title('Heatmap for Correlation')

plt.show()
df.drop(['Country','Firstname','Lastname'],axis=1,inplace=True)
df['Sex'] = pd.get_dummies(df.Sex,drop_first=True)

df['Category'] = pd.get_dummies(df.Category,drop_first=True)



df.head()
y = df['Survived']

X = df.drop(['Survived'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)



# Model List

models = [] 

models.append(('Logistic Regression', LogisticRegression())) 

models.append(('KNearest Neighbour', KNeighborsClassifier()))

models.append(('SVM', SVC())) 

models.append(('Naive Bayes', GaussianNB()))

models.append(('Decision Tree', DecisionTreeClassifier()))

models.append(('Random Forest', RandomForestClassifier()))
results = [] 

names = []

for name, model in models:

    kfold = KFold(n_splits=10, random_state=42) 

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy') 

    results.append(cv_results) 

    names.append(name) 

      

    print("\n Model Name:{}\n Accuracy :{:.3f} \n Std:{:.3f}".format(name, cv_results.mean(), cv_results.std()))
 

from sklearn.metrics import confusion_matrix,classification_report

classifier = RandomForestClassifier()

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

report = classification_report(y_test,y_pred)

print('\n Confusion Matrix is \n',cm)

print('\n Classification Matrix \n',report)
