import numpy as np 

import pandas as pd

import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%config InlineBackend.figure_format = 'retina'
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

df.head(10)
df.info()
df.isnull().sum()
df['Category'] = df['Category'].map(lambda x : 0 if x == 'P' else 1)

df['Sex'] = df['Sex'].map(lambda x : 0 if x == 'M' else 1)

df.head()
# 데이터 분포 파악

fig = df.hist(figsize= (20, 10), bins = 20 ,

             xlabelsize= 15, ylabelsize= 15)
col_list = df.columns.tolist()

col_list
fig = px.pie(values = df['Survived'].value_counts(), names =['Death', 'Survived'])

fig.update_traces(textinfo = ('label + percent'))

fig.show()
sns.set(font_scale=1.5)

plt.figure(figsize= (10, 8))

fig = sns.countplot(data = df, y = 'Country')
pd.DataFrame(df['Country'].value_counts()).style.background_gradient('Reds')
# 생존자

df_survived = df[df['Survived'].isin([1])]



# 사망자

df_death = df[df['Survived'].isin([0])]
fig = sns.violinplot(y = df['Age'], x = df['Survived'])

plt.title('Age, 0 : Death, 1 : Survived')

plt.show()
man_S = len(df[df['Sex'].isin([0]) & df['Survived'].isin([1])])

man_D = len(df[df['Sex'].isin([0]) & df['Survived'].isin([0])])

woman_S = len(df[df['Sex'].isin([1]) & df['Survived'].isin([1])])

woman_D = len(df[df['Sex'].isin([1]) & df['Survived'].isin([0])])
values = [man_S, man_D, woman_S, woman_D]

names =  ['Survived_Man', 'Death_Man', 'Survived_Woman', 'Death_Woman']

color = ['blue', 'red', 'blue', 'red']

fig = px.pie(values= values, names= names, color= color, )

fig.update_traces(textinfo = ('label + percent'))

fig.show()



sns.countplot(data=df, x = 'Sex', )

plt.show()
# Crew vs Passenger

pas_S = len(df[df['Category'].isin([0]) & df['Survived'].isin([1])])

pas_D = len(df[df['Category'].isin([0]) & df['Survived'].isin([0])])

crew_S = len(df[df['Category'].isin([1]) & df['Survived'].isin([1])])

crew_D = len(df[df['Category'].isin([1]) & df['Survived'].isin([0])])
import plotly.graph_objects as go

from plotly.subplots import make_subplots





fig = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],

                    subplot_titles=['Crew', 'Passenger'])



fig.add_trace(go.Pie(labels = ['crew_S', 'crew_D'], values=[crew_S, crew_D],

                     pull = [0, 0.2],

                     scalegroup ='one',

                     name='Passenger'), 1, 1)



fig.add_trace(go.Pie(labels= ['pas_S', 'pas_D'], values=[pas_S, pas_D],

                     pull = [0, 0.2],

                     scalegroup ='one',

                     name='Passenger'), 1, 2)



fig.update_traces(textinfo='label+percent', textfont_size=15,

                  marker=dict(colors=['red','blue'], line=dict(color='#000000', width=2)))



fig.update_layout(title_text='Crew vs Passenger')



fig.show()
print('Crew : ', crew_D + crew_S)

print('Passenger :', pas_D + pas_S)
col_list
# 학습모델에 필요한 feature만 선택

features_list = col_list[4:]

features_list
from sklearn.model_selection import train_test_split



X = df[features_list].iloc[:,:-1]

y = df[features_list].iloc[:,-1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 15)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix



clf = LogisticRegression().fit(X_train, y_train)



y_pred = clf.predict(X_test)

ac = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)



print('accuracy : {:.4f}'.format(ac))

print(cm)
from sklearn.tree import DecisionTreeClassifier



clf_score = []



for depth in range(3, 10):

    clf = DecisionTreeClassifier(max_depth= depth, random_state = 15)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    ac = accuracy_score(y_test, y_pred)

    

    clf_score.append(ac)

    

plt.plot(range(3,10), clf_score)

plt.show()

print(max(clf_score))
from sklearn.ensemble import RandomForestClassifier



clf_score = []



for depth in range(3, 10):

    clf = RandomForestClassifier(max_depth= depth, random_state = 15)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    ac = accuracy_score(y_test, y_pred)

    

    clf_score.append(ac)

    

plt.plot(range(3,10), clf_score)

plt.show()

print(max(clf_score))
from sklearn.svm import SVC



clf = SVC().fit(X_train, y_train)

y_pred = clf.predict(X_test)

ac = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)



print('accuracy : {:.4f}'.format(ac))

print(cm)