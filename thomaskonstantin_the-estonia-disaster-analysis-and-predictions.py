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
p_data = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
p_data.info()
p_data.describe(include='all')
p_data.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

plt.figure(figsize=(11,5))

sns.boxplot(x='Survived',y='Age',data=p_data)

plt.show()
print('Age Kurt: ' ,p_data['Age'].kurt())

print('Age Skew: ' ,p_data['Age'].skew())
plt.figure(figsize=(11,7))

ax = sns.countplot(x='Category',hue='Survived',data= p_data)

ax.set_xticklabels(['Passenger','Crew'])

ax.text(-0.22,700,p_data['Survived'][(p_data['Category']=='P') & (p_data['Survived']==0) ].count(),fontsize=12, fontweight='bold')

ax.text(0.19,100,p_data['Survived'][(p_data['Category']=='P') & (p_data['Survived']==1) ].count(),fontsize=12, fontweight='bold')

ax.text(0.79,157,p_data['Survived'][(p_data['Category']=='C') & (p_data['Survived']==0) ].count(),fontsize=12, fontweight='bold')

ax.text(1.19,45,p_data['Survived'][(p_data['Category']=='C') & (p_data['Survived']==1) ].count(),fontsize=12, fontweight='bold')



plt.show()
print("Precent Of Passenger Survived: ","{:.2%}".format(98/698))

print("Precent Of Crew Survived: ","{:.2%}".format(39/154))
plt.figure(figsize=(11,7))

cdata = p_data.groupby('Country').sum()

ax = sns.barplot(x=cdata.index,y='Survived',data=cdata)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

ax.set(title='Survivers Per Country')

plt.show()
#lets check our age range first

print("Ages Range between",p_data['Age'].min(),',',p_data['Age'].max())
#we will divide ages into groups

age_groups = pd.cut(p_data['Age'], 5,labels=['0-18','19-35','36-52','53-69','70-87'])

pa_data = p_data.copy()

pa_data['Age Groups'] = age_groups

p_data['Age Groups'] = age_groups

plt.figure(figsize=(12,8))

ax = sns.countplot(pa_data['Age Groups'][pa_data['Survived']==1],hue=pa_data['Category'][pa_data['Survived']==1])

ax.set_ylabel('Number Of Survivers')

ax.set_title('Number Of Survivers According To Age Group And Individual Category')

plt.show()
plt.figure(figsize=(12,8))

ax = sns.countplot(pa_data['Age Groups'][pa_data['Survived']==1],hue=pa_data['Sex'][pa_data['Survived']==1])

ax.set_ylabel('Number Of Survivers')

ax.set_title('Number Of Survivers According To Age Group And Individual Geneder')

plt.show()
#firt we will select the features that will be used in our models

features = ['Sex','Age','Category','Age Groups','Country']

y = p_data.pop('Survived')

p_data
p_data['Country'] = p_data.Country.astype('category').cat.codes

p_data['Sex'] = p_data.Sex.astype('category').cat.codes

p_data['Category'] = p_data.Category.astype('category').cat.codes

p_data['Age Groups'] = p_data['Age Groups'].astype('category').cat.codes



p_data
#We Will Add A Feature Called "Survival Chance Using The Insight We Have Learnd From The EDA"

high_survival_chance   =    p_data[(p_data['Age Groups']== 1) & (p_data['Sex']== 1) & (p_data['Category']== 0) ].copy()

high_survival_chance['Survival Chance'] = 3

medium_survival_chance =    p_data[(p_data['Age Groups']!= 1) & (p_data['Sex']== 1) & (p_data['Category']== 0) ].copy()

medium_survival_chance['Survival Chance'] = 2

low_survival_chance    =    p_data[(p_data['Age Groups']!= 1) & (p_data['Sex']!= 1) & (p_data['Category']== 0) ].copy()

low_survival_chance['Survival Chance'] = 1

c_target = [high_survival_chance,medium_survival_chance,low_survival_chance]

common = pd.concat(c_target)

very_low_survival_chance =  p_data[~p_data.isin(common)].dropna().copy()

very_low_survival_chance['Survival Chance'] = 0

p_data = pd.concat([very_low_survival_chance,high_survival_chance,medium_survival_chance,low_survival_chance])

features.append('Survival Chance')
X = p_data[features]
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import f1_score as f1

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.3)

#scaler = StandardScaler()

#train_x = scaler.fit_transform(train_x)

#test_x = scaler.fit_transform(test_x)
def optimal_knn(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = KNeighborsClassifier(n_neighbors=n)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(f1(np.round(pred),test_y))

    return results

def optimal_n(train_x,test_x,train_y,test_y,n_list):

    results = []

    for n in n_list:

        model = AdaBoostClassifier(n_estimators=n)

        model.fit(train_x,train_y)

        pred = model.predict(test_x)

        results.append(f1(np.round(pred),test_y))

    return results
KNN_scores = optimal_knn(train_x,test_x,train_y,test_y,[2,3,6,10])

KNN_scores

ada_scores = optimal_n(train_x,test_x,train_y,test_y,[2,3,6,10])

ada_scores

plt.plot(KNN_scores)

plt.plot(ada_scores)

plt.legend(['KNN','ADABOOST'])

import keras

from keras.models import Sequential

from keras.layers import Dense

np.random.seed(42)
model = Sequential()

model.add(Dense(6,activation='tanh',input_dim=6))

model.add(Dense(4,activation='tanh'))

model.add(Dense(4,activation='tanh'))

model.add(Dense(1,activation='tanh'))

model.compile(optimizer='adam', 

              loss='BinaryCrossentropy', 

              metrics=['accuracy'])

model.fit(train_x,train_y,epochs=25)
pred = model.predict(test_x)
pred = (pred>0.14).astype(int)

cm = confusion_matrix(pred,test_y)

plt.figure(figsize=(10,6))

sns.heatmap(cm,annot=True,fmt='g',cmap='YlGnBu',xticklabels=['Predicted Not Survived','Predicted Survived'],yticklabels=['Not Survived','Survived'])

print(classification_report(pred,test_y))