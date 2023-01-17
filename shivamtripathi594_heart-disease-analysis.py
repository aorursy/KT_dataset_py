# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importing dataset

df = pd.read_csv('../input/heart.csv')

df.head()
df.shape
df.info()
#Dataset has 14 columns consisting of 13 independant parameters and target as dependent variable

df.describe()
#Grouping data with respect to target and calculating mean

df.groupby('target').mean()
df.target.value_counts()
sns.countplot(x = 'target', data = df)

plt.title('People having disease')

plt.xlabel('People')

plt.ylabel('Count')

plt.xticks([0,1], ['Not having disease', 'Having Disease'])

plt.show()
count_having_disease = df.target[df.target == 1].count()

count_not_having_disease = df.target[df.target == 0].count()

percentage_having_disease = (count_having_disease/df.target.count())*100

percentage_not_having_disease = (count_not_having_disease/df.target.count())*100

print('Percentage of people having heart disease = {:.2f}%'.format(percentage_having_disease))

print('Percentage of people not having heart disease = {:.2f}%'.format(percentage_not_having_disease))
df.sex.value_counts()
sns.countplot(x = 'sex', data = df)

plt.title('Sex')

plt.xlabel('Sex')

plt.ylabel('Count')

plt.xticks([0,1], ['Female', 'Male'])

plt.show()
males = df.sex[df.sex == 1].count()

females = df.sex[df.sex == 0].count()

per_males = (males/df.sex.count())*100

per_females = (females/df.sex.count())*100

print('Percentage of males = {:.2f}%'.format(per_males))

print(' Percentage of females = {:.2f}%'.format(per_females))
pd.crosstab(df.sex, df.target).plot(kind='bar')

plt.title('Disease vs Sex')

plt.legend(['Not Having Disease', 'Having Disease'])

plt.ylabel('No. of persons having disease')

plt.xlabel('Sex')

plt.xticks((0,1),['Female', 'Male'], rotation =0)
male_having_disease = (df.sex[(df.sex ==1) & (df.target ==1)].count()/df.sex[df.sex == 1].count()) * 100

female_having_disease = (df.sex[(df.sex ==0) & (df.target ==1)].count()/df.sex[df.sex == 0].count()) * 100

print('There are {:.2f}% male and {:.2f}% female having heart disease.'.format(male_having_disease, female_having_disease))
pd.crosstab(df.thalach, df.target).plot(kind = 'bar',figsize = (20,12))

plt.title('Thalach vs Disease')

plt.xlabel('Thalach Value')

plt.ylabel('Count')

plt.legend(['Not having disease', 'Havving disease'])

plt.show()
#Visualing the maximum heart rate along with age

plt.scatter(df.age[df.target == 1], df.thalach[df.target == 1], c = 'red')

plt.scatter(df.age[df.target == 0], df.thalach[df.target == 0])

plt.xlabel('Age')

plt.ylabel('Maximum Heart Rate')

plt.title('Maximum Heart Rate vs Age')

plt.legend(['Having Disease', 'Not Having Disease'])
pd.crosstab(df.cp, df.target).plot(kind='bar')

plt.title('Chest Pain and Disease')

plt.xlabel('Chest Pain Type')

plt.ylabel('Count')

plt.legend(['Not Having Disease', 'Having Disease'])

plt.xticks((0,1,2,3), ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptotic'], rotation=90)

plt.show()
chestpain = []

for i in range(0,4):

    chestpain.append((df.cp[(df.cp == i) & (df.target == 1)].count()/df.cp[df.cp == i].count())*100)

chestpain
total_no_people_having_type_cp = []

for i in range(0,4):

    total_no_people_having_type_cp.append(df.cp[df.cp == i].count())



total_people_having_disease_with_cp_type = []

for i in range(0,4):

    total_people_having_disease_with_cp_type.append(df.cp[(df.cp == i) & (df.target == 1)].count())
CP = pd.DataFrame({'Total No. of People having Chest Pain':total_no_people_having_type_cp,

                  'Total No. of People having disease with Chest Pain':total_people_having_disease_with_cp_type,

                  'Percentage':chestpain})

CP
indices = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptotic']

CP.set_index(pd.Index(indices), inplace=True)

CP
pd.crosstab(df.fbs, df.target).plot(kind='bar')

plt.title('Fasting Blood Sugar vs Disease')

plt.xlabel('Fasting blood Sugar')

plt.ylabel('Count')

plt.xticks((0,1), ['<120 mg/dl', '>120 mg/dl'], rotation = 0)

plt.legend(['Not Having Disease', 'Having Disease'])

plt.show()
pd.crosstab(df.slope, df.target).plot(kind='bar')

plt.title('Slope vs Disease')

plt.xlabel('Slope Type')

plt.ylabel('Count')

plt.legend(['Not Having Disease', 'Having Disease'])

plt.show()
#In this data Chest pain(cp), Slope and thal are actually categorical data

#Conveting the categoraical values in dummy variable

chest = pd.get_dummies(df.cp, prefix='cp', drop_first= True) #Exluding first column in dummy varaible to avoid Dummy varaible trap 

th = pd.get_dummies(df.thal, prefix='thal', drop_first=True)

sl = pd.get_dummies(df.slope, prefix='slope', drop_first=True)
#Adding the all the parameters

merged = pd.concat([df, chest, th, sl], axis = 1)

merged.head()
#Dropping thal,cp,slope for data

merged.drop(['thal', 'cp', 'slope'], axis = 1, inplace=True)

merged.head()
#Splitting data into independant and dependant variables

X = merged.drop('target', axis = 1)

y = merged.target

print(X.head(), y.head())
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
#Splitting dataset into traing and test set

#We will split 80% data into training set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 0)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)

log_reg_acc = lr.score(X_test, y_test)

print('Logistic regression accuracy is{}'.format(lr.score(X_test, y_test)))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)

knn_acc = knn.score(X_test, y_test)

print('2-nn Classifier accuracy is {}'.format(knn.score(X_test, y_test)))
#Checking the number of neighbors with max accuracy

score_list = []

for i in range(1, 20):

    knnn = KNeighborsClassifier(n_neighbors=i)

    knnn.fit(X_train, y_train)

    score_list.append(knnn.score(X_test, y_test))

score_list
plt.bar(np.arange(1,20), score_list)

plt.xticks(np.arange(1,20))

plt.yticks(np.arange(0,1,.1))
print('Maximum score of KNN is {} and for index {}'.format(max(score_list), score_list.index(max(score_list))))
knn_acc = .8688
from sklearn.svm import SVC

sv = SVC(random_state=0)

sv.fit(X_train, y_train)

svm_acc = sv.score(X_test, y_test)

print('SVC Accuracy is {}'.format(sv.score(X_test, y_test)))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy', random_state=0)

dt.fit(X_train, y_train)

dt_acc = dt.score(X_test, y_test)

print('Decision Tree Accuracy is {}'.format(dt.score(X_test, y_test)))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0)

rfc.fit(X_train, y_train)

rfc_acc = rfc.score(X_test, y_test)

print('RFC Accuracy is {}'.format(rfc.score(X_test, y_test)))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

nb_acc = nb.score(X_test, y_test)

print('NaiveBayes Accuracy is {}'.format(nb.score(X_test, y_test)))
models = ['Logistic Regression', 'KNN', 'SVM', 'Decision Tree', 'Random Forest', 'Naive Bayes']

accuracy = [log_reg_acc, knn_acc, svm_acc, dt_acc, rfc_acc, nb_acc]

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]



plt.figure(figsize=(16,5))

plt.ylabel("Accuracy ")

plt.xlabel("Algorithms")

plt.yticks(np.arange(0,2,.1))

plt.grid()

sns.barplot(x=models, y=accuracy, palette=colors)

plt.show()