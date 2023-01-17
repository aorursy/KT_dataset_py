# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plot

import seaborn as sns

%matplotlib inline

sns.set(style="ticks")



from scipy.stats import zscore

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection
# Original data: 

data = pd.read_csv('../input/bank-loan-modeling-synthetic-demo/Loan.csv')



# Fabricated data: 

#data = pd.read_csv('../input/bank-loan-modeling-synthetic-demo/synth_loan.csv')







data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]
data.head()
data.columns
data.shape
data.info()
# No columns have null data in the file

data.apply(lambda x : sum(x.isnull()))
# Eye balling the data

data.describe().transpose()
#finding unique data

data.apply(lambda x: len(x.unique()))
sns.pairplot(data.iloc[:,1:])
# there are 52 records with negative experience. Before proceeding any further we need to clean the same

data[data['Experience'] < 0]['Experience'].count()
#clean the negative variable

dfExp = data.loc[data['Experience'] >0]

negExp = data.Experience < 0

column_name = 'Experience'

mylist = data.loc[negExp]['ID'].tolist() # getting the customer ID who has negative experience
# there are 52 records with negative experience

negExp.value_counts()
for id in mylist:

    age = data.loc[np.where(data['ID']==id)]["Age"].tolist()[0]

    education = data.loc[np.where(data['ID']==id)]["Education"].tolist()[0]

    df_filtered = dfExp[(dfExp.Age == age) & (dfExp.Education == education)]

    exp = df_filtered['Experience'].median()

    data.loc[data.loc[np.where(data['ID']==id)].index, 'Experience'] = exp
# checking if there are records with negative experience

data[data['Experience'] < 0]['Experience'].count()
data.describe().transpose()
sns.boxplot(x='Education',y='Income',hue='PersonalLoan',data=data)
sns.boxplot(x="Education", y='Mortgage', hue="PersonalLoan", data=data,color='yellow')
sns.countplot(x="SecuritiesAccount", data=data,hue="PersonalLoan")
sns.countplot(x='Family',data=data,hue='PersonalLoan',palette='Set1')
sns.countplot(x='CDAccount',data=data,hue='PersonalLoan')
sns.distplot( data[data.PersonalLoan == 0]['CCAvg'], color = 'r')

sns.distplot( data[data.PersonalLoan == 1]['CCAvg'], color = 'g')
print('Credit card spending of Non-Loan customers: ',data[data.PersonalLoan == 0]['CCAvg'].median()*1000)

print('Credit card spending of Loan customers    : ', data[data.PersonalLoan == 1]['CCAvg'].median()*1000)
fig, ax = plot.subplots()

colors = {1:'red',2:'yellow',3:'green'}

ax.scatter(data['Experience'],data['Age'],c=data['Education'].apply(lambda x:colors[x]))

plot.xlabel('Experience')

plot.ylabel('Age')
# Correlation with heat map

import matplotlib.pyplot as plt

import seaborn as sns

corr = data.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
sns.boxplot(x=data.Family,y=data.Income,hue=data.PersonalLoan)

# Looking at the below plot, families with income less than 100K are less likely to take loan,than families with 

# high income
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data.drop(['ID','Experience'], axis=1), test_size=0.3 , random_state=100)
train_labels = train_set.pop('PersonalLoan')

test_labels = test_set.pop('PersonalLoan')
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



dt_model=DecisionTreeClassifier(criterion = 'entropy',max_depth=3)

dt_model.fit(train_set, train_labels)
dt_model.score(test_set , test_labels)
y_predict = dt_model.predict(test_set)

y_predict[:5]
test_set.head(5)
naive_model = GaussianNB()

naive_model.fit(train_set, train_labels)



prediction = naive_model.predict(test_set)

naive_model.score(test_set,test_labels)
randomforest_model = RandomForestClassifier(max_depth=2, random_state=0)

randomforest_model.fit(train_set, train_labels)
Importance = pd.DataFrame({'Importance':randomforest_model.feature_importances_*100}, index=train_set.columns)

Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
predicted_random=randomforest_model.predict(test_set)

randomforest_model.score(test_set,test_labels)
train_set_indep = data.drop(['Experience' ,'ID'] , axis = 1).drop(labels= "PersonalLoan" , axis = 1)

train_set_dep = data["PersonalLoan"]

X = np.array(train_set_indep)

Y = np.array(train_set_dep)

X_Train = X[ :3500, :]

X_Test = X[3501: , :]

Y_Train = Y[:3500, ]

Y_Test = Y[3501:, ]
knn = KNeighborsClassifier(n_neighbors= 21 , weights = 'uniform', metric='euclidean')

knn.fit(X_Train, Y_Train)    

predicted = knn.predict(X_Test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(Y_Test, predicted)

print(acc)
X=data.drop(['PersonalLoan','Experience','ID'],axis=1)

y=data.pop('PersonalLoan')
models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=12345)

	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()