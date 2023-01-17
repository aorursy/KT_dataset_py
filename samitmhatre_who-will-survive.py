import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
trainData = pd.read_csv('../input/titanic/train.csv')
testData = pd.read_csv('../input/titanic/test.csv')
trainData.head()
trainData.Survived
plt.hist(trainData.Survived, bins = 2)
trainData.describe()
pie_data = trainData.drop(columns=['PassengerId', 'Name', 'Ticket', 'Age','Fare','Cabin'])
fig = plt.figure(figsize=(15,15))

for i in range(1, pie_data.shape[1] +1):

    plt.subplot(2, 3, i)

    

    fig = plt.gca()

    fig.axes.get_yaxis().set_visible(False)

    

    fig.set_title(pie_data.columns.values[i-1])

    values =  pie_data.iloc[:, i-1].value_counts(normalize = True).values

    index =  pie_data.iloc[:, i-1].value_counts(normalize = True).index

    fig = plt.pie(values, labels = index, autopct = '%1.1f%%')

    

    plt.axis('equal')

plt.tight_layout(rect = [0, 0.03, 1, 0.95])
sns.set_style('darkgrid')

plt.figure(figsize=(12,12))

trainData.drop(columns=['Survived','PassengerId']).corrwith(trainData.Survived).plot.bar()

plt.title('Correlation with the dependent vairable')

plt.show()
plt.figure(figsize=(12,12))

sns.heatmap(trainData.corr(),annot = True, fmt='g')
trainData.isnull().sum()
trainData2 = trainData.copy(deep = True)
testData2 = testData.copy(deep=True)
trainData2.drop(columns = ['Cabin'], inplace = True)
testData2.drop(columns = ['Cabin'], inplace = True)
trainData2.Age.fillna(trainData2.Age.mean(),inplace = True)
testData2.Age.fillna(testData2.Age.mean(),inplace = True)

testData2.Fare.fillna(testData2.Fare.mean(), inplace = True)
trainData2.isnull().sum()
testData2.isnull().sum()
trainData2[trainData2.Embarked.isnull()].index
trainData2.drop(index=[61,829],inplace=True)
trainData2.isnull().sum()
trainData2.shape
testData2.shape
name = trainData.Name
title = np.asarray([])

for i in name:

    if i.find('Mrs') != -1:

        r = i.find('Mrs')

        re = i[r: r+3]

    elif i.find('Mr') != -1:

        r = i.find('Mr')

        re = i[r:r+2]

    elif i.find('Master') != -1:

        r = i.find('Master')

        re = i[r:r+6]

    else:

        r = i.find('Miss')

        re = i[r:r+4]

    title = np.append(title, re)

    
title.head()
title = pd.DataFrame(title)
title[0].unique()
indexes = title[title[0] ==''].index
for i in indexes:

    if trainData.iloc[i, 4] == 'male':

        if trainData.iloc[i, 5] >= 18:

            title.iloc[i,0] = 'Mr'

        else:

            title.iloc[i,0] = 'Master'

    else:

        title.iloc[i,0] = 'Miss'
title[0].unique()
name_test = testData2.Name

title_test = np.asarray([])

for i in name_test:

    if i.find('Mrs') != -1:

        r = i.find('Mrs')

        re = i[r: r+3]

    elif i.find('Mr') != -1:

        r = i.find('Mr')

        re = i[r:r+2]

    elif i.find('Master') != -1:

        r = i.find('Master')

        re = i[r:r+6]

    else:

        r = i.find('Miss')

        re = i[r:r+4]

    title_test = np.append(title_test, re)
title_test = pd.DataFrame(title_test)

title_test[0].unique()
indexes_test = title_test[title_test[0] ==''].index

for i in indexes_test:

    if testData2.iloc[i, 3] == 'male':

        if testData2.iloc[i, 4] >= 18:

            title_test.iloc[i,0] = 'Mr'

        else:

            title_test.iloc[i,0] = 'Master'

    else:

        title_test.iloc[i,0] = 'Miss'
title_test[0].unique()
trainData2['Title'] = title[0]
trainData2.head()
testData2['Title'] = title_test[0]

testData2.head()
y=trainData2.Survived

Id = trainData2.PassengerId

x = trainData2.drop(columns = ['PassengerId', 'Name', 'Ticket','Survived'])
Id_test = testData2.PassengerId

testData2.drop(columns = ['PassengerId','Name','Ticket'], inplace = True)
x_dummies = pd.get_dummies(x, drop_first=True)

testData_dummies = pd.get_dummies(testData2, drop_first = True)
x_dummies.head()
testData_dummies.head()
x = x_dummies
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
sc = StandardScaler()
x_train2 = pd.DataFrame(sc.fit_transform(x_train), columns = x_train.columns.values)
x_test2 = pd.DataFrame(sc.transform(x_test), columns = x_test.columns.values)
testData_dummies2 = pd.DataFrame(sc.transform(testData_dummies), columns = testData_dummies.columns.values)
x_train2
x_test2
testData_dummies2
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()
classifier.fit(x_train2,y_train)
y_pred = classifier.predict(x_test2)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
confusion_matrix(y_test,y_pred)
precision_score(y_test,y_pred)
accuracy_score(y_test, y_pred)
from sklearn.model_selection import RandomizedSearchCV
param = {'n_estimators':[100,300,500],'criterion':['gini','entropy'],'max_depth':[3,5,7],'max_features':['sqrt','log2','auto']}
classifier = RandomForestClassifier()

random = RandomizedSearchCV(estimator=classifier,param_distributions=param,n_iter=10,scoring='accuracy',n_jobs = -1,cv = 5)
random.fit(x_train2,y_train)
random.best_params_
classifier2 = RandomForestClassifier(n_estimators=100,

                                     max_features='sqrt',

                                     max_depth=5,

                                     criterion='entropy')

classifier2.fit(x_train2,y_train)
y_pred2 = classifier2.predict(x_test2)
confusion_matrix(y_test,y_pred2)
precision_score(y_test,y_pred2)
accuracy_score(y_test,y_pred2)
classifier.fit(x_train2,y_train)
prediction = classifier.predict(testData_dummies2)
Id = pd.DataFrame(Id_test, columns = ['Id'])



predi = pd.DataFrame(prediction, columns = ['Survived'])
result = pd.concat([Id,predi],axis = 1)
