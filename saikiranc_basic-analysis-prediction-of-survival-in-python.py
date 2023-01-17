#import all the necessary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#read the files using pandas to data frames 

df  = pd.read_csv("../input/train.csv")

df1 = pd.read_csv("../input/test.csv")
#study information of data

df.info()
df1.info()
#drop all the unneccesary information

df.drop("Name",axis=1,inplace=True)

df.drop("Ticket",axis=1,inplace=True)

df.drop("Cabin",axis=1,inplace=True)

df.drop("PassengerId",axis=1,inplace=True)
print (df.head(2))
#replace all the NaN values with mean of the values

#df = df.fillna(df.mean())
#classification of required data

df['Sex'] = np.where(df['Sex']== 'male', 1 , 0  )
df['Embarked'] = np.where(df['Embarked']== 'Q', 1 ,df['Embarked']  )

df['Embarked'] = np.where(df['Embarked']== 'S', 2 ,df['Embarked']  )

df['Embarked'] = np.where(df['Embarked']== 'C', 0 ,df['Embarked']  )
for i  in range (10):

    df['Age'] = np.where((df['Age']>10*i) & (df['Age']<=(i+1)*10) ,((10*i) + ((i+1)*10))/2 , df['Age'])
for i  in range (10):

    df['Fare'] = np.where((df['Fare']>10*i) & (df['Fare']<=(i+1)*10) ,((10*i) + ((i+1)*10))/2 , df['Fare'])

df['Fare'] = np.where(df['Fare']>100 ,100, df['Fare'])
print (df.head(1))
#define correlation graph to remove multi colinearity

corr = df.corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr,cbar = True, square = True, annot = True, fmt = '.2f', annot_kws = {'size':15},

           xticklabels = True, yticklabels = True,

           cmap = 'coolwarm')
#drop unnecessary data from test data

del df1['PassengerId']

del df1['Cabin']

del df1['Name']

del df1['Ticket']
#replace all the NaN values with mean of the values

df = df.fillna(df.mean())

df1 = df1.fillna(df1.mean())
#classify the test data

for i  in range (10):

    df1['Age'] = np.where((df1['Age']>10*i) & (df1['Age']<=(i+1)*10) ,((10*i) + ((i+1)*10))/2 , df1['Age'])
df1['Embarked'] = np.where(df1['Embarked']== 'Q', 1 ,df1['Embarked']  )

df1['Embarked'] = np.where(df1['Embarked']== 'S', 2 ,df1['Embarked']  )

df1['Embarked'] = np.where(df1['Embarked']== 'C', 0 ,df1['Embarked']  )
for i  in range (10):

    df1['Fare'] = np.where((df1['Fare']>10*i) & (df1['Fare']<=(i+1)*10) ,((10*i) + ((i+1)*10))/2 , df1['Fare'])

df1['Fare'] = np.where(df1['Fare']>100 ,100, df1['Fare'])
df1['Sex'] = np.where(df1['Sex']== 'male', 1 , 0  )
df
df1
prediction_var = ['Pclass', 'Sex', 'Age', 'Parch','SibSp',  'Fare', 'Embarked']
df3 = pd.read_csv("../input/gendermodel.csv")
train_X = df[prediction_var]

train_y = df.Survived

test_X = df1[prediction_var]

test_y = df3.Survived
#from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

#from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

#from sklearn import svm

from sklearn import metrics
model=RandomForestClassifier(n_estimators=100)

#print(train_X)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
#find the important features in the data

featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)

print(featimp)
#consider top 4 important features 

prediction_var = ['Pclass', 'Sex', 'Age',  'Fare' ]
train_X = df[prediction_var]

train_y = df.Survived

test_X = df1[prediction_var]

test_y = df3.Survived
model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
def model(model,data,prediction,outcome):

    kf = KFold(data.shape[0], n_folds=10) 
def classification_model(model,data,prediction_input,output):

    model.fit(data[prediction_input],data[output]) 

    predictions = model.predict(data[prediction_input])

    accuracy = metrics.accuracy_score(predictions,data[output])

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    kf = KFold(data.shape[0], n_folds=5)

    error = []

    for train, test in kf:

        train_X = (data[prediction_input].iloc[train,:])

        train_y = data[output].iloc[train]

        model.fit(train_X, train_y)

        test_X=data[prediction_input].iloc[test,:]

        test_y=data[output].iloc[test]

        error.append(model.score(test_X,test_y))

        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    
model = RandomForestClassifier(n_estimators=100)

prediction_var = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

outcome_var= "Survived"

classification_model(model,df,prediction_var,outcome_var)