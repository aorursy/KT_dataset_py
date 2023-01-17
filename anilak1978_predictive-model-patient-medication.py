# importing neccessary libraries

import pandas as pd

import numpy as np
df=pd.read_csv('../input/drug200.csv')

df.head(5)
# summary of the dataframe

df.info()
# looking to see if there are any missing values

missing_data=df.isnull()

missing_data.head(5)
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print('')
df.describe()
df.corr()
# defining our feature matrix that will predict the target y value(drug)

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

X[0:5]
# turn the categorical variables to numeric variables

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()

le_sex.fit(['F','M'])

X[:,1] = le_sex.transform(X[:,1]) 





le_BP = preprocessing.LabelEncoder()

le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])

X[:,2] = le_BP.transform(X[:,2])





le_Chol = preprocessing.LabelEncoder()

le_Chol.fit([ 'NORMAL', 'HIGH'])

X[:,3] = le_Chol.transform(X[:,3]) 



X[0:5]

# defining the y target value

y=df['Drug']

y[0:5]
# importing neccessary libraries

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset=train_test_split(X, y, test_size=0.3, random_state=3)
X_trainset[0:5]
# creating the Decision Tree Clasifier instance

from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree # it shows the default parameters
# we will fit the X, y trainset. (training the dataset with X, y trainset values)

drugTree.fit(X_trainset,y_trainset)
# our model is ready and we can start defining predictions

predTree = drugTree.predict(X_testset)
print (predTree [0:5])

print (y_testset [0:5])
# our model is ready and we can check the accuracy of the model by importing metrics from sklearn

from sklearn import metrics

import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))