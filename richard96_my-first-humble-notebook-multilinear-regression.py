import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

print(len(dataset.index))

dataset.head()
dataset.isna().values.any()
print(dataset['Country'].value_counts())

plt.bar(dataset['Country'].value_counts().index,dataset['Country'].value_counts().values)

plt.xticks(rotation=60)

plt.xlabel('Country')

plt.ylabel('Number of passengers')
print("The number of people on the boat divided by sex:" )

print(dataset['Sex'].value_counts())



fig1, ax1 = plt.subplots()

ax1.pie(dataset['Sex'].value_counts(), explode=(0.1, 0), labels=['Men', 'Woman'], autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.title('Sex')

plt.show()
print("Age range:" )

print(dataset['Age'].value_counts())

k=dataset['Age'].value_counts()

plt.bar(k.index,k.values)

plt.xlabel('Age')

plt.ylabel('Number of occurences')

plt.title('Age range on the boat')
print("The number of people on the boat divided by the category atribute:" )

print(dataset['Category'].value_counts())



fig1, ax1 = plt.subplots()

ax1.pie(dataset['Category'].value_counts(), explode=(0.1, 0), labels=['Passenger', 'Crew'], autopct='%1.1f%%',

        shadow=True, startangle=90,colors=['grey','Yellow'])

ax1.axis('equal')

plt.title('Category')

plt.show()
print(dataset['Survived'].value_counts())

fig1, ax1 = plt.subplots()

ax1.pie(dataset['Survived'].value_counts(), explode=(0.1, 0), labels=['Dead', 'Alive'], autopct='%1.1f%%',

        shadow=True, startangle=90,colors=['orange','Grey'])



ax1.axis('equal')

plt.title('Survived')

plt.show()
dataset=dataset.drop(['Firstname','Lastname','PassengerId'],axis=1)

dataset.head()
dataset["Country_Sweden"] = np.where(dataset["Country"]=="Sweden",1, 0)

dataset["Country_Estonia"] = np.where(dataset["Country"]=="Estonia", 1 ,0)

dataset["Other_Country"] = np.where((dataset["Country"]!="Sweden") & (dataset["Country"]!="Estonia")  , 1, 0)

dataset=dataset.drop("Country",axis=1)



dataset.head()
dataset=pd.get_dummies(dataset, columns=["Category"],drop_first=True)

dataset=pd.get_dummies(dataset, columns=["Sex"],drop_first=True)



dataset.head()
print(dataset['Age'].describe()[['mean']])



dataset["Age_under44"] = np.where(dataset["Age"]<45, 1, 0)

dataset["Age_over44"] = np.where(dataset["Age"]>=45, 1, 0)

dataset=dataset.drop("Age",axis=1)

dataset.head()
y = dataset.iloc[:, 0].values

X = dataset.iloc[:, 1:].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y,random_state=6)





from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)

score = regressor.score(X_test,y_test)

print("Multilinear regression score is: %.2f%% " % (score* 100.00))
from sklearn import tree

clf = tree.DecisionTreeClassifier()

fig, ax = plt.subplots(figsize=(24, 12))

tree.plot_tree(clf.fit(X_train, y_train), max_depth=4, fontsize=10)

plt.show()



print("Mean average accuracy is: %.2f%% "% (clf.score(X_test,y_test)*100.0))
import xgboost as xgb



dtrain = xgb.DMatrix(X_train,y_train)

dtest = xgb.DMatrix(X_test,y_test)

param = {'max_depth': 6, 'eta': 0.3, 'objective': 'binary:logistic'}

param['eval_metric'] = ['auc', 'rmse','map']



evallist = [(dtest, 'eval'), (dtrain, 'train')]

num_round = 100

evals_result ={}

bst = xgb.train(param, dtrain, num_round, evallist, evals_result=evals_result)

from sklearn.metrics import accuracy_score

y_pred = bst.predict(dtest)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

print("Area under curve score is:"+"{}".format(evals_result['eval']['auc'][-1]*100.0)+"%")

print("Mean average precision is:"+"{}".format(evals_result['eval']['map'][-1]*100.0)+"%")

ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

xgb.plot_importance(bst)
from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense

import tensorflow as tf



model = Sequential()

model.add(Dense(24, input_dim=7, activation='relu'))

model.add(Dense(18, activation='relu'))

model.add(Dense(12, activation='relu'))

model.add(Dense(6, activation='relu'))

model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=3)



score = model.evaluate(X_test, y_test,verbose=0)



print('Test loss: %.2f%%'% (score[0]*100)) 

print('Test accuracy: %.2f%%'% (score[1]*100))
clas = ['0','1']

import sklearn

from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict(X_test)

rounded_predictions = np.argmax(predictions, axis=-1)

cm = confusion_matrix(y_test, rounded_predictions)

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay



disp = ConfusionMatrixDisplay(confusion_matrix=cm,

                              display_labels=clas)

disp = disp.plot(cmap='Greens')

plt.show()
