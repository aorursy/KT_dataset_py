import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
%matplotlib inline

dfReset= pd.read_csv("../input/titanic/train.csv")
df = dfReset.copy()
df.head()
df.isna().sum().sort_values(ascending = False)
df[['Cabin', 'Age', 'Embarked']].head()
# Exploring mean values

sea.pairplot(df[['Age', 'Sex']], hue = 'Sex')
sea.pairplot(df[['Age', 'Pclass']], hue = 'Pclass')

groupbySex = df.groupby('Sex').mean()['Age']
groupbyPclass = df.groupby('Pclass').mean()['Age']

print(groupbySex)
print('\n')
print(groupbyPclass)
groupbySexPclass = df.groupby(['Sex', 'Pclass']).describe()['Age']
groupbySexPclass
meanAgeSexPclass = df.groupby(['Sex', 'Pclass']).mean()['Age']
df.set_index(['Sex', 'Pclass'], inplace = True)
df['Age'].fillna(meanAgeSexPclass, axis = 'index',inplace = True)
df['Age'] = df['Age'].round()
df.reset_index(inplace = True)

df['Age'].isna().sum()
# Dropping Embarked and Cabin

df.drop(['Embarked', 'Cabin'], axis = 'columns', inplace = True)
df.head()
df.drop(['Ticket', 'Pclass', 'Name', 'PassengerId'], axis = 'columns', inplace = True)
df.head()
df['Female'] = np.where(df['Sex'] == 'female', 1, 0)
df.drop('Sex', axis = 'columns', inplace = True)
df.head()
df.corr()['Survived'].sort_values(ascending = False)
X = df.drop('Survived', axis = 'columns')
y = df['Survived']

from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 1)
dfTrain = pd.concat([xTrain, yTrain], axis = 'columns') 
dfTest = pd.concat([xTest, yTest], axis = 'columns')

import statsmodels.formula.api as smf

logitModel = smf.logit('Survived ~ Female + Fare', data = dfTrain).fit()
logitModel.summary()
from sklearn.metrics import classification_report, confusion_matrix

def Results(results, predictions):
    print(classification_report(results, predictions))
    print('\n')
    print('Confusion Matrix')
    print(confusion_matrix(results, predictions))

pred = logitModel.predict(dfTest)
pred = np.where(pred > 0.5, 1, 0)

Results(pred, yTest)
df['FamilySize'] = df['SibSp'] + df['Parch']

plt.figure(figsize = (8, 4))
sea.countplot('FamilySize', data = df, hue = 'Survived')
plt.legend(loc = 'upper right')

plt.figure(figsize = (8, 4))
sea.countplot('FamilySize', data = df[df['Female'] == 1], hue = 'Survived')
plt.legend(loc = 'upper right')

plt.figure(figsize = (8, 4))
sea.countplot('FamilySize', data = df[df['Female'] == 0], hue = 'Survived')
plt.legend(loc = 'upper right')

df[['FamilySize', 'Survived']].corr()
df[df['Female'] == 1].corr()['Survived'].sort_values(ascending = False)
df.drop(['SibSp', 'Parch'], axis = 'columns', inplace = True)

X = df.drop('Survived', axis = 'columns')
y = df['Survived']
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 1)
dfTrain = pd.concat([xTrain, yTrain], axis = 'columns') 
dfTest = pd.concat([xTest, yTest], axis = 'columns')

logitModel2 = smf.logit('Survived ~ Female + Fare + FamilySize', data = dfTrain).fit()
logitModel2.summary()
pred = logitModel2.predict(dfTest)
pred = np.where(pred > 0.5, 1, 0)

Results(pred, yTest)
from sklearn.svm import SVC

X = df.drop('Survived', axis = 'columns')
y = df['Survived']
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 1)

svc = SVC()
svc.fit(xTrain, yTrain)

pred = svc.predict(xTest)
Results(pred, yTest)
from sklearn.model_selection import GridSearchCV

paramGrid = {'C':[0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), paramGrid)
grid.fit(xTrain, yTrain)

pred = grid.predict(xTest)
Results(pred, yTest)
print(grid.best_params_)
print('\n')
print(grid.best_estimator_)
from sklearn.tree import DecisionTreeClassifier

x = df.drop('Survived', axis = 'columns')
y = df['Survived']

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 1)

dtree = DecisionTreeClassifier()
dtree.fit(xTrain, yTrain)

pred = dtree.predict(xTest)
Results(pred, yTest)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

pred = rfc.predict(xTest)
Results(pred, yTest)
from sklearn.preprocessing import scale

dfScaled = df.copy()
dfScaled[['Age', 'Fare', 'FamilySize']] = dfScaled[['Age', 'Fare', 'FamilySize']].apply(lambda x: scale(x))

X = dfScaled.drop('Survived', axis = 'columns')
y = dfScaled['Survived']
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.neighbors import KNeighborsClassifier

errorRate = []

for i in range(1, 50):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(xTrain, yTrain)
    pred = knn.predict(xTest)
    errorRate.append((pred != yTest).mean())

plt.figure(figsize = (10, 6))
plt.plot(range(1,50), errorRate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red', 
         markersize = 10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
optimalNeighbors = errorRate.index(min(errorRate))
knn = KNeighborsClassifier(n_neighbors = optimalNeighbors)
knn.fit(xTrain, yTrain)
pred = knn.predict(xTest)
Results(pred, yTest)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

xTrain = xTrain.values
xTest = xTest.values
yTrain = yTrain.values
yTest = yTest.values

model = Sequential()
earlyStop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 25)

model.add(Dense(5, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

model.fit(x = xTrain, y = yTrain, validation_data = (xTest, yTest), epochs = 1000, batch_size = 100, callbacks = [earlyStop])

pd.DataFrame(model.history.history).plot()
pred = model.predict_classes(xTest)

Results(pred, yTest)
df['Name'] = dfReset['Name']
df['Title'] = df['Name'].apply(lambda x: x.split())
df['Title'] = df['Title'].apply(lambda x: [word for word in x if '.' in word][0][:-1])
df.drop('Name', axis = 'columns', inplace = True)