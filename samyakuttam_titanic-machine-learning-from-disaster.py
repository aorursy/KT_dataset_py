# import all required libraries for reading, analysing and visualizing data

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
# read the data from the train and test csv files

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.head()
print('Training dataset shape: ', train_df.shape)
train_df.info()
# describe gives statistical information about all columns in the dataset

train_df.describe(include = 'all')
# check if any of the columns has null values

train_df.isnull().sum()
# drop the columns which are not needed

train_df.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1, inplace = True)
# Replace the colums 'SibSp' & 'Parch' with a single column 'Family' which represents whether the passenger had any family

# member aboard or not

train_df['Family'] = train_df['SibSp'] + train_df['Parch']



# drop tht 'SibSp' and 'Parch' columns

train_df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)
z = lambda x: x.strip().split(',')[1].split('.')[0]

train_df['Title'] = train_df['Name'].apply(z)



# drop tht 'Name' column

train_df.drop('Name', axis = 1, inplace = True)
# fill the null values of 'Age' column with the mean of the column values

train_df.Age.fillna(train_df['Age'].mean(), inplace = True)

train_df.Age.fillna('S', inplace = True)
# convert the values of 'Embarked' column to numerical format

z = lambda x: 1 if x == 'C' else (2 if x == 'Q' else 3)

train_df['Embarked'] = train_df['Embarked'].apply(z)
# convert the values of 'Sex' column to numerical format

z = lambda x: 1 if x == 'male' else 0

train_df['Sex'] = train_df['Sex'].apply(z)
plt.figure(figsize = [8,5])

sns.distplot(train_df['Age']);
# (Pclass, Sex, Embarked) Vs Survived

fig, (axis1,axis2,axis3) = plt.subplots(1, 3, figsize = (18,5))

sns.barplot(x = 'Pclass', y = 'Survived', data = train_df, ax = axis1)

sns.barplot(x = 'Sex', y = 'Survived', data = train_df, ax = axis2)

sns.barplot(x = 'Embarked', y = 'Survived', data = train_df, ax = axis3);
fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (16, 5))

sns.countplot(x = 'Family', data = train_df, ax = axis1)

sns.barplot(x = 'Family', y = 'Survived', data = train_df, ax = axis2);
fig, (axis1, axis2) = plt.subplots(2, 1, figsize = (16, 10))

sns.countplot(x = 'Title', data = train_df, ax = axis1)

sns.barplot(x = 'Title', y = 'Survived', data = train_df, ax = axis2);
y = train_df.Title.value_counts() > 10

train_df['Title'] = train_df['Title'].apply(lambda x: 'Others' if y[x] == False else x)



train_df = pd.concat([train_df, pd.get_dummies(train_df['Title'])], axis = 1)



# drop the 'Name' column

train_df.drop('Title', axis = 1, inplace = True)
train_df.head()
# import the required modules

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# get the training and test data from the dataframes

(Y_train, X_train) = (train_df['Survived'].values, train_df.drop(['Survived'], axis = 1).values)

Y_train = Y_train.reshape(X_train.shape[0], 1)

(X_train, X_test, Y_train, Y_test) = train_test_split(X_train,Y_train)



print("X_train shape:" + str(X_train.shape))

print("Y_train shape:" + str(Y_train.shape))

print("X_test shape:" + str(X_test.shape))
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
logreg = LogisticRegression(max_iter = 1000)

logreg.fit(X_train, Y_train)



# train score

lr_train_score = round(logreg.score(X_train, Y_train) * 100, 2)

lr_test_score = round(logreg.score(X_test, Y_test) * 100, 2)

# predicted output

Y_pred_lr = logreg.predict(X_test)



print('Logistic Regression train score: ', lr_train_score)

print('Logistic Regression test score: ', lr_test_score)

print('Classification Report: \n', classification_report(Y_test, Y_pred_lr))

print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred_lr))
rf_cl = RandomForestClassifier()

rf_cl.fit(X_train, Y_train)



# train score

rf_train_score = round(rf_cl.score(X_train, Y_train) * 100, 2)

rf_test_score = round(rf_cl.score(X_test, Y_test) * 100, 2)

# predicted output

Y_pred_rf = rf_cl.predict(X_test)



print('Random Forest train score: ', rf_train_score)

print('Random Forest test score: ', rf_test_score)

print('Classification Report: \n', classification_report(Y_test, Y_pred_rf))

print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred_rf))
svm_cl = svm.SVC(kernel = 'linear')

svm_cl.fit(X_train, Y_train)



# train score

svm_train_score = round(svm_cl.score(X_train, Y_train) * 100, 2)

svm_test_score = round(svm_cl.score(X_test, Y_test) * 100, 2)

# predicted output

Y_pred_svm = svm_cl.predict(X_test)



print('SVM train score: ', svm_train_score)

print('SVM test score: ', svm_test_score)

print('Classification Report: \n', classification_report(Y_test, Y_pred_svm))

print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred_svm))
mpl_clf = MLPClassifier(hidden_layer_sizes = (20, 10), max_iter=1000, activation='logistic')

mpl_clf.fit(X_train, Y_train)



# train score

mpl_train_score = round(mpl_clf.score(X_train, Y_train) * 100, 2)

mpl_test_score = round(mpl_clf.score(X_test, Y_test) * 100, 2)

# predicted output

Y_pred_mpl = svm_cl.predict(X_test)



print('MPL train score: ', mpl_train_score)

print('MPL test score: ', mpl_test_score)

print('Classification Report: \n', classification_report(Y_test, Y_pred_svm))

print('Confusion Matrix:\n', confusion_matrix(Y_test, Y_pred_mpl))
from keras.models import Sequential

from keras.layers import Input, Dense
model = Sequential()

model.add(Input(shape = (11, )))

model.add(Dense(20, activation = 'relu'))

model.add(Dense(15, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))
# summary of the model

model.summary()
# compile the model

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# iterate on training data with mini batch size of 32

history = model.fit(X_train, Y_train, epochs = 50, batch_size = 16, validation_split = 0.2)
# plot training and validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Val'], loc = 'upper left')

plt.show()



# plot training and validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train', 'Val'], loc = 'upper left')

plt.show()
# find the accuracy on train and test set

train_loss, train_acc = model.evaluate(X_train, Y_train)

test_loss, test_acc = model.evaluate(X_test, Y_test)

print("Accuracy on train set is %f" %(train_acc * 100)  + "%")

print("Accuracy on test set is %f" %(test_acc * 100)  + "%")
ids = np.array([i + X_train.shape[0] + 1 for i in range(X_test.shape[0])])

ans = {'PassengerId': ids, 'Survived': Y_pred_lr}
df = pd.DataFrame(ans)
df.to_csv('output.csv', index = False)