import pandas as pd 
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data.describe()
train_data.head()
train_data.info()
train_data.isnull().sum()
target = train_data['Survived']
X_data = train_data.drop(columns='Survived')
print(X_data[X_data["Embarked"].isnull()]['Embarked'])
X_data['Embarked'].fillna('NaN', inplace=True)
from sklearn.model_selection import train_test_split



X_train0, X_test, y_train0, y_test = train_test_split(X_data, target, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train0, y_train0, test_size=0.25, random_state=0)
print(len(y_train))

print(len(y_val))

print(len(y_test))
X_train.isnull().sum()
print('nr of "Ticket" entries = {}'.format(len(X_train['Ticket'])))

print('nr of unique "Ticket" entries = {}'.format(len(X_train['Ticket'].unique())))
X_train.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)

X_val.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True)
X_train.info()
X_val.info()
s = (X_train.dtypes == 'object')

object_columns = list(s[s].index)

object_columns
from sklearn.preprocessing import LabelEncoder



# I'll make a copy of 'X_train' to avoid changing the data

le_X_train = X_train.copy()

le_X_val = X_val.copy()



# define the label encoder

label_encoder = LabelEncoder()



for column in object_columns:

    print(column)

    label_encoder.fit(X_train0[column])

    le_X_train[column] = label_encoder.transform(X_train[column])

    le_X_val[column]   = label_encoder.transform(X_val[column])
le_X_val.info()
X_train_mean = le_X_train.copy()

X_val_mean = le_X_val.copy()



X_train_neg = le_X_train.copy()

X_val_neg = le_X_val.copy()



X_train_mean['Age'].fillna(X_train_mean['Age'].mean(), inplace=True)

X_val_mean['Age'].fillna(X_val_mean['Age'].mean(), inplace=True)



X_train_neg['Age'].fillna(0, inplace=True)

X_val_neg['Age'].fillna(0, inplace=True)
X_train_neg.describe()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



model_mean = RandomForestClassifier(random_state=0)

model_mean.fit(X_train_mean, y_train)

pred_mean = model_mean.predict(X_val_mean)

acc_mean = accuracy_score(y_val, pred_mean)

print('accuracy for mean is: {:.4}'.format(acc_mean))
model_neg = RandomForestClassifier(random_state=0)

model_neg.fit(X_train_neg, y_train)

pred_neg = model_neg.predict(X_val_neg)

acc_neg = accuracy_score(y_val, pred_neg)

print('accuracy for mean is: {:.4}'.format(acc_neg))
from sklearn.metrics import confusion_matrix



print('CM mean:{}'.format(confusion_matrix(y_val, pred_mean)))

print()

print('CM neg:{}'.format(confusion_matrix(y_val, pred_neg)))
# first I copy the dataframe to avoid modifying the data

X_train_mean2 = X_train_mean.copy()

X_val_mean2 = X_val_mean.copy()



X_train_mean2.drop(columns=['Ticket', 'Embarked'], inplace=True)

X_val_mean2.drop(columns=['Ticket', 'Embarked'], inplace=True)



# now for the negative one

X_train_neg2 = X_train_neg.copy()

X_val_neg2 = X_val_neg.copy()



X_train_neg2.drop(columns=['Ticket', 'Embarked'], inplace=True)

X_val_neg2.drop(columns=['Ticket', 'Embarked'], inplace=True)
X_train_mean2.head()
model_mean2 = RandomForestClassifier(random_state=0)

model_mean2.fit(X_train_mean2, y_train)

pred_mean2 = model_mean2.predict(X_val_mean2)

acc_mean2 = accuracy_score(y_val, pred_mean2)

print('accuracy mean2 is: {:.4}'.format(acc_mean2))
model_neg2 = RandomForestClassifier(random_state=0)

model_neg2.fit(X_train_mean2, y_train)

pred_neg2 = model_neg2.predict(X_val_mean2)

acc_neg2 = accuracy_score(y_val, pred_neg2)

print('accuracy neg2 is: {:.4}'.format(acc_neg2))
print('CM mean:{}'.format(confusion_matrix(y_val, pred_mean2)))

print()

print('CM neg:{}'.format(confusion_matrix(y_val, pred_neg2)))