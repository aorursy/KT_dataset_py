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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head(6)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head(6)
df = train_data.drop(columns=["Name", "Ticket", "Cabin"])

dft = test_data.drop(columns=["Name", "Ticket", "Cabin"])

df.head(6)
survivors = df[df.Survived == 1]



women_survivors = df[(df.Survived == 1) & (df.Sex == "female")]

percent_women_survivors = len(women_survivors) / len(survivors)



men_survivors = df[(df.Survived == 1) & (df.Sex == "male")]

percent_men_survivors = len(men_survivors) / len(survivors)



print(percent_women_survivors, percent_men_survivors)
class1_survivors = df[(df.Survived == 1) & (df.Pclass == 1)]

percent_class1_survivors = len(class1_survivors) / len(survivors)



class2_survivors = df[(df.Survived == 1) & (df.Pclass == 2)]

percent_class2_survivors = len(class2_survivors) / len(survivors)



class3_survivors = df[(df.Survived == 1) & (df.Pclass == 3)]

percent_class3_survivors = len(class3_survivors) / len(survivors)



print(percent_class1_survivors, percent_class2_survivors, percent_class3_survivors)
df.isna().sum()
dft.isna().sum()
df2 = df.copy()

df2.Age = df.Age.fillna(df.Age.mean())

df2.Embarked = df.Embarked.fillna(df.Embarked.mode())



dft2 = dft.copy()

dft2.Age = dft.Age.fillna(dft.Age.mean())

dft2.Fare = dft.Fare.fillna(dft.Fare.mean())



df2.head(6)
df3 = df2.copy()

df3.Sex = df3.Sex.map({"male":0, "female":1})



dft3 = dft2.copy()

dft3.Sex = dft3.Sex.map({"male":0, "female":1})



df3.head(6)
df4 = pd.get_dummies(df3)

dft4 = pd.get_dummies(dft3)

df4.head(6)
dft4.head(6)
X_train = df4.drop(columns=["PassengerId", "Survived"])

X_test = dft4.drop(columns=["PassengerId"])

y_train = df4["Survived"]

X_train.head(6)
from sklearn.preprocessing import MinMaxScaler, StandardScaler



X_train_sc = X_train.copy()

X_test_sc = X_test.copy()

numerical_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

sc = StandardScaler()

X_train_sc[numerical_features] = sc.fit_transform(X_train[numerical_features])

X_test_sc[numerical_features] = sc.transform(X_test[numerical_features])
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, StratifiedKFold



skf = StratifiedKFold(n_splits=8, shuffle=True)

log = LogisticRegression(max_iter=1000, C=0.1)

scores = cross_val_score(log, X_train, y_train, cv=skf)



print(scores)

print(scores.mean())
X_train.head(10)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(criterion='entropy', min_impurity_decrease=0.0004)

scores = cross_val_score(rfc, X_train, y_train, cv=skf)



print(scores)

print(scores.mean())
from sklearn.ensemble import AdaBoostClassifier



abc = AdaBoostClassifier(n_estimators=100)

scores = cross_val_score(abc, X_train, y_train, cv=skf)



print(scores)

print(scores.mean())
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(learning_rate=0.2, min_impurity_decrease=0.0005)

scores = cross_val_score(gbc, X_train, y_train, cv=skf)



print(scores)

print(scores.mean())
from xgboost import XGBClassifier



xgb = XGBClassifier(max_depth=4, learning_rate=0.3, gamma=0.0004, objective='binary:logistic')

scores = cross_val_score(xgb, X_train_sc, y_train, cv=skf)



print(scores)

print(scores.mean())
from sklearn.svm import SVC



svm = SVC()

scores = cross_val_score(svm, X_train_sc, y_train, cv=skf)



print(scores)

print(scores.mean())
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

scores = cross_val_score(mlp, X_train_sc, y_train, cv=skf)



print(scores)

print(scores.mean())
os.mkdir("/kaggle/temp")
from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.callbacks import ModelCheckpoint



checkpoint = ModelCheckpoint("/kaggle/temp/checkpoint.h5", 

                             monitor='val_accuracy', mode='max',

                             save_best_only=True, save_weights_only=True)



ann = Sequential()



ann.add(Dense(10, activation='relu', input_shape=(9,)))

ann.add(BatchNormalization())

ann.add(Dense(10, activation='relu'))

ann.add(BatchNormalization())

ann.add(Dense(1, activation='sigmoid'))



ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train_sc, y_train, 

          batch_size=100, epochs=30, 

          validation_split=0.1, verbose=0, 

          callbacks=[checkpoint])



print(max(ann.history.history["val_accuracy"]))

ann.load_weights("/kaggle/temp/checkpoint.h5")
model = ann

x_train = X_train_sc

x_test = X_test_sc



#model.fit(x_train, y_train)

predictions = model.predict_classes(x_test).ravel()

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")