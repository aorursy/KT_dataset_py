# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn import linear_model

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

import utils



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_t = pd.read_csv('../input/test.csv')
df.count()
df.head(3)
df_t.head(3)
import matplotlib.pyplot as plt



fig = plt.figure(figsize=(18, 6)) # create figure



plt.subplot2grid((2,3), (0,0))

df.Survived.value_counts(normalize=True).plot(kind='bar', alpha=0.5) # data that we want to plot

plt.title("Survived")



plt.subplot2grid((2,3), (0,1))

plt.scatter(df.Survived, df.Age, alpha=0.1)

plt.title("Age-Survived")



plt.subplot2grid((2,3), (0,2))

df.Pclass.value_counts(normalize=True).plot(kind='bar', alpha = 0.5)

plt.title("Classes")



plt.subplot2grid((2,3), (1,0), colspan=2)

for x in [1,2,3]:

    # find list of age where Pclass value is 1,2,3

    df.Age[df.Pclass == x].plot(kind="kde")

plt.title("Class-Age")

plt.legend(("1", "2", "3"))



# where did the passengers got on the ship (3 different locations)

plt.subplot2grid((2,3), (1,2))

df.Embarked.value_counts(normalize=True).plot(kind='bar', alpha=0.5)

plt.title("Embarked")



plt.show() # show figure
df.Survived.value_counts()
fig = plt.figure(figsize=(18,6))



plt.subplot2grid((3,4), (0,0))

df.Survived.value_counts(normalize=True).plot(kind='bar', alpha=0.5)

plt.title("Survived (regardless of sex)")



plt.subplot2grid((3,4), (0,1))

df.Survived[df.Sex=="male"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("men survived")



plt.subplot2grid((3,4), (0,2))

df.Survived[df.Sex=="female"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("women survived")



plt.subplot2grid((3,4), (0,3))

df.Sex[df.Survived==1].value_counts(normalize=True).plot(kind="bar", alpha=0.5)

plt.title("sex of survived")



plt.subplot2grid((3,4), (1,0), colspan=3)

for x in [1,2,3]:

    df.Survived[df.Pclass == x].plot(kind="kde")

plt.title("class-survived")

plt.legend(("1", "2", "3"))



plt.subplot2grid((3,4), (2,0))

df.Survived[(df.Sex =='male') & (df.Pclass == 1)].value_counts(normalize=True).plot(kind="bar")

plt.title("Rich men survived")



plt.subplot2grid((3,4), (2,1))

df.Survived[(df.Sex =='male') & (df.Pclass == 3)].value_counts(normalize=True).plot(kind="bar")

plt.title("Poor men survived")



plt.subplot2grid((3,4), (2,2))

df.Survived[(df.Sex =='female') & (df.Pclass == 1)].value_counts(normalize=True).plot(kind="bar")

plt.title("Rich women survived")



plt.subplot2grid((3,4), (2,3))

df.Survived[(df.Sex =='female') & (df.Pclass == 3)].value_counts(normalize=True).plot(kind="bar")

plt.title("Poor women survived")



plt.show()
df_t.head()
# clean data and fill empty columns

df["Fare"] = df["Fare"].fillna(df["Fare"].dropna().median())

df["Age"] = df["Age"].fillna(df["Age"].dropna().median())



# encode categorical data into one hot vectors

labelencoder_1 = LabelEncoder()

df['Sex'] = labelencoder_1.fit_transform(df['Sex'])

# label data : 0,1,2,..

labelencoder_2 = LabelEncoder()

df['Embarked'] = labelencoder_2.fit_transform(df['Embarked'].astype(str))

# slect X, Y

Y = df.iloc[:,1]

X = df.iloc[:, [2,4,5,6,7,9,11]]

# encode the data

onehotencoder = OneHotEncoder(categorical_features=[-1])

X = onehotencoder.fit_transform(X).toarray()



# scale the data

sc = StandardScaler()

X = sc.fit_transform(X)
classifier = linear_model.LogisticRegression()

classifier.fit(X, Y)
classifier.score(X, Y)
print(classifier.coef_)

# pd.DataFrame(classifier.coef_, columns=['Coeff'])
dtree = DecisionTreeClassifier()

dtree.fit(X, Y)

dtree.score(X, Y)



accuracies = cross_val_score(estimator = dtree, X = X, y = Y, cv=10, n_jobs=1, scoring='accuracy')

print(accuracies)



mean = accuracies.mean()

# if we have high variance between different K-fold sets its a sign of overfitting in our training set

variance = accuracies.std()  

print(mean, variance)
# use gridsearch to find better parameters

depths = np.arange(1, 10)

num_leafs = [2, 3, 5, 10, 20, 25]



param_grid = [{'max_depth':depths,

              'max_leaf_nodes':num_leafs,

               'min_samples_split':num_leafs

              }]



grid = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring="accuracy", cv=3)

grid.fit(X, Y)



best_accuracy = grid.best_score_

best_parameters = grid.best_params_



print(best_parameters, best_accuracy)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout 

from keras.wrappers.scikit_learn import KerasClassifier
classifier = Sequential()

classifier.add(Dense(output_dim=100, init='uniform', activation='relu', input_dim=10))

classifier.add(Dropout(p=0.4))

classifier.add(Dense(output_dim=50, init='uniform', activation='relu'))

classifier.add(Dropout(p=0.2))

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X, Y,batch_size=10, epochs=100, validation_split=0.2)