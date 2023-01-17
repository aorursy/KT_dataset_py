# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import string

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
def prepareFeatuers(df):

    df['CabinPrefix'] = [str(cabinname)[0] for cabinname in df.Cabin]

    df.loc[df.Cabin.isnull(), 'CabinPrefix'] = 'None'

    df['CabinKnown'] = [value for value in df.Cabin.isnull()]

    df['TicketSplitLen'] = [len(t.split()) for t in df.Ticket]

    df['Sex_Ind'] = -1

    df.loc[df.Sex=='female', 'Sex_Ind'] = 1

    df.loc[df.Sex=='male', 'Sex_Ind'] = 2

    df['Age'] = df.Age.fillna(0)

    df['Fare'] = df.Fare.fillna(0)

    return df



train = prepareFeatuers(train)

test = prepareFeatuers(test)
cols = ['Pclass','Sex', 'SibSp', 'Parch', 'Embarked', 'CabinPrefix', 'TicketSplitLen', 'CabinKnown']

for col in cols:

    q = train.groupby(col).Survived.sum()

    # 

    t = train.groupby(col).Survived.sum() + train.groupby(col).Survived.count()

    fig, ax = plt.subplots()

    pos = [i for i,name in enumerate(q.index)]

    vals = [name for i,name in enumerate(q.index)]

    ax.barh(pos, t, color='r', label='died')

    ax.barh(pos, q, label='survived')

    ax.set_yticks(pos)

    ax.set_yticklabels(vals)

    ax.set_ylabel(col)

    ax.legend()
# Create a map to "transform" letters into numbers

letter_map = {}

for i,letter in enumerate(list(string.ascii_lowercase.upper())):

    letter_map[letter] = i+1

letter_map[None] = -1

#df['CabinPrefixInd'] = [letter_map[cabin_prefix] for cabin_prefix in df.CabinPrefix]
# dont forget about 

for col in ['Pclass', 'SibSp','Parch', 'TicketSplitLen', 'Sex', 'CabinPrefix']:

    unique_vals = np.array(train[col].unique())

    unique_vals.sort()

    for unique_value in unique_vals:

        # Perform simultaneously on both sets, to be sure to have the same features

        for df in [train, test]:

            df.loc[df[col] == unique_value, f'{col} {unique_value}'] = 1

            df.loc[df[col] != unique_value, f'{col} {unique_value}'] = 0
print(train.columns)

# Found out what SibSp and Parch means and decided to keep it as one column

feature_cols = ['Age','Fare','Pclass 1', 'Pclass 2', 'Pclass 3',

                'SibSp', 'Parch',

               # 'SibSp 0', 'SibSp 1', 'SibSp 2', 'SibSp 3', 'SibSp 4', 

               # 'SibSp 5', 'SibSp 8', 'Parch 0', 'Parch 1', 'Parch 2', 

               # 'Parch 3', 'Parch 4', 'Parch 5', 'Parch 6',  

                'TicketSplitLen 1', 'TicketSplitLen 2', 'TicketSplitLen 3', 

                'Sex female', 'Sex male', 'CabinPrefix A', 'CabinPrefix B',

                'CabinPrefix C', 'CabinPrefix D', 'CabinPrefix E', 

                'CabinPrefix F', 'CabinPrefix G', 'CabinPrefix None']
from sklearn.model_selection import train_test_split

X = np.array(train[feature_cols])

y = np.array(train.Survived)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.shape



X_submission = np.array(test[feature_cols])
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)

predictions = lr_model.predict(X_test)

accuracy = accuracy_score(y_test,predictions)

precision = precision_score(y_test,predictions)

recall = recall_score(y_test,predictions)

f1 = f1_score(y_test,predictions)

parameters = lr_model.coef_

comparison = pd.DataFrame([['LR', accuracy, precision, recall, f1]], columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])

print(f'Accuracy with LR: {accuracy}')

print(f'Precision with LR: {precision}')

print(f'Recall with LR: {recall}')

print(f'F1 with LR: {f1}')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout



nn_model = Sequential()



# input layer

nn_model.add(Dense(20, activation = "relu", input_shape=(20, )))



# hidden layers

nn_model.add(Dropout(0.3, noise_shape=None, seed=None))

nn_model.add(Dense(64, activation = "relu"))

#nn_model.add(Dropout(0.3, noise_shape=None, seed=None))

#nn_model.add(Dense(128, activation = "relu"))

#nn_model.add(Dropout(0.2, noise_shape=None, seed=None))

#nn_model.add(Dense(64, activation = "relu"))

#nn_model.add(Dropout(0.3, noise_shape=None, seed=None))

nn_model.add(Dense(32, activation = "relu"))

nn_model.add(Dropout(0.2, noise_shape=None, seed=None))

nn_model.add(Dense(16, activation = "relu"))

nn_model.add(Dropout(0.3, noise_shape=None, seed=None))

#nn_model.add(Dense(8, activation = "relu"))

#nn_model.add(Dropout(0.2, noise_shape=None, seed=None))



# output layer

nn_model.add(Dense(1, activation = "sigmoid"))



nn_model.summary()



nn_model.compile(

 optimizer = "adam",

 loss = "binary_crossentropy",

 metrics = ["accuracy"]

)
results = nn_model.fit(

 X_train, y_train,

 epochs= 200,

 batch_size = 48,

 validation_data = (X_test, y_test)

)



f, axes = plt.subplots(1,2, figsize=(10,5))

axes[0].plot(results.history['loss'])

axes[0].plot(results.history['val_loss'])

axes[0].set_title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

axes[0].grid(color='grey')





axes[1].plot(results.history['acc'])

axes[1].plot(results.history['val_acc'])

axes[1].set_title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

axes[1].grid(color='grey')

plt.show()



predictions = nn_model.predict(X_test)

accuracy = accuracy_score(y_test,predictions.round())

precision = precision_score(y_test,predictions.round())

recall = recall_score(y_test,predictions.round())

f1 = f1_score(y_test,predictions.round())

comparison = comparison.append({'Model':'NN', 'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1':f1}, ignore_index=True)

print(f'Accuracy with NN: {accuracy}')

print(f'Precision with NN: {precision}')

print(f'Recall with NN: {recall}')

print(f'F1 with NN: {f1}')
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)



# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)



# Fit the random search model

rf_random.fit(X_train, y_train)

best_random = rf_random.best_estimator_
predictions = best_random.predict(X_test)

accuracy = accuracy_score(y_test,predictions)

precision = precision_score(y_test,predictions)

recall = recall_score(y_test,predictions)

f1 = f1_score(y_test,predictions)

comparison = comparison.append({'Model':'RF', 'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1':f1}, ignore_index=True)

print(f'Accuracy with RF: {accuracy}')

print(f'Precision with RF: {precision}')

print(f'Recall with RF: {recall}')

print(f'F1 with RF: {f1}')
if 'Model' in comparison.columns:

    comparison = comparison.set_index('Model')

for col in comparison.columns:

    fig, ax = plt.subplots()

    pos = [i for i,name in enumerate(comparison.index)]

    vals = [name for i,name in enumerate(comparison.index)]

    ax.barh(pos, comparison[col], label=col)

    ax.set_yticks(pos)

    ax.set_yticklabels(vals)

    ax.set_ylabel(col)

    ax.set_xlim(0.6,1)
#predictions = nn_model.predict(X_submission)

#submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': [int(p > 0.65) for p in predictions]})

predictions = best_random.predict(X_submission)

submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

submission.to_csv('submission.csv', index=False)

submission.shape
submission.Survived.sum()