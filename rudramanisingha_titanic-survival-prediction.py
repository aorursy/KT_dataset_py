#Imports

#!pip install scikit-learn==0.21.2

import numpy as np

import pandas as pd

import seaborn as sns

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, StratifiedKFold

pd.set_option('max_columns', None, "max_rows", None)

from numpy.random import seed

seed(1002)

tf.random.set_seed(1002)
#Declare Global variables 

train_path = "../input/titanic/train.csv"

test_path = "../input/titanic/test.csv"

mapping = {'Col': 'Other',

           'Major': 'Other',

           'Ms': 'Miss',

           'Mlle': 'Miss',

           'Sir': 'Royal',

           'Jonkheer': 'Royal',

           'Countess': 'Royal',

           'Lady': 'Royal',

           'Capt': 'Other',

           'Dona': 'Royal',

           'Mme': 'Mrs',

           'Don': 'Royal',

           'Dr': 'Other',

           'Rev' : 'Other'}

continuous = ['Age', 'Fare', 'Parch', 'SibSp', 'Family_Size', "Family_Survival"]
def prepare_data(train_path,test_path):

    train = pd.read_csv(train_path)

    test = pd.read_csv(test_path)

    df = pd.concat([train, test], axis = 0, sort = False)

    df["Title"] = df["Name"].str.extract(r"([a-zA-Z]+)\.", expand = True)

    df.replace({"Title": mapping}, inplace = True)

    title_ages = dict(df.groupby('Title')['Age'].median())

    df["age_med"] = df["Title"].apply(lambda a : title_ages[a])

    df["Age"].fillna(df["age_med"], inplace = True)

    #df["Pclass_rel"] = df["Pclass"]

    submit = df[pd.isnull(df["Survived"])][["PassengerId","Survived"]]

    df["Fare"].fillna(df["Fare"][df["Pclass"] == 3].median(), inplace = True)

    df['Family_Size'] = df['Parch'] + df['SibSp'] + 1

    df.loc[:,'FsizeD'] = 'Alone'

    df.loc[(df['Family_Size'] > 1),'FsizeD'] = 'Small'

    df.loc[(df['Family_Size'] > 4),'FsizeD'] = 'Big'

    # Family Survival (https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83)

    df['Last_Name'] = df['Name'].apply(lambda x: str.split(x, ",")[0])

    DEFAULT_SURVIVAL_VALUE = 0.5

    df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

    for grp, grp_df in df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId', 

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

        if (len(grp_df) != 1):

            # A Family group is found.

            for ind, row in grp_df.iterrows():

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin == 0.0):

                    df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

    for _, grp_df in df.groupby('Ticket'):

        if (len(grp_df) != 1):

            for ind, row in grp_df.iterrows():

                if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                    smax = grp_df.drop(ind)['Survived'].max()

                    smin = grp_df.drop(ind)['Survived'].min()

                    passID = row['PassengerId']

                    if (smax == 1.0):

                        df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1

                    elif (smin == 0.0):

                        df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

    df['Embarked'].fillna(method='backfill', inplace=True)

    df['Sex'] = df['Sex'].astype('category').cat.codes

    df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId', 'age_med', 'Last_Name'], axis=1, inplace=True)

    df = pd.get_dummies(df, columns = ["Embarked", "Pclass", "Title", "FsizeD"])

    scaler = StandardScaler()

    for var in continuous:

        df[var] = scaler.fit_transform(df[var].astype('float64').values.reshape(-1, 1))

    x_train = df[pd.notnull(df["Survived"])].drop("Survived",axis = 1)

    y_train = df[pd.notnull(df["Survived"])]["Survived"].astype(int)

    x_test = df[pd.isnull(df["Survived"])].drop("Survived",axis = 1)

    return x_train, y_train, x_test, submit
x_train, y_train, x_test, submit = prepare_data(train_path, test_path)
layers = [[8],[16],[8,4],[16,8],[24,16,8],[24,8]]

activation = ["relu","linear","tanh"]

optimizer = ["SGD","RMSprop","Adam"]

dropout = [0.0,0.2]

batch_size = [32,64,128]

epochs = [50,75]

param_grid = dict(batch_size = batch_size, 

                  epochs = epochs,

                  lyr = layers,

                  act = activation,

                  opt = optimizer, 

                  dr = dropout)
def create_model(lyr = [13,8], act = "relu", opt = "adam", dr = 0.2):

    model = Sequential()

    model.add(Dense(lyr[0], input_dim = 22, activation = act))

    model.add(Dropout(dr))

    for i in lyr[1:]:

        model.add(Dense(i, activation = act))

    model.add(Dense(1, activation = "sigmoid"))

    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

    return model
def search():

  model = KerasClassifier(build_fn = create_model, verbose = 1)

  grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1, cv = 2, verbose=2)

  grid_result = grid.fit(x_train, y_train)

  return grid_result
estimator = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 10, verbose = 0)

kfold = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = False)

results = cross_val_score(estimator, x_train, y_train, cv = kfold)

print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
estimator.fit(x_train, y_train, epochs = 100, batch_size = 10)
submit["Survived"] = estimator.predict(x_test)

submit["Survived"] = [int(np.round(x,0)) for x in submit["Survived"]]

submit.to_csv('predictions.csv', index=False)

submit.head()