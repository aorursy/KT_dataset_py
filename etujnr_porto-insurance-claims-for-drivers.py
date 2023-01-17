# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Step 1: Data Preparation

# Loading the required python package for analysis

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.offline as py



py.init_notebook_mode(connected=True)

from plotly.graph_objs import Scatter, Figure, Layout



import plotly.tools as tls

import warnings



import seaborn as sns



plt.style.use('fivethirtyeight')

sns.set_style("whitegrid")



from collections import Counter



warnings.filterwarnings('ignore')



import plotly.graph_objs as go

import plotly.plotly as plpl



# Step 2: Data Overview: file structure & content

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head(10)

print(train.head(10))

pd.set_option('precision', 3)

train.describe()

print(train.describe())

id_test = test['id'].values

target_train = train['target'].values





# Step 3: Data Validation Checks

# To check if there is any null information in the dataset

train.isnull().any().any()

print(train.isnull().any().any())

# We check if there's any NaN in the dataset

train_cp = train

# train_cp = train_cp.replace(-1, np.NaN)

(train_cp == -1).sum()



data = train

col_with_nan = train_cp.columns[train_cp.isnull().any()].tolist()

print("this dataset has %s Rows. \n" % (train_cp.shape[0]))



vars_with_missing = []



for f in train.columns:

    missings = train[train[f] == -1][f].count()

    if missings > 0:

        vars_with_missing.append(f)

        missings_perc = missings / train.shape[0]



print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))

print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))



f, ax = plt.subplots(1, 2, figsize=(20, 15))

train['target'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('target')

ax[0].set_ylabel('')

sns.countplot('target', data=train, ax=ax[1])

ax[1].set_title('target')

plt.show()



# Also, we can prepare a lists of numeric, categorical and binary columns

# All features

all_features = train.columns.tolist()

all_features.remove('target')

# Numerical features

numeric_features = [x for x in all_features if x[-3:] not in ['bin', 'cat']]

# Categorical features

categorical_features = [x for x in all_features if x[-3:]=='cat']

# Binary Features

binary_features = [x for x in all_features if x[-3:]=='bin']

train['target_name'] = train['target'].map({0: 'Not Filed', 1: 'Filed'})



# Very big imbalance in the dataset as we can see from the plot

train_float = train.select_dtypes(include=['float64'])

train_int = train.select_dtypes(include=['int64'])

Counter(train.dtypes.values)

print(Counter(train.dtypes.values))



# Step 4: Feature Inspection

# We would be using correlation plots to inspect the data

# Getting correlation matrix

cor_matrix = train[numeric_features].corr().round(2)

# Plotting heatmap

fig = plt.figure(figsize=(18,18));

sns.heatmap(cor_matrix, annot=True, center=0, cmap=sns.diverging_palette(250, 10, as_cmap=True), ax=plt.subplot(111));

plt.show()



# Exploring the numerical features in the dataset

# Looping through and plotting the numerical features

for column in numeric_features:

    fig = plt.figure(figsize=(20,12))



    # Plotting the Distribution

    sns.distplot(train[column], ax=plt.subplot(221));

    # Label for X-axis

    plt.xlabel(column, fontsize=16);

    # Label for Y-axis

    plt.ylabel('Density', fontsize=16);

    # Adding a title (One for the figure)

    plt.suptitle('Plots for '+column, fontsize=20);



    # The distribution per claim value

    # When the claim is not filed

    sns.distplot(train.loc[train.target==0, column], color='red', label='Claim not filed', ax=plt.subplot(222));

    # When the claim is filed

    sns.distplot(train.loc[train.target==1, column], color='green', label='Claim filed', ax=plt.subplot(222));

    # Legend

    plt.legend(loc='best')

    # Labelling the X-axis

    plt.xlabel(column, fontsize=16);

    # Labelling the Y-axis

    plt.ylabel('Density per Claim Value', fontsize=16);



    # Preparing a boxplot of column per claim value

    sns.boxplot(x="target_name", y=column, data=train, ax=plt.subplot(224));

    # Labelling the X-axis

    plt.xlabel('Is Filed Claim?', fontsize=16);

    # Labelling the Y-axis

    plt.ylabel(column, fontsize=16);

    plt.show()



# Exploring the categorical features

# Looping through and Plotting Categorical features

for column in categorical_features:

    # Figure initiation

    fig = plt.figure(figsize=(18, 12))



    # Number of occurrences per category

    ax = sns.countplot(x=column, hue="target_name", data=train, ax=plt.subplot(211));

    # Labelling the X-axis

    plt.xlabel(column, fontsize=16);

    # Labelling the Y-axis

    plt.ylabel('Number of occurrences', fontsize=16)

    # Adding Title

    plt.suptitle('Plots for ' + column, fontsize=16);



    # Adding the percents over each bar

    # Getting heights of the bars

    height = [p.get_height() for p in ax.patches]

    # Counting number of bar groups

    ncol = int(len(height) / 2)

    # Counting total height of groups

    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2

    # Looping through bars

    for i, p in enumerate(ax.patches):

        # Adding percentages

        ax.text(p.get_x() + p.get_width() / 2, height[i] * 1.01 + 1000,

                '{:1.0%}'.format(height[i] / total[i]), ha="center", size=14)



    # Filed Claims percentage for every value of feature in teh dataset

    sns.pointplot(x=column, y='target', data=train, ax=plt.subplot(212));

    # Labelling the X-axis

    plt.xlabel(column, fontsize=16);

    # Labelling the Y-axis

    plt.ylabel('Filed Claims Percentage', fontsize=16);

    plt.show()



# Exploring the Binary Features in the dataset

# looping through and plotting binary features

for column in binary_features:

    fig = plt.figure(figsize=(18, 12))



    # Finding the number of occurrences per binary value

    ax = sns.countplot(x=column, hue="target_name", data=train, ax=plt.subplot(211));

    # Labelling the X-axis

    plt.xlabel(column, fontsize=16);

    # Labelling the Y-axis

    plt.ylabel('Number of occurrences', fontsize=16)

    # Adding title

    plt.suptitle('Plots for ' + column, fontsize=16);



    # Adding percents over bars

    # Getting heights of our bars

    height = [p.get_height() for p in ax.patches]

    # Counting number of bar groups

    ncol = int(len(height) / 2)

    # Counting total height of groups

    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2

    # Looping through bars

    for i, p in enumerate(ax.patches):

        # Adding percentages

        ax.text(p.get_x() + p.get_width() / 2, height[i] * 1.01 + 1000,

                '{:1.0%}'.format(height[i] / total[i]), ha="center", size=14)



    # Filed Claims percentage for every value of feature

    sns.pointplot(x=column, y='target', data=train, ax=plt.subplot(212));

    # Labelling the X-axis

    plt.xlabel(column, fontsize=16);

    # Labelling the Y-axis

    plt.ylabel('Filed Claims Percentage', fontsize=16);

    plt.show()





# Step 5: Feature Importance

# In this step, we would be creating/trying out different baseline of performance on the problem and check it using some algorithms

# For now, I'll use the following algorithms

# Linear Algorithms

# a. Logistic Regression

# b. Linear Discriminant Analysis

# Nonlinear Algorithms

# a. Gaussian Naive Bayes

# b. Classification & Regression Trees (CART)



'''

# Getting the required python packages

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB



# Preparing the test options and evaluation metrics

num_folds = 10

seed = 8

scoring = 'Accuracy'



X = np.asarray(train.drop(['id', 'target'], axis=1))

Y = np.asarray(train['target'])



validation_size = 0.4

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)



# We generate some results with the linear and non-linear algorithms

models = [('LR', LogisticRegression()),

          ('LDA', LinearDiscriminantAnalysis()),

          ('CART', DecisionTreeClassifier()),

          ('NB', GaussianNB())]

results = []

names = []



for name, model in models:

    print("Training model %s" % (name))

    model.fit(X_train, Y_train)

    result = model.score(X_validation, Y_validation)

    kfold = KFold(n_splits=num_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold)

    results.append(cv_results)

    names.append(name)

    msg = "Classifier score %s: %f" % (name, result)

    print(msg)

print("----- Training Completed!! -----")

'''



# Trying out a different Machine Learning Algorithm learned in class

# to predict the outcome for the drivers

# Multilayer Perceptron using Keras is Implemented

import keras

import sklearn.model_selection

import pandas as pd



# Load Datasets

df_train  = pd.read_csv('../input/train.csv')

df_test   = pd.read_csv('../input/test.csv')

df_submit = pd.read_csv('../input/sample_submission.csv')



# To numpy array - dataset of train

x_all = df_train.drop(['target', 'id'], axis=1).values

y_all = keras.utils.np_utils.to_categorical(df_train['target'].values)



# Catering for imbalanced data

y_all_0 = y_all[y_all[:,1]==0]

y_all_1 = y_all[y_all[:,1]==1]

x_all   = np.concatenate([x_all[y_all[:,1]==0], np.repeat(x_all[y_all[:,1]==1], repeats=int(len(y_all_0)/len(y_all_1)), axis=0)], axis=0)

y_all   = np.concatenate([y_all[y_all[:,1]==0], np.repeat(y_all[y_all[:,1]==1], repeats=int(len(y_all_0)/len(y_all_1)), axis=0)], axis=0)



# Split train/valid datasets

x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x_all, y_all, test_size=0.4, random_state=0)



# Defining the model

model = keras.models.Sequential()

model.add(keras.layers.normalization.BatchNormalization(input_shape=tuple([x_train.shape[1]])))

model.add(keras.layers.core.Dense(32, activation='relu'))

model.add(keras.layers.core.Dropout(rate=0.5))

model.add(keras.layers.normalization.BatchNormalization())

model.add(keras.layers.core.Dense(32, activation='relu'))

model.add(keras.layers.core.Dropout(rate=0.5))

model.add(keras.layers.normalization.BatchNormalization())

model.add(keras.layers.core.Dense(32, activation='relu'))

model.add(keras.layers.core.Dropout(rate=0.5))

model.add(keras.layers.core.Dense(2,   activation='sigmoid'))

model.compile(loss="categorical_crossentropy", optimizer="adadelta",metrics=["accuracy"])

print(model.summary())



# Use Early-Stopping

callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')



# Training the model

model.fit(x_train, y_train, batch_size=1024, epochs=200, validation_data=(x_valid, y_valid), verbose=1, callbacks=[callback_early_stopping])



# Predict test dataset

x_test = df_test.drop(['id'], axis=1).values

y_test = model.predict(x_test)



# Output

df_submit['target'] = y_test[:, 1]

df_submit.to_csv('Output_Submission.csv', index=False)