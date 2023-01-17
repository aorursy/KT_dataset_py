########################################## KAGGLE SPECIFIC CODE ####################################################

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



###################################################################################################################





##### Let's follow a typical data science workflow #####



##### 1. problem description and objective

# * problem description: Titanic sank on maiden voyage after colliding with an iceberg, killing 2/3 of the passengers

# * objective: predict if a passenger survived or not (binary classification problem)



##### 2. data acquisition

# 2a. early negotiating data access with stakeholders too reduce risk

# 2b. storing the data securely and according to privacy regulations (GDPR)

# 2c. spend your time wisely. More data samples? More features? Artificially augment data?



# in this case Kaggle has peformed data acquisition for us

from sklearn.utils import shuffle

df = pd.read_csv("/kaggle/input/titanic/train.csv")  # it has features and labels

kaggle_competition_data = pd.read_csv("/kaggle/input/titanic/test.csv")  # for the Kaggle competition; it features but has no labels!





##### 3. data exploration using descriptive statistics and visualization

# Some options:

# 3a. find out for each feature: data type

# 3b. find out for each feature: numerical, nominal categorical, ordinal categorical

# 3c. heat map to find correlations between features and between features and target variable



# just a pretty random tryout

men = df.loc[df.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)



# show some data samples to get a feeling about the data

print('\n************************* df.head():')

print(df.head())

# to note:

# * what is the difference between passenger id and ticket number?



# some descriptive statistics

print('\n************************* df.describe():')

print(df.describe())



# find out the types of the features and find the features that contain missing data (NaN)

print('\n************************* df.info():')

print(df.info())

# to note:

# * feature Cabin has very many missing values

# * feature Age has quite some missing values

# * feature Embarked has missing values

# we also need to investigate the kaggle_competition_data

print('\n************************* kaggle_competition_data.info():')

print(kaggle_competition_data.info())

# to note:

# * feature Fare has missing values



# use google to find information about the feature

# http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf

# (int)(numerical)           PassengerId

# (int)(target variable)     Survived  Survival (0 = No; 1 = Yes)

# (int)(ordinal categorical) Pclass    Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)

# (str)(nominal categorical) Name      

# (str)(nominal categorical) Sex       

# (flt)(numerical)           Age       

# (int)(numerical)           SibSp     Number of Siblings/Spouses Aboard

# (int)(numerical)           Parch     Number of Parents/Children Aboard

# (str)(nominal categorical) Ticket    Ticket Number

# (flt)(numerical)           Fare      Passenger Fare (British pound)

# (str)(nominal categorical) Cabin     Cabin (letter of the cabin number indicates the deck)

# (str)(nominal categorical) Embarked  Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)



# heatmap: visualization of correlation between features

# heatmap ignores categorical features that are not numbers

import seaborn as sns

sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1, center= 0)





##### 4. data preprocessing 

# 4a. dealing with missing data

# 4b. encode categorical features as dummy variables



# below we're going to split df into a training and test set to allow model evaluation. To avoid differences between train 

# and test dataset let's shuffle df so that the split will be random. 

df = shuffle(df)



# it is handy to perform data processing on the complete data set at once, so on df and kaggle_competition_data together

# however df contains labels and kaggle_competition_data does not, so let's split df into features and labels

df_y = df['Survived']

df_X = df.drop('Survived', axis=1)

# to remember what part is df and what part is kaggle_competition_data

df_len = len(df)



# * split the data into training, validation and test data sets to allow hyperparameter turning and model evaluation

# * be very careful that no knowledge of the test_set spills into training of the model

# * note that df_X, train_X, validation_X and test_X are REFERENCES pointing to the same dataset. So a change to df will be visible

#   in train_data and test_data and vice versa! This is handy, as it allows to perform some actions on the complete data set

#   at once, but also something to continuously be aware of!!

train_X = df_X[:int(df_len * 0.6)]

validation_X = df_X[int(df_len * 0.6):int(df_len * 0.8)]

test_X = df_X[int(df_len * 0.8):]

train_y = df_y[:int(df_len * 0.6)]

validation_y = df_y[int(df_len * 0.6):int(df_len * 0.8)]

test_y = df_y[int(df_len * 0.8):]

print('length train_X:', len(train_X))

print('length validation_X:', len(validation_X))

print('length test_X:', len(test_X))

print('length test_y:', len(test_y))



# add kaggle_competition_data to df_X

df_X = pd.concat([df_X, kaggle_competition_data], axis=0)



# impute train_data and test_data as random forest alg can't handle missing values

# remember from above, feature Age (numerical), Cabin (categorical) and Embarked (categorical) and Fare (numerical) contain missing values

print('\n************************* df_X.info() before imputation:')

print(df_X.info())

train_X_age_mean = train_X['Age'].mean()

df_X['Age'].fillna(train_X_age_mean, inplace=True)  # use the mean of train data on validation_data and test_data to avoid leakage!

df_X['Cabin'].fillna('VAL', inplace=True)

df_X['Embarked'].fillna('EM', inplace=True)

train_X_fare_mean = train_X['Fare'].mean()

df_X['Fare'].fillna(train_X_fare_mean, inplace=True)  # use the mean of train data on validation_data and test_data to avoid leakage!

print('\n************************* df_X.info() after imputation:')

print(df_X.info())



# hot encode categorical features as random forest cannot handle strings

# * also ordinal categorical features, although of type numerical, need to be one hot encoded as well, 

#   as the distance between the numbers provide wrong information to the training algorithm, affecting the model we train

# * note that get_dummies encodes a missing value into 0,0,0 instead of nan,nan,nan; that's why we perform imputation 

#   *before* one hot encoding

# * what about the features 'Ticket' and 'Cabin'?? 

categorical_features = ['Pclass', 'Sex', 'Embarked']

df_X = pd.get_dummies(df_X, columns=categorical_features)

print('\n************************* df_X.info() after one hot encoding:')

print(df_X.info())

print('\n************************* train_X.info() after one hot encoding:')

print(train_X.info())

# we see that the changes to df_X are not visible from train_X, so do split again

train_X = df_X[:int(df_len * 0.6)]

validation_X = df_X[int(df_len * 0.6):int(df_len * 0.8)]

test_X = df_X[int(df_len * 0.8):df_len]

kaggle_competition_data = df_X[df_len:]

print('\n************************* train_X.info() after one hot encoding and after resplit:')

print(train_X.info())





##### 5. modeling: feature selection



# Based on the data exploration, let's skip the features 'PassengerId', 'Name', 'Ticket', 'Cabin'

features = train_X.columns

features = [f for f in features if f not in ['PassengerId', 'Name', 'Ticket', 'Cabin']]

train_X = train_X[features]

validation_X = validation_X[features]

test_X = test_X[features]

train_X = train_X[features]

kaggle_competition_X = kaggle_competition_data[features]  # renamed to kaggle_competition_X as we need kaggle_competition_data later



##### 6. modeling: model training

# 6a. algorithm selection

# 6b. hyperparameter tuning



# algorithm selection - random forest is flexible, easy to use with great result most of the time

# hyperparameter tuning - for random forest some hyperparameters are n_estimators=100, max_depth=5. A validation data set is used for this.

# Question: why don't we use the test data to perform hyperparameter tuning? Answer: because otherwise the model might become 

# optimized for the test data; this is not good as the model must remain fully independent of the test data to see how well

# the model *generalizes* (== works well on unseen data).

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

for n_estimators in [10, 100, 500]:

    for max_depth in [1, 5, 50]:

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)

        model.fit(train_X, train_y)

        predictions = model.predict(validation_X)

        print('accuracy_score of prediction on validation data, using n_estimators =', n_estimators, 'and max_depth =', max_depth, ':', accuracy_score(validation_y, predictions))

# using n_estimators = 500 and max_depth = 50 gives highest accuracy on the validation data (several attempts), so we'll use these values

model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=1)

model.fit(train_X, train_y)



##### 7. modeling: model evaluation

# 7a. choosing evaluation metric: in this case, Kaggle chooses the evaluation metric "accuracy" for us



# evaluating against the *TRAIN_DATA*

# gives an impression of the BIAS/VARIANCE of the model: to what extent the model is able to fit the train_data

predictions = model.predict(train_X)

print('accuracy_score of prediction on train_data', accuracy_score(train_y, predictions))



# evaluating against the *TEST_DATA*

# shows how well the model GENERALIZES, ability to predict using unseen data

predictions = model.predict(test_X)

print('accuracy_score of prediction on test_data', accuracy_score(test_y, predictions))





##### 8. deployment of the model

# will be done in a separate hands-on workshop

# very important, but easily forgotten, ALL data transformations that are part of data preparation (e.g. feature normalization,

# imputing, one hot encoding) must be done as well when using the model for predictions, otherwise the predictions will be incorrect.





##### 9. Kaggle competition

# This step is not part of the data science workflow, but is specific to a Kaggle competition

# 9a. create a submission file. 



kaggle_competition_predictions = model.predict(kaggle_competition_X)



output = pd.DataFrame({'PassengerId': kaggle_competition_data.PassengerId, 'Survived': kaggle_competition_predictions})  # kaggle_competition_X has no feature PassengerId any more

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")



# 9b. you upload the submission file to the Kaggle competition using the Kaggle GUI. You'll find the submission file in the right sidebar and then

#     under 'data'. After submission, your result will show up in the leader board.
