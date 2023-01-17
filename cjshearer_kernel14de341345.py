# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
combine = [train_df, test_df]
combine_df = pd.concat(combine)
# Let's compare the group sizes found by summing the known number of parents and siblings to the group
# sizes found by counting duplicate tickets.

# we add one here to represent the individual themself, this makes the comparison from Ticket data more 'fair'
sibpar = combine_df.Parch+combine_df.SibSp+1

tik = combine_df.groupby('Ticket', sort=False)['Ticket'].transform('count')

diff = tik-sibpar

plt.title('Group size ')
plt.hist([sibpar, tik, diff], label=['Immediate family (including self)',  'Passengers per Ticket', 'difference'])
plt.legend(loc='upper right')
plt.show()
#combine_df.loc[diff<0].describe()
def create_titles(df):
    titles = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
    titles = titles.replace(['Capt', 'Col', 'Countess', 'Don', 'Dr','Jonkheer','Lady','Major','Rev', 'Sir'], 'Rare')
    titles = titles.replace(['Mlle', 'Ms'], 'Miss')
    titles = titles.replace('Mme', 'Mrs')
    
    titles = titles.map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})    
    return titles

def create_gender(df):
    return df.Sex.map({"male": 0, "female": 1})

def create_origin(df):
    return df.Embarked.map({np.nan : 0, 'S': 1, 'C' : 2, 'Q' : 3})

def process_labeled_data(df):
    # create new features
    df['titles'] = create_titles(df)
    df['gender'] = create_gender(df)
    df['origin'] = create_origin(df)
    
    # Extract unused features and labels from training set
    X = df.drop(['Name','PassengerId','Cabin', 'Sex', 'Embarked','Survived'], axis = 1)
    y = df['Survived']
    return X, y
    
X, y = process_labeled_data(train_df)
# code adapted from https://medium.com/@literallywords/stratified-k-fold-with-keras-e57c487b1416

from sklearn.model_selection import StratifiedKFold

kfold_splits = 10

# Instatiate the cross validator
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)

# Get indicies of split data
indices = list(skf.split(X, y))
# by subtracting the number of parents and siblings from the number of those a given passenger shares a ticket with,
# we find the number of friends, extended family, and others who are traveling with a given passenger 
def create_others(df):
    return df.groupby('Ticket', sort=False)['Ticket'].transform('count')-(df.Parch+df.SibSp)

from sklearn.impute import KNNImputer
def imputation(df):
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df = imputer.fit_transform(df)
    return df

def process_training_fold(X, train_ind):
    # get slice of dataframe from indices
    xtrain = X.iloc[train_ind].copy()
    
    # create new features
    xtrain['others'] = create_others(xtrain)

    # drop unused features
    xtrain.drop(['Ticket'], axis = 1, inplace = True)
    
    # impute missing ages and place them in bins
    age_bin_size = 4
    columns = xtrain.columns
    xtrain = pd.DataFrame(data=imputation(xtrain), columns=columns)
    xtrain.Age, age_bins = pd.qcut(xtrain.Age, q = age_bin_size, retbins=True, labels=range(age_bin_size))
    xtrain.rename(columns={"Age": "age_range"}, inplace=True)
        # set upper and lower bounds for possible ages, we assume all ages will fall between 0 and 200
         # bins[0]= 0
         # bins[bin_size] = 200
    
    # place fare into bins
    fare_bin_size = 5
    xtrain.Fare, fare_bins = pd.qcut(xtrain.Fare, q=fare_bin_size, retbins=True, labels=range(fare_bin_size))
    xtrain.rename(columns={"Fare": "fare_range"}, inplace=True)
    return age_bins, fare_bins, xtrain

#Loop through the indices the split() method returns
# for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):
#     print(X[val_indices])
#     break
#     print("Training on fold",index+1,"out of",kfold_splits)
#     xtrains, xvals = X[train_indices], X[val_indices]

# extract first set of training and validation indices
(train_ind, val_ind) = indices[0]
# get pre-processed dataframe for training data
age_bins, fare_bins, xtrain = process_training_fold(X, train_ind)

xtrain.head()
# Split variables into subject/target

X = train_df.drop(['PassengerId','Name','Ticket','Cabin','Survived'], 1)
y = train_df['Survived']

# Get one-hot encodings

X = pd.get_dummies(X)

# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# Define the keras model

model = Sequential()
model.add(Dense(12, input_dim=10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the keras model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the keras model on the dataset

model.fit(X, y, epochs=150, batch_size=10)
# Evaluate the keras model

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))