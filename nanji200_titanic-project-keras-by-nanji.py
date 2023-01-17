# This kernel borrows heavily from https://www.kaggle.com/nanji200/titanic-neural-networks-keras-81-8-e46e71/edit

# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Data processing, metrics and modeling

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import StratifiedKFold

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense, Dropout



# Model evaluations

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



pd.set_option('display.max_colwidth', 0)



hypertuning = 0
test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")
def exploreData(df,dfName):

    #Set display

    #pd.options.display.max_columns=15

    #pd.options.display.max_rows=892



    print('SNAPSHOT OF '+ dfName)

    df.info()

    print('\n')



    print('BASIC DESCRIPTION')

    print(df.describe())

    print('\n')



    print('SNAPSHOT OF FIRST 8 RECORDS')

    print(df.head(8))

    print('\n')



    # List missing values

    print('MISSING VALUE SUMMARY')

    total = df.isnull().sum().sort_values(ascending=False)

    percent_1 = df.isnull().sum()/df.isnull().count()*100

    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

    print(missing_data.head(5))   
 # Describe features

des='''DESCRIPTION OF FEATURES:

survival:    Survival 

PassengerId: Unique Id of a passenger. 

pclass:    Ticket class     

sex:    Sex     

Age:    Age in years     

sibsp:    # of siblings / spouses aboard the Titanic     

parch:    # of parents / children aboard the Titanic     

ticket:    Ticket number     

fare:    Passenger fare     

cabin:    Cabin number     

embarked:    Port of Embarkation'''

print(des)
exploreData(train_df,'TRAIN Dataset')
exploreData(test_df,'TEST Dataset')
import re

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

genders = {"male": 0, "female": 1}

ports = {"S": 0, "C": 1, "Q": 2}

data = [train_df, test_df]



for dataset in data:

#add new column "relatives"

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']   

#Since the Embarked feature has only 2 missing values, we will just fill these with the most common one

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#Create new column IsThereCabinData

    dataset.loc[dataset['Cabin'].isna(), 'IsThereCabinData'] = 0

    dataset.loc[dataset['Cabin'].notna(), 'IsThereCabinData'] = 1

    dataset['IsThereCabinData'] = dataset['IsThereCabinData'].astype(int)

#create new column "deck"

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].fillna('U')

# Title

    dataset['Title'] = dataset['Name']

# Cleaning name and extracting Title

    for name_string in dataset['Name']:

        dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=True)    

# Replacing rare titles 

    mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 

               'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 

               'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 

               'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}           

    dataset.replace({'Title': mapping}, inplace=True)

    titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']

# Replacing missing age by median/title 

    for title in titles:

        age_to_impute = dataset.groupby('Title')['Age'].median()[titles.index(title)]

        dataset.loc[(dataset['Age'].isnull()) & (dataset['Title'] == title), 'Age'] = age_to_impute    

# New feature : Family_size

    dataset['Family_Size'] = dataset['Parch'] + dataset['SibSp'] + 1

    dataset.loc[:,'FsizeD'] = 'Alone'

    dataset.loc[(dataset['Family_Size'] > 1),'FsizeD'] = 'Small'

    dataset.loc[(dataset['Family_Size'] > 4),'FsizeD'] = 'Big'

# Replacing missing Fare by median/Pclass 

    fa = dataset[dataset["Pclass"] == 3]

    dataset['Fare'].fillna(fa['Fare'].median(), inplace = True)

#  New feature : Child

    dataset.loc[:,'Child'] = 1

    dataset.loc[(dataset['Age'] >= 18),'Child'] =0
#drop columns

col_list = ['Cabin','Name','Parch', 'SibSp','Ticket', 'Family_Size', 'Embarked', 'Deck']

train_df = train_df.drop(columns = col_list)

test_df = test_df.drop(columns = col_list)
le = LabelEncoder()

data = [train_df, test_df]

for dataset in data:

    # Binary columns with 2 values like Sex

    bin_cols = dataset.nunique()[dataset.nunique() == 2].keys().tolist()

    # Multi value columns with 3 to 11 values. i.e columns with category data like Pclass

    multi_cols = dataset.nunique()[(dataset.nunique() > 2) & (dataset.nunique() < 12)].keys().tolist()  

    # numerical columns with > 11 values like Fare, Age etc.

    num_cols = dataset.nunique()[dataset.nunique() > 11].keys().tolist() 

    

std = StandardScaler()



# TRAIN

# Binary columns - perform label encoding

for i in bin_cols :

    train_df[i] = le.fit_transform(train_df[i])

# Multi value columns - perform one hot encoding

train_df = pd.get_dummies(train_df,columns = multi_cols)  

# Numerical columns - perfrom scaling/normalization

scaled = std.fit_transform(train_df[num_cols])

scaled = pd.DataFrame(scaled,columns = num_cols)

# dropping original values merging scaled values for numerical columns

train_df = train_df.drop(columns = num_cols,axis = 1)

train_df = train_df.merge(scaled,left_index = True,right_index = True,how = "left")

train_df = train_df.drop(columns = ['PassengerId'],axis = 1)



# TEST

# Binary columns - perform label encoding

for i in bin_cols :

    test_df[i] = le.fit_transform(test_df[i])

# Multi value columns - perform one hot encoding

test_df = pd.get_dummies(test_df,columns = multi_cols)   

# Numerical columns - perfrom scaling/normalization

scaled = std.fit_transform(test_df[num_cols])

scaled = pd.DataFrame(scaled,columns = num_cols)

# dropping original values merging scaled values for numerical columns

test_df = test_df.drop(columns = num_cols,axis = 1)

test_df = test_df.merge(scaled,left_index = True,right_index = True,how = "left")

test_df = test_df.drop(columns = ['PassengerId'],axis = 1)
# Deck_T feature is in train but not in test, let us remove it from train

#train_df = train_df.drop(columns = ['Deck_T'],axis = 1)

# X and Y

X_train = train_df.iloc[:, 1:39].as_matrix()

y_train = train_df.iloc[:,0].as_matrix()
# baseline model

def create_baseline():

    # create model

    model = Sequential()

    model.add(Dense(13, input_dim = 26, activation = 'relu')) # 26: number of input columns in train/test

    model.add(Dropout(0.2))

    model.add(Dense(8, activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    return model



estimator = KerasClassifier(build_fn = create_baseline, epochs = 20, batch_size = 10, verbose = 1)

kfold = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = False)

results = cross_val_score(estimator, X_train, y_train, cv = kfold)

print("Results: Mean score: %.2f%% (STD score: %.2f%%)" % (results.mean()*100, results.std()*100))
# define the grid search parameters

batch_size = [20, 40, 60, 80]

epochs = [20, 50, 100, 150, 200]

param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)

grid_result = grid.fit(X_train, y_train)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# X Test

X_test = test_df.as_matrix()

#estimator.fit(X_train, y_train, epochs = 20, batch_size = 10)

estimator.fit(X_train, y_train, epochs = 150, batch_size = 40)

score = estimator.model.evaluate(X_train, y_train, batch_size=40)

print("Score=", score)

# Predicting y_test

prediction = estimator.predict(X_test).tolist()
se = pd.Series(prediction)

# Creating new column of predictions in data_check dataframe

data_check =  pd.read_csv("../input/test.csv")

data_check['check'] = se

data_check['check'] = data_check['check'].str.get(0)

series = []

for val in data_check.check:

    if val >= 0.5:

        series.append(1)

    else:

        series.append(0)

data_check['final'] = series

temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])

temp['Survived'] = data_check['final']

temp.to_csv("submission.csv", index = False)
train_df.columns.values
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')
FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)

FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )

FacetGrid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_df)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
data = [train_df, test_df]

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)
train_df
train_df['not_alone'].value_counts()
axes = sns.factorplot('relatives','Survived', 

                      data=train_df, aspect = 2.5, )