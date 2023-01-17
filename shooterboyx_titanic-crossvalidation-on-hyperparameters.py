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
# Load data

train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

train_data.head()
import re



def preprocessing(data):

    """

    Preprocessing of the Titanic data

    

    Parameters:

        data (DataFrame): Titanic data

    """

    # Replace NaN values by mean of values if possible

    data['Cabin']=data['Cabin'].replace({np.NaN:'none'})

    data['Embarked']=data['Embarked'].replace({np.NaN:'none'})

    data['Age']=data['Age'].replace({np.NaN:np.mean(data['Age'])})

    data['Fare']=data['Fare'].replace({np.NaN:np.mean(data['Fare'])})

    # Names preprocessing 

    data.loc[:,'Name']=data.loc[:,'Name'].str.lower()

    data.loc[:,'Name']=data.loc[:,'Name'].str.replace('\W+', ' ') # remove special chars

    # Splitting data in attributes and labels

    data_x = data.loc[:, data.columns != 'Survived']

    data_y = data.loc[:, data.columns == 'Survived']

    return data_x, data_y
train_data_x, train_data_y = preprocessing(train_data)

test_data_x, test_data_y = preprocessing(test_data)

train_data_x.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer



def encode(train_x, test_x):

    """

    Encode the gender and embark labels

    Count vectorize their names

    

    Parameters:

        train_x (DataFrame): Titanic training data

        test_x  (DataFrame): Titanic testing data

    """

    gender_encoder = LabelEncoder()

    embark_encoder = LabelEncoder()

    ticket_encoder = LabelEncoder()

    cabin_encoder = LabelEncoder()

    

    ### Labelling

    ## Gender

    train_x, test_x = train_x.copy(), test_x.copy()

    train_x.loc[:,'Sex'] = gender_encoder.fit_transform(train_x.loc[:,'Sex'])

    test_x.loc[:,'Sex'] = gender_encoder.transform(test_x.loc[:,'Sex'])

    

    ## Embark

    train_x.loc[:,'Embarked'] = embark_encoder.fit_transform(train_x.loc[:,'Embarked'])

    test_x.loc[:,'Embarked'] = embark_encoder.transform(test_x.loc[:,'Embarked'])

    

    ## Ticket (assume that their ticket number is at least 3 numbers long)

    # Extract the numbers and  add to new column

    train_ticket_numbers = train_x.loc[:,'Ticket'].str.extract('(\d{3,})')

    train_x.insert(7, "Ticket Nr.", train_ticket_numbers, True)

    train_x.loc[:,'Ticket Nr.'] = train_x.loc[:,'Ticket Nr.'].replace(np.nan, '0')

    # Remove the numbers from Ticket column and fit the encoder

    train_x.loc[:,'Ticket'] = train_x.loc[:,'Ticket'].str.replace('(\d{3,})','')

    

    

    # Extract the numbers and add to new column

    test_ticket_numbers = test_x.loc[:, 'Ticket'].str.extract('(\d{3,})')

    test_x.insert(7, "Ticket Nr.", test_ticket_numbers, True)

    test_x.loc[:,'Ticket Nr.'] = test_x.loc[:,'Ticket Nr.'].replace(np.nan, '0')

    # Remove the numbers from Ticket column and label encode the rest

    test_x.loc[:, 'Ticket'] = test_x.loc[:,'Ticket'].str.replace('(\d{3,})', '')

    

    # Fit to both

    ticket_encoder = ticket_encoder.fit(test_x.append(train_x).loc[:,'Ticket'])

    

    # Set the transformed values

    train_x.loc[:,'Ticket'] = ticket_encoder.transform(train_x.loc[:,'Ticket'])

    test_x.loc[:, 'Ticket'] = ticket_encoder.transform(test_x.loc[:,'Ticket'])

    

    ## Cabin

    # Extract the letters and  add to new column

    train_cabin_letter = train_x.loc[:, 'Cabin'].str.extract('([A-Z])')

    train_x.insert(10, "Cabin Letter", train_cabin_letter, True)

    train_x.loc[:,'Cabin Letter'] = train_x.loc[:,'Cabin Letter'].replace({np.NaN:'none'})

    # Remove the letters from the cabin number

    train_x.loc[:,'Cabin'] = train_x.loc[:,'Cabin'].str.replace('([A-Z])', '')

    train_x.loc[:,'Cabin'] = train_x.loc[:,'Cabin'].str.replace('none', '0')

    train_x.loc[:,'Cabin'] = train_x.loc[:,'Cabin'].str.replace(' ', '')

    train_x.loc[:,'Cabin'] = train_x.loc[:,'Cabin'].replace(r'^\s*$', '0', regex=True)

    

    # Extract the letters and  add to new column

    test_cabin_letter = test_x.loc[:, 'Cabin'].str.extract('([A-Z])')

    test_x.insert(10, "Cabin Letter", test_cabin_letter, True)

    test_x.loc[:,'Cabin Letter'] = test_x.loc[:,'Cabin Letter'].replace({np.NaN:'none'})

    # Remove the letters from the cabin number

    test_x.loc[:,'Cabin'] = test_x.loc[:,'Cabin'].str.replace('([A-Z])', '')

    test_x.loc[:,'Cabin'] = test_x.loc[:,'Cabin'].str.replace('none', '0')

    test_x.loc[:,'Cabin'] = test_x.loc[:,'Cabin'].str.replace(' ', '')

    test_x.loc[:,'Cabin'] = train_x.loc[:,'Cabin'].replace(r'^\s*$', '0', regex=True)

    

    # Fit to both

    cabin_encoder = cabin_encoder.fit(test_x.append(train_x).loc[:,'Cabin Letter'])

    

    # Set the transformed values

    train_x.loc[:,'Cabin Letter'] = cabin_encoder.transform(train_x.loc[:,'Cabin Letter'])

    test_x.loc[:,'Cabin Letter'] = cabin_encoder.transform(test_x.loc[:,'Cabin Letter'])

    

    

    ### Count vectorizing

    ## Names

    vectorizer = CountVectorizer()

    train_names = vectorizer.fit_transform(train_x['Name'])

    train_names_vec = pd.DataFrame(train_names.todense(), columns=vectorizer.get_feature_names())

    train_x = pd.concat([train_x, train_names_vec], axis=1)

    train_x = train_x.drop(['Name'], axis=1)

    

    test_names = vectorizer.transform(test_x['Name'])

    test_names_vec = pd.DataFrame(test_names.todense(), columns=vectorizer.get_feature_names())

    test_x = pd.concat([test_x, test_names_vec], axis=1)

    test_x = test_x.drop(['Name'], axis=1)

    return train_x, test_x

    

train_x, test_x = encode(train_data_x, test_data_x)

train_y = train_data_y
from sklearn.naive_bayes import MultinomialNB



def naiveBayes(train_x, train_y):

    """

    Fits a Multinomial Naive Bayes classifier to the given data

    

    Parameters:

        train_x (DataFrame): Titanic training data

        train_y (DataFrame): Titanic training data labels

    """

    bayes = MultinomialNB(alpha=0.4)

    bayes = bayes.fit(train_x, train_y['Survived'].to_numpy())

    return bayes



bayesModelFinal = naiveBayes(train_x, train_y)
from sklearn.tree import DecisionTreeClassifier



def decisionTree(train_x, train_y):

    """

    Fits a Decision tree classifier to the given data

    

    Parameters:

        train_x (DataFrame): Titanic training data

        train_y (DataFrame): Titanic training data labels

    """

    decisionTree = DecisionTreeClassifier()#max_depth= 6, max_features= 800, min_samples_leaf=9)

    decisionTree.fit(train_x, train_y['Survived'].to_numpy())

    return decisionTree



decistionTreeModelFinal = decisionTree(train_x, train_y)
from sklearn.ensemble import RandomForestClassifier



def randomForest(train_x, train_y):

    """

    Fits a Random Forest classifier to the given data

    

    Parameters:

        train_x (DataFrame): Titanic training data

        train_y (DataFrame): Titanic training data labels

    """

    randomForest = RandomForestClassifier()

    randomForest = randomForest.fit(train_x, train_y['Survived'].to_numpy())

    return randomForest



randomForestModelFinal = randomForest(train_x, train_y)
from sklearn.neural_network import MLPClassifier



def mlpClassifier(train_x, train_y):

        mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=100)

        mlp.fit(train_x, train_y['Survived'].to_numpy())

        return mlp



mlpModelFinal = mlpClassifier(train_x, train_y)
from sklearn.model_selection import GridSearchCV, StratifiedKFold



def crossVal(train_x, train_y):

    # Bayes

    bayes = MultinomialNB()

    alpha = np.arange(0.1, 1.1, 0.05)

    n_features = range(100, train_x.shape[1])

    param_grid = dict(alpha=alpha

                      )

    # No real best params

    

    # DecisionTree

    decisionTree = DecisionTreeClassifier()

    max_depth = range(0,200)

    min_samples_leaf = range(0,10)

    max_features = np.arange(0,train_x.shape[0],100)

    param_grid = dict(max_depth=max_depth,

                      min_samples_leaf=min_samples_leaf,

                      max_features=max_features

                     )

    # Best Score:  0.863551672073986

    # Best Params:  {'max_depth': 6, 'max_features': 800, 'min_samples_leaf': 9}

    

    # MLP

    mlp = MLPClassifier()

    hidden_layer_sizes=[(10,10,10), (10,10,10,10), (10,10,10,10,10), (10,10,10,10,10,10),

                        (100,100,100), (100,100,100,100), (100,100,100,100,100), (100,100,100,100,100,100,100)]

    max_iter=[100, 500, 1000]

    param_grid = dict(hidden_layer_sizes=hidden_layer_sizes,

                      max_iter=max_iter

                     )

    # Best Score:  0.6310246674729652

    # Best Params:  {'hidden_layer_sizes': (10, 10, 10, 10, 10, 10), 'max_iter': 500}

    

    # Random Forest

    randomForest = RandomForestClassifier()

    n_estimators = [10,100,500,1000,1500,2000]

    max_depth = [None, 1,2,5,10,15,20,50,100]

    min_samples_leaf = np.arange(0.1,1.1,0.1)

    param_grid = dict(n_estimators=n_estimators,

                      max_depth=max_depth,

                      min_samples_leaf=min_samples_leaf

                      )

    # Best Score:  0.8461495797506732

    # Best Params:  {'max_depth': 10, 'min_samples_leaf': 0.2, 'n_estimators': 2000}

    



    grid = GridSearchCV(estimator=randomForest,

                        param_grid=param_grid,

                        scoring='roc_auc',

                        verbose=1,

                        n_jobs=-1,

                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

                        )

    grid_result = grid.fit(train_x, train_y['Survived'].to_numpy())

    print(grid_result.cv_results_)

    print('Best Score: ', grid_result.best_score_)

    print('Best Params: ', grid_result.best_params_)    

    return



# crossVal(train_x, train_y)
def split(percentage, train_df_x, train_df_y):

    """

    Splits the train data into evaluation and training chunks according to the percentage

    

    Parameters:

        percentage   (float): percentage of the dataframe that should be kept for training

        train_df (DataFrame): the training dataframe that has to be split

        train_y  (DataFrame): the labels for the dataframe that need to be spit

    """

    train_x = train_df_x.sample(frac=percentage,replace=False)

    train_y = train_df_y.sample(frac=percentage,replace=False)

    eval_x = train_df_x.drop(train_x.index)

    eval_y = train_df_y.drop(train_y.index)

    return train_x, train_y, eval_x, eval_y



train_x, train_y, eval_x, eval_y = split(0.7, train_x, train_y)
from sklearn.metrics import classification_report



def predict(model, test_x, test_y, report=False):

    """

    Runs a prediction given the model and the test set

    """

    prediction = model.predict(test_x)

    if report:

        print(np.unique(prediction))

        print(np.unique(test_y))

        print(classification_report(test_y, prediction, digits=20))

    else:

        return prediction

    

    
print("Bayes (Unoptimized):")

bayesModel = naiveBayes(train_x, train_y)

predict(bayesModel, eval_x, eval_y, True)

print("\nBayes (Optimized):")

bayesModelOptimized = naiveBayes(train_x, train_y)

predict(bayesModel, eval_x, eval_y, True)
print("\nDecisionTree (Unoptimized):")

decisionTreeModel = decisionTree(train_x, train_y)

predict(decisionTreeModel, eval_x, eval_y, True)

print("\nDecisionTree (Optimized):")

decisionTreeOptimized = DecisionTreeClassifier(max_depth= 6, max_features= 10, min_samples_leaf=9)

predict(decisionTreeOptimized.fit(train_x, train_y['Survived'].to_numpy()), eval_x, eval_y, True)
print("\nRandomForest (Unoptimized):")

randomForestModel = randomForest(train_x, train_y)

predict(randomForestModel, eval_x, eval_y, True)

print("\nRandomForest (Optimized):")

randomForestModelOptimized = RandomForestClassifier(max_depth=10,  n_estimators=20, random_state=1)

predict(randomForestModelOptimized.fit(train_x, train_y['Survived'].to_numpy()), eval_x, eval_y, True)
print("\nMLPClassifier (Unoptimizied):")

mlpModel = mlpClassifier(train_x, train_y)

predict(mlpModel, eval_x, eval_y, True)

print("\nMLPClassifier (Optimizied):")

mlpModelOptimized = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10, 10), max_iter=500)

predict(mlpModelOptimized.fit(train_x, train_y['Survived'].to_numpy()), eval_x, eval_y, True)
# Load data

train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

test_data=pd.read_csv('/kaggle/input/titanic/test.csv')



# Preprocessing

train_data_x, train_data_y = preprocessing(train_data)

test_data_x, test_data_y = preprocessing(test_data)



# Encoding

train_x, test_x = encode(train_data_x, test_data_x)

train_y = train_data_y



# Training

model = RandomForestClassifier(max_depth=10,  n_estimators=20, random_state=1)

model = model.fit(train_x, train_y['Survived'].to_numpy())



# Predicting

pred = model.predict(test_x)



output = pd.DataFrame({'PassengerId': test_x.PassengerId, 'Survived': pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")