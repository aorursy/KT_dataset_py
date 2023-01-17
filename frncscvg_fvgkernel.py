# TL;DR: 

#        Model 1 kaggle results: 0.73205

#        Model 2 kaggle results: 0.74641     <- este resultado envié
# Imports

import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

import warnings

warnings.simplefilter("ignore")



# Data paths

train_path = '/Users/pacovg/Desktop/ML/train.csv'

test_path = '/Users/pacovg/Desktop/ML/test.csv'



# Original Data Frames

train = pd.read_csv(train_path)

test = pd.read_csv(test_path)



# Target and features

y = train.Survived

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']

X = train[features]
# Getting data ready



def status(feature):

    # To confirm progress while running code

    print('Processing', feature, ': Done')



def get_combined_data():

    train = pd.read_csv('/Users/pacovg/Desktop/ML/train.csv')

    targets = train.Survived

    train.drop(['Survived'], 1, inplace=True)

    combined = train

    combined.reset_index(inplace=True)

    combined.drop(['index', 'PassengerId'], inplace=True, axis=1)

    

    return combined



combined = get_combined_data()



def get_combined_data2():

    test = pd.read_csv('/Users/pacovg/Desktop/ML/test.csv')

    combined2 = test

    combined2.reset_index(inplace=True)

    combined2.drop(['index', 'PassengerId'], inplace=True, axis=1)

    

    return combined2



combined2 = get_combined_data2()
# Filling missing ages in training data



import numpy as np



grouped_train = combined.iloc[:892].groupby(['Sex', 'Pclass'])

grouped_median_train = grouped_train.median()

grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Age']]



def fill_age(row):

    condition = (

        (grouped_median_train['Sex'] == row['Sex']) & 

        (grouped_median_train['Pclass'] == row['Pclass'])

    ) 

    return grouped_median_train[condition]['Age'].values[0]





def process_age():

    global combined

    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    status('age')

    return combined



combined = process_age()
# Filling missing ages in validation data



import numpy as np



grouped_train2 = combined2.iloc[:892].groupby(['Sex', 'Pclass'])

grouped_median_train2 = grouped_train2.median()

grouped_median_train2 = grouped_median_train2.reset_index()[['Sex', 'Pclass', 'Age']]



def fill_age2(row):

    condition2 = (

        (grouped_median_train2['Sex'] == row['Sex']) & 

        (grouped_median_train2['Pclass'] == row['Pclass'])

    ) 

    return grouped_median_train2[condition2]['Age'].values[0]





def process_age2():

    global combined2

    combined2['Age'] = combined2.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

    status('age')

    return combined2



combined2 = process_age2()

# Dropping unused columns



combined.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], inplace=True, axis=1)

combined2.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], inplace=True, axis=1)
# Changing strings to numbers for analysis 



def process_sex():

    global combined

    combined['Sex'] = combined['Sex'].map({'male':1, 'female':0})

    status('Sex')

    return combined

combined = process_sex()



def process_sex2():

    global combined2

    combined2['Sex'] = combined2['Sex'].map({'male':1, 'female':0})

    status('Sex')

    return combined2

combined2 = process_sex2()
# Model 1 and fitting

sur_model = DecisionTreeRegressor(random_state=1)

sur_model.fit(combined,y)
# Test predictions and exporting results from Model 1

val_predictions = sur_model.predict(combined2).astype(int)



output = sur_model.predict(combined2).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('/Users/pacovg/Desktop/ML/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('/Users/pacovg/Desktop/ML/resultss.csv', index=False)



# Model 2 and fitting



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



train_X, test_X, train_y, test_y = train_test_split(combined.as_matrix(), y.as_matrix(), test_size=0.25)



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

surviv_preds = forest_model.predict(test_X).astype(int)

# Test predictions and exporting results from Model 2



output2 = forest_model.predict(combined2).astype(int)

df_output2 = pd.DataFrame()

aux2 = pd.read_csv('/Users/pacovg/Desktop/ML/test.csv')

df_output2['PassengerId'] = aux['PassengerId']

df_output2['Survived'] = output2

df_output2[['PassengerId','Survived']].to_csv('/Users/pacovg/Desktop/ML/resultsss.csv', index=False)
