import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

perfect = pd.read_csv('/kaggle/input/perfecttitanic/PerfectScoreTitanic.csv')
# credit: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction. 

# One of the best notebooks on getting started with a ML problem.



def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns



train_missing= missing_values_table(train)

train_missing
test_missing= missing_values_table(test)

test_missing
def combine(train,test):

    merged = pd.concat([train,test])

    merged.drop('Survived',axis=1,inplace=True)

    return merged

merged = combine(train,test)
def encoding_label(merged, baseline= True):

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()

    

    if baseline == True:

        merged['Sex'] = encoder.fit_transform(merged['Sex'])

    else:

        merged['Sex'] = encoder.fit_transform(merged['Sex'])

        merged['Cabin'] = [0 if 'Cabin' in i else 1 for i in merged['Cabin']]

        merged['Embarked'] = encoder.fit_transform(merged['Embarked'])

    return merged
def scoring(merged):

    trained = merged.iloc[:len(train)]

    tested = merged.iloc[len(train):]

    y = train.Survived

    y_perfect = perfect.Survived

    

    from sklearn.linear_model import LogisticRegression

    logreg = LogisticRegression()

    

    logreg.fit(trained,y)

    return logreg.score(tested,y_perfect)
name = list()

scores = list()
merged = combine(train,test)

merged.drop(columns = ['PassengerId','Name','Cabin','Ticket','Fare','Embarked','Age'], inplace=True)
merged = encoding_label(merged)

scores.append(scoring(merged))

name.append('Base_Model')
merged = combine(train,test)

merged = merged[['Sex','Pclass','Age','SibSp','Parch','Fare','Cabin','Embarked']]
# Replacing the null values in the Age column with Mean

from sklearn.impute import SimpleImputer

from array import array



# Imputers for Cabin

imputer = SimpleImputer(missing_values= np.nan, strategy='constant', fill_value='No_Cabin')



# Fit and transform to the parameters

merged['Cabin'] = imputer.fit_transform(np.array(merged['Cabin']).reshape(-1,1))



# Imputers for Embarked

imputer = SimpleImputer(missing_values= np.nan, strategy='most_frequent')



# Fit and transform to the parameters

merged['Embarked'] = imputer.fit_transform(np.array(merged['Embarked']).reshape(-1,1))



# Checking for any null values

merged.head()
merged = encoding_label(merged, baseline=False)
merged_simple = merged.copy()

merged_simple['Fare'] = merged.Fare.fillna(merged.Fare.mean())

merged_simple['Age'] = merged_simple.Age.fillna(merged.Age.mean())
scores.append(scoring(merged_simple))

name.append('Simple_Mean')
merged_pclass = merged.copy()

merged_pclass['Age'] = merged.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.mean()))

merged_pclass['Fare'] = merged.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.mean()))
scores.append(scoring(merged_pclass))

name.append('Mean_Based_on_Pclass')
merged_simple = merged.copy()

merged_simple['Fare'] = merged.Fare.fillna(merged.Fare.median())

merged_simple['Age'] = merged_simple.Age.fillna(merged.Age.median())
scores.append(scoring(merged_simple))

name.append('Simple_Median')
merged_pclass = merged.copy()

merged_pclass['Age'] = merged.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

merged_pclass['Fare'] = merged.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
scores.append(scoring(merged_simple))

name.append('Median_Based_on_Pclass')
merged_ffill = merged.copy()

merged_ffill['Age'].fillna(method='ffill',inplace=True)

merged_ffill['Fare'].fillna(method='ffill',inplace=True)
scores.append(scoring(merged_ffill))

name.append('FFILL_Method')
merged_bfill = merged.copy()

merged_bfill['Age'].fillna(method='bfill',inplace=True)

merged_bfill['Fare'].fillna(method='bfill',inplace=True)



# Because it still leaves 2 missing values using bfill, I use ffill to mask it

merged_bfill['Age'].fillna(method='ffill',inplace=True)

scores.append(scoring(merged_bfill))

name.append('BFILL_Method')
from sklearn.impute import KNNImputer

merged_knn = merged.copy(deep=True)



knn_imputer = KNNImputer(n_neighbors=2, weights="uniform")



merged_knn['Age'] = knn_imputer.fit_transform(merged_knn[['Age']])

merged_knn['Fare'] = knn_imputer.fit_transform(merged_knn[['Fare']])
scores.append(scoring(merged_knn))

name.append('KNN_Imputation')
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

merged_mice = merged.copy(deep=True)



mice_imputer = IterativeImputer()

merged_mice['Age'] = mice_imputer.fit_transform(merged_mice[['Age']])



mice_imputer = IterativeImputer()

merged_mice['Fare'] = mice_imputer.fit_transform(merged_mice[['Fare']])
scores.append(scoring(merged_mice))

name.append('MICE_Imputation')
merged_regression = merged.copy()

merged_regression_train = merged_regression.iloc[:len(train)]

merged_regression_test = merged_regression.iloc[:len(test)]
from sklearn.linear_model import LinearRegression

merged_regression_train_age = merged_regression_train[merged_regression_train["Age"].isna() == False]

merged_regression_test_age = merged_regression_test[merged_regression_test["Age"].isna() == False]



merged_regression_new = merged_regression_train_age.append(merged_regression_test_age)



merged_regression_age_X = merged_regression_new.drop(["Age"], axis = 1)

merged_regression_age_y = merged_regression_new["Age"]



merged_regression_age_X["Fare"].fillna(merged_regression_age_X["Fare"].median(), inplace = True)



linear_reg_model = LinearRegression().fit(merged_regression_age_X, merged_regression_age_y)
# get indexes of rows that have NaN value



def get_age_indexes_to_replace(df):

    age_temp_list = df["Age"].values.tolist()

    indexes_age_replace = []

    age_temp_list = [str(x) for x in age_temp_list]

    for i, item in enumerate(age_temp_list):

        if item == "nan":

            indexes_age_replace.append(i)

    return indexes_age_replace



indexes_to_replace_main = get_age_indexes_to_replace(merged_regression_train)

indexes_to_replace_test = get_age_indexes_to_replace(merged_regression_test)
# make predictions on the missing values

def linear_age_predictions(reg_df, indexes_age_replace):

    reg_df_temp = reg_df.drop(["Age"], axis = 1)

    age_predictions = []

    for i in indexes_age_replace:

        x = reg_df_temp.iloc[i]

        x = np.array(x).reshape(1,-1)

        pred = linear_reg_model.predict(x)

        age_predictions.append(pred)

    return age_predictions



age_predictions_main = linear_age_predictions(merged_regression_train, indexes_to_replace_main)

age_predictions_test = linear_age_predictions(merged_regression_test, indexes_to_replace_test)
# fill the missing values with predictions

def fill_age_nan(merged_regression, indexes_age_replace, age_predictions):



    for i, item in enumerate(indexes_age_replace):

        merged_regression["Age"][item] =  age_predictions[i]



    return merged_regression



merged_regression_train = fill_age_nan(merged_regression_train, indexes_to_replace_main, age_predictions_main)

merged_regression_test = fill_age_nan(merged_regression_test, indexes_to_replace_test, age_predictions_test)
merged_regression = pd.concat([merged_regression_train, merged_regression_test])
scores.append(scoring(merged_regression))

name.append('LinearRegression_Imputation')
comparison = pd.DataFrame([scores],columns=name)

comparison