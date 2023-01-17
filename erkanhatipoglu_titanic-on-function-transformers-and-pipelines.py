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



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

pd.set_option('display.max_rows', None)
def save_file (predictions):

    """Save submission file."""

    # Save test predictions to file

    output = pd.DataFrame({'PassengerId': sample_sub_file.PassengerId,

                       'Survived': predictions})

    output.to_csv('submission.csv', index=False)

    print ("Submission file is saved")

    

def transform_age(df):

    ''' A function that transforms the Age column of the Titanic dataset.

        'Age' feature is transformed into a categorical data of the passengers

        such that masters and people whose age are smaller than 16 is defined

        as child.'''

    # Make a copy to avoid changing original data

    X_temp = df.copy()

    

    # Create Age_new column

    pd.DataFrame.insert(X_temp, len(X_temp.columns),"Age_new","",False)    

    

    # Get the index values

    index_values = X_temp.index.values.astype(int)

    

    for i in index_values:

        age = X_temp.at[i, 'Age'].astype(float)

        name = X_temp.loc[i,'Name']

        if name.find('.'):

            title = name.split('.')[0].split()[-1]



        if np.isnan(age):

            if title == "Master":

                X_temp.loc[i,'Age_new'] = "Child"

            else:

                X_temp.loc[i,'Age_new'] = "Adult"

        else:

            if age < 16:

                X_temp.loc[i,'Age_new'] = "Child"

            else:

                X_temp.loc[i,'Age_new'] = "Adult"

        

    drop = ["Age", "Name"]

    X_temp.drop(drop, axis=1, inplace=True)

    X_temp.rename(columns={'Age_new':'Age'}, inplace=True)

    return X_temp



def transform_family(df):

    '''A funtion that calculates the family size by summing Parch and SibSp columns into the 'Fcount' column. Afterward Parch 

    and SibSp columns are dropped.'''

    # Make a copy to avoid changing original data

    X_temp = df.copy()

    

    # Create Fcount column

    pd.DataFrame.insert(X_temp, len(X_temp.columns),"Fcount","",False)    

    

    # Get the index values

    index_values = X_temp.index.values.astype(int)

    

    for i in index_values:

        X_temp.loc[i, 'Fcount'] = X_temp.loc[i, 'Parch'] + X_temp.loc[i,'SibSp']

        

    X_temp["Fcount"] = X_temp["Fcount"].astype('int64')

    X_temp.drop(['Parch', 'SibSp'], axis=1, inplace=True)



    return X_temp



print("Functions loaded")
# Loading data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

sample_sub_file = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



# Make a copy to avoid changing original data

X = train_data.copy()

y = X.Survived

X_test = test_data.copy()



# Remove target from predictors

X.drop(['Survived'], axis=1, inplace=True)

print("['Survived'] column dropped from training data!")



# Remove Ticket, Cabin, Embarked columns. We will not use them.

cols_dropped = ["Ticket", "Cabin", "Embarked"]

X.drop(cols_dropped, axis = 1, inplace = True)

X_test.drop(cols_dropped, axis = 1, inplace = True)

print("{} dropped from both training and test data!".format(cols_dropped))



print("\nShape of training data: {}".format(X.shape))

print("Shape of target: {}".format(y.shape))

print("Shape of test data: {}".format(X_test.shape))

print("Shape of submission data: {}".format(sample_sub_file.shape))



# Split the data for validation

X_train, X_valid, y_train, y_valid = train_test_split(X,y, random_state=2)



print("\nShape of X_train data: {}".format(X_train.shape))

print("Shape of X_valid: {}".format(X_valid.shape))

print("Shape of y_train: {}".format(y_train.shape))

print("Shape of y_valid: {}".format(y_valid.shape))



print("\nFiles Loaded")
X_train.head()
# Define the custom transformers for the pipeline

age_transformer = FunctionTransformer(transform_age)

family_transformer = FunctionTransformer(transform_family)
X_temp = age_transformer.fit_transform(X)

X_temp = family_transformer.fit_transform(X_temp)
X_temp[5:10]
# Define transformers



# Define the custom transformers for the pipeline

age_transformer = FunctionTransformer(transform_age)

family_transformer = FunctionTransformer(transform_family)



# Define transformer for categorical columns using a pipeline

cat_cols = ["Sex", "Age", "Pclass"]

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(drop = 'first', sparse = False))

])



# Define column transformer for categorical data

column_transformer = ColumnTransformer(transformers=[('cat', categorical_transformer, cat_cols)], remainder='passthrough')
# Define Model

model = XGBClassifier(seed=42)
# Define preprocessor

preprocessor = Pipeline(steps=[('age', age_transformer),

                              ('family', family_transformer),

                              ('column', column_transformer)])



# Make a copy to avoid changing original data 

X_valid_eval=X_valid.copy()



# Preprocessing of validation data

X_valid_eval = preprocessor.fit(X_train, y_train).transform (X_valid_eval)



# Display the number of remaining columns after transformation 

print("We have", X_valid_eval.shape[1], "features left")
# Create and Evaluate the Pipeline

# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])
# Preprocessing of training data, fit model 

X_cv = X.copy()

X_sub = X_test.copy()
# Cross-validation

scores = cross_val_score(my_pipeline, X_cv, y,

                              cv=5,

                              scoring='accuracy')



print("MAE score:\n", scores)

print("MAE mean: {}".format(scores.mean()))

print("MAE std: {}".format(scores.std()))
# Preprocessing of training data, fit model 

my_pipeline.fit(X_cv, y)



# Get predictions

preds = my_pipeline.predict(X_sub)
# Use predefined utility function

save_file(preds)