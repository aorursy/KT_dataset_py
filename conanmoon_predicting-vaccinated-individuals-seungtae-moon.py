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
# Import pandas and sklearn's train_test_split libraries

import pandas as pd

from sklearn.model_selection import train_test_split



# Load the datasets

target = 'vacc_h1n1_f'

df = pd.merge(pd.read_csv('../input/prediction-of-h1n1-vaccination/train.csv'), 

                 pd.read_csv('../input/prediction-of-h1n1-vaccination/train_labels.csv')[target], left_index=True, right_index=True)

test = pd.read_csv('../input/prediction-of-h1n1-vaccination/test.csv')
# Separate data for train 

train, val = train_test_split(df, train_size=0.80, test_size=0.20, stratify=df[target], random_state=2)



train.shape, val.shape, test.shape
# Quick overview of the training dataset

from pandas_profiling import ProfileReport as pr

profile = pr(train, minimal=True).to_notebook_iframe()
# Check duplicates

train.T.duplicated()
# Check cardinality

train.describe(exclude='number')
# Feature engineering



def engineer(df):

    

    # Create "behaviorals" feature

    behaviorals = [col for col in df.columns if 'behavioral' in col] 

    df['behaviorals'] = df[behaviorals].sum(axis=1)

    

    # Transform employment_status feature values("Not in Labor Force" -> "Unemployed")

    fixed_data = []

    for i in df["employment_status"]:

      if i == "Not in Labor Force":

        fixed_data.append("Unemployed")

      else:

        fixed_data.append(i)

    df["employment_status"] = fixed_data

    

    # Remove any feature with cardinality of over 30

    selected_cols = df.select_dtypes(include=['number', 'object'])

    colnames = selected_cols.columns.tolist()

    labels = selected_cols.nunique()

    

    selected_features = labels[labels <= 30].index.tolist()

    df = df[selected_features]

        

    return df





train = engineer(train)

val = engineer(val)

test = engineer(test)
# Separate the target feature from the training data

features = train.drop(columns=[target]).columns



# Diving training, validation, and testing data into X and y

X_train = train[features]

y_train = train[target]

X_val = val[features]

y_val = val[target]

X_test = test[features]
# Import libraries for OrdinalEncoder, SimpleImputer, RandomForestClassifier, and make_pipeline

from category_encoders import OrdinalEncoder

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
%%time



# ordinal encoding

pipe_ord = make_pipeline(

    OrdinalEncoder(), 

    SimpleImputer(), 

    RandomForestClassifier(n_estimators=100, random_state=10, max_depth=14, oob_score=True, n_jobs=-1, criterion="gini", min_samples_split=5, max_features=6)

)



pipe_ord.fit(X_train, y_train)

print('검증 정확도', pipe_ord.score(X_val, y_val))
# Out-of-Bag samples accuracy

pipe_ord.named_steps['randomforestclassifier'].oob_score_
# Predict the target using the testing data

y_pred_test = pipe_ord.predict(X_test)

y_pred_test = pd.Series(y_pred_test)

y_pred_test.value_counts()
# Create the DataFrame including predictions with "id" feature from the original data as index 

id = pd.Series(range(len(y_pred_test)))

y_pred_test = pd.Series(y_pred_test)

submission = pd.concat([id, y_pred_test], axis=1)

submission.rename(columns={0:"id", 1:target}, inplace=True)



# Display the submission data information

print(submission.shape)

print(submission.value_counts(target))



# Export the dataset to a *csv* file.

submission.to_csv("./submission.csv",index=False)