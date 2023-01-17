# conventional way to import pandas

import pandas as pd

#read file.

df = pd.read_csv('../input/anonymized_full_release_competition_dataset20181128.csv')

#replace spaces with underscores for all columns 

df.columns = df.columns.str.replace(' ', '_')

df.head()
#locate a value in a column as Nan https://stackoverflow.com/questions/45416684/python-pandas-replace-multiple-columns-zero-to-nan?rq=1

import numpy as np

df.loc[df['MCAS'] == -999.0,'MCAS'] = np.nan

# create the 'genderFemale' dummy variable using the 'map' method

df['genderFemale'] = df.InferredGender.map({'Female':1, 'Male':0})

# Removing unused columns

list_drop = ['InferredGender']

df.drop(list_drop, axis=1, inplace=True)

# create dummy variables for multiple categories; this drops nominal columns and creates dummy variables

dfDummy=pd.get_dummies(df, columns=['MiddleSchoolId'], drop_first=True)



#use observations only with no missing in isSTEM

stud=df.dropna(subset=['isSTEM'], how='any')

stud.shape
# list(stud) to copy column names

feature_cols = [

 'RES_BORED',

 'RES_CONCENTRATING',

 'RES_CONFUSED',

 'RES_FRUSTRATED',

 'RES_OFFTASK',

 'RES_GAMING']

X = stud[feature_cols]

y = stud.isSTEM
!pip install eli5
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

#https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(my_model)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_values = explainer.shap_values(val_X)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_values[1], val_X)