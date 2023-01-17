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
import numpy as np

female=stud[stud.genderFemale == 1]

male=stud[stud.genderFemale == 0]
female.shape
male.shape
feature_colsM = [

 'RES_BORED',

 'RES_CONCENTRATING',

 'RES_CONFUSED',

 'RES_FRUSTRATED',

 'RES_OFFTASK',

 'RES_GAMING']

XM = male[feature_colsM]

yM = male.isSTEM
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_XM, val_XM, train_yM, val_yM = train_test_split(XM, yM, random_state=1)

my_modelM = RandomForestClassifier(random_state=0).fit(train_XM, train_yM)



#https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights

import eli5

from eli5.sklearn import PermutationImportance



permM = PermutationImportance(my_modelM, random_state=1).fit(val_XM, val_yM)

eli5.show_weights(permM, feature_names = val_XM.columns.tolist())
import shap 

# Create object that can calculate shap values

explainerM = shap.TreeExplainer(my_modelM)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_valuesM = explainerM.shap_values(val_XM)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_valuesM[1], val_XM)