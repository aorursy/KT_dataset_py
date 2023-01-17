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
feature_colsF = [

 'RES_BORED',

 'RES_CONCENTRATING',

 'RES_CONFUSED',

 'RES_FRUSTRATED',

 'RES_OFFTASK',

 'RES_GAMING']

XF = female[feature_colsF]

yF = female.isSTEM
!pip install eli5
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



train_XF, val_XF, train_yF, val_yF = train_test_split(XF, yF, random_state=1)

my_modelF = RandomForestClassifier(random_state=0).fit(train_XF, train_yF)



#https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights

import eli5

from eli5.sklearn import PermutationImportance



permF = PermutationImportance(my_modelF, random_state=1).fit(val_XF, val_yF)

eli5.show_weights(permF, feature_names = val_XF.columns.tolist())
import shap  # package used to calculate Shap values

# Create object that can calculate shap values

explainerF = shap.TreeExplainer(my_modelF)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_valuesF = explainerF.shap_values(val_XF)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_valuesF[1], val_XF)