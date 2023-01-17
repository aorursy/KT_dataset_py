# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
adult_df = pd.read_csv('../input/adult.csv')
adult_df.head()
adult_df.info()
#lets go through each of the columns and decide what action to be taken
import matplotlib.pyplot as plt
plt.boxplot(adult_df['age'])
plt.show()
plt.hist(adult_df['age'])
plt.show()
#i dont see any problems with this col; we can leave it as is
obj_cols_to_be_treated = []
adult_df['workclass'].value_counts()
obj_cols_to_be_treated.append('workclass')
adult_df.columns
adult_df['education'].value_counts()
obj_cols_to_be_treated.append('education')
adult_df['educational-num'].value_counts()
#it is clear that the educational-num is an encoded col of education; so we will jus use this and not encode education again
obj_cols_to_be_treated.remove('education')
adult_df['marital-status'].value_counts()
obj_cols_to_be_treated.append('marital-status')
adult_df['occupation'].value_counts()
obj_cols_to_be_treated.append('occupation')
#'relationship', 'race', 'gender',

adult_df['relationship'].value_counts()
obj_cols_to_be_treated.append('relationship')
adult_df['race'].value_counts()
obj_cols_to_be_treated.append('race')
adult_df['gender'].value_counts()
obj_cols_to_be_treated.append('gender')
#all but one of the obj cols are reviewed
plt.boxplot(adult_df['capital-gain'])
plt.show()
plt.hist(adult_df['capital-gain'])
plt.show()
#this is a skewed column; but no action for now; 
#lets jus move on to prediction after reviewing the output col
adult_df['income'].describe()
adult_df['income'].value_counts()
adult_df['income_less_than_50K_1_0'] = adult_df['income'].map({'<=50K':1, '>50K':0})
adult_df['income_less_than_50K_1_0'].value_counts()
#OK, lets numericalize the obj cols
obj_cols_to_be_treated
for obj_col in obj_cols_to_be_treated:
    adult_df[obj_col + '_cat'] = adult_df[obj_col].astype('category').cat.as_ordered()
    adult_df[obj_col + '_cat_codes'] = adult_df[obj_col + '_cat'].cat.codes
encoded_cols = [col for col in adult_df.columns if '_cat_codes' in col]
encoded_cols
#add the encoded cols and original numeric cols from df to create the input vars list
input_vars = encoded_cols + ['age','educational-num','capital-gain','capital-loss','hours-per-week']
output_var = 'income_less_than_50K_1_0'
from sklearn.ensemble import RandomForestClassifier
#lets start with a single decision tree with max depth of 3
rf1 = RandomForestClassifier(n_estimators=1, max_depth=3)
rf1.fit(X=adult_df[input_vars], y=adult_df[output_var])
rf1.score(X=adult_df[input_vars], y=adult_df[output_var])
from treeinterpreter import treeinterpreter as ti
adult_df.head()
sample_record = adult_df.loc[0:1]
sample_record
prediction, bias, contribution = ti.predict(X=sample_record[input_vars], model=rf1)
prediction.shape
prediction
bias.shape
bias
adult_df['income_less_than_50K_1_0'].value_counts()
adult_df['income_less_than_50K_1_0'].value_counts()[0] / adult_df.shape[0]
#we can see that the bias value doesnt change across instances; it is the proportion of that output variable
contribution.shape
contribution[0].shape
contribution[0]
input_vars
#get more trees
rf2 = RandomForestClassifier(n_estimators=10, max_depth=3)
rf2.fit(X=adult_df[input_vars], y=adult_df[output_var])
rf2.score(X=adult_df[input_vars], y=adult_df[output_var])
rf3 = RandomForestClassifier(n_estimators=10, max_depth=5)
rf3.fit(X=adult_df[input_vars], y=adult_df[output_var])
rf3.score(X=adult_df[input_vars], y=adult_df[output_var])
feat_imp_dict ={}
for i in range(len(input_vars)):
    feat_imp_dict[input_vars[i]] = rf3.feature_importances_[i]
feat_imp_df = pd.DataFrame.from_dict(feat_imp_dict, orient= 'index')
feat_imp_df.reset_index(inplace=True)
feat_imp_df.columns = ['feature', 'feat_imp']
feat_imp_df.sort_values(ascending=False, by=['feat_imp'], inplace=True)
feat_imp_df.head()
from treeinterpreter import treeinterpreter as ti
prediction, bias, contribution = ti.predict(X=sample_record[input_vars], model=rf3)
prediction
contribution[0]
#lets look at a few cases where output var is 0
adult_df[adult_df['income_less_than_50K_1_0'] ==0]
adult_df[adult_df['income_less_than_50K_1_0'] ==0].loc[2:3]
sample_record_0s = adult_df[adult_df['income_less_than_50K_1_0'] ==0].loc[2:3]
prediction, bias, contribution = ti.predict(X=sample_record_0s[input_vars], model=rf3)
prediction
bias
contribution[1]
#the third item from last is the one which impacts the most
input_vars[-3]
sample_record_0s
from sklearn.tree import export_graphviz
s=export_graphviz(rf1.estimators_[0], out_file=None, feature_names=input_vars, filled=True,
                      special_characters=True, rotate=True, precision=3)
from IPython.display import display
import IPython
from sklearn.tree import export_graphviz
import graphviz
IPython.display.display(graphviz.Source(s))
