# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the read-only "../input/" directory

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 



%matplotlib inline
train_df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')

train_df
train_df.head()
train_df.info()
train_df.describe()
NaN_sum = train_df.isna().sum()
for i in range(len(NaN_sum)):

    if(NaN_sum[i] > 0):

        print(i)
train_df = train_df.set_index('id')

train_df
x_train = train_df.drop('target',axis=1)

x_train
y_train = train_df[['target']]

y_train
x_test =  pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

x_test
x_test = x_test.set_index('id')
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

scaled_x_train = scalar.fit_transform(x_train)

scaled_x_test = scalar.transform(x_test)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
# parameters = {'criterion': ("gini", "entropy"), 'max_depth': (50,300)}



# dt_cv = DecisionTreeClassifier()



# clf = GridSearchCV(dt_cv, parameters, verbose=1)



# clf.fit(scaled_x_train, y_train)
print(clf.score(scaled_x_train, y_train))
y_pred = clf.predict(scaled_x_test)
y_pred_series = pd.Series(y_pred)
df00 = pd.DataFrame(columns = ['target'])

df00['target'] = y_pred_series

df00

x_test_copy =  pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

x_test_copy
combined_x_test = pd.concat([x_test_copy, df00], axis=1)

combined_x_test
combined_x_test['target'].value_counts()
temp_df = combined_x_test[['id', 'target']]

# temp_df = temp_df.set_index('id')

temp_df
temp_df.to_csv('submission_x.csv', index=False)
submission_x_df =  pd.read_csv('/kaggle/working/submission_x.csv')

submission_x_df
# from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# dt = DecisionTreeClassifier()

# dt.fit(scaled_x_train, y_train)

# y_pred = dt.predict(scaled_x_test)

# y_pred_series = pd.Series(y_pred)

# df00['target'] = y_pred_series

# df00

# x_test_2 =  pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# x_test_2

# combined_x_test = pd.concat([x_test_2, df00], axis=1)

# combined_x_test

# combined_x_test['target'].value_counts()

# temp_df = combined_x_test[['id', 'target']]

# temp_df = temp_df.set_index('id')

# temp_df

# temp_df.to_csv('submission_y.csv', index=False)

# column_names = scaled_x_train.columns

# feature_importances = pd.DataFrame(dt.feature_importances_,

#                                    index = column_names,

#                                     columns=['importance'])

# feature_importances.sort_values(by='importance', ascending=False).head(10)
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier(n_estimators=200)

# rf.fit(scaled_x_train, y_train)

# y_pred = rf.predict(scaled_x_test)

# y_pred_series = pd.Series(y_pred)

# df00['target'] = y_pred_series

# df00

# x_test_2 =  pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# x_test_2

# combined_x_test = pd.concat([x_test_2, df00], axis=1)

# combined_x_test

# combined_x_test['target'].value_counts()

# temp_df = combined_x_test[['id', 'target']]

# temp_df = temp_df.set_index('id')

# temp_df

# temp_df.to_csv('submission_y.csv', index=False)
# from xgboost import XGBClassifier

# xgb = XGBClassifier()

# xgb.fit(scaled_X_train, y_train)

# y_pred = xgb.predict(scaled_x_test)

# y_pred_series = pd.Series(y_pred)

# df00['target'] = y_pred_series

# df00

# x_test_2 =  pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# x_test_2

# combined_x_test = pd.concat([x_test_2, df00], axis=1)

# combined_x_test

# combined_x_test['target'].value_counts()

# temp_df = combined_x_test[['id', 'target']]

# temp_df = temp_df.set_index('id')

# temp_df

# temp_df.to_csv('submission_y.csv', index=False)
# from sklearn.model_selection import GridSearchCV

# from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# parameters = {'criterion': ("gini", "entropy"), 'max_depth': (50,300)}



# dt_cv = DecisionTreeClassifier()



# clf = GridSearchCV(dt_cv, parameters, verbose=1)



# clf.fit(scaled_x_train, y_train)

# y_pred = clf.predict_proba(scaled_x_test)[:,1]

# y_pred_series = pd.Series(y_pred)

# df00['target'] = y_pred_series

# df00

# x_test_2 =  pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

# x_test_2

# combined_x_test = pd.concat([x_test_2, df00], axis=1)

# combined_x_test

# combined_x_test['target'].value_counts()

# temp_df = combined_x_test[['id', 'target']]

# temp_df = temp_df.set_index('id')

# temp_df

# temp_df.to_csv('submission_y.csv', index=False)