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
#import required libraries
import pandas as pd
import numpy as np 
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

#Read training data
d = pd.read_csv('../input/train.csv')

# Fill missing data with appropriate equivalents according to feature descriptions
d['meaneduc']=d['meaneduc'].fillna(0)
d['SQBmeaned']=d['SQBmeaned'].fillna(0)
d['rez_esc']=d['rez_esc'].fillna(0)
d['v18q1']=d['v18q1'].fillna(0)
d['v2a1']=d['v2a1'].fillna(0)

d.loc[d['dependency'] == 'yes', 'dependency'] = 1
d.loc[d['dependency'] == 'no', 'dependency'] = 0
d['dependency'] = pd.to_numeric(d['dependency'])

d.loc[d['edjefe'] == 'yes', 'edjefe'] = 1
d.loc[d['edjefe'] == 'no', 'edjefe'] = 0
d['edjefe'] = pd.to_numeric(d['edjefe'])

d.loc[d['edjefa'] == 'yes', 'edjefa'] = 1
d.loc[d['edjefa'] == 'no', 'edjefa'] = 1
d['edjefa'] = pd.to_numeric(d['edjefa'])


# Import and use ExtraTreesClassifier for finding feature importances
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Create features and labels
features=d.drop(['Target','Id','idhogar'], axis=1)
labels=d.Target

# Split for training and test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

# Fit model
model = ExtraTreesClassifier(random_state=42)
model.fit(X_train, y_train)

# Get importances table
imp=pd.DataFrame(({'features': X_train.columns, 'importances': model.feature_importances_}))

#Decided to use 16 features with top scores
imp[imp.importances>0.016].features

#Create dataframe with 16 features
model_df1=d[imp[imp.importances>0.016].features] 

# Add household head id and Target columns to source dataframe
model_df1 =pd.concat([model_df1, d[['idhogar','Target']]],axis=1)

# Create a new dataframe with aggregate functions - Trying to find best transformed features
top16features_group = (model_df1.groupby(['idhogar','Target'])
                    .agg({'meaneduc':np.sum,
                         'cielorazo':np.sum,
                         'r4t1':np.sum,
                         'overcrowding':np.median,
                         'hogar_nin':np.median,
                         'edjefe':np.median,
                         'SQBmeaned':np.median,
                         'paredblolad':np.sum,
                         'SQBovercrowding':np.median,
                         'dependency':np.median,
                         'qmobilephone':np.sum,
                         'SQBdependency':np.median,
                         'v18q':np.mean,
                         'r4m3':np.median,
                         'SQBedjefe':np.median,
                         'SQBhogar_nin':np.sum
                         }))

#Create features and labels
features=top16features_group.reset_index().drop(['Target','idhogar'], axis=1)
labels=top16features_group.reset_index().Target

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)


# Use Calibrated XGB Classifier for training
clf=XGBClassifier(random_state=1,objective='multi:softprob', max_depth= 8, n_estimators= 25, colsample_bytree=0.8,learning_rate=0.1)
cal_clf = CalibratedClassifierCV(clf, cv=3) 
cal_clf.fit(X_train, y_train)

#Print scores
print("Train score : " + str(cal_clf.score(X_train, y_train)))
print("Test score : " + str(cal_clf.score(X_test, y_test)))
#Read test set for prediction
t= pd.read_csv('../input/test.csv')

# Clean and replace missing values
t.loc[t['dependency'] == 'yes', 'dependency'] = 1
t.loc[t['dependency'] == 'no', 'dependency'] = 0

t['meaneduc']=t['meaneduc'].fillna(0)
t['SQBmeaned']=t['SQBmeaned'].fillna(0)
t['rez_esc']=t['rez_esc'].fillna(0)
t['v18q1']=t['v18q1'].fillna(0)
t['v2a1']=t['v2a1'].fillna(0)

t.loc[t['dependency'] == 'yes', 'dependency'] = 1
t.loc[t['dependency'] == 'no', 'dependency'] = 0
t['dependency'] = pd.to_numeric(t['dependency'])

t.loc[t['edjefe'] == 'yes', 'edjefe'] = 1
t.loc[t['edjefe'] == 'no', 'edjefe'] = 0
t['edjefe'] = pd.to_numeric(t['edjefe'])

t.loc[t['edjefa'] == 'yes', 'edjefa'] = 1
t.loc[t['edjefa'] == 'no', 'edjefa'] = 1
t['edjefa'] = pd.to_numeric(t['edjefa'])

# Get target features
tfeatures = t[imp[imp.importances>0.016].features]

# Add household head id and Target columns to source dataframe
tfeatures =pd.concat([tfeatures, t[['idhogar']]],axis=1)

# Create a new dataframe with aggregate functions - Trying to find best transformed features
top16features_testgroup = (tfeatures.groupby(['idhogar'])
                         .agg({'meaneduc':np.sum,
                         'cielorazo':np.sum,
                         'r4t1':np.sum,
                         'overcrowding':np.median,
                         'hogar_nin':np.median,
                         'edjefe':np.median,
                         'SQBmeaned':np.median,
                         'paredblolad':np.sum,
                         'SQBovercrowding':np.median,
                         'dependency':np.median,
                         'qmobilephone':np.sum,
                         'SQBdependency':np.median,
                         'v18q':np.mean,
                         'r4m3':np.median,
                         'SQBedjefe':np.median,
                         'SQBhogar_nin':np.sum
                         }))
# Predict test group
y_pred_test = cal_clf.predict(top16features_testgroup)

#Create dataframe for merging predictions with original household head ids
submission_df = pd.DataFrame({'Id':t.Id,'idhogar':t.idhogar})
#Set default Target value as zero
submission_df['Target']=0
submission_df.head()
#Fill submission Targets with predicted values
prediction_index=0
for household_index,row in top16features_testgroup.iterrows():
    submission_df.loc[submission_df['idhogar']==household_index,'Target']=y_pred_test[prediction_index]    
    prediction_index+=1

#Control Target data
submission_df.Target.value_counts()
#Export
submission_df[['Id','Target']].to_csv('submission.csv',sep=',',encoding='utf-8')
