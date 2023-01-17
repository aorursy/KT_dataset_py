#imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.columns
train.head()
#look at distribution of the target
train.groupby('Target')['Id'].agg('count') #imbalanced classes here
#create simple training, test sets, based on subset of features
X = train.iloc[:,1:142] #got worse after 50
y = train['Target']

X2 = test.iloc[:,1:142]

#fill in NaNs
X = X.fillna(-1)
X2 = X2.fillna(-1)
X.head()
# for x in X.columns:
#     print(x)
feature_columns = X.columns.tolist()
#feature_columns.remove('rooms')
feature_columns.remove('idhogar')
feature_columns.remove('dependency')
feature_columns.remove('edjefe') 
feature_columns.remove('edjefa') 
#feature engineering and extraction

#get dummies for rooms
#room_dummies = pd.get_dummies(train['rooms'],prefix='room')
#room_dummies2 = pd.get_dummies(test['rooms'],prefix='room')

#select columns and combine
X = X[feature_columns]
# X = pd.concat([X,room_dummies],axis=1)

X2 = X2[feature_columns]
# X2 = pd.concat([X2,room_dummies2],axis=1)

#reduce the columns in test data
#make columns the same as training data
X2 = X2[X.columns]
#check all data types
# for x,y in zip(X.columns.tolist(),X.dtypes.tolist()):
#     print(x,y)
X.head()
#standard scale all features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_ = scaler.fit_transform(X)
X2_ = scaler.fit_transform(X2)
#get highly correlated columns to eliminate
scaled_features_df = pd.DataFrame(X_, index=X.index, columns=feature_columns)
corr_df = scaled_features_df.corr()
corr_df['vs'] = corr_df.columns
#restart the loop after taking action to remove 2nd column from feature_columns
for col1 in feature_columns:
    for col2 in feature_columns:
        if col1 != col2:
            print(col1+' '+col2)
            print(corr_df.loc[col2,col1])
            
# i=2
# while i < n:
#     if something:
#        do something
#        i += 1
#     else: 
#        do something else  
#        i = 2 #restart the loop  

#create data sets from training to test fitting 
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.30, random_state=42)
#try a logistic on the simple feature set
# clf = LogisticRegressionCV(cv=5
#                            ,solver='newton-cg'
#                            ,multi_class='multinomial'
#                            ,class_weight='balanced').fit(X_train, y_train)

# #predict classes
# predictions = clf.predict(X_test)
#evaluate logistic regression
# print(metrics.balanced_accuracy_score(y_test, predictions)) #accuracy
# print(metrics.f1_score(y_test,predictions,average='macro')) #weighted avg of precision and recall
#try a simple RF
clf2 = RandomForestClassifier(n_estimators=10, class_weight='balanced') #{1:6,2:2,3:4,4:1}
clf2.fit(X_train,y_train)

#predict classes
predictions2 = clf2.predict(X_test)
#evaluate RF
print(metrics.balanced_accuracy_score(y_test, predictions2)) #accuracy
print(metrics.f1_score(y_test,predictions2,average='macro')) #weighted avg of precision and recall
#try first submission without tuning hyperparams
#get a baseline of performance without much jiggery-pokery
# from sklearn.linear_model import LogisticRegressionCV
# clf = LogisticRegressionCV(cv=5
#                            ,solver='newton-cg'
#                            ,multi_class='multinomial'
#                            ,class_weight='balanced').fit(X,y)

# test_predictions = clf.predict(X2)
#submit predictions
# test['Target'] = test_predictions
# submission_df = test[['Id','Target']]
# submission_df.to_csv('logistic_submission.csv', index=False)
#try submitting RF
clf2 = RandomForestClassifier(n_estimators=20
                              ,max_depth=50
                              ,class_weight='balanced')
clf2.fit(X_,y)
test_predictions2 = clf2.predict(X2_)
#submit predictions
test['Target'] = test_predictions2
submission_df = test[['Id','Target']]
submission_df.to_csv('RF_submission.csv', index=False)
