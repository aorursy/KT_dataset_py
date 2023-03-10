# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import fbeta_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")

df_train.head()
working_df_train = df_train[['Pclass','Sex','Age']].copy() #These Fields actually contribute to the predictions of survival. Hence only utilising them

survived=df_train['Survived'].values.ravel() # Target Label as a 1-D Array

working_df_train = pd.get_dummies(working_df_train).fillna(-1) #Apply One HOT Encoding to convert Text Columns to Numerical and replace NaN with -1

working_df_train.head()
from sklearn import preprocessing

#Scale value of features so that different values due to different units do not cause an issue

min_max_scaler = preprocessing.MinMaxScaler()

working_df_train_scaled = min_max_scaler.fit_transform(working_df_train)
from sklearn.cross_validation import train_test_split

#Split Data Set into Train and Test Sets

X_train, X_test,y_train, y_test = train_test_split(working_df_train_scaled, survived, test_size=0.25, random_state=0)
from sklearn.svm import SVC

clf_SVC=SVC()

clf_SVC.fit(X_train, y_train)

print("SVC Score: ",clf_SVC.score(X_test,y_test))

#print(fbeta_score(y_train[:223], clf_SVC.predict(X_test),beta=0.5))
from sklearn.ensemble import RandomForestClassifier

clf_RF = RandomForestClassifier(random_state=0,n_estimators = 100)

clf_RF.fit(X_train, y_train)

print("RF Score: ",clf_RF.score(X_test,y_test))

for name, importance in zip(working_df_train.columns, clf_RF.feature_importances_):

    print(name, "=", importance)

#print(fbeta_score(y_train[:223], clf_RF.predict(X_test),beta=0.5))    
from sklearn.ensemble import AdaBoostClassifier

clf_AB = AdaBoostClassifier(random_state=0,n_estimators = 100)

clf_AB.fit(X_train, y_train)

print("Ada Booster Score: ",clf_AB.score(X_test,y_test))

for name, importance in zip(working_df_train.columns, clf_AB.feature_importances_):

    print(name, "=", importance)

#print(fbeta_score(y_train[:223], clf_AB.predict(X_test),beta=0.5))       
from sklearn.ensemble import GradientBoostingClassifier

clf_GB = GradientBoostingClassifier(random_state=0)

clf_GB.fit(X_train, y_train)

print("Gradient Boost Score: ",clf_GB.score(X_test,y_test))

for name, importance in zip(working_df_train.columns, clf_GB.feature_importances_):

    print(name, "=", importance)

#print(fbeta_score(y_train[:223], clf_GB.predict(X_test),beta=0.5))   
from xgboost import XGBClassifier

clf_XGB = XGBClassifier(random_state=0)

clf_XGB.fit(X_train, y_train)

print("XGradient Boost Score: ",clf_XGB.score(X_test,y_test))
# Read Test File

df_test = pd.read_csv("../input/test.csv")

working_df_test = df_test[['Pclass','Sex','Age']].copy() #These Fields actually contribute to the predictions of survival. Hence only utilising them

working_df_test = pd.get_dummies(working_df_test).fillna(-1)  # Hot Encode Test Data

working_df_test_scaled = min_max_scaler.fit_transform(working_df_test) # Scale Test Data



output = clf_XGB.predict(working_df_test_scaled) #Predict using Gradient Boosting



# Begin Formatting Output File

columns=['Survived']

output_df=pd.concat([df_test[['PassengerId']],pd.DataFrame(output,columns=columns)], axis=1)

output_df.to_csv("gender_submission.csv", index = False)
df_actual_output = pd.read_csv("../input/gender_submission.csv")

actual_array = np.array(df_actual_output)

pred_array = np.array(output_df)

result_array=np.equal(actual_array,pred_array) # Compare output given by Kaggle and Output Generated by me for the number of matches

count=0

for x in result_array[:,[1]]:

    if x == True:

        count = count+1

print('My Output Prediction Accuracy: ',(count/len(result_array))*100,'%')