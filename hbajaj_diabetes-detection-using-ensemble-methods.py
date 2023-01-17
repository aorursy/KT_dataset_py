# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

print(df.head())
df.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness","insulin","bmi","Diabetes_Pedigree_Function","age","outcome"]

df.head()
df.shape
df.info()
df.describe().T
for cols in ['glucose','blood_pressure','skin_thickness','insulin','bmi']:

    df.loc[df[cols] == 0,cols]= df[cols].mean(skipna=True)
df.describe().T
df.corr()
from sklearn.model_selection import train_test_split

Y = df['outcome']

X = df.drop(columns=['outcome'])





X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)



print('X_train size {} , X_test size {}'.format(X_train.shape,X_test.shape))
from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier(criterion='entropy',random_state=1)



dtc.fit(X_train,y_train)

print('Accuracy on train data:',dtc.score(X_train,y_train))

print('Accuracy on test data:',dtc.score(X_test,y_test))



from sklearn.metrics import confusion_matrix,f1_score

y_pred_dtc = dtc.predict(X_test)

dtc_cm = confusion_matrix(y_test,y_pred_dtc)

print('Confusion Matrix\n {}'.format(dtc_cm))



f1score_dtc = f1_score(y_test,y_pred_dtc)

print('F1 Score for Decision Tree:',f1score_dtc)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=100,criterion='gini',min_samples_split=5,min_samples_leaf=5,

                             max_features='auto',random_state=1)

rfc.fit(X_train,y_train)

print('Accuracy on train data:',rfc.score(X_train,y_train))

print('Accuracy on test data:',rfc.score(X_test,y_test))

y_pred_rfc = rfc.predict(X_test)

rfc_cm = confusion_matrix(y_test,y_pred_rfc)

print('Confusion Matrix\n {}'.format(rfc_cm))



f1score_rfc = f1_score(y_test,y_pred_rfc)

print('F1 Score for RandomForest:',f1score_rfc)
from sklearn.ensemble import AdaBoostClassifier



abc = AdaBoostClassifier(n_estimators=110,learning_rate=1,random_state=1)

abc.fit(X_train,y_train)



print('Accuracy on train data:',abc.score(X_train,y_train))

print('Accuracy on test data:',abc.score(X_test,y_test))

y_pred_abc = abc.predict(X_test)

abc_cm = confusion_matrix(y_test,y_pred_abc)

print('Confusion Matrix\n {}'.format(abc_cm))



f1score_abc = f1_score(y_test,y_pred_abc)

print('F1 Score for AdaBoost:',f1score_abc)
from sklearn.ensemble import BaggingClassifier



bgc = BaggingClassifier(n_estimators=100,max_samples=0.7,max_features=0.5,random_state=1)

bgc.fit(X_train,y_train)

print('Accuracy for train data {}'.format(bgc.score(X_train,y_train)))

print('Accuracy for test data {}'.format(bgc.score(X_test,y_test)))

y_pred_bgc = bgc.predict(X_test)

bgc_cm = confusion_matrix(y_test,y_pred_bgc)

print('Confusion Matrix\n {}'.format(bgc_cm))



f1score_bgc = f1_score(y_test,y_pred_bgc)

print('F1 Score for BaggingClassifier:',f1score_bgc)
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier(loss='deviance',learning_rate=0.1, n_estimators=90,subsample=0.8,

                                criterion="friedman_mse",min_samples_split=0.2,min_samples_leaf=5,max_depth=3,random_state=1)

gbc.fit(X_train,y_train)

print('Accuracy for train data {}'.format(gbc.score(X_train,y_train)))

print('Accuracy for test data {}'.format(gbc.score(X_test,y_test)))

y_pred_gbc = gbc.predict(X_test)

gbc_cm = confusion_matrix(y_test,y_pred_gbc)

print('Confusion Matrix\n {}'.format(gbc_cm))



f1score_gbc = f1_score(y_test,y_pred_gbc)

print('F1 Score for GradientBoostingClassifier:',f1score_gbc)
print('F1 Score for Decision Tree:',f1score_dtc)

print('F1 Score for RandomForest:',f1score_rfc)

print('F1 Score for AdaBoost:',f1score_abc)

print('F1 Score for BaggingClassifier:',f1score_bgc)

print('F1 Score for GradientBoostingClassifier:',f1score_gbc)