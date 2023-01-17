import pandas as pd

import numpy as np

from sklearn import linear_model

from sklearn.model_selection import train_test_split 

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")

df2=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")
df.info()
df2.info()
df.isnull().values.any()
df2.isnull().values.any()
high_corr=df.corr()

high_corr['class']
Y=df['class']

X= df.drop(['id','chem_2','chem_3','chem_7','class'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
#KNN

from sklearn.neighbors import KNeighborsClassifier

#splitting

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)

classifier = KNeighborsClassifier(n_neighbors=9,metric='euclidean',weights='distance')



classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Score is",classifier.score(X_test,y_test))
 #RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier



rforest=RandomForestClassifier(n_estimators=100)

rforest.fit(X_train,y_train)
# Actual class predictions

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

rf_predictions = rforest.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, rf_predictions))
#GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

gb_clf = GradientBoostingClassifier().fit(X_train,y_train)

y_pred_gb = gb_clf.predict(X_test)

gb_acc = accuracy_score(y_test,y_pred_gb)



print(gb_acc)
# Stacking



# We'll split the training dataset into two parts - A & B. The base models will be trained on A. Their predictions on B will be used to train a meta model.



n = len(X_train)

X_A = X_train[:n//2]

y_A = y_train[:n//2]

X_B = X_train[n//2:]

y_B = y_train[n//2:]
# Train the base models on dataset A and generate predictions on dataset B



clf_1 = KNeighborsClassifier().fit(X_A, y_A)

y_pred_1 = clf_1.predict(X_B)

clf_2 = RandomForestClassifier(n_estimators=100).fit(X_A, y_A)

y_pred_2 = clf_2.predict(X_B)

clf_3 = GradientBoostingClassifier().fit(X_A, y_A)

y_pred_3 = clf_3.predict(X_B)
# Create a new dataset C with predictions of base models on B

X_C = pd.DataFrame({'RandomForest': y_pred_2, 'KNN': y_pred_1, 'GradientBoost': y_pred_3})

y_C = y_B

X_C.head()
# Combine predictions made by base models on validation set to create a dataset D

X_D = pd.DataFrame({'RandomForest': clf_2.predict(X_test), 'KNN': clf_1.predict(X_test), 'GradientBoost': clf_3.predict(X_test)})

y_D = y_test
# Train a meta model on C and print its accuracy on D.

 #RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier



rf_new=RandomForestClassifier(n_estimators=100)

rf_new.fit(X_train,y_train)
# Actual class predictions

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

rf_predict_new = rf_new.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, rf_predict_new))
df2=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

ID=df2['id']

df2.drop(['id','chem_2','chem_3','chem_7'],axis=1,inplace=True)

 
Y_dfpred=pd.DataFrame()

Y_dfpred['id']=ID



pred=rf_new.predict(df2)



Y_dfpred['class']=list(pred)
Y_dfpred.to_csv('Output_newstk.csv',index=False)#output1 is by  random_forest