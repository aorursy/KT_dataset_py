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
import numpy as np

import pandas as pd



data = pd.read_csv('../input/train.csv')



data.head()
data.info()
data.tail()
#cabin conversion



import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}



data['Cabin'] = data['Cabin'].fillna("U0")

data['Deck'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

data['Deck'] = data['Deck'].map(deck)

data['Deck'] = data['Deck'].fillna(0)

data['Deck'] = data['Deck'].astype(int)

data['Deck']



# drop passenger names , Cabin 

data = data.drop(['Name'], axis=1)

data = data.drop(['Cabin'], axis=1)



data.head(10)
# Convert Gender 



#data['Sex'].fillna("0")



gender = {"male": 1, "female": 2 }



data['Sex'] = data['Sex'].map(gender)



data.head(10)

data['Sex'].value_counts()
# Convert Embarked 



data['Embarked'] = data['Embarked'].fillna("0")

embark = {"S": 1, "C": 2 , "Q" :3 , "0" :1}



data['Embarked'] = data['Embarked'].map(embark)



data.head(10)
# drop ticket for now 



data = data.drop(['Ticket'], axis=1)



data.head(10)
# Fix Age



data.describe()  # take average age - approx 30 

data['Age'] = data['Age'].fillna(30)



data[['Age','Sex']].groupby(['Sex']).agg(['median'])

# check data cleansing 



data.info()
data.isnull().sum()

# looks 2 records are not correct in Embarked column

#data['Embarked'] =data['Embarked'].fillna(1)

#data[data['Embarked'].isnull()]

# split the data 



from  sklearn.model_selection import train_test_split



X = data[['Pclass' , 'Sex', 'Age' ,'SibSp' , 'Parch' , 'Fare' ,'Embarked' , 'Deck']]

y = data['Survived']



X_train, X_test, y_train, y_test = train_test_split(X,y,random_state =0)



print (X_train.shape , X_test.shape , y_train.shape , y_test.shape)

#X_test
# Classifier



from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(random_state=0)



clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)







# test accuracy

from sklearn.metrics import accuracy_score



accuracy_score(y_test , y_pred )

final = pd.read_csv('../input/test.csv')

# drop Name  

final = final.drop(['Name'], axis=1) #Input data





# Cabin to Deck 

import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}



final['Cabin'] = final['Cabin'].fillna("U0")

final['Deck'] = final['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

final['Deck'] = final['Deck'].map(deck)

final['Deck'] = final['Deck'].fillna(0)

final['Deck'] = final['Deck'].astype(int)

#final['Deck']



# drop Cabin

final = final.drop(['Cabin'], axis=1) #Input data

# Convert Gender 



final['Sex'].fillna("0")



gender = {"male": 1, "female": 2 , "0" :"female"}



final['Sex'] = final['Sex'].map(gender)





# Convert Embarked 



final['Embarked'] = final['Embarked'].fillna("0")

embark = {"S": 1, "C": 2 , "Q" :3 , "0" :1}



final['Embarked'] = final['Embarked'].map(embark)





# Fix Age





final['Age'] = final['Age'].fillna(30)





final.head()
# check Final Data 



final.isnull().sum()



# Fare is having NaN - so find mean fare i.e. 36

final['Fare'] = final['Fare'].fillna(14)



final.describe()



final['Fare'].median()
final.isnull().sum()
#All Good  

final_X = final[['Pclass' , 'Sex', 'Age' ,'SibSp' , 'Parch' , 'Fare' ,'Embarked' , 'Deck']]



final_y = clf.predict (final_X)

len(final_y )



final_y
# print output 



data_to_submit = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(final_y)})
# write to csv



data_to_submit.to_csv('titanic.csv' , index=False)
# Classifier



from sklearn.tree import DecisionTreeClassifier



w_clf = DecisionTreeClassifier(random_state=0)



w_clf.fit(X, y)

w_y_pred = w_clf.predict(final_X)



data_to_submit2 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(w_y_pred)})



data_to_submit2.to_csv('titanic_whole_training.csv' , index=False)


# Classifier



from sklearn.ensemble import RandomForestClassifier



r_clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)



r_clf.fit(X_train, y_train)

r_y_pred = r_clf.predict(final_X)



data_to_submit3 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(r_y_pred)})



data_to_submit3.to_csv('titanic_random_training.csv' , index=False)

# Classifier



from sklearn.ensemble import RandomForestClassifier



r_w_clf = RandomForestClassifier(n_estimators=50, max_depth=4,random_state=0)



r_w_clf.fit(X, y)

r_w_y_pred = r_w_clf.predict(final_X)



data_to_submit4 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(r_w_y_pred)})



data_to_submit4.to_csv('titanic_random_whole_training.csv' , index=False)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(X, y)

y_knn_pred = neigh.predict(final_X)



#accuracy_score(y_test, y_knn_pred, normalize=True)





data_to_submit5 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_knn_pred)})



data_to_submit5.to_csv('titanic_knn_whole_training.csv' , index=False)



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



lr = LogisticRegression()

lr.fit(X_train, y_train)

y_lr_pred = lr.predict(final_X)



#accuracy_score(y_test, y_lr_pred, normalize=True)





data_to_submit6 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_lr_pred)})



data_to_submit6.to_csv('titanic_lr_training.csv' , index=False)



from sklearn.tree import DecisionTreeClassifier



feature_clf = DecisionTreeClassifier(random_state=0)

feature_clf.fit(X_train, y_train)

feature_clf.feature_importances_



#X_train.columns



feat_dict = dict(zip(X_train.columns,feature_clf.feature_importances_ ))

feat_dict



X_3_train = X_train[['Sex','Age','Fare','Pclass']]

X_3_test = X_test[['Sex','Age','Fare','Pclass']]



imp_clf =  DecisionTreeClassifier(random_state=0)

imp_clf.fit(X_3_train, y_train)



imp_y_pred = imp_clf.predict(X_3_test)



from sklearn.metrics import accuracy_score



accuracy_score(y_test , imp_y_pred ) 



y_imp_feat_pred = imp_clf.predict(final_X[['Sex','Age','Fare','Pclass']])



#accuracy_score(y_test, y_lr_pred, normalize=True)





data_to_submit7 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_imp_feat_pred)})



data_to_submit7.to_csv('titanic_feat_imp_training.csv' , index=False)







X_4_train = X[['Sex','Age','Fare','Pclass']]

imp_clf.fit(X_4_train, y)

y_imp_feat_pred = imp_clf.predict(final_X[['Sex','Age','Fare','Pclass']])





data_to_submit8= pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_imp_feat_pred)})



data_to_submit8.to_csv('titanic_feat_imp_whole_training.csv' , index=False)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



lr_feat = LogisticRegression()

lr_feat.fit(X[['Sex','Age','Fare','Pclass']], y)

y_lr_pred = lr_feat.predict(final_X[['Sex','Age','Fare','Pclass']])



#accuracy_score(y_test, y_lr_pred, normalize=True)





data_to_submit9 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_lr_pred)})



data_to_submit9.to_csv('titanic_lr_feat_training.csv' , index=False)

from sklearn.ensemble import RandomForestClassifier



r_f_w_clf = RandomForestClassifier(n_estimators=20, max_depth=4,random_state=0)



from sklearn.metrics import accuracy_score



r_f_w_clf.fit(X[['Sex','Age','Fare','Pclass','Deck']], y)

y_f_w_pred = r_f_w_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])



#accuracy_score(y_test, y_lr_pred, normalize=True)





data_to_submit10 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_f_w_pred)})



data_to_submit10.to_csv('titanic_rf_f_w_feat_training.csv' , index=False)

from sklearn.ensemble import RandomForestClassifier



r_f_clf = RandomForestClassifier(n_estimators=75, max_depth=5,random_state=0)



from sklearn.metrics import accuracy_score



r_f_clf.fit(X_train[['Sex','Age','Fare','Pclass','Deck']], y_train)

y_f_pred = r_f_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])



#accuracy_score(y_test, y_f_pred)





data_to_submit11 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_f_pred)})



data_to_submit11.to_csv('titanic_rf_f_feat_training.csv' , index=False)

from sklearn.ensemble import GradientBoostingClassifier



gbclf = GradientBoostingClassifier(n_estimators = 50, learning_rate  =0.2 , max_features  =5)



#X_train.info()

gbclf.fit(X_train,y_train)



y_gb_pred = gbclf.predict(X_test)



accuracy_score (y_gb_pred, y_test)





y_gb_final_pred = gbclf.predict(final_X)





data_to_submit12 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_gb_final_pred)})



data_to_submit12.to_csv('titanic_gb_training.csv' , index=False)



from sklearn.ensemble import GradientBoostingClassifier



gbclf = GradientBoostingClassifier()



#X_train.info()

gbclf.fit(X,y)





y_gb_full_train_final_pred = gbclf.predict(final_X)





data_to_submit13 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_gb_full_train_final_pred)})



data_to_submit13.to_csv('titanic_gb_full_training.csv' , index=False)

from sklearn.naive_bayes import GaussianNB



gnbclf = GaussianNB()



#X_train.info()

gnbclf.fit(X_train[['Sex','Age','Fare','Pclass']],y_train)



y_gnb_pred = gnbclf.predict(final_X[['Sex','Age','Fare','Pclass']])



#accuracy_score (y_gnb_pred, y_test)





data_to_submit14 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_gnb_pred)})



data_to_submit14.to_csv('titanic_gnb_training.csv' , index=False)

from sklearn.ensemble import GradientBoostingClassifier



gbclf2 = GradientBoostingClassifier(n_estimators = 100, learning_rate  =0.2 , max_features  =4)



#X_train.info()

gbclf2.fit(X_train[['Sex','Age','Fare','Pclass','Deck']],y_train)





y_gb2_final_pred = gbclf2.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])





data_to_submit15 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_gb2_final_pred)})



data_to_submit15.to_csv('titanic_gb2_training.csv' , index=False)
from sklearn.ensemble import GradientBoostingClassifier



gbclf3 = GradientBoostingClassifier(n_estimators = 100, learning_rate  =0.2 , max_features  =3)



#X_train.info()

gbclf3.fit(X_train,y_train)





y_gb3_final_pred = gbclf3.predict(final_X)





data_to_submit16 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_gb3_final_pred)})



data_to_submit16.to_csv('titanic_gb3_training.csv' , index=False)
from sklearn import svm

svm_clf = svm.SVC(gamma='scale')



svm_clf.fit(X_train, y_train) 



y_svm_pred = svm_clf .predict(X_test)



print(accuracy_score(y_test, y_svm_pred))







y_svm_final_pred = svm_clf.predict(final_X)





data_to_submit17 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_svm_final_pred)})



data_to_submit17.to_csv('titanic_svm_training.csv' , index=False)



from sklearn import svm

svm_clf2 = svm.SVC(gamma='scale')



svm_clf2.fit(X, y) 



y_svm2_final_pred = svm_clf2.predict(final_X)





data_to_submit18 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(y_svm2_final_pred)})



data_to_submit18.to_csv('titanic_svm2_training.csv' , index=False)



from sklearn.model_selection import cross_val_score



from sklearn.ensemble import RandomForestClassifier



rr_clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)



print(cross_val_score(rr_clf, X_train, y_train, cv=5))  





from sklearn.ensemble import GradientBoostingClassifier



ens_gbclf = GradientBoostingClassifier(n_estimators = 100, learning_rate  =0.2 , max_features  =4)



#X_train.info()

ens_gbclf.fit(X_train[['Sex','Age','Fare','Pclass','Deck']],y_train)





from sklearn.ensemble import RandomForestClassifier



ens_rf_clf = RandomForestClassifier(n_estimators=20, max_depth=4,random_state=0)





ens_rf_clf.fit(X[['Sex','Age','Fare','Pclass','Deck']], y)

#y_f_w_pred = ens_rf_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])





ens_rf2_clf = RandomForestClassifier(n_estimators=75, max_depth=5,random_state=0)







ens_rf2_clf.fit(X_train[['Sex','Age','Fare','Pclass','Deck']], y_train)



from xgboost import XGBClassifier





xgb = XGBClassifier()

xgb.fit(X_train[['Sex','Age','Fare','Pclass','Deck']], y_train)





from sklearn.ensemble import AdaBoostClassifier



ens_ada_clf = AdaBoostClassifier()

ens_ada_clf.fit(X_train[['Sex','Age','Fare','Pclass','Deck']], y_train)



ens_gb_pred = ens_gbclf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])

ens_rf_pred = ens_rf_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])

ens_rf2_pred = ens_rf2_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])

ens_ada_pred = ens_ada_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])

ens_xgb_pred = xgb.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])











predictions=[]



for n in range(len(ens_gb_pred)):

            combined = ens_gb_pred[n] + ens_rf_pred[n] + ens_rf2_pred[n] +ens_ada_pred[n] +ens_xgb_pred[n]

            p = 0 if combined <= 2 else 1

            predictions.append(p)



#print(predictions)



data_to_submit19 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(predictions)})



data_to_submit19.to_csv('titanic_ens_t_training.csv' , index=False)







from sklearn.ensemble import GradientBoostingClassifier



ens_gbclf = GradientBoostingClassifier(n_estimators = 100, learning_rate  =0.2 , max_features  =4)



#X_train.info()

ens_gbclf.fit(X[['Sex','Age','Fare','Pclass','Deck']],y)





from sklearn.ensemble import RandomForestClassifier



ens_rf_clf = RandomForestClassifier(n_estimators=20, max_depth=4,random_state=0)





ens_rf_clf.fit(X[['Sex','Age','Fare','Pclass','Deck']], y)

#y_f_w_pred = ens_rf_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])





ens_rf2_clf = RandomForestClassifier(n_estimators=75, max_depth=5,random_state=0)







ens_rf2_clf.fit(X[['Sex','Age','Fare','Pclass','Deck']], y)





from sklearn.ensemble import AdaBoostClassifier



ens_ada_clf = AdaBoostClassifier()

ens_ada_clf.fit(X[['Sex','Age','Fare','Pclass','Deck']], y)



ens_gb_pred = ens_gbclf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])

ens_rf_pred = ens_rf_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])

ens_rf2_pred = ens_rf2_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])

ens_ada_pred = ens_ada_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])





from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X[['Sex','Age','Fare','Pclass','Deck']], y)

ens_xgb_pred = xgb.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])









predictions=[]



for n in range(len(ens_gb_pred)):

            combined = ens_gb_pred[n] + ens_rf_pred[n] + ens_rf2_pred[n] +ens_ada_pred[n] + ens_xgb_pred[n]

            p = 0 if combined <= 2 else 1

            predictions.append(p)



#print(predictions)



data_to_submit19 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(predictions)})



data_to_submit19.to_csv('titanic_ens_f_training.csv' , index=False)







from sklearn.ensemble import GradientBoostingClassifier



ens_gbclf = GradientBoostingClassifier(n_estimators = 100, learning_rate  =0.2 , max_features  =4)



#X_train.info()

ens_gbclf.fit(X,y)





from sklearn.ensemble import RandomForestClassifier



ens_rf_clf = RandomForestClassifier(n_estimators=20, max_depth=4,random_state=0)





ens_rf_clf.fit(X, y)

#y_f_w_pred = ens_rf_clf.predict(final_X[['Sex','Age','Fare','Pclass','Deck']])





ens_rf2_clf = RandomForestClassifier(n_estimators=75, max_depth=5,random_state=0)







ens_rf2_clf.fit(X, y)





ens_gb_pred = ens_gbclf.predict(final_X)

ens_rf_pred = ens_rf_clf.predict(final_X)

ens_rf2_pred = ens_rf2_clf.predict(final_X)











predictions=[]



for n in range(len(ens_gb_pred)):

            combined = ens_gb_pred[n] + ens_rf_pred[n] + ens_rf2_pred[n]

            p = 0 if combined == 1 or combined == 0 else 1

            predictions.append(p)



#print(predictions)



data_to_submit19 = pd.DataFrame({'PassengerId':final['PassengerId'],

                               'Survived':pd.Series(predictions)})



data_to_submit19.to_csv('titanic_ens2_training.csv' , index=False)









from xgboost import XGBClassifier



from sklearn.metrics import accuracy_score



xgb = XGBClassifier()

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)



print(accuracy_score(y_test, xgb_pred, normalize=True))





from sklearn.ensemble import AdaBoostClassifier



ens_ada_clf = AdaBoostClassifier()



ens_ada_clf.fit(X_train, y_train)

y_ada_pred = ens_ada_clf.predict(X_test)



accuracy_score(y_test, y_ada_pred, normalize=True)





from xgboost import XGBClassifier



from sklearn.metrics import accuracy_score



xgb = XGBClassifier()

xgb.fit(X_train[['Sex','Age','Fare','Pclass','Deck']], y_train)

xgb_pred = xgb.predict(X_test[['Sex','Age','Fare','Pclass','Deck']])



print(accuracy_score(y_test, xgb_pred, normalize=True))