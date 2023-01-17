import os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

print(os.listdir("../input"))

pd.set_option('display.float_format',lambda x:'%.3f'%x)

pd.set_option('display.max_columns',500)

pd.set_option('display.max_colwidth',500)

titan_df=pd.read_csv("../input/train.csv")

test_titan_df=pd.read_csv("../input/test.csv")

titan_df.drop(['PassengerId','Name','Ticket','Cabin'],1,inplace=True)

test_titan_df.drop(['Name','Ticket','Cabin'],1,inplace=True)

#ways of exploring data

#print(titan_df.head())

titan_df.describe()

print(titan_df.columns)

#converting columns to numeric 

#titan_df.convert_objects(convert_numeric=True)

titan_df.fillna(0,inplace=True)

test_titan_df.fillna(0,inplace=True)

titan_df1=pd.get_dummies(titan_df,columns=['Sex','Embarked',],drop_first=True)

print(titan_df1.columns)

test_df1=pd.get_dummies(test_titan_df,columns=['Sex','Embarked'],drop_first=True)

#print(test_df1.head(20))

X_train, X_test, y_train,y_test = train_test_split(titan_df1.drop(columns=['Survived']),titan_df1['Survived'],test_size =0.20, shuffle =True)

#print(X_train)





#Visualize data

from matplotlib import pyplot as plt

x=titan_df.Sex

y=titan_df.Survived

for x in titan_df:

    if x ==1:

        plt.bar(x,y,color='green',linewidth=5)

        plt.show()

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score

print('Logistic accuracy score is {:4f}'.format(accuracy_score(y_test,pred)))

print('Logistic Precision score: ', precision_score(y_test, pred,average='micro'))

print(' Logistic Recall score: ', recall_score(y_test, pred,average='micro'))

print('Logistic f1 score is {:4f}'.format(f1_score(y_test,pred)))



predict_test = model.predict(test_df1)

#print(predict_test)

out_df = pd.DataFrame()

out_df['PassengerId'], out_df['Survived'] =test_df1['PassengerId'],pd.DataFrame(predict_test) 

#print(out_df)

out_df.to_csv('./Submission_logistic.csv', index =False)
from sklearn.ensemble import RandomForestClassifier

class_titan=RandomForestClassifier(n_estimators=100,oob_score=True)

ct=class_titan.fit(X_train, y_train)

y_test_rand=class_titan.predict(X_test)

print('Rand accuracy score is {:4f}'.format(accuracy_score(y_test,y_test_rand)))

print('Rand Precision score: ', precision_score(y_test, y_test_rand,average='micro'))

print('Rand Recall score: ', recall_score(y_test, y_test_rand,average='micro'))

print('Rand f1 score is {:4f}'.format(f1_score(y_test,y_test_rand)))

predict_test = class_titan.predict(test_df1)

#print(predict_test)

out_df = pd.DataFrame()

out_df['PassengerId'], out_df['Survived'] =test_df1['PassengerId'],pd.DataFrame(predict_test) 

#print(out_df)

out_df.to_csv('./Submission_random.csv', index =False)

from sklearn.svm import LinearSVC

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5,max_iter = 10000))

clf.fit(X_train, y_train)

print('Accuracy lsvc {}'.format (clf.score(X_test, y_test)))

y_pred_LS = clf.predict(X_test)

print('LinearSVC accuracy score is {:4f}'.format(accuracy_score(y_test,y_pred_LS)))

print('LinearSVC Precision score: ', precision_score(y_test, y_pred_LS,average='micro'))

print('LinearSVC Recall score: ', recall_score(y_test, y_pred_LS,average='micro'))

print('LinearSVC f1 score is {:4f}'.format(f1_score(y_test,y_pred_LS)))

predict_test_lsvc = clf.predict(test_df1)

out_df = pd.DataFrame()

out_df['PassengerId'], out_df['Survived'] =test_df1['PassengerId'],pd.DataFrame(predict_test_lsvc) 

#print(out_df)

out_df.to_csv('./Submission_lsvc.csv', index =False)
from sklearn import svm

clf = svm.SVC(kernel='rbf',gamma = 0.1 ,C = 1.0)

clf.fit(X_train, y_train)

y_pred_svc= clf.predict(X_test)

print('SVC accuracy score is {:4f}'.format(accuracy_score(y_test,y_pred_svc)))

print('SVC Precision score: ', precision_score(y_test, y_pred_svc,average='micro'))

print('SVC Recall score: ', recall_score(y_test, y_pred_svc,average='micro'))

print('SVC f1 score is {:4f}'.format(f1_score(y_test,y_pred_svc)))

#print(clf.support_vectors_)

predict_test_svc = clf.predict(test_df1)

out_df = pd.DataFrame()

out_df['PassengerId'], out_df['Survived'] =test_df1['PassengerId'],pd.DataFrame(predict_test_svc) 

#print(out_df)

out_df.to_csv('./Submission_svc.csv', index =False)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets

model.fit(X_train,y_train)

#Predict Output

predicted= model.predict(X_test)

print('KNN accuracy score is {:4f}'.format(accuracy_score(y_test,y_pred_svc)))

print('KNN Precision score: ', precision_score(y_test, y_pred_svc,average='micro'))

print('KNN Recall score: ', recall_score(y_test, y_pred_svc,average='micro'))

print('KNN f1 score is {:4f}'.format(f1_score(y_test,y_pred_svc)))



predict_test_knn = model.predict(test_df1)

out_df = pd.DataFrame()

out_df['PassengerId'], out_df['Survived'] =test_df1['PassengerId'],pd.DataFrame(predict_test_knn) 

#print(out_df)

out_df.to_csv('./Submission_knn.csv', index =False)