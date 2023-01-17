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
data=np.loadtxt('../input/faults_data.txt')
np.random.shuffle(data)
columns=['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum' ,'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter','Sum_of_Luminosity', 'Minimum_of_Luminosity','Maximum_of_Luminosity', 'Length_of_Conveyer' ,'TypeOfSteel_A300' ,'TypeOfSteel_A400','Steel_Plate_Thickness' ,'Edges_Index' ,'Empty_Index' ,'Square_Index' ,'Outside_X_Index','Edges_X_Index' ,'Edges_Y_Index','Outside_Global_Index' ,'LogOfAreas' ,'Log_X_Index' ,'Log_Y_Index' ,'Orientation_Index' ,'Luminosity_Index','SigmoidOfAreas','Pastry','Z_Scratch','K_Scratch','Stains','Dirtiness','Bumps','Other_Faults']
data=pd.DataFrame(data,columns=columns)
data.head()
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
X=data[columns[:27]]
y=data[columns[27:34]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
cv_score_mean=[]
acc_score=[]
y_pred=[]
models_list=[LogisticRegression(solver='sag'),SVC(gamma='auto'),RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0),DecisionTreeClassifier(random_state=0)]
for model in models_list:
    pipeline=Pipeline([('scaler',StandardScaler()),('clf',OneVsRestClassifier(model))])
    cvx = KFold(n_splits=10, shuffle=False, random_state=0)
    mean_cv_score = cross_val_score( pipeline, X_train, y_train, cv=cvx).mean()
    cv_score_mean.append(mean_cv_score)
    pipeline.fit(X_train,y_train)
    print('Training set accuracy_score: ',accuracy_score(pipeline.predict(X_train),y_train))
    pred=pipeline.predict(X_test)
    acc=accuracy_score(pred,y_test)
    print('Cross Validation score :',mean_cv_score)
    print('Test set accuracy_score:',acc)
    #pred_proba=pipeline.predict_proba(X_test)
    acc_score.append(acc)
    y_pred.append(pred)

   
y_pred[0].shape
from sklearn.metrics import confusion_matrix
y_test_non_category = [ np.argmax(t) for t in np.array(y_test)]
conf_mats=[]
for pred in y_pred:
    y_predict_non_category = [ np.argmax(t) for t in pred]
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
    print(conf_mat)
    conf_mats.append(conf_mat)
#The best model is svm.Last two classes are having low accuracy compared to others in classification. 