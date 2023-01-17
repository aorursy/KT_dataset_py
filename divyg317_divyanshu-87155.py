# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
original = "../input/dataset/final_project_dataset.pkl"


destination = "word_data_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

with open('./word_data_unix.pkl','rb') as infile :
    data_dict=pickle.load(infile)

infile.close()
df=pd.DataFrame.from_dict(data_dict)
print(df)
df = df.replace('NaN', np.nan)
df=df.fillna(value="0")
df2=df.T
df2=df2.drop(['email_address','loan_advances','restricted_stock_deferred','director_fees','deferral_payments'], axis = 1)
df2[df2.columns]=df2[df2.columns].astype(int)

df2=df2.replace("False","0").replace("True","1")
df2['poi']=df2['poi'].astype(int)

from sklearn.model_selection import train_test_split
features=list(df2.columns)
y=df2['poi']
df3=df2
df2=df2.drop(['poi'],axis=1)
X=df2[df2.columns]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer,RobustScaler
scaler=PowerTransformer(method='yeo-johnson').fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


model=KNeighborsClassifier()
model1=GradientBoostingClassifier()

model3=RandomForestClassifier()
model4=LogisticRegression(solver='liblinear')
model5=SVC()
model6=DecisionTreeClassifier()
b=[]
for i in range(5,11):
    acc=cross_val_score(model,X,y,cv=i)
    acc1=cross_val_score(model1,X,y,cv=i)
    
    acc3=cross_val_score(model3,X,y,cv=i)
    acc4=cross_val_score(model4,X,y,cv=i)
    acc5=cross_val_score(model5,X,y,cv=i)
    acc6=cross_val_score(model6,X,y,cv=i)
    b.extend([acc.mean(),acc1.mean(),acc3.mean(),acc4.mean(),acc5.mean(),acc6.mean()])
b=np.asarray(b).reshape(-1,6)
print("Best cross validation score:- "+str(np.amax(b)))
a1= np.where(b==np.amax(b))[1]
a2= np.where(b==np.amax(b))[0]
if a1==[0]:
    print("K nearest neighbors")
elif a1==[1]:
    print("Gradient Boosting Classifier")

elif a1==[2]:
    print("Random Forest Classifier")
elif a1==[3]:
    print("Logistic Regression")
elif a1==[4]:
    print("Support Vector Classifier")
else:
    print("Decision Tree Classifier")
if a2==[0]:
    print("No. of folds:- 5")
elif a2==[1]:
    print("No. of folds:- 6")
elif a2==[2]:
    print("No. of folds:- 7")
elif a2==[3]:
    print("No. of folds:- 8")
elif a2==[4]:
    print("No. of folds:- 9")
elif a2==[5]:
    print("No. of folds:- 10")
else:
    print("No. of folds:- 11")
from sklearn.model_selection import GridSearchCV

grid_values={'n_neighbors':list(range(1,70)),
              }
grid_model=GridSearchCV(model,param_grid=grid_values,scoring='accuracy',cv=8).fit(X_train,y_train)
print("Best paramters:- "+str(grid_model.best_params_))
model=KNeighborsClassifier(n_neighbors=grid_model.best_estimator_.get_params()['n_neighbors'])
model.fit(X_train,y_train)
y_pred=model.predict_proba(X_test)
print(model.predict(X_test))

from sklearn.metrics import roc_auc_score,roc_curve,classification_report
print("Accuracy achieved on training set:- "+str(model.score(X_train,y_train)))
print("Accuracy achieved on testing set:- "+str(model.score(X_test,y_test)))
print("ROC Score:- "+str(roc_auc_score(y_test,y_pred[:,1])))
pickle.dump(model,open("my_classifier.pkl","wb"))
pickle.dump(new_dict,open("my_dataset.pkl","wb"))
pickle.dump(features,open("my_feature_list","wb"))