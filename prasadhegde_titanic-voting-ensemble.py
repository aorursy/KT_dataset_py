# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
gnd_df=pd.read_csv("../input/gender_submission.csv")
all_data=[train_df,test_df]
print(train_df.head(3))
null_cols=[col for col in test_df.columns if train_df[col].isnull().any()]
print(null_cols)
print(train_df.info())


for df in all_data:
    df["Cabin"] = df["Cabin"].fillna("X")
    df["Fare"]=df["Fare"].fillna(df["Fare"].mean())
    df["Age"] = df["Age"].fillna(25)
    df["Embarked"]=df["Embarked"].fillna("S")
ls= preprocessing.LabelEncoder()
le = preprocessing.LabelEncoder()
ls.fit(["male","female"])

le.fit(all_data[0]["Embarked"])
for df in all_data:
    df["Agebin"]=pd.cut(df["Age"],4,labels=[1,2,3,4])
    df["Farebin"]=pd.qcut(df["Fare"],4,labels=[1,2,3,4])
    df["en_sex"]=ls.transform(df["Sex"])
    df["en_Embarked"]=le.transform(df["Embarked"])
import seaborn as sns
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(all_data[0][["Pclass","Agebin","Farebin","en_sex","SibSp","en_Embarked","Parch","Survived"]],hue="Survived")
X=all_data[0][["Pclass","Agebin","Farebin","en_sex","SibSp","en_Embarked","Parch"]]
Y=all_data[0][["Survived"]]
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=.20,random_state=1)


params = {'min_samples_split':range(100,200,10), 'min_samples_leaf':range(10,20,3),'n_estimators':range(60,80,5)}
gs=GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10),
param_grid = params, scoring='roc_auc',iid=False, cv=5)
gs.fit(X_train,y_train.values.ravel())
gs.grid_scores_, gs.best_params_, gs.best_score_
print(gs.best_params_["min_samples_split"],gs.best_params_["min_samples_leaf"],gs.best_params_["n_estimators"])
clf1=GradientBoostingClassifier(min_samples_split=gs.best_params_["min_samples_split"],min_samples_leaf=gs.best_params_["min_samples_leaf"],
                               n_estimators=gs.best_params_["n_estimators"])
clf1.fit(X_train,y_train.values.ravel())

ypred1=clf1.predict(X_test)
# print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred1))
final_test=all_data[1][["Pclass","Agebin","Farebin","en_sex","SibSp","en_Embarked","Parch"]]
null_cols1=[col for col in final_test.columns if final_test[col].isnull().any()]
# print(np.where(all_data[1]["Farebin"].isnull())[0])
# print(final_test.iloc[150:160,:])
print(null_cols1)
clf2=GradientBoostingClassifier(min_samples_split=gs.best_params_["min_samples_split"],min_samples_leaf=gs.best_params_["min_samples_leaf"],
                               n_estimators=gs.best_params_["n_estimators"])


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
parameters = {'solver': ['lbfgs'], 'max_iter': [50,100,150], 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12), 'random_state':[0,1,2]}
clf_grid = GridSearchCV(MLPClassifier(), parameters)
scaler = StandardScaler() 
scaler.fit(X)
X=scaler.transform(X)
final_test=scaler.transform(final_test)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('gb', clf1), ('mlp',clf_grid), ('gnb', clf3)], voting='hard')
eclf.fit(X,Y.values.ravel())
ypred2=eclf.predict(final_test)
print("Done ")

PassengerId=test_df['PassengerId']
sub_csv=pd.DataFrame({'PassengerId':PassengerId,'Survived':ypred2})
print(sub_csv.head())
sub_csv.to_csv('Submission.csv',index=False)
