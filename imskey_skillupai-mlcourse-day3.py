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
 #import basic library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
from IPython.core.display import display,Image
import scipy as sp

#preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedShuffleSplit,GridSearchCV

#model
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV

#evaluation & vizualization
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.externals.six import StringIO
import pydotplus
import graphviz
from sklearn.model_selection import GridSearchCV
#set data
data = pd.read_csv("../input/mushrooms.csv")

#data overview
print("shape of data:{}".format(data.shape))
display(data.head())
display(data.describe())

#cheking null
def isnull_table(df): 
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    isnull_table = pd.concat([null_val, percent], axis=1)
    isnull_table_ren_columns = isnull_table.rename(
    columns = {0 : 'NaN', 1 : '%'})
    return isnull_table_ren_columns

isnull_table(data)
#checking featrues
print("shape of data:{}".format(data.shape))
print("names of columns:\n{}".format(data.columns))
data.isnull().sum()
for column in data:
    print("unique value in {} data:\n{}".format(column,data[column].value_counts()))
# making histgram
x = 8
y = 3
figsize = (20,50)

def hist_dataframe(df,x,y,figsize):
    plt.figure(figsize =figsize)
    for i,j in zip(range(1,len(df.columns)+1),range(len(df.columns)+1)):
        plt.subplot(x,y,i)
        plt.hist(df.iloc[:,j])
        plt.title(df.columns[j])

hist_dataframe(data,x,y,figsize)     
# delete veli-type 
data = data.drop("veil-type",axis = 1)
# making features dummies
print("Features before get dummies:\n{}".format(data.columns))
data_dummies = pd.get_dummies(data)
print("Features after get dummies:\n{}".format(data_dummies.columns))
display(data_dummies.head())
#descrese features ( class_e >= 0.4 | class_e <= -0.4)
data_dummies_all = data_dummies #reserve the data all
data_dummies_corr = data_dummies.corr()
data_index = data_dummies.corr().query("class_e >= 0.4 | class_e <= -0.4").index
data_dummies = data_dummies[data_index].drop(["class_p",'bruises_f'],axis = 1)
print("shape of data;{}".format(data_dummies.shape))
print("columns:{}".format(data_dummies.columns))
data_dummies.corr().style.background_gradient().format('{:.2f}')
X = data_dummies.drop("class_e",axis = 1).values
y = data_dummies["class_e"].values
print("X.shape:{} y.shape:{}".format(X.shape,y.shape))
# set logistic regression model
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

print("Train score:{:.4f}".format(logreg.score(X_train,y_train)))
print("Test score:{:.4f}".format(logreg.score(X_test,y_test)))

#cross validation
shuffle_split = StratifiedShuffleSplit()
scores = cross_val_score(logreg,X,y,cv=shuffle_split)
print("Cross-validation score:{}".format(scores))
print("Average Cross-validation score:{}".format(scores.mean()))
#evaluating logistic regression model

y_pred = logreg.predict(X_test)
mse = mean_squared_error(y_pred,y_test)
mae = mean_absolute_error(y_pred,y_test)

print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )


cmx = confusion_matrix(y_test,y_pred,)

sns.heatmap(cmx,annot=True,fmt='d')
plt.title("confusion matrix")
print(classification_report(y_true = y_test,y_pred = y_pred))
# vizualizing Coefficient magnitude 
n_features = X.shape[1]
feature_names = data_dummies.drop("class_e",axis =1).columns

plt.figure(figsize = (10,10))
plt.plot(logreg.coef_.T,'o')
plt.xticks(range(n_features),feature_names,rotation = 90)
plt.hlines(0,0,n_features)
plt.ylim(-7,6)
plt.xlabel("feature")
plt.ylabel("Coefficient magnitude")
#setting decision tree model
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_split=3, min_samples_leaf=3, random_state=0)
clf = clf.fit(X_train, y_train)
print("Train score:{:.4f}".format(clf.score(X_train,y_train)))
print("Test score:{:.4f}".format(clf.score(X_test,y_test)))

#cross validation
shuffle_split = StratifiedShuffleSplit()
scores = cross_val_score(clf,X,y,cv=shuffle_split)
print("Cross-validation score:\n{}".format(scores))
print("Average Cross-validation score:{}".format(scores.mean()))
#evaluating clf model
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_pred,y_test)
mae = mean_absolute_error(y_pred,y_test)

print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )

cmx = confusion_matrix(y_test,y_pred,)

sns.heatmap(cmx,annot=True,fmt='d')
plt.title("confusion matrix")
print(classification_report(y_true = y_test,y_pred = y_pred))
n_features = X.shape[1]
feature_names = data_dummies.drop("class_e",axis =1).columns

plt.figure(figsize = (10,10))
plt.barh(range(n_features),clf.feature_importances_,align = 'center')
plt.yticks(np.arange(n_features),feature_names)
plt.xlabel("Featue importance")
# grid seach 
best_score = 0
for depth in[3,4,5,6,7,8,9,10]:
    for split in[3,4,5,6,7,8,9,10]:
        for leaf in [3,4,5,6,7,8,9,10]:
            clf = DecisionTreeClassifier(max_depth = depth ,
                                         min_samples_split = split,
                                         min_samples_leaf = leaf)
            
            scores = cross_val_score(clf,X_train,y_train,cv=5)
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_parameters = {'max_depth':depth,
                                   'min_samples_split':split,
                                   'min_samples_leaf':leaf}
            
clf = DecisionTreeClassifier(**best_parameters)
clf.fit(X_train,y_train)
#evaluating the model
print("Train score:{:.4f}".format(clf.score(X_train,y_train)))
print("Test score:{:.4f}".format(clf.score(X_test,y_test)))

#cross validation
shuffle_split = StratifiedShuffleSplit()
scores = cross_val_score(clf,X,y,cv=shuffle_split)
print("Cross-validation score:\n{}".format(scores))
print("Average Cross-validation score:{}".format(scores.mean()))

#evaluating clf model
y_pred = clf.predict(X_test)
mse = mean_squared_error(y_pred,y_test)
mae = mean_absolute_error(y_pred,y_test)

print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )

cmx = confusion_matrix(y_test,y_pred,)

sns.heatmap(cmx,annot=True,fmt='d')
plt.title("confusion matrix")
print(classification_report(y_true = y_test,y_pred = y_pred))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                     feature_names = data_dummies.drop("class_e",axis =1).columns,
                     class_names=["p","e"],  
                     filled=True, rounded=True,  
                     special_characters=True) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())
n_features = X.shape[1]
feature_names = data_dummies.drop("class_e",axis =1).columns

plt.figure(figsize = (10,10))
plt.barh(range(n_features),clf.feature_importances_,align = 'center')
plt.yticks(np.arange(n_features),feature_names)
plt.xlabel("Featue importance")
#setting XGboosting
mod = xgb.XGBClassifier()

#Grid Search
param_dist = {
    'learning_rate':sp.stats.uniform(0.1,1),
    'max_depth':sp.stats.randint(1,11),
    'subsample':sp.stats.uniform(0.5,0.5),
    'colsample_bytree':sp.stats.uniform(0.5,0.5),
    'reg_lambda':sp.stats.uniform(0.1,10)
}
rand_search = RandomizedSearchCV(xgb.XGBClassifier(),
                                 param_dist,
                                 cv =10,
                                 n_iter = 20,
                                scoring ="accuracy",
                                verbose=0)
rand_search.fit(X_train,y_train)
print("Best estimator:\n{}".format(rand_search.best_estimator_))

#evaluating the model
print("Train score:{:.4f}".format(rand_search.score(X_train,y_train)))
print("Test score:{:.4f}".format(rand_search.score(X_test,y_test)))

y_pred = rand_search.predict(X_test)

mse = mean_squared_error(y_pred,y_test)
mae = mean_absolute_error(y_pred,y_test)

print("MSE=%s"%round(mse,3) )
print("RMSE=%s"%round(np.sqrt(mse), 3) )
print("MAE=%s"%round(mae,3) )


cmx = confusion_matrix(y_test,y_pred,)
print(cmx)
sns.heatmap(cmx,annot=True,fmt='d')
plt.title("confusion matrix")
print(classification_report(y_true = y_test,y_pred = y_pred))