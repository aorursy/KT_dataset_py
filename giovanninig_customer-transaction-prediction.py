import numpy as np
import pandas as pd
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.model_selection import cross_val_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train_df = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")
test_df = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
train_df.head(10)
train_df = train_df.drop(["ID_code"] , axis = 1)
test_df.head(10)
test_ID = test_df["ID_code"]
test_df = test_df.drop(["ID_code"], axis= 1)
train_df.describe()
test_df.describe()
train_df.isnull().sum().sort_values()
test_df.isnull().sum().sort_values()
sns.countplot(train_df["target"])
train_df["target"].value_counts()
def plot_feature_distribution(df1, df2, label1, label2, features):   
    i = 0                                   
    sns.set_style('whitegrid')              
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(15,17))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False, label = label1)
        sns.distplot(df2[feature], hist=False, label = label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]


features = train_df.columns.values[1:101]
plot_feature_distribution(t0, t1, '0', '1', features)  
features = train_df.columns.values[101:201]
plot_feature_distribution(t0, t1, '0', '1', features)  
correlations = train_df.corr()
correlations.sort_values(by=["target"]).tail(10)
correlations.sort_values(by=["target"]).head(10)
X = train_df.drop(["target"] ,axis = 1)
y = train_df["target"].values
from sklearn.preprocessing import RobustScaler 
rb = RobustScaler()
X_scaled = rb.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns = X.columns) 
X = X_scaled

#X.describe()
for v in X.columns:
    variance = X.var()
variance = variance.sort_values(ascending = False)
   
plt.figure(figsize=(12,5))
plt.plot(variance)  

variance
trans = train_df.loc[train_df["target"] == 1]

no_trans = train_df.loc[train_df["target"] == 0]

no_trans = no_trans.sample(n = 20098 , random_state = 42)

train_df = pd.concat([trans , no_trans])
sns.countplot(train_df["target"])
train_df["target"].value_counts()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


classifiers =  [
       ['Logistic Regression Classifier :', LogisticRegression()] ,
       ['K-Neighbors Classifier :', KNeighborsClassifier()] ,
       ['Support Vector Classifier :', SVC()] ,
       ['Naive Bayes :' , GaussianNB()] ,
       ['XGB Classifier :', XGBClassifier()]      
       ]

       

for name,model in classifiers:    

    model = model
    
    model.fit(X_train,y_train)
    
    y_pred_train = model.predict(X_train)

    y_pred = model.predict(X_test)
     
    print('-----------------------------------')
    print(name)
    
    print(" -- TRAINING SET --")
    print('Accuracy: ', accuracy_score( y_train , y_pred_train))
    print("f1: ",f1_score( y_train , y_pred_train))
    print("precision: ", precision_score( y_train , y_pred_train))
    print("recall: ", recall_score( y_train , y_pred_train))
    print("ROC AUC: ", roc_auc_score( y_train , y_pred_train))
    print('---------------------------------')
        
     
    print(" --  TEST SET --  ")
    print('Accuracy: ', accuracy_score( y_test, y_pred))
    print("f1:      ",f1_score( y_test, y_pred))
    print("precision: ", precision_score( y_test, y_pred))
    print("recall: ", recall_score( y_test, y_pred))
    print("ROC AUC: ", roc_auc_score( y_test, y_pred))
    print('---------------------------------')
# FEATURES IMPORTANCE

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit( X_train , y_train)

importances = model.feature_importances_
index = np.argsort(importances)[::-1][0:10]
feature_names = X.columns.values

plt.figure(figsize=(10,5))
sns.barplot(x = feature_names[index], y = importances[index]);
plt.title("Top important features ");

importances = pd.Series(importances)

importances = importances.sort_values(ascending = False)
importances
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(model, threshold=0.001)   

X_train = X_train.loc[ :, sfm.fit(X_train , y_train).get_support()]
X_train.describe()
X_test = X_test[X_train.columns]
X_test.describe()
from sklearn.model_selection import RandomizedSearchCV


colsample_bylevel = [1 , 0.5]
colsample_bytree = [1 , 0.5]
gamma = [0 , 1 , 5]
learning_rate = [0.1 , 0.01 , 0.001]
max_delta_step = [0]
max_depth = [1 , 5 , 10 ]
min_child_weight = [1]
n_estimators = [ 50 , 100 , 250 , 500 , 750]
objective = ['binary:logistic']
random_state = [42]     
reg_alpha = [0, 1]
reg_lambda = [0 , 1]
scale_pos_weight = [1]
subsample = [0.5, 0.8 ,  1 ]


param_distributions = dict(
                           colsample_bylevel = colsample_bylevel,
                           colsample_bytree = colsample_bytree,
                           gamma = gamma, 
                           learning_rate = learning_rate,
                           max_depth = max_depth,
                           min_child_weight = min_child_weight,
                           n_estimators = n_estimators,
                           objective = objective,
                           random_state = random_state,
                           reg_alpha = reg_alpha,
                           reg_lambda = reg_lambda,
                           scale_pos_weight = scale_pos_weight,
                           subsample = subsample , 
                           ) 



estimator = XGBClassifier()     


RandomCV = RandomizedSearchCV(
                            estimator = estimator,         
                            param_distributions = param_distributions,
                            n_iter = 10,
                            cv = 5,
                            scoring = "roc_auc",   
                            random_state = 42, 
                            verbose = 1, 
                            n_jobs = -1,
                            )


hyper_model = RandomCV.fit(X_train, y_train)                   
                                              

print('Best Score: ', hyper_model.best_score_)    

print('Best Params: ', hyper_model.best_params_)

hyper_model.best_estimator_.fit(X_train,y_train)

y_pred_train_hyper = hyper_model.best_estimator_.predict(X_train)  

y_pred_hyper = hyper_model.best_estimator_.predict(X_test)  



print("HYPER   TRAIN")
print('Accuracy Score ', accuracy_score( y_train , y_pred_train_hyper))
print("f1: ",f1_score(y_train , y_pred_train_hyper))
print("precision: ", precision_score(y_train , y_pred_train_hyper))
print("recall_score: ", recall_score( y_train, y_pred_train_hyper))
print("ROC AUC: ", roc_auc_score( y_train, y_pred_train_hyper))


print(" HYPER  TEST")
print('Accuracy Score ', accuracy_score( y_test, y_pred_hyper))
print("f1: ",f1_score(y_test, y_pred_hyper))
print("precision: ", precision_score(y_test, y_pred_hyper))
print("recall_score: ", recall_score( y_test, y_pred_hyper))
print("ROC AUC: ", roc_auc_score( y_test, y_pred_hyper))

test_df = test_df[X_train.columns]

from sklearn.preprocessing import RobustScaler 
rb = RobustScaler()
test_df_scaled = rb.fit_transform(test_df)

test_df_scaled = pd.DataFrame(test_df_scaled, columns = test_df.columns)
test_df = test_df_scaled
final_pred = hyper_model.best_estimator_.predict(test_df)
sub_df = pd.DataFrame({"ID_code": test_ID.values})

sub_df["target"] = final_pred
sub_df.to_csv("Final Prediction.csv", index=False)