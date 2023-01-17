from IPython.display import IFrame

IFrame('https://public.tableau.com/views/TelecomChurnEDAAndInsightStory/TelecomChurnEDAAndInsightStory?:language=en&:display_count=y&:origin=viz_share_link', width=1000, height=925)
import numpy as np # For data manipluation

import pandas as pd # for data manipulation

import matplotlib.pyplot as plt #plot libary 

%matplotlib inline 

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import balanced_accuracy_score,roc_auc_score,make_scorer # for scoring 

from sklearn.model_selection import GridSearchCV  # for cross validation 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix
df = pd.read_csv(r"../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
df.shape
df.columns
df["Contract"].unique()
df["gender"].unique()
df["SeniorCitizen"].unique()
df["Partner"].unique()
df.info()
df.columns
df.drop(["customerID"],axis=1,inplace=True)

df.head()
df.columns=df.columns.str.replace(" ","_")
df.columns[(df.isnull().any())].tolist()  # doesnt mean there is no Blank Values they can be in " " form to.
df.columns
df["TotalCharges"].unique()
len(df.loc[df["TotalCharges"]==" "])
df.loc[df["TotalCharges"]==" "]  # these are the rows where we have black in total charges.
#lets make the charges 0



df.loc[(df["TotalCharges"]== " "),"TotalCharges"]=0
df.loc[df["tenure"]== 0]
# lets change the data type to numeric as xgboost dont take objects or strings



df["TotalCharges"]=pd.to_numeric(df["TotalCharges"])

df.dtypes
df.replace(" ","_",regex=True,inplace=True)

df.head()
df["Churn"]=df["Churn"].apply(lambda x: 0 if x=="No" else 1)

df.head()
X= df.drop("Churn",axis=1).copy()

X.head()
y=df["Churn"].copy()

y.head()
pd.get_dummies(X,columns=["PaymentMethod"],drop_first=True).head()
X.info()
X_encoded=pd.get_dummies(X,drop_first=True)
X_encoded.shape
y.unique()
#lets check if the data is our dependent variable is Balanced



sum(y)/len(y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,random_state=42,stratify=y)
#now lets check if it got statified or not



sum(y_train)/len(y_train)
sum(y_test)/len(y_test)
clf_xgb=xgb.XGBClassifier(objective="binary:logistic",missing=None,seed=42)

clf_xgb.fit(X_train,

            y_train,

            verbose=True,

            early_stopping_rounds=10,

            eval_metric="aucpr",

            eval_set=[(X_test,y_test)]

            )
plot_confusion_matrix(clf_xgb,

                      X_test,

                      y_test,

                      values_format="d",

                      display_labels=["Did not leave","Left"])
#Round 1

param_grid={

    "max_depth":[3,4,5],

    "learning_rate":[0.1,0.01,0.05],

    "gamma":[0,0.25,1],

    "reg_lambda":[0,1.0,10.0],

    'scale_pos_weight':[1,3,5]

}
grid_search=GridSearchCV(clf_xgb,param_grid=param_grid,n_jobs=-1,cv=2,scoring="accuracy")

grid_search.fit(X_train,y_train)
print(grid_search.best_params_)
#Round 1

param_grid_2={

    "max_depth":[3,4,5],

    "learning_rate":[0.1,0.01,0.05],

    "gamma":[0,0.25,1],

    "reg_lambda":[0,1.0,10.0],

    'scale_pos_weight':[1,3,5]

}
grid_search=GridSearchCV(estimator=xgb.XGBClassifier(objective="binary:logistic",

                                                     missing=None,

                                                     seed=42,

                                                     subsample=0.9,

                                                     colsample_bytree=0.5

                                                     ),param_grid=param_grid_2,n_jobs=-1,cv=10,scoring="roc_auc",verbose=0)

grid_search.fit(X_train,y_train)

print(grid_search.best_params_)
clf_xgb=xgb.XGBClassifier(objective="binary:logistic",missing=None,seed=42,

                          gamma=1,

                          learning_rate=0.1,

                          max_depth=4,

                          reg_lambda=10,

                          scale_pos_weight=5

                        )











clf_xgb.fit(X_train,

            y_train,

            verbose=True,

            early_stopping_rounds=10,

            eval_metric="aucpr",

            eval_set=[(X_test,y_test)]

            )
plot_confusion_matrix(clf_xgb,

                      X_test,

                      y_test,

                      values_format="d",

                      display_labels=["Did not leave","Left"])
405+62
414/467
851+443
851/1294
node_param={"style":"filled"}
xgb.to_graphviz(clf_xgb,num_trees=0)