import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import balanced_accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, make_scorer

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
file_path = "/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)
df.head()
df.info()
df.describe().T
for cols in df.columns:
    print(cols, " : ", df[cols].unique())
df.replace(to_replace=["No_phone_service","No_internet_service"], value="No", inplace=True)
for cols in df.columns:
    print(cols, " : ", df[cols].unique())
len(df.loc[df["TotalCharges"]==" "])
df.loc[df["TotalCharges"]==" "]
df.loc[(df["TotalCharges"]==" "), "TotalCharges"] = 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df.info()
df.replace(' ', '_', regex=True, inplace=True)
df.head()
df["Churn"] = df["Churn"].replace(to_replace=["Yes", "No"], value=[1, 0])
df["Churn"].head()
sns.boxplot(x=df["gender"], y=df["TotalCharges"], hue=df["Churn"]);
sns.boxplot(x=df["SeniorCitizen"], y=df["TotalCharges"], hue=df["Churn"]);
sns.boxplot(y=df["MonthlyCharges"], x=df["Churn"]);
sns.boxplot(y=df["TotalCharges"], x=df["Churn"]);
sns.pairplot(df);
X = df.drop(columns="Churn", axis=1).copy()
X.head()
y = df["Churn"].copy()
y.head()
X.drop(columns="customerID", inplace=True)
cat_cols = list(X.columns[X.dtypes==object])
cat_cols
X_encoded = pd.get_dummies(X, columns=cat_cols)
X_encoded.head(10)
y.unique()
sum(y)/len(y) * 100
display(X_encoded.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, stratify=y, test_size=0.3, random_state=24)
sum(y_train)/len(y_train) * 100
sum(y_test)/len(y_test) * 100
clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic',
                           missing=None,
                           seed=24)
clf_xgb.fit(X_train,
           y_train,
           verbose=True,
           early_stopping_rounds=10,
           eval_metric='aucpr',
           eval_set=[(X_test, y_test)])
plot_confusion_matrix(clf_xgb,
                     X_test,
                     y_test,
                     values_format='d',
                     display_labels=['Churned','Not Churned'])
param_grid = {
    'max_depth' : [3,4,5],
    'learning_rate' : [0,1,0.01,0.05],
    'gamma' : [0,0.25,1.0],
    'reg_lambda' : [0,1.0,10.0],
    'scale_pos_weight' : [1,3,5]
}
xgb_estimator = xgb.XGBClassifier(objective='binary:logistic',
                                  seed=24,
                                  subsample=0.9,
                                  colsample_bytree=0.5)
clf_xgb_tuned = GridSearchCV(estimator=xgb_estimator,
                             param_grid=param_grid,
                             scoring='roc_auc',
                             verbose=2,
                             n_jobs=-1,
                             cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
)
clf_xgb_tuned.fit(X_train,
                  y_train,
                  verbose=True,
                  early_stopping_rounds=10,
                  eval_metric='aucpr',
           eval_set=[(X_test, y_test)]
)
clf_xgb_tuned.best_estimator_
clf_xgb_tuned.best_params_
clf_xgb_tuned.best_score_
plot_confusion_matrix(clf_xgb_tuned,
                     X_test,
                     y_test,
                     values_format='d',
                     display_labels=['Churned','Not Churned'])
y_pred = clf_xgb_tuned.predict(X_test)
print(classification_report(y_pred, y_test))
plot_roc_curve(clf_xgb_tuned,
               X_test,
               y_test,
               name='XGB Tuned ROC AUC');
plot_precision_recall_curve(clf_xgb_tuned,
               X_test,
               y_test,
               name='XGB Precision-Recall Curve');
xgb_lone_estimator = xgb.XGBClassifier(
    objective='binary:logistic', 
    seed=24, 
    subsample=0.9, 
    colsample_bytree=0.5, 
    gamma = 0.25, 
    learning_rate=0.05, 
    max_depth=4,
    reg_lambda=10.0, 
    scale_pos_weight=3,
    n_estimator=1
)
xgb_lone_estimator.fit(X_train, y_train)
xgb_bst = xgb_lone_estimator.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s:  ' % importance_type, xgb_bst.get_score(importance_type=importance_type))
node_params = {
    'shape' : 'box',
    'style' : 'filled, rounded',
    'fillcolor' : '#78cbe'
}

leaf_params = {
    'shape' : 'box',
    'style' : 'filled',
    'fillcolor' : '#e48038'
}
xgb.to_graphviz(xgb_lone_estimator,
                num_trees=0,
                size="5,5",
                condition_node_params=node_params,
                leaf_node_params=leaf_params
)
