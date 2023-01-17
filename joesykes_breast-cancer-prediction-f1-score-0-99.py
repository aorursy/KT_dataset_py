import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.isnull().sum()
df = df.drop('Unnamed: 32',axis=1)
df.head()
df = df.drop('id',axis=1)
df['diagnosis'].unique()
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == "M" else 0)
df.info()
sns.countplot(x='diagnosis',data=df)
df['diagnosis'].value_counts()
len(df[df['diagnosis']==1]) * 100 / len(df)
df.describe()
y = df['diagnosis']              # Target Variable

X = df.drop('diagnosis',axis=1)  # Independent Variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

df = pd.DataFrame(X,columns=df.columns[1:])

df['diagnosis'] = y
df.head()
df.describe()
x = df.drop('diagnosis',axis=1)

y = df['diagnosis']

data = pd.concat([y,x.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)
data = pd.concat([y,x.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)
data = pd.concat([y,x.iloc[:,20:30]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)
plt.figure(figsize=(20,20))

sns.heatmap(df.corr(), annot=True,cmap='viridis')
for i in df.columns:

    print("Features highly related to column {}:".format(i))

    related_list = []

    for j in df.columns:

        if (i != j) & (abs(df.corr()[i][j]) > 0.9):

            related_list.append(j)

    print(related_list)

    print("-" * 50)
sns.jointplot(df['texture_worst'],df['texture_mean'],kind='regg',color='purple')
sns.jointplot(df['concave points_mean'],df['concave points_worst'],kind='regg')
sns.pairplot(df[['radius_mean','radius_se','radius_worst','perimeter_mean','perimeter_se','perimeter_worst','area_mean','area_se','area_worst','diagnosis']],hue='diagnosis')
X = df.iloc[:,:-1]

y = df.iloc[:,-1]



#from sklearn.model_selection import train_test_split

#from sklearn.model_selection import GridSearchCV

# 
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=5000,random_state=11)

rf.fit(X,y)

feat_imp = pd.DataFrame(rf.feature_importances_)

feat_imp.index = pd.Series(df.iloc[:,:-1].columns)

feat_imp = (feat_imp*100).copy().sort_values(by=0,ascending=False)

feat_imp = feat_imp.reset_index()

feat_imp
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



results_list = []

for var in np.arange(feat_imp.shape[0],9,-1):

    X_new = X[feat_imp.iloc[:var,0]].copy()

    X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

    final_rf = RandomForestClassifier(random_state=11)

    gscv = GridSearchCV(estimator=final_rf,param_grid={

        "n_estimators":[100,500,1000,5000],

        "criterion":["gini","entropy"]

    },cv=5,n_jobs=-1,scoring="f1_weighted")



    model = gscv.fit(X_train,y_train)

    

    results_list.append((var, model.best_score_))

    print("Model Created using the top {} variables".format(var))

    print("F1 Score: {}".format(model.best_score_))

    print("-"*30)

    

    #print(str(var)+" variables:  "+str(model.best_estimator_)+"  F1 score: "+str(model.best_score_))
from imblearn.over_sampling import SMOTE

SMOTE_list = []

for var in np.arange(feat_imp.shape[0],9,-1):

    X_new = X[feat_imp.iloc[:var,0]].copy()

    X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

    smote = SMOTE(random_state = 11) 

    X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

    final_rf = RandomForestClassifier(random_state=11)

    gscv = GridSearchCV(estimator=final_rf,param_grid={

        "n_estimators":[100,500,1000,5000],

        "criterion":["gini","entropy"]

    },cv=5,n_jobs=-1,scoring="f1_weighted")



    model = gscv.fit(X_train_smote,y_train_smote)

    SMOTE_list.append((var, model.best_score_))

    print("SMOTE Model Created using the top {} variables".format(var))

    print("F1 Score: {}".format(model.best_score_))

    print("Best Model {}".format(model.best_estimator_))

    print("-"*30)
x_plot = range(10,31)

y_results = [] 

for i in range(20,-1,-1):

    y_results.append(results_list[i][1])

y_results

y_results_SMOTE = [] 

for i in range(20,-1,-1):

    y_results_SMOTE.append(SMOTE_list[i][1])

y_results

y_1 = y_results

y_2 = y_results_SMOTE



plt.figure(figsize=(10,6))

plt.plot(x_plot, y_1, '-b', label='Without SMOTE')

plt.plot(x_plot, y_2, '-r', label='With SMOTE')

plt.legend()

plt.xlabel('Number of Variables')

plt.ylabel('F1 Score')

plt.title('Figure Comparing the F1 Scores obtained with and without using SMOTE')

from xgboost import XGBClassifier



xgb = XGBClassifier(n_estimators=5000,random_state=11)

xgb.fit(X,y)

feat_imp_xgb = pd.DataFrame(xgb.feature_importances_)

feat_imp_xgb.index = pd.Series(df.iloc[:,:-1].columns)

feat_imp_xgb = (feat_imp_xgb*100).copy().sort_values(by=0,ascending=False)

feat_imp_xgb = feat_imp_xgb.reset_index()

feat_imp_xgb
SMOTE_list_xgb = []

for var in np.arange(feat_imp.shape[0],9,-1):

    X_new = X[feat_imp.iloc[:var,0]].copy()

    X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

    smote = SMOTE(random_state = 11) 

    X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

    final_xgb = XGBClassifier(random_state=11)

    gscv = GridSearchCV(estimator=final_xgb,param_grid={

        "n_estimators":[100,500,1000,5000],

        "criterion":["gini","entropy"]

    },cv=5,n_jobs=-1,scoring="f1_weighted")



    model = gscv.fit(X_train_smote,y_train_smote)

    SMOTE_list_xgb.append((var, model.best_score_))

    print("SMOTE XGB Model Created using the top {} variables".format(var))

    print("F1 Score: {}".format(model.best_score_))

    print("Best Model {}".format(model.best_estimator_))

    print("-"*30)
xgb_results = []

for i in range(20,-1,-1):

    xgb_results.append(SMOTE_list_xgb[i][1])

    

plt.figure(figsize=(10,6))

plt.plot(x_plot, xgb_results, '-b')

plt.xlabel('Number of Variables')

plt.ylabel('F1 Score')

plt.title('Figure Comparing the F1 Scores obtained using XGBoost for various numbers of input variables')
X_new = X[feat_imp.iloc[:18,0]].copy()

X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

smote = SMOTE(random_state = 11) 

X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

final_rf = RandomForestClassifier(random_state=11)

gscv = GridSearchCV(estimator=final_rf,param_grid={

    "n_estimators":[100,500,1000,5000],

    "criterion":["gini","entropy"]

},cv=5,n_jobs=-1,scoring="f1_weighted")



model = gscv.fit(X_train_smote,y_train_smote)

final_rfc_model = model.best_estimator_

    
rfc_preds = final_rfc_model.predict(X_test)
X_new = X[feat_imp.iloc[:26,0]].copy()

X_train, X_test, y_train,y_test = train_test_split(X_new,y,test_size=0.2,random_state=11)

smote = SMOTE(random_state = 11) 

X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

final_xgb = XGBClassifier(random_state=11)

gscv = GridSearchCV(estimator=final_xgb,param_grid={

   "n_estimators":[100,500,1000,5000],

   "criterion":["gini","entropy"]

},cv=5,n_jobs=-1,scoring="f1_weighted")



model = gscv.fit(X_train_smote,y_train_smote)

final_xgb_model = model.best_estimator_
xgb_preds = final_xgb_model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, rfc_preds))
plt.figure(figsize=(10,6))

sns.heatmap(confusion_matrix(y_test, rfc_preds),annot=True)

plt.ylabel('Actual Class')

plt.xlabel('Predicted Class')

plt.title('Predictions Using the Random Forest Classifier')
print(classification_report(y_test, xgb_preds))
plt.figure(figsize=(10,6))

sns.heatmap(confusion_matrix(y_test, xgb_preds),annot=True)

plt.ylabel('Actual Class')

plt.xlabel('Predicted Class')

plt.title('Predictions Using the XGBoost Classifier')