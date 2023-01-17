# Import the necessary packages
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
from lightgbm.sklearn import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')
# Import and read dataset
input_ = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(input_)

df.head(10)
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)
plt.show()
df.describe()
x = df.drop(columns='DEATH_EVENT')
y = df['DEATH_EVENT']

model = LGBMClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
for i in range(0,len(df.columns)):
    print("{} = {}".format(i,df.columns[i]))
# Delete outlier
df = df[df['ejection_fraction']<70]
inp_data = df.drop(df[['DEATH_EVENT']], axis=1)
#inp_data = df.iloc[:,[11,6,2,4,7,0,8]]
out_data = df[['DEATH_EVENT']]

X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2, random_state=0, shuffle=True)

## Applying Transformer
#sc= StandardScaler()
#sc = MinMaxScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)
## X_train, X_test, y_train, y_test Shape

print("X_train Shape : ", X_train.shape)
print("X_test Shape  : ", X_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)
# I coded this method for convenience and to avoid writing the same code over and over again

def result(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print('Accuracy Score    : {:.4f}'.format(accuracy_score(y_test, y_pred)))
    print('LightGBM f1-score      : {:.4f}'.format(f1_score( y_test , y_pred)))
    print('LightGBM precision     : {:.4f}'.format(precision_score(y_test, y_pred)))
    print('LightGBM recall        : {:.4f}'.format(recall_score(y_test, y_pred)))
    print("LightGBM roc auc score : {:.4f}".format(roc_auc_score(y_test,y_pred)))
    print("\n",classification_report(y_pred, y_test))
    
    plt.figure(figsize=(6,6))
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
    plt.title("LightGBM Confusion Matrix (Rate)")
    plt.show()
    
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=["FALSE","TRUE"],
                yticklabels=["FALSE","TRUE"],
                cbar=False)
    plt.title("LightGBM Confusion Matrix (Number)")
    plt.show()
    
    
def report(**params):
    scores = [] 
    for i in range(0,250): # 250 samples
        X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2, shuffle=True)
        sc = StandardScaler()
        clf = LGBMClassifier(**params)
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        clf.fit(X_train, y_train)
        scores.append(accuracy_score(clf.predict(X_test), y_test)) 
        
    Importance = pd.DataFrame({'Importance':clf.feature_importances_*100},index=df.columns[:12])
    Importance.sort_values(by='Importance',axis=0,ascending=True).plot(kind='barh',color='lightblue')
    plt.xlabel('Importance for variable');
    plt.hist(scores)
    plt.show()
    print("Best Score: {}\nMean Score: {}".format(np.max(scores), np.mean(scores)))
clf = LGBMClassifier()
result(clf)
report()
param_grid = {
    'min_child_weight': np.arange(1,20,1),
    'colsample_bytree': np.linspace(0.5,2,11)
}

clf = LGBMClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = LGBMClassifier(
    min_child_weight= 0.6,
    colsample_bytree= 0.65,
    n_jobs=-1
)

result(clf)
report(
    max_depth= 1,
    min_child_weight= 1,
)
param_grid = {
    "n_estimators": [10,100,1000,10000 ]
}

clf = LGBMClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = LGBMClassifier(
    max_depth= 1,
    min_child_weight= 1,
    gamma = 0.0,
    colsample_bytree= 0.5,
    n_estimators=10
)

result(clf)
report(
    max_depth= 1,
    min_child_weight= 1,
    gamma = 0.0,
    colsample_bytree= 0.5,
    n_estimators=10
)
param_grid = {
 'reg_alpha': [0.001, 0.005, 0.01, 0.05]
}

clf = LGBMClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = LGBMClassifier(
    max_depth= 1,
    min_child_weight= 1,
    gamma = 0.0,
    colsample_bytree= 0.5,
    n_estimators=10,
    reg_alpha=0.001
)

result(clf)
report(
    max_depth= 1,
    min_child_weight= 1,
    colsample_bytree= 0.5,
    n_estimators=10,
    reg_alpha=0.001
)
clf = LGBMClassifier(
    max_depth= 1,
    n_estimators=100,
    colsample_bytree=0.9,
    gamma=0.5,
    learning_rate=0.01,
    
)

result(clf)
report(
    max_depth= 1,
    n_estimators=100,
    colsample_bytree=0.9,
    gamma=0.5,
    learning_rate=0.01,
)