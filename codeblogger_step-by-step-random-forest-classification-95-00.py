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
from sklearn.ensemble import RandomForestClassifier

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

model = RandomForestClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
# Delete outlier
df=df[df['ejection_fraction']<70]
#inp_data = df.drop(df[['DEATH_EVENT']], axis=1)
inp_data = df.iloc[:,[0,4,7,11]]
out_data = df[['DEATH_EVENT']]

X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2, random_state=0)

## Applying Transformer
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
## X_train, X_test, y_train, y_test Shape

print("X_train Shape : ", X_train.shape)
print("X_test Shape  : ", X_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)
## I coded this method for convenience and to avoid writing the same code over and over again

def result(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
    print('Random Forest Classifier f1-score      : {:.4f}'.format(f1_score( y_test , y_pred)))
    print('Random Forest Classifier precision     : {:.4f}'.format(precision_score(y_test, y_pred)))
    print('Random Forest Classifier recall        : {:.4f}'.format(recall_score(y_test, y_pred)))
    print("Random Forest Classifier roc auc score : {:.4f}".format(roc_auc_score(y_test,y_pred)))
    print("\n",classification_report(y_pred, y_test))
    
    plt.figure(figsize=(6,6))
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
    plt.title("RandomForestClassifier Confusion Matrix (Rate)")
    plt.show()
    
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=["FALSE","TRUE"],
                yticklabels=["FALSE","TRUE"],
                cbar=False)
    plt.title("RandomForestClassifier Confusion Matrix (Number)")
    plt.show()
    
def sample_result(
    n_estimators=100,
    max_features='auto',
    max_depth=None,
    min_samples_split=2):    
    
    scores = [] 
    for i in range(0,100): # 100 samples
        n_estimators, max_features, max_depth, min_samples_split
        X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2)
        clf = RandomForestClassifier(n_estimators= n_estimators,
                                     max_features=max_features,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split) 
        sc=StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        clf.fit(X_train, y_train)
        scores.append(accuracy_score(clf.predict(X_test), y_test)) 
    
    plt.hist(scores)
    plt.show()
    print("Best Score: {}\nMean Score: {}".format(np.max(scores), np.mean(scores)))
clf = RandomForestClassifier(random_state=0)
result(clf)
sample_result()
param_grid = {
    "n_estimators": [100, 500, 1000],
    "max_features": [0.5,1,'auto'],
    "max_depth": [1,2,3,4,None],
    "min_samples_split": [2,5,8]
}

clf = RandomForestClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = RandomForestClassifier(
    n_estimators=1000,
    max_features=0.5,
    max_depth=3,
    min_samples_split=5,
    random_state=0
)

result(clf)
sample_result(1000,0.5,3,5)
Importance = pd.DataFrame({'Importance':clf.feature_importances_*100},index=df.iloc[:,[0,4,7,11]].columns)
Importance.sort_values(by='Importance',axis=0,ascending=True).plot(kind='barh',color='lightblue')
plt.xlabel('Importance for variable');