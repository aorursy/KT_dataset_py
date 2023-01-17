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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

import warnings
warnings.filterwarnings('ignore')
# Import and read dataset

input_ = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(input_)

df.head(10)
import pandas_profiling as pdp
report = pdp.ProfileReport(df, title='Pandas Profiling Report')
report.to_widgets() 
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(),annot=True)
plt.show()
x = df.drop(columns='DEATH_EVENT')
y = df['DEATH_EVENT']

model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
df.describe()
df=df[df['ejection_fraction']<70]
## data preprocessing

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
    print('Decision Tree Classifier f1-score      : {:.4f}'.format(f1_score( y_test , y_pred)))
    print('Decision Tree Classifier precision     : {:.4f}'.format(precision_score(y_test, y_pred)))
    print('Decision Tree Classifier recall        : {:.4f}'.format(recall_score(y_test, y_pred)))
    print("Decision Tree Classifier roc auc score : {:.4f}".format(roc_auc_score(y_test,y_pred)))
    print("\n",classification_report(y_pred, y_test))
    
    plt.figure(figsize=(6,6))
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")
    plt.title("DecisionTreeClassifier Confusion Matrix (Rate)")
    plt.show()
    
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=["FALSE","TRUE"],
                yticklabels=["FALSE","TRUE"],
                cbar=False)
    plt.title("DecisionTreeClassifier Confusion Matrix (Number)")
    plt.show()
    
def sample_result(class_weight=None,criterion='gini',max_depth=None,max_features=None,max_leaf_nodes=None,min_samples_split=2):    
    scores = [] 
    for i in range(0,10000): # 10.000 samples
        X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2)
        clf = DecisionTreeClassifier(class_weight= class_weight,
                                     criterion=criterion,
                                     max_depth=max_depth,
                                     max_features=max_features,
                                     max_leaf_nodes=max_leaf_nodes,
                                     min_samples_split=min_samples_split) 
        sc=StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)
        clf.fit(X_train, y_train)
        scores.append(accuracy_score(clf.predict(X_test), y_test)) 
    
    plt.hist(scores)
    plt.show()
    print("Best Score: {}\nMean Score: {}".format(np.max(scores), np.mean(scores)))
clf = DecisionTreeClassifier(random_state=0)
result(clf)
sample_result()
param_grid = {
    "max_depth": np.arange(1,10),
    "min_samples_split": [0.001, 0.01, 0.1, 0.2, 0.02, 0.002],
    "criterion": ["gini", "entropy", None],
    "max_leaf_nodes": np.arange(1,10),
    "class_weight": ["balanced", None]
}

clf = DecisionTreeClassifier()
grid = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2, cv=10)
grid.fit(X_train, y_train)
grid.best_params_
clf = DecisionTreeClassifier(
    class_weight='balanced',
    criterion='gini',
    max_depth=1,
    max_leaf_nodes=2,
    min_samples_split=0.001,
    random_state=0
)

result(clf)
# class_weight=None,criterion='gini',max_depth=None,max_features=None,max_leaf_nodes=None,min_samples_split=2
sample_result('balanced',"gini",1 ,None , 2,  0.001)