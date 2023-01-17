import pandas as pd

df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 15
MEDIUM_SIZE = 17
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

df_eda = df.copy()
cols = df_eda.columns.tolist()
cols = set(cols) - set(['customerID', 'TotalCharges', 'tenure', 'MonthlyCharges', 'Churn'])

fig, *axes = plt.subplots(nrows=8, ncols=2, figsize=(20,35))
axes = np.array(axes).ravel()

for c,i in enumerate(cols):
    df_eda.groupby(['Churn', i]).size().unstack(level=0).plot(kind='barh', ax=axes[c])
plt.tight_layout()
Y = df['Churn'].map({"Yes": 1, "No": 0})
df = df.drop('Churn', 1);
df = df.drop('customerID', 1);
#Check dtypes of columns
df.dtypes
#Cast `TotalCharges` to float
df['TotalCharges'] = df['TotalCharges'].apply(pd.to_numeric, errors='coerce')
#Check NaN values in columns
df.isnull().sum()
#Fill NaN with 0 in `TotalCharges`
df['TotalCharges'] = df['TotalCharges'].fillna(0)
c_single = ('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'gender')
c_all = ('MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod')

df_categorical = df.join(pd.concat([pd.get_dummies(df[col], prefix=col, drop_first=True) for col in c_single] + [pd.get_dummies(df[col], prefix=col) for col in c_all], axis=1))
df_categorical.drop(list(c_single + c_all), inplace=True, axis=1)
df_categorical.head()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(df_categorical, Y, test_size=0.2, random_state=42)
clfs = [LogisticRegression(C=0.1, random_state=0), SVC(C=0.01, degree=2, gamma='auto', probability=True, random_state=0), DecisionTreeClassifier(random_state=0, max_depth=5, criterion='entropy'), RandomForestClassifier(max_depth=6, criterion='entropy', n_estimators=10, random_state=0)]
clf_labels = ['Logistic Regression', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier']
for clf, label in zip(clfs, clf_labels):
    score = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10, scoring='roc_auc')
    print("ROC AUC {} Score Mean: {}, Std: {}".format(label, round(score.mean(),2), round(score.std(),3)))
from sklearn.metrics import roc_curve, auc

plt.rc('legend', fontsize=10)
colors = ['black', 'orange', 'blue', 'green']
line_styles = [':', '--', '-.', '-']

for clf, label, clr, ls in zip(clfs, clf_labels, colors, line_styles):
    y_pred = clf.fit(x_train, y_train).predict_proba(x_test)[:,1]
    fpr, tpr, threashold = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label="{} (auc = {})".format(label, round(roc_auc, 2)))

plt.legend(loc='lower right')
plt.plot([0,1], [0,1], linestyle='--', color='gray', linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sn

y_pred = clfs[-1].fit(x_train, y_train).predict(x_test)

ax= plt.subplot()
sn.heatmap(confusion_matrix(y_test, y_pred), annot=True, ax=ax, cmap='Blues', fmt='g')
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix');