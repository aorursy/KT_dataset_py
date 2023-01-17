# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
df = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
t_df = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
t_df.head()
df.head()
df.info()
df.isna().mean()
df.describe()
df['Vehicle_Age'].value_counts()
df['Response'].value_counts()
t_df.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
t_df['Gender'] = le.transform(t_df['Gender'])
df['Vehicle_Age'] = le.fit_transform(df['Vehicle_Age'])
t_df['Vehicle_Age'] = le.transform(t_df['Vehicle_Age'])
df['Vehicle_Damage'] = le.fit_transform(df['Vehicle_Damage'])
t_df['Vehicle_Damage'] = le.transform(t_df['Vehicle_Damage'])
df.info()
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Response',data=df)
plt.show()
df['Response'].value_counts(normalize=True)*100
ext = df[df['Response'] == 1]
train_final = pd.concat([df,ext,ext,ext,ext,ext,ext])
train_final['Response'].value_counts(normalize=True)*100
import seaborn as sns
sns.countplot(x='Response',data=train_final)
plt.show()
dff = train_final
dff.corr().round(2)
sns.heatmap(dff.corr())
   
cat_features = dff[[ 'Vehicle_Damage', 'Previously_Insured', 'Gender','Vehicle_Damage', 'Vehicle_Age', 'Driving_License']].columns
for i in cat_features:
    sns.barplot(x="Response",y=i,data=dff)
    plt.title(i+" by "+"Response")
    plt.show()
X = dff.drop(['Response','id'],axis = 1)
y = dff['Response']
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier().fit(X,y)
dt.feature_importances_
X.columns
pd.DataFrame(dt.feature_importances_,index=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured','Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium','Policy_Sales_Channel', 'Vintage']).plot.bar()
X_new = X.drop(['Gender','Driving_License','Vehicle_Age'],axis = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_new,y,test_size = 0.33,random_state = 42)
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_train
x_test
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
dt = DecisionTreeClassifier(random_state=42).fit(x_train,y_train)
dt.score(x_test,y_test)
dt_pred = dt.predict(x_test)
from sklearn.metrics import roc_auc_score
print('DT_Score :',roc_auc_score(y_test,dt_pred))
forest = RandomForestClassifier(random_state=42).fit(x_train,y_train)
forest.score(x_test,y_test)
forest_pred = forest.predict(x_test)
print('Forest_Score :',roc_auc_score(y_test,forest_pred))
grad = GradientBoostingClassifier().fit(x_train,y_train)
grad_pred = grad.predict(x_test)
grad.score(x_test,y_test)
print('GradientBoosting_Score :',roc_auc_score(y_test,grad_pred))
ada = AdaBoostClassifier().fit(x_train,y_train)
ada_pred = ada.predict(x_test)
ada.score(x_test,y_test)
print('AdaBoost_Score :',roc_auc_score(y_test,ada_pred))
xgb = XGBClassifier().fit(x_train,y_train)
xgb_pred = xgb.predict(x_test)
xgb.score(x_test,y_test)
print('xgb_Score :',roc_auc_score(y_test,xgb_pred))
dt = DecisionTreeClassifier(random_state=42).fit(x_train,y_train)
dt.score(x_test,y_test)
print('ROC_AUC_Score :',roc_auc_score(y_test,dt_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, dt_pred).ravel()
confusion_matrix
#Predicting proba
y_pred_prob = dt.predict(x_test)
from sklearn.metrics import roc_curve
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
from sklearn import metrics
cols = ['Model', 'ROC Score', 'Precision Score', 'Recall Score','Accuracy Score','Kappa Score']
models_report = pd.DataFrame(columns = cols)

tmp5 = pd.Series({'Model': " Decision Tree",
                 'ROC Score' : metrics.roc_auc_score(y_test, dt_pred),
                 'Precision Score': metrics.precision_score(y_test, dt_pred),
                 'Recall Score': metrics.recall_score(y_test, dt_pred),
                 'Accuracy Score': metrics.accuracy_score(y_test, dt_pred),
                 'Kappa Score':metrics.cohen_kappa_score(y_test, dt_pred)})

model5_report = models_report.append(tmp5, ignore_index = True)
model5_report
t_df = t_df[X_new.columns]
scc = StandardScaler().fit(X_new)
X_new_sc = scc.transform(X_new)
t_sc = scc.transform(t_df)
dtt = DecisionTreeClassifier(random_state=42).fit(X_new_sc,y)
pred = dtt.predict(t_sc)
pd.Series(pred).to_csv('Prediction.csv')
