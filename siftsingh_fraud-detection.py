import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, recall_score, classification_report

%matplotlib inline
data = pd.read_csv("../input/creditcard.csv")
data.head()
target = data['Class']
features = data.drop(["Class"],axis=1)
# Checking for Null Values
data.isna().sum()
modelxgb = XGBClassifier()
modelxgb.fit(features, target)

print(modelxgb.feature_importances_)
from xgboost import plot_importance
plot_importance(modelxgb)
f_xgb = pd.DataFrame(data={'feature':features.columns,'value':modelxgb.feature_importances_})
f_xgb = f_xgb.sort_values(['value'],ascending=False )
plt.figure(figsize=(15,8))
sns.barplot(f_xgb['feature'],f_xgb['value'])
etcmodel = ExtraTreesClassifier()
etcmodel.fit(features,target)
print(etcmodel.feature_importances_)
f_etc = pd.DataFrame(data={'feature':features.columns,'value':etcmodel.feature_importances_})
f_etc = f_etc.sort_values(['value'],ascending=False )
plt.figure(figsize=(15,8))
sns.barplot(f_etc['feature'],f_etc['value'])
ft = pd.merge(f_xgb, f_etc, how='inner', on=["feature"])
ft.sort_values(["value_x","value_y"],ascending=False, inplace=True)
top15ft = ft.head(15)
top15ft
plt.figure(figsize=(20,10))
sns.heatmap(features[top15ft.feature].corr(), square=True, annot=True,robust=True, yticklabels=1)
X=features[top15ft.feature]
Y=target.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=7)
modelsvm = svm.SVC(kernel='rbf', gamma='auto')
modelsvm.fit(X_train, y_train)
y_svm = modelsvm.predict(X_test)
accuracy_score(y_test, y_svm)
svm_conf_matrix = confusion_matrix(y_test, y_svm)
svm_conf_matrix
modelrf = RandomForestClassifier(max_depth = 2, random_state = 0)
modelrf.fit(X_train, y_train)
y_rf = modelrf.predict(X_test)
accuracy_score(y_test, y_rf)
class_count = pd.value_counts(data['Class'], sort = True).sort_index()
class_count.plot(kind = 'bar')
plt.title("Fraud Classes")
plt.xlabel("Class")
plt.ylabel("Frequency")
class_count
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data.head()