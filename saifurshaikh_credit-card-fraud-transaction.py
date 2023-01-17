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
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
print(data.shape)
data.head()
data.Class.value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
corr = data.corr()
corr
plt.figure(figsize=(20,12))
sns.heatmap(corr, annot=True, cmap='viridis', linewidths=0.5)
plt.show()
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
X = data.drop('Class', axis=1)
y = data.Class
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
def model_name(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred))
    print('\t')
    print('ROC_AUC_Score:', metrics.roc_auc_score(y_test, y_pred))
    print('\t')
    print(metrics.confusion_matrix(y_test, y_pred))
    print('\t')
    print('\t\t Classification Report:')
    print(metrics.classification_report(y_test, y_pred))
    print('\t')
    print('Training Score:', model.score(X_train, y_train))
    print('Testing Score:', model.score(X_test, y_test))
log_reg = LogisticRegression(class_weight={0:1, 1:250})
ml_nb = MultinomialNB()
gb_nb = GaussianNB()
svc_rbf = SVC(kernel='rbf', probability=True)
svc_lin = SVC(kernel='linear', probability=True)
rf = RandomForestClassifier(random_state=2, class_weight={0:1, 1:250})
etclf = ExtraTreesClassifier(random_state=2, class_weight={0:1, 1:250})
gboost = GradientBoostingClassifier(random_state=2)
xgboost = XGBClassifier()
estimators = [log_reg, ml_nb, gb_nb, svc_rbf, svc_rbf, rf, etclf, gboost, xgboost]
for i in estimators: 
    print(i)
    model_name(i, X_train, y_train, X_test, y_test)
    print('\t')
    print('\t')
clf = VotingClassifier(estimators=[('lr', log_reg), ('svc_lin', svc_lin), ('svc_rbf', svc_rbf), ('rf', rf), ('etree', etclf), 
                                   ('xbg',xgboost)], voting='hard')
model_name(clf, X_train, y_train, X_test, y_test)
clf1 = VotingClassifier(estimators=[('lr', log_reg), ('svc_lin', svc_lin), ('svc_rbf', svc_rbf), ('rf', rf), ('etree', etclf), 
                                   ('xbg',xgboost)], voting='soft')
model_name(clf1, X_train, y_train, X_test, y_test)
y_pred_prob = clf1.predict_proba(X_test)[:,1]
y_pred_prob
y_hat = np.where(y_pred_prob>0.25, 1, 0)
y_hat
print('Accuracy Score:', metrics.accuracy_score(y_test, y_hat))
print('\t')
print('ROC_AUC_Score:', metrics.roc_auc_score(y_test, y_hat))
print('\t')
print(metrics.confusion_matrix(y_test, y_hat))
print('\t')
print('\t\t\t Classification Report:')
print(metrics.classification_report(y_test, y_hat))
print('\t')
print('Training Score:', clf1.score(X_train, y_train))
print('Testing Score:', clf1.score(X_test, y_hat))
