!pip3 install lightgbm

!pip3 install xgboost

!pip3 install catboost
import pandas as pd

import numpy as np

from pandas_profiling import ProfileReport



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics
df = pd.read_csv("../input/drug200/drug200.csv")

df.head()
profile = ProfileReport(df, title='EDA Profiling')
profile.to_widgets()
df.describe().transpose()
features = df.copy()

labels = features.pop('Drug')

features = features.values
sexo = preprocessing.LabelEncoder()

sexo.fit(['F','M'])

features[:,1] = sexo.transform(features[:,1]) 





pressao = preprocessing.LabelEncoder()

pressao.fit([ 'LOW', 'NORMAL', 'HIGH'])

features[:,2] = pressao.transform(features[:,2])





colesterol = preprocessing.LabelEncoder()

colesterol.fit([ 'NORMAL', 'HIGH'])

features[:,3] = colesterol.transform(features[:,3]) 
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.5, random_state=0)
tree = DecisionTreeClassifier()

lgbm = LGBMClassifier()

xgb = XGBClassifier()

cat = CatBoostClassifier()

rf = RandomForestClassifier()

svm = SVC()
tree.fit(features_train, labels_train)

lgbm.fit(features_train, labels_train)

xgb.fit(features_train, labels_train)

cat.fit(features_train, labels_train)

rf.fit(features_train, labels_train)

svm.fit(features_train, labels_train)
test_results = {}



test_results['Decision Tree'] = metrics.accuracy_score(labels_test, tree.predict(features_test))

test_results['LightGBM'] = metrics.accuracy_score(labels_test, lgbm.predict(features_test))

test_results['XGBoost'] = metrics.accuracy_score(labels_test, xgb.predict(features_test))

test_results['CatBoost'] = metrics.accuracy_score(labels_test, cat.predict(features_test))

test_results['Random Forest'] = metrics.accuracy_score(labels_test, rf.predict(features_test))

test_results['SVM'] = metrics.accuracy_score(labels_test, svm.predict(features_test))
print_df = pd.DataFrame(test_results, index=['Acurácia']).T

print_df['Acurácia'].map('{:,.2f}'.format)

print_df