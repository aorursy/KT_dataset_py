import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

pd.set_option('display.max_columns', None)
health =pd.read_csv('/kaggle/input/fetal-health-classification/fetal_health.csv')

health.head()
health.info()
health.describe()
healthd=health.copy()

healthd.drop_duplicates(inplace=True)

healthd.shape
health.shape
health=healthd

del healthd

health.head()
health.shape
import pandas_profiling as pp 

profile = pp.ProfileReport(health) 

profile.to_file("EDA.html")
X=health.drop('fetal_health',axis=1)

y=health['fetal_health']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.70,stratify=y,shuffle=True, random_state = 42)
X_train.info()
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(random_state=42)
model.fit(X_train,y_train)
import sklearn.metrics

score_cbc=model.score(X_test,y_test)

print('Score :',score_cbc)
sklearn.metrics.confusion_matrix(y_test,model.predict(X_test))
roc_auc_score_cbc=sklearn.metrics.roc_auc_score(y_test,model.predict_proba(X_test),multi_class='ovr')

print('Compute Area Under the Receiver Operating Characteristic Curve',roc_auc_score_cbc)
cohen_kappa_score_cbc=sklearn.metrics.cohen_kappa_score(model.predict(X_test),y_test)

print('Cohen’s kappa :',cohen_kappa_score_cbc)
sklearn.metrics.f1_score(y_test,model.predict(X_test),average='macro')
sklearn.metrics.f1_score(y_test,model.predict(X_test),average='micro')
sklearn.metrics.f1_score(y_test,model.predict(X_test),average='weighted')
sklearn.metrics.f1_score(y_test,model.predict(X_test),average=None)
sklearn.metrics.fbeta_score(y_test,model.predict(X_test),average='macro',beta=0.5)
sklearn.metrics.fbeta_score(y_test,model.predict(X_test),average='micro',beta=0.5)
sklearn.metrics.fbeta_score(y_test,model.predict(X_test),average='weighted',beta=0.5)
sklearn.metrics.fbeta_score(y_test,model.predict(X_test),average=None,beta=0.5)
sklearn.metrics.hamming_loss(y_test,model.predict(X_test))
sklearn.metrics.hinge_loss(y_test,model.predict_proba(X_test))
sklearn.metrics.jaccard_score(y_test,model.predict(X_test),average='macro')
sklearn.metrics.jaccard_score(y_test,model.predict(X_test),average='micro')
sklearn.metrics.jaccard_score(y_test,model.predict(X_test),average='weighted')
sklearn.metrics.jaccard_score(y_test,model.predict(X_test),average=None)
sklearn.metrics.log_loss(y_test,model.predict_proba(X_test))
sklearn.metrics.multilabel_confusion_matrix(y_test,model.predict(X_test))
sklearn.metrics.matthews_corrcoef(y_test,model.predict(X_test))
sklearn.metrics.precision_recall_fscore_support(y_test,model.predict(X_test))
sklearn.metrics.precision_score(y_test,model.predict(X_test),average='macro')
sklearn.metrics.precision_score(y_test,model.predict(X_test),average='micro')
sklearn.metrics.precision_score(y_test,model.predict(X_test),average='weighted')
sklearn.metrics.precision_score(y_test,model.predict(X_test),average=None)
sklearn.metrics.recall_score(y_test,model.predict(X_test),average='macro')
sklearn.metrics.recall_score(y_test,model.predict(X_test),average='micro')
sklearn.metrics.recall_score(y_test,model.predict(X_test),average='weighted')
sklearn.metrics.recall_score(y_test,model.predict(X_test),average=None)
sklearn.metrics.zero_one_loss(y_test,model.predict(X_test))
print(sklearn.metrics.classification_report(y_test,model.predict(X_test)))
# predict result

X_test['Predicted fetal_health']= model.predict(X_test)

X_test.head()

plt.figure(figsize = (10,5))

ax= sns.countplot(X_test['Predicted fetal_health'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 45)

ax.set_yscale('log')

plt.show()
plt.figure(figsize = (10,5))

ax= sns.countplot(y_test)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 45)

ax.set_yscale('log')

plt.show()
X_test['Predicted fetal_health'].value_counts(),y_test.value_counts()