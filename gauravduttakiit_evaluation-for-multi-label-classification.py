import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
seed = pd.read_csv('/kaggle/input/seed-from-uci/Seed_Data.csv')

seed.head()
seed.shape
seed.info()
seed.describe()
seedd=seed.copy()

seedd.drop_duplicates(subset=None,inplace=True)

seedd.shape
del seedd

seed.shape
seed.isnull().sum()
seed.isnull().sum(axis=1)
sns.violinplot(x='A',data=seed)

plt.show()
sns.violinplot(y='A',x='target',data=seed)

plt.show()
sns.violinplot(x='P',data=seed)

plt.show()
sns.violinplot(y='P',x='target',data=seed)

plt.show()
sns.violinplot(x='C',data=seed)

plt.show()
percentiles = seed['C'].quantile([0.05,0.95]).values

seed['C'][seed['C'] <= percentiles[0]] = percentiles[0]

seed['C'][seed['C'] >= percentiles[1]] = percentiles[1]
sns.violinplot(x='C',data=seed)

plt.show()
sns.violinplot(y='C',x='target',data=seed)

plt.show()
sns.violinplot(x='LK',data=seed)

plt.show()
sns.violinplot(y='LK',x='target',data=seed)

plt.show()
sns.violinplot(x='WK',data=seed)

plt.show()
sns.violinplot(y='WK',x='target',data=seed)

plt.show()
sns.violinplot(x='A_Coef',data=seed)

plt.show()
percentiles = seed['A_Coef'].quantile([0.01,0.99]).values

seed['A_Coef'][seed['A_Coef'] <= percentiles[0]] = percentiles[0]

seed['A_Coef'][seed['A_Coef'] >= percentiles[1]] = percentiles[1]
sns.violinplot(x='A_Coef',data=seed)

plt.show()

sns.violinplot(x='LKG',data=seed)

plt.show()
sns.violinplot(y='LKG',x='target',data=seed)

plt.show()
ax=sns.countplot('target',data=seed)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.show()
seed.describe()
plt.figure(figsize = (10,5))

sns.heatmap(seed.corr(), annot = True, cmap="rainbow")

plt.show()

from sklearn.model_selection import train_test_split

train,test = train_test_split(seed, train_size=0.7, random_state=11)

X_train=train.drop('target',axis=1)

X_test=test.drop('target',axis=1)

y_train=train['target']

y_test=test['target']
# Standarisation technique for scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[:]=scaler.fit_transform(X_train[:])

X_test[:]=scaler.transform(X_test[:])
X_train.info()
from catboost import CatBoostClassifier

model=CatBoostClassifier()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

categorical_features_indices
model.fit(X_train,y_train,cat_features=([]))
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