import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(r'../input/creditcardfraud/creditcard.csv')
data.head().transpose()
data['Class'].value_counts()
data['Class'].value_counts().plot(kind = 'barh')
train = data.copy()
from sklearn.utils import resample
train_minor = train[train['Class']==1]
train_major = train[train['Class']==0]
train_minor = resample(train_minor, n_samples = 284315)
train = pd.concat([train_minor, train_major], 0)
train.shape
sample = train.sample(frac = 0.2)
sample['Class'].value_counts()
corr = sample.corr()
sns.heatmap(corr, cmap = 'coolwarm')
import pandas as pd
from xgboost import XGBClassifier as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier as cbc
models = [
    (xgb(), 'xgb'),
    (GradientBoostingClassifier(), 'gbt'),
    (RandomForestClassifier(), 'rf'),
    (BernoulliNB(), 'bnb'),
    (GaussianNB(), 'gb'),
    #(SVC(), 'svc'),
    (LogisticRegression(), 'lr')]
score = {'Model': [],
         'train': [],
         'test': []}
def model_score(model_info):

    name = model_info[1]
    model = model_info[0]
    
    model.fit(X_train , y_train)
    
    score['Model'].append(name)
    score['train'].append(model.score(X_train, y_train))
    score['test'].append(model.score(X_test, y_test))
from sklearn.model_selection import train_test_split as tst
target_column = 'Class'

X_train, X_test, y_train, y_test = tst(sample.drop(target_column, 1), sample[target_column], random_state = 42, test_size = 0.2)

for model in models:
    print(model[1])
    model_score(model)
cat = cbc(iterations = 500)
cat.fit(X_train, y_train, plot = True, verbose = 50, eval_set = (X_test, y_test), )
score['Model'].append('cbc')
score['train'].append(cat.score(X_train, y_train))
score['test'].append(cat.score(X_test, y_test))
pd.DataFrame(score)
from sklearn.ensemble import VotingClassifier
models = [
    ('xgb', xgb()),
    ('rf', RandomForestClassifier()),
    ('lr', LogisticRegression())]
model = VotingClassifier(models, voting= 'soft')
model.fit(X_train, y_train)
from sklearn.metrics import roc_auc_score, classification_report
roc_auc_score(y_test, model.predict(X_test))
print(classification_report(y_test, model.predict(X_test)))
labels = data['Class']
X = data.drop('Class', 1)

print(roc_auc_score(labels, model.predict(X)))
print(classification_report(labels, model.predict(X)))
print(roc_auc_score(labels, cat.predict(X)))
print(classification_report(labels, cat.predict(X)))
from imblearn.over_sampling import SMOTE
models = [
    (xgb(), 'xgb'),
    (GradientBoostingClassifier(), 'gbt'),
    (RandomForestClassifier(), 'rf'),
    (BernoulliNB(), 'bnb'),
    (GaussianNB(), 'gb'),
    #(SVC(), 'svc'),
    (LogisticRegression(), 'lr')]
score = {'Model': [],
         'train': [],
         'test': []}
OS = SMOTE(random_state=12)
X_os, y_os = OS.fit_sample(data.drop('Class', 1), data['Class'])
df = pd.concat([X_os, y_os], 1)
df.head().transpose()
sample = df.sample(frac = 0.2)
target_column = 'Class'

X_train, X_test, y_train, y_test = tst(sample.drop(target_column, 1), sample[target_column], random_state = 42, test_size = 0.2)

for model in models:
    print(model[1])
    model_score(model)
cat = cbc(iterations = 500)
cat.fit(X_train, y_train, plot = True, verbose = 50, eval_set = (X_test, y_test))
score['Model'].append('cbc')
score['train'].append(cat.score(X_train, y_train))
score['test'].append(cat.score(X_test, y_test))
pd.DataFrame(score)
labels = data['Class']
X = data.drop('Class', 1)

print(roc_auc_score(labels, cat.predict(X)))
print(classification_report(labels, cat.predict(X)))