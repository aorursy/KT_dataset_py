import sklearn
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import roc_curve, auc
train=pd.read_csv("../input/naivebayes/train_dataNB.csv")
test=pd.read_csv("../input/naivebayes/test_featuresNB.csv")
train.head()
print(train.shape)
print(test.shape)
train.drop('Id',axis=1);
test.drop('Id',axis=1);
train['ham'].value_counts().plot('bar')
plt.matshow(train.corr())
lista = list(train)
for i in lista:
    if(abs(train[i].corr(train['ham'])) <= 0.2):
        train.pop(i)
        test.pop(i)
HAM = train.pop('ham')
train.head()

bnb = BernoulliNB()
train_scores = cross_val_score(bnb, train, HAM, cv=10)
train_scores.mean()
t=pd.read_csv("../input/naivebayes/train_dataNB.csv")
t_scores = cross_val_score(bnb, t, HAM, cv=10)
t_scores.mean()


f3 = make_scorer(fbeta_score, beta=3)
t_f3scores = cross_val_score(bnb, t, HAM, cv=10, scoring = f3)
t_f3scores.mean()
proba = cross_val_predict(bnb, t, HAM, cv=10, method = 'predict_proba')
fpr, tpr ,thresholds =roc_curve(HAM,proba[:,1]);
lw=2
plt.plot(fpr,tpr, color='red')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

roc_auc=auc(fpr,tpr)
roc_auc
plt.plot(thresholds)
plt.show()
