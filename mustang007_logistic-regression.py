import pandas as pd
from sklearn.datasets import load_wine
wine_dataset = load_wine()
wine = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)

wine['quality'] = wine_dataset.target
wine.head()

wine = wine[wine.quality !=2]
wine.quality.value_counts()
X = wine.drop(columns='quality')
Y = wine['quality']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
pred
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, auc, roc_curve
confusion_matrix(pred, Y_test)
precision_score(Y_test, pred)
recall_score(Y_test, pred)
f1_score(pred,Y_test)
classification_report(pred, Y_test)
# a = auc(pred, Y_test)
from sklearn.metrics import roc_auc_score
fpr, tpr, thres = roc_curve(Y_test,  pred)
plt.scatter(fpr, tpr)
thres
roc_auc_score(pred, Y_test)

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba)
auc = roc_auc_score(Y_test, pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
plt.scatter(fpr, tpr)
plt.show()

# Print ROC curve
import numpy as np
plt.plot(fpr,tpr)
plt.show() 

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.pipeline import make_pipeline
# anova_filter = SelectKBest(f_regression, k=3)
# anova_svm = make_pipeline(anova_filter, model)
# anova_svm.fit(X_train, Y_train)
# pred = anova_svm.predict(X_test)
# from sklearn.linear_model import RidgeClassifierCV
# model = RidgeClassifierCV()
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform
# logistic = LogisticRegression(solver='liblinear',multi_class='ovr')
# distributions = dict(class_weight=['balanced'],penalty=['l2', 'l1'],max_iter=[10,50,100,300,500,900])
# clf = RandomizedSearchCV(logistic, distributions, random_state=0)
# search = clf.fit(X_train,Y_train)
# search.best_params_
# model = LogisticRegression(solver='liblinear',multi_class='ovr',penalty='l2',max_iter=300,class_weight='balanced')
# model.fit(X_train, Y_train)