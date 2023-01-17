import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


df=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
print(df.info())
features = df.drop(columns='quality').columns
#print(df.describe())
plt.figure(figsize=(20,14))
sns.boxplot(data=df)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(df.quality, bins=20, kde=False)
plt.ylabel('The number of wines')
plt.xlabel('Wine quality')
plt.title('The number of wines per quality')
plt.show()
plt.figure(figsize=(12,8))
corr_mat = df.corr(method='pearson')
sns.heatmap(corr_mat, annot=True) 
plt.xticks(rotation=30)
plt.show()
# Split-out validation df
array = df.values
X = array[:,0:11]
y = array[:,11]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

models = []
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto', probability=True))) 
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_log_loss')
    results.append(cv_results)
    names.append(name)

plt.figure(figsize=(12,10))
plt.boxplot(results, labels=names)
plt.title('Performance comparison of different algorithms')
plt.show()
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('SVM', SVC(gamma='auto', probability=True))) 
results = []
names = []
for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict_proba(X_validation)
    score=log_loss(Y_validation, predictions)
    results.append(cv_results)
    names.append(name)
    print('logloss score of {} on unseen data is: '.format(name) + str(score))

model=LogisticRegression(solver='liblinear', multi_class='ovr') ### Note: default solver is 'lbfgs' which only supports L2 (Ridge);  solver 'liblinear' supports both 'L1' and 'L2' ; Usually Ridge is better than Lasso, Lasso is good for feature reduction
parameters = {'C': np.logspace(-5, 8, 15, 10, 12)}

# Create grid search using 5-fold cross validation
lr_cv = GridSearchCV(model, parameters, cv=5, verbose=0)
best_model = lr_cv.fit(X_train, Y_train)

# View best parameters
print('Best C:', best_model.best_estimator_.get_params()['C'])

# Evaluate predictions
predictions = lr_cv.predict_proba(X_validation)
print('logloss score on unseen data is: ' + str(log_loss(Y_validation, predictions)))


xgb = XGBClassifier(max_depth=35, random_state=42, n_estimators=1500, learning_rate=0.005, booster='gbtree', objective='multi:softprob', min_child_weight=0.1, n_jobs=10)
xgb.fit(X_train, Y_train)
y_pred=xgb.predict_proba(X_validation)

print("logloss score XGBoost: {}".format(log_loss(Y_validation, y_pred)))
plt.figure(figsize=(12,10))
feat_imp = pd.Series(xgb.feature_importances_, index=features).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances XGBoost')
plt.show()