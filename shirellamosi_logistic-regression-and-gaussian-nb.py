import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import scikitplot as skplt

import seaborn as sns



from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler



from sklearn.decomposition import PCA

from sklearn.feature_selection import RFE



from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report



from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.naive_bayes import GaussianNB



%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv',index_col=0)

print('Data Size: ', data.shape,'\n')



data.head()
pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

data.describe()
print(data.target.unique())

data.target.value_counts(normalize=True)
data.isna().sum().sum()
corrmat = data.corr()  

  

f, ax = plt.subplots(figsize =(20, 20)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths =0.5)
((corrmat>=0.5).sum() !=1 ).sum()
train, test = train_test_split(data, test_size = 0.2,

                        stratify = data['target'])



train.shape, test.shape
train.target.value_counts(normalize=True)
test.target.value_counts(normalize=True)
X_train = train.drop(columns=['target'])

y_train = train.target



X_test = test.drop(columns=['target'])

y_test = test.target



X_train.shape, y_train.shape, X_test.shape, y_test.shape
scaler = StandardScaler() #MaxAbsScaler() # MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
model1 = LogisticRegressionCV(cv = 5,

                            solver='lbfgs', 

                            class_weight='balanced', 

                            n_jobs=4, 

                            random_state=77)

model1.fit(X_train_scaled,y_train)
print(pd.DataFrame(confusion_matrix(y_train, model1.predict(X_train_scaled), labels=model1.classes_)))
print(classification_report(y_train, model1.predict(X_train_scaled), labels=model1.classes_))
X_train_prob = model1.predict_proba(X_train_scaled)

X_train_prob = pd.DataFrame(X_train_prob)

X_train_prob.head()
print(f'Train AUC= {roc_auc_score(y_train, X_train_prob[1]):.4f}')
print(pd.DataFrame(confusion_matrix(y_test, model1.predict(X_test_scaled), labels=model1.classes_)))
print(classification_report(y_test, model1.predict(X_test_scaled), labels=model1.classes_))
X_test_prob = model1.predict_proba(X_test_scaled)

X_test_prob = pd.DataFrame(X_test_prob)

X_test_prob.head()
print(f'Test AUC= {roc_auc_score(y_test, X_test_prob[1]):.4f}')
pca = PCA(random_state=77)

pca.fit(X_train_scaled)

skplt.decomposition.plot_pca_2d_projection(pca, X_train_scaled, y_train) 

plt.show()

pca = PCA(n_components = 180)

X_train_pca = pca.fit_transform(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)
pd.DataFrame(X_train_pca).head()
model2 = LogisticRegressionCV(cv = 5,

                              solver='lbfgs', 

                              class_weight='balanced', 

                              n_jobs=4, 

                              random_state=77)

model2.fit(X_train_pca, y_train)
print(pd.DataFrame(confusion_matrix(y_train, model2.predict(X_train_pca), labels=model2.classes_)))
print(classification_report(y_train, model2.predict(X_train_pca), labels=model2.classes_))
print(pd.DataFrame(confusion_matrix(y_test, model2.predict(X_test_pca), labels=model2.classes_)))
print(classification_report(y_test, model2.predict(X_test_pca), labels=model2.classes_))
X_train_prob = pd.DataFrame(model2.predict_proba(X_train_pca))

X_test_prob = pd.DataFrame(model2.predict_proba(X_test_pca))



print(f'Train AUC= {roc_auc_score(y_train, X_train_prob[1]):.4f}')

print(f'Test AUC= {roc_auc_score(y_test, X_test_prob[1]):.4f}')
selector = RFE(model1)

selector = selector.fit(X_train_scaled, y_train)
X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)

X_train_scaled.head(3)
ranking = list(zip(X_train_scaled.columns, selector.ranking_))



mask = [x[0] for x in ranking if x[1]>1] #drop features that weren't ranked as 1 (1=highest feature importance)

print('Number of unimportant features: ', len(mask),'\n')

print(mask)
X_train_rfe = X_train_scaled.drop(columns=mask)

X_train_rfe.columns
X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)

X_test_rfe = X_test_scaled.drop(columns=mask)

len(X_test_rfe.columns)
model3 = LogisticRegressionCV(cv = 5,

                              class_weight= 'balanced', 

                              random_state=55, 

                              n_jobs=4)

model3.fit(X_train_rfe, y_train)
X_train_prob = pd.DataFrame(model3.predict_proba(X_train_rfe))

X_test_prob = pd.DataFrame(model3.predict_proba(X_test_rfe))



print(f'Train AUC= {roc_auc_score(y_train, X_train_prob[1]):.4f}')

print(f'Test AUC= {roc_auc_score(y_test, X_test_prob[1]):.4f}')
skplt.metrics.plot_roc(y_test, X_test_prob)

plt.show()
selector2 = RFE(model2)

selector2 = selector2.fit(X_train_pca, y_train)
X_train_pca = pd.DataFrame(X_train_pca)

X_train_pca.head(2)
ranking = list(zip(X_train_pca.columns, selector2.ranking_))



mask = [x[0] for x in ranking if x[1]>1] #drop features that weren't ranked as 1 (1=highest feature importance)

print('Number of unimportant features: ', len(mask),'\n')

print(mask)
X_train_pca_rfe = X_train_pca.drop(columns=mask)

X_train_pca_rfe.columns
X_test_pca = pd.DataFrame(X_test_pca)

X_test_pca_rfe = X_test_pca.drop(columns=mask)

len(X_test_pca_rfe.columns)
model4 = LogisticRegressionCV(cv = 5,

                              class_weight= 'balanced', 

                              random_state=55, 

                              n_jobs=4)

model4.fit(X_train_pca_rfe, y_train)
X_train_prob = pd.DataFrame(model4.predict_proba(X_train_pca_rfe))

X_test_prob = pd.DataFrame(model4.predict_proba(X_test_pca_rfe))



print(f'Train AUC= {roc_auc_score(y_train, X_train_prob[1]):.4f}')

print(f'Test AUC= {roc_auc_score(y_test, X_test_prob[1]):.4f}')
model5 = GaussianNB()

params = {}

skf = StratifiedKFold(n_splits=10)



model5_cv = GridSearchCV(model5, params, cv=skf, scoring='roc_auc', n_jobs=4)

model5_cv.fit(X_train_pca_rfe, y_train)
X_train_prob = pd.DataFrame(model5_cv.predict_proba(X_train_pca_rfe))

X_test_prob = pd.DataFrame(model5_cv.predict_proba(X_test_pca_rfe))



print(f'Train AUC= {roc_auc_score(y_train, X_train_prob[1]):.4f}')

print(f'Test AUC= {roc_auc_score(y_test, X_test_prob[1]):.4f}')
model5_cv.fit(X_train_pca, y_train)

X_train_prob = pd.DataFrame(model5_cv.predict_proba(X_train_pca))

X_test_prob = pd.DataFrame(model5_cv.predict_proba(X_test_pca))



print(f'Train AUC= {roc_auc_score(y_train, X_train_prob[1]):.4f}')

print(f'Test AUC= {roc_auc_score(y_test, X_test_prob[1]):.4f}')
model6 = GaussianNB()

params = {}



model6_cv = GridSearchCV(model6, params, cv=skf, scoring='roc_auc', n_jobs=4)

model6_cv.fit(X_train_scaled, y_train)
X_train_prob = pd.DataFrame(model6_cv.predict_proba(X_train_scaled))

X_test_prob = pd.DataFrame(model6_cv.predict_proba(X_test_scaled))



print(f'Train AUC= {roc_auc_score(y_train, X_train_prob[1]):.4f}')

print(f'Test AUC= {roc_auc_score(y_test, X_test_prob[1]):.4f}')
skplt.metrics.plot_roc(y_test, X_test_prob)

plt.show()