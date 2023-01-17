import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import os
# List files available

print(os.listdir("../input/"))
filepath = "../input/creditcard.csv"

df = pd.read_csv(filepath, engine="python")
df.shape
df.head()
plt.figure(figsize=(15,7))

sns.distplot(df[df["Class"]==1]["Amount"])

plt.title("Distribution of fraudulent payment amount")

plt.legend()

plt.show()
def time_decompo(x):

    nb_min, nb_secs = x//60, x%60

    nb_hour, nb_mins = nb_min//60, nb_min%60

    nb_day, nb_hours = nb_hour//24, nb_hour%24

    return nb_day, nb_hours, nb_mins, nb_secs
df.apply(lambda x: [1, 2], axis=1)

df["Day"] = df["Time"].apply(lambda x: time_decompo(x)[0])

df["Hour"] = df["Time"].apply(lambda x: time_decompo(x)[1])

df["Min"] = df["Time"].apply(lambda x: time_decompo(x)[2])

df["Sec"] = df["Time"].apply(lambda x: time_decompo(x)[3])
plt.figure(figsize=(15,7))

sns.scatterplot(x=df[df["Class"]==1]["Hour"], y=df[df["Class"]==1]["Amount"])

plt.title("Amount of credit card payment depending on Hour")

plt.legend()

plt.show()
plt.figure(figsize=(15,7))

sns.distplot(df["Hour"])

plt.title("Distribution of payment hour")

plt.show()
plt.figure(figsize=(15,7))

plt.subplots_adjust(hspace=0.4)



sns.distplot(df[df["Class"]==1]["Hour"], label="fraudulent", bins=20)

sns.distplot(df[df["Class"]==0]["Hour"], label="non fraudulent", bins=20)

plt.title("Distribution of fraudulent payment hour")



plt.legend()

plt.show()
def night_time(x):

    if x < 7 or x > 0:

        return 1

    else:

        return 0
df["NightTime"] = df["Hour"].apply(lambda x: night_time(x)) 
df.shape
column_drop = ["Day", "Min", "Sec"]

df = df.drop(column_drop, axis=1)
y = df["Class"]

X = df.drop("Class", axis=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X=X)
X.shape
X_centered = X - X.mean()
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_centered)
pca.explained_variance_ratio_.sum()
plt.figure()

plt.scatter(X_pca[y==0][:,0], X_pca[y==0][:,1], label="non fraudulent")

plt.scatter(X_pca[y==1][:,0], X_pca[y==1][:,1], label="fraudulent")

plt.legend(loc=0)

plt.title('PCA projection n_dym = 2')

plt.show()
# Let's create a 3d-plot

%matplotlib inline 

#to print inlin mode rather than notebook mode

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X_pca[y==0][:,0], X_pca[y==0][:,1], X_pca[y==0][:,2], label="non fraudulent")

ax.scatter(X_pca[y==1][:,0], X_pca[y==1][:,1], X_pca[y==1][:,2], label="fraudulent") 



plt.legend(loc=0)

plt.title('PCA projection n_dym = 3')

plt.show()
y.sum(), y.count()
X_sample = (pd.DataFrame(df)).sample(n=1000)
y_sample = y[X_sample.index]
from sklearn.manifold import TSNE

tsne = TSNE(n_components=3)

X_tsne = tsne.fit_transform(X_sample)
X_tsne.shape
# Then we plot the results of t-SNE

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X_tsne[y_sample==0][:,0], X_tsne[y_sample==0][:,1], X_tsne[y_sample==0][:,2], label="non fraudulent")

ax.scatter(X_tsne[y_sample==1][:,0], X_tsne[y_sample==1][:,1], X_tsne[y_sample==1][:,2], label="fraudulent") 



plt.legend(loc=0)

plt.title('t-SNE projection')

plt.show()
X.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"Ratio of Class 1 in y: {round((y_train.sum()/len(y_train))*100, 2)} %")
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy='auto', return_indices=False, random_state=0, replacement=False, ratio=0.01)
X_undersamp, y_undersamp = rus.fit_resample(X_train, y_train)

X_undersamp.shape, y_undersamp.shape
print(f"Ratio of Class 1 in y_undersamp: {round((y_undersamp.sum()/len(y_undersamp))*100, 2)} %")
from imblearn import over_sampling

smote = over_sampling.SMOTE(k_neighbors=5, ratio=0.01, random_state=0)
X_oversamp, y_oversamp = smote.fit_resample(X_train, y_train)

X_oversamp.shape, y_oversamp.shape
print(f"Ratio of Class 1 in y_oversamp: {round((y_oversamp.sum()/len(y_oversamp))*100, 2)} %")
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_oversamp, y_oversamp)

y_pred_lr = lr.predict_proba(X_test)[:,1]
print(classification_report(y_test,np.round(y_pred_lr)))
import xgboost as xgb

xg = xgb.XGBClassifier()

xg.fit(X_oversamp, y_oversamp)

y_pred_xg = xg.predict_proba(X_test)[:,1]
print(classification_report(y_test, np.round(y_pred_xg)))
import lightgbm as lgb

lgbm = lgb.LGBMClassifier()

lgbm.fit(X_oversamp, y_oversamp)

y_pred_lgbm = lgbm.predict_proba(X_test)[:,1]
print(classification_report(y_test, np.round(y_pred_lgbm)))
from sklearn.metrics import precision_recall_curve, auc

precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_test, y_pred_lr)

precision_xg, recall_xg, thresholds_xg = precision_recall_curve(y_test, y_pred_xg)

precision_lgbm, recall_lgbm, thresholds_lgbm = precision_recall_curve(y_test, y_pred_lgbm)



print(f"Logistic Regression AUPRC: {auc(recall_lr, precision_lr)}")

print(f"XG Boost AUPRC: {auc(recall_xg, precision_xg)}")

print(f"LGBM AUPRC: {auc(recall_lgbm, precision_lgbm)}")
plt.figure(figsize=(15,7))

plt.plot(recall_lr, precision_lr, label="Logistic regression", linewidth=2)

plt.plot(recall_xg, precision_xg, label="XG Boost", linewidth=2)

plt.plot(recall_lgbm, precision_lgbm, label="LGBM", linewidth=2)

plt.title("Precision recall curve")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.legend()

plt.show()
X_train = X[y==0][:5000]
X_train.shape
X_test_outliers = X[y==1]

X_test_inliers = X[y==0][1000:5000]

X_test = np.concatenate([X_test_outliers, X_test_inliers])
X_test.shape
y_test = np.zeros(len(X_test_outliers)+len(X_test_inliers))

y_test[:len(X_test_outliers)] = 1
print(f"Contanination ratio: {(y.sum()/len(y))*100}")
from sklearn.neighbors import LocalOutlierFactor

LOF = LocalOutlierFactor(n_neighbors=20, contamination=0.0017, novelty=True) 

#depending on novelty you fit your model differently, 

#        if novelty = False you can do fit_predict(X)

#        if novelty = True we need to prepare our X_tain / X_test as done above

LOF.fit(X_train)

y_predLOF = LOF.predict(X_test)
y_predLOF[y_predLOF == 1] = 0

y_predLOF[y_predLOF == -1] = 1
n_errors = (y_predLOF != y_test).sum()

X_scores = LOF.negative_outlier_factor_

n_errors, LOF.offset_
y_test.sum(), len(y_test)
print(classification_report(y_test, y_predLOF))
from sklearn.ensemble import IsolationForest
IF = IsolationForest(n_estimators=100, contamination=0.0018, behaviour='new', random_state=0)

IF.fit(X_train)

y_predIF = IF.predict(X_test)
y_predIF[y_predIF == 1] = 0

y_predIF[y_predIF == -1] = 1
print(classification_report(y_test, y_predIF))