import pandas as pd

test = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/train.csv")
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.utils import resample

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
train.head(5)
train.isnull().sum()
train.isna().sum()
train = train.drop(columns=['ID_code'])
#corr = train2.corr()

#corr.style.background_gradient(cmap='coolwarm')
#train2.boxplot(by='target', figsize=(60,55))

#plt.show()
def get_redundant_pairs(df):

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop





def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]
Q1 = train2.quantile(0.25)

Q3 = train2.quantile(0.75)

IQR = Q3 - Q1

#print(IQR)



#train2 = train2[~((train2 < (Q1-1.5 * IQR)) | (train2 > (Q3+1.5 * IQR))).any(axis=1)]

#train2.shape
top_corr = get_top_abs_correlations(train, 180)
corr_features = [indx[1] for indx in top_corr.index if indx[0]=='target']

corr_features.insert(0, 'target')



rel_features = [indx[0] for indx in top_corr.index if indx[0]!='target'] + [indx[1] for indx in top_corr.index if indx[0]!='target']

corr_features2 = [x for x in corr_features if x not in rel_features]



corr_features = corr_features[:100] # change number of the features

train2 = train[corr_features]



df_majority = train2[train2.target==0]

df_minority = train2[train2.target==1]



df_majority_downsampled = resample(df_majority,

                                   replace=False,

                                   n_samples=20098,

                                   random_state=123)



'''

df_minority_upsampled = resample(df_minority, 

                                 replace=True,

                                 n_samples=23000,

                                 random_state=123)

'''



df_downsampled = pd.concat([df_majority_downsampled, df_minority])

#df_upsampled = pd.concat([df_majority, df_minority_upsampled])



df_sampled = pd.concat([df_majority_downsampled, df_minority])

df_sampled.target.value_counts()





scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(df_sampled.values)

X = train_scaled[:, 1:]

y = train_scaled[:, 0]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)



target_names = ['Negative', 'Positive']
knn = KNeighborsClassifier(5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)



print("KNN accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_knn)), "\n")

print("KNN confusion matrix:\n", confusion_matrix(y_test, y_pred_knn, labels=[0,1]), "\n")

print(classification_report(y_test, y_pred_knn, target_names=target_names))
logreg = LogisticRegression(random_state=5, solver="sag")

logreg.fit(X_train, y_train)

y_pred_lr = logreg.predict(X_test)



print("LogReg accuracy score: {:.2f}".format(logreg.score(X_test, y_test)), "\n")

print("LogReg confusion matrix:\n", confusion_matrix(y_test, y_pred_lr, labels=[0,1]), "\n")

print(classification_report(y_test, y_pred_lr, target_names=target_names))
svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train, y_train)

y_pred_svm = svclassifier.predict(X_test)



print("SVM accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_svm)), "\n")

print("SVM confusion matrix:\n", confusion_matrix(y_test, y_pred_svm, labels=[0,1]), "\n")

print(classification_report(y_test, y_pred_svm, target_names=target_names))
nbclassifier = BernoulliNB()

nbclassifier.fit(X_train, y_train)

y_pred_nb = nbclassifier.predict(X_test)



print("Naive Bayes accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_nb)), "\n")

print("Naive Bayes confusion matrix:\n", confusion_matrix(y_test, y_pred_nb, labels=[0,1]), "\n")

print(classification_report(y_test, y_pred_nb, target_names=target_names))
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)



print("Decision Tree accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_dt)), "\n")

print("Decision Tree confusion matrix:\n", confusion_matrix(y_test, y_pred_dt, labels=[0,1]), "\n")

print(classification_report(y_test, y_pred_dt, target_names=target_names))
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)



print("Random Forest accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_rf)), "\n")

print("Random Forest confusion matrix:\n", confusion_matrix(y_test, y_pred_rf, labels=[0,1]), "\n")

print(classification_report(y_test, y_pred_rf, target_names=target_names))
params = {"n_estimators":1000, "max_depth":6, "seed":123, "tree_method":"gpu_hist", "predictor":"gpu_predictor", "n_gpus":1}

xg_cl = xgb.XGBClassifier(**params)

xg_cl.fit(X_train, y_train)

y_pred_xgb = xg_cl.predict(X_test)



accuracy = float(np.sum(y_pred_xgb==y_test))/y_test.shape[0]

print("XGBoost accuracy score: {:.2f}".format(accuracy), "\n")

print("XGBoost confusion matrix:\n", confusion_matrix(y_test, y_pred_xgb, labels=[0,1]), "\n")

print(classification_report(y_test, y_pred_xgb, target_names=target_names))
print("KNN accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_knn)))

print("LogReg accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_lr)))

print("SVM accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_svm)))

print("Naive Bayes accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_nb)))

print("Decision Tree accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_dt)))

print("Random Forest accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_rf)))

print("XGBoost accuracy score: {:.2f}".format(accuracy))
dataset_dmatrix = xgb.DMatrix(data=X, label=y)

params = {"objective":"binary:logistic", "max_depth":6}



cv_results = xgb.cv(dtrain=dataset_dmatrix, params=params, num_boost_round=10, nfold=3, metrics="rmse", as_pandas=True, seed=123)

print(cv_results, "\n")

print(1-cv_results["test-rmse-mean"].tail(1))
cv_results = xgb.cv(dtrain=dataset_dmatrix, params=params, num_boost_round=50, nfold=3, metrics="auc", as_pandas=True, seed=123)

print(cv_results["test-auc-mean"].tail(1))
params = {"n_estimators":1000, "max_depth":6, "seed":123, "tree_method":"gpu_hist", "predictor":"gpu_predictor", "n_gpus":1}

xg_cl2 = xgb.XGBClassifier(**params)

xg_cl2.fit(train.drop(columns='target'), train['target'])



weight_features = xg_cl2.get_booster().get_score(importance_type='weight')

weight_features = [it[0] for it in sorted(weight_features.items(), key=lambda kv:(kv[1]), reverse=True)]

weight_features.insert(0, 'target')
weight_features = weight_features[:70]

train3 = train[weight_features]



df_majority = train3[train2.target==0]

df_minority = train3[train2.target==1]



df_majority_downsampled = resample(df_majority,

                                   replace=False,

                                   n_samples=20098, #30000

                                   random_state=123)



df_downsampled = pd.concat([df_majority_downsampled, df_minority])
train3 = train[corr_features]



df_majority = train3[train2.target==0]

df_minority = train3[train2.target==1]



df_majority_downsampled = resample(df_majority,

                                   replace=False,

                                   n_samples=20098, #30000

                                   random_state=123)



df_downsampled = pd.concat([df_majority_downsampled, df_minority])





scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(df_downsampled.values)

X_train = train_scaled[:, 1:]

y_train = train_scaled[:, 0]



corr_features2 = [it for it in corr_features if it!='target']



scaler_test = MinMaxScaler()

test2 = test.drop(columns='ID_code')

test2 = test2[corr_features2]

test_scaled = scaler_test.fit_transform(test2.values)



X_test = test_scaled



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)



target_names = ['Negative', 'Positive']
from sklearn.model_selection import validation_curve



param_range = np.arange(20, 30, 2)



train_scores, test_scores = validation_curve(

    RandomForestClassifier(),

    X=X_train,

    y=y_train,

    param_name='n_estimators',

    param_range=param_range,

    cv=3,

    scoring="accuracy",

    n_jobs=-1)



train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)



test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)



plt.plot(param_range, train_mean, label="Training score", color="black")

plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")



plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")

plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")



plt.title("Validation Curve With Random Forest")

plt.xlabel("Number Of Trees")

plt.ylabel("Accuracy Score")

plt.tight_layout()

plt.legend(loc="best")

#plt.show()
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)



#print("Random Forest accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_rf)), "\n")

#print("Random Forest confusion matrix:\n", confusion_matrix(y_test, y_pred_rf, labels=[0,1]), "\n")

#print(classification_report(y_test, y_pred_rf, target_names=target_names))
params = {"n_estimators":1000, "max_depth":6, "seed":123, "tree_method":"gpu_hist", "predictor":"gpu_predictor", "n_gpus":1}

xg_cl = xgb.XGBClassifier(**params)

xg_cl.fit(X_train, y_train)

y_pred_xgb = xg_cl.predict(X_test)



#accuracy = float(np.sum(y_pred_xgb==y_test))/y_test.shape[0]

#print("XGBoost accuracy score: {:.2f}".format(accuracy), "\n")

#print("XGBoost confusion matrix:\n", confusion_matrix(y_test, y_pred_xgb, labels=[0,1]), "\n")

#print(classification_report(y_test, y_pred_xgb, target_names=target_names))
logreg = LogisticRegression(random_state=5, solver="sag")

logreg.fit(X_train, y_train)

y_pred_lr = logreg.predict(X_test)



#print("LogReg accuracy score: {:.2f}".format(logreg.score(X_test, y_test)), "\n")

#print("LogReg confusion matrix:\n", confusion_matrix(y_test, y_pred_lr, labels=[0,1]), "\n")

#print(classification_report(y_test, y_pred_lr, target_names=target_names))
knn = KNeighborsClassifier(5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)



#print("KNN accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_knn)), "\n")

#print("KNN confusion matrix:\n", confusion_matrix(y_test, y_pred_knn, labels=[0,1]), "\n")

#print(classification_report(y_test, y_pred_knn, target_names=target_names))
svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train, y_train)

y_pred_svm = svclassifier.predict(X_test)



#print("SVM accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_svm)), "\n")

#print("SVM confusion matrix:\n", confusion_matrix(y_test, y_pred_svm, labels=[0,1]), "\n")

#print(classification_report(y_test, y_pred_svm, target_names=target_names))
nbclassifier = BernoulliNB()

nbclassifier.fit(X_train, y_train)

y_pred_nb = nbclassifier.predict(X_test)



#print("Naive Bayes accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_nb)), "\n")

#print("Naive Bayes confusion matrix:\n", confusion_matrix(y_test, y_pred_nb, labels=[0,1]), "\n")

#print(classification_report(y_test, y_pred_nb, target_names=target_names))
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)



#print("Decision Tree accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_dt)), "\n")

#print("Decision Tree confusion matrix:\n", confusion_matrix(y_test, y_pred_dt, labels=[0,1]), "\n")

#print(classification_report(y_test, y_pred_dt, target_names=target_names))
submission_rf = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_rf

})

submission_xgb = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_xgb

})

submission_lr = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_lr

})

submission_knn = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_knn

})

submission_svm = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_svm

})

submission_nb = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_nb

})

submission_dt = pd.DataFrame({

        "ID_code": test["ID_code"],

        "target": y_pred_dt

})

submission_rf.to_csv('submission_rf.csv', index=False)

submission_xgb.to_csv('submission_xgb.csv', index=False)

submission_lr.to_csv('submission_lr.csv', index=False)

submission_knn.to_csv('submission_knn.csv', index=False)

submission_svm.to_csv('submission_svm.csv', index=False)

submission_nb.to_csv('submission_nb.csv', index=False)

submission_dt.to_csv('submission_dt.csv', index=False)