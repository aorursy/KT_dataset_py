import numpy as np # linear algebra

import pandas as pd 

from sklearn import preprocessing

import matplotlib.pyplot as plt 

import seaborn as sns

import missingno

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import model_selection, metrics, preprocessing



#Machine Learning

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier



#preprocessing

from sklearn.impute import KNNImputer
df = pd.read_csv('../input/diabetes-dataset/diabetes2.csv')

df.dtypes
df.head()
df.shape
df.isnull().sum()
df.describe()
missingno.matrix(df) #nice way of visualizing missing values
plt.figure()

ax = sns.distplot(df['Pregnancies'][df.Outcome == 1], color ="darkturquoise", rug = True)

sns.distplot(df['Pregnancies'][df.Outcome == 0], color ="lightcoral",rug = True)

plt.legend(['Diabetes', 'No Diabetes'])
df['Pregnancies'].value_counts()

#df['Pregnancies'].unique()
sns.boxplot(x = df['Pregnancies'])
plt.figure()

ax = sns.distplot(df['Glucose'][df.Outcome == 1], color ="darkturquoise", rug = True)

sns.distplot(df['Glucose'][df.Outcome == 0], color ="lightcoral", rug = True)

plt.legend(['Diabetes', 'No Diabetes'])
min(df['Glucose']) 
df[df['Glucose'] == 0]
df[df['Glucose'].lt(60)]
df['Glucose'].value_counts() #not significant
sns.boxplot(x = df['Glucose'])
plt.figure()

ax = sns.distplot(df['BloodPressure'][df.Outcome == 1], color ="darkturquoise", rug=True)

sns.distplot(df['BloodPressure'][df.Outcome == 0], color ="lightcoral", rug=True)

plt.legend(['Diabetes', 'No Diabetes'])
min(df['BloodPressure']) #Again, seems absurd
print(df.loc[df['BloodPressure'] == 0].shape[0])

print(df.loc[df['BloodPressure'] == 0].shape[0]/df.shape[0])
df[df['BloodPressure'].lt(40)]
df[df['BloodPressure'].lt(40)].shape
df[df['BloodPressure'].gt(120)]
sns.boxplot(x = df['BloodPressure'])
plt.figure()

ax = sns.distplot(df['SkinThickness'][df.Outcome == 1], color ="darkturquoise", rug=True)

sns.distplot(df['SkinThickness'][df.Outcome == 0], color ="lightcoral", rug=True)

plt.legend(['Diabetes', 'No Diabetes'])
sns.boxplot(x = df['SkinThickness'])
df[df['SkinThickness'] == 0].shape
df[df['SkinThickness'].lt(2)]
plt.figure()

ax = sns.distplot(df['Insulin'][df.Outcome == 1], color ="darkturquoise", rug=True)

sns.distplot(df['Insulin'][df.Outcome == 0], color ="lightcoral", rug=True)

plt.legend(['Diabetes', 'No Diabetes'])
sns.boxplot(x = df['Insulin'])
df[df['Insulin'].lt(16)]
df[df['Insulin'] == 0]
df[df['Insulin'] == 0].shape
plt.figure()

ax = sns.distplot(df['BMI'][df.Outcome == 1], color ="darkturquoise", rug=True)

sns.distplot(df['BMI'][df.Outcome == 0], color ="lightcoral", rug=True)

plt.legend(['Diabetes', 'No Diabetes'])
sns.boxplot(x = df['BMI'])
df[df.BMI == 0]
df[df.BMI == 0].shape
plt.figure()

ax = sns.distplot(df['DiabetesPedigreeFunction'][df.Outcome == 1], color ="darkturquoise", rug=True)

sns.distplot(df['DiabetesPedigreeFunction'][df.Outcome == 0], color ="lightcoral", rug=True)

plt.legend(['Diabetes', 'No Diabetes'])
sns.boxplot(x = df['DiabetesPedigreeFunction'])
plt.figure()

ax = sns.distplot(df['Age'][df.Outcome == 1], color ="darkturquoise", rug=True)

sns.distplot(df['Age'][df.Outcome == 0], color ="lightcoral", rug=True)

sns.distplot(df['Age'], color ="green", rug=True)

plt.legend(['Diabetes', 'No Diabetes', 'all'])
sns.boxplot(x = df['Age'])
df_with_na = df.copy(deep = True)

df_with_na['Insulin'] = df['Insulin'].map(lambda i: np.nan if i==0 else i)

df_with_na['SkinThickness'] = df['SkinThickness'].map(lambda i: np.nan if i==0 else i)

df_with_na['BloodPressure'] = df['BloodPressure'].map(lambda i: np.nan if i==0 else i)

df_with_na['BMI'] = df['BMI'].map(lambda i: np.nan if i==0 else i)

df_with_na['Glucose'] = df['Glucose'].map(lambda i: np.nan if i==0 else i)



missingno.matrix(df_with_na) #nice way of visualizing missing values
sns.heatmap(df.corr(), annot = True)
sns.heatmap(df_with_na.corr(), annot = True)
sns.pairplot(df, hue='Outcome')
clf = RandomForestClassifier()

clf.fit(df.drop('Outcome', axis = 1), df['Outcome'])

plt.figure()

importance = clf.feature_importances_

print(df.drop('Outcome', axis=1).columns)

print(clf.feature_importances_)

importance = pd.DataFrame(importance, index=df.drop('Outcome', axis=1).columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))
df_no_BMI0 = df[df.BMI != 0]



clf = RandomForestClassifier()

clf.fit(df_no_BMI0.drop('Outcome', axis = 1), df_no_BMI0['Outcome'])

plt.figure()

importance = clf.feature_importances_

print(df_no_BMI0.drop('Outcome', axis=1).columns)

print(clf.feature_importances_)

importance = pd.DataFrame(importance, index=df_no_BMI0.drop('Outcome', axis=1).columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))

df_no_BP0 = df[df.BloodPressure != 0]



clf = RandomForestClassifier()

clf.fit(df_no_BP0.drop('Outcome', axis = 1), df_no_BP0['Outcome'])

plt.figure()

importance = clf.feature_importances_

print(df_no_BP0.drop('Outcome', axis=1).columns)

print(clf.feature_importances_)

importance = pd.DataFrame(importance, index=df_no_BP0.drop('Outcome', axis=1).columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))
df_no_IN0 = df[df.Insulin != 0]



clf = RandomForestClassifier()

clf.fit(df_no_IN0.drop('Outcome', axis = 1), df_no_IN0['Outcome'])

plt.figure()

importance = clf.feature_importances_

print(df_no_IN0.drop('Outcome', axis=1).columns)

print(clf.feature_importances_)

importance = pd.DataFrame(importance, index=df_no_IN0.drop('Outcome', axis=1).columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))
df_no_SK0 = df[df.SkinThickness != 0]



clf = RandomForestClassifier()

clf.fit(df_no_SK0.drop('Outcome', axis = 1), df_no_SK0['Outcome'])

plt.figure()

importance = clf.feature_importances_

print(df_no_SK0.drop('Outcome', axis=1).columns)

print(clf.feature_importances_)

importance = pd.DataFrame(importance, index=df_no_SK0.drop('Outcome', axis=1).columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))
df_no_missing = df_with_na.dropna()

print(df_no_missing.shape)



clf = RandomForestClassifier()

clf.fit(df_no_missing.drop('Outcome', axis = 1), df_no_missing['Outcome'])

plt.figure()

importance = clf.feature_importances_

print(df_no_missing.drop('Outcome', axis=1).columns)

print(clf.feature_importances_)

importance = pd.DataFrame(importance, index=df_no_missing.drop('Outcome', axis=1).columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))
X, X_test, y, y_test = train_test_split(df.drop('Outcome', axis= 1), df['Outcome'], test_size=0.20, random_state=42)

X, X_test_with_na, y, _ = train_test_split(df_with_na.drop('Outcome', axis= 1), df_with_na['Outcome'], test_size=0.20, random_state=42)
df_train = pd.concat([X, y], axis = 1)

df_train_without_missing = df_train.dropna()



y_wo_missing_train = df_train_without_missing['Outcome']

X_wo_missing_train = df_train_without_missing.drop(columns = ['Outcome'])



scaler = preprocessing.StandardScaler().fit(X_wo_missing_train)



#DATASET 1

X_wo_missing_train = scaler.transform(X_wo_missing_train)

X_wo_missing_test = scaler.transform(X_test)

X_wo_missing_test_with_na = scaler.transform(X_test_with_na)



print(X.shape)

print(df_train_without_missing.shape)
imputer = KNNImputer(n_neighbors=5)

X_knn_imp = imputer.fit_transform(X)

df_knn_imp = pd.DataFrame(X_knn_imp, columns = df.drop('Outcome', axis = 1).columns) #DATASET 2



clf = RandomForestClassifier()

clf.fit(df_knn_imp, y)

plt.figure()

importance = clf.feature_importances_

print(df_knn_imp.columns)

print(clf.feature_importances_)

importance = pd.DataFrame(importance, index=df_knn_imp.columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))



scaler = preprocessing.StandardScaler().fit(df_knn_imp)

X_knn_imp_train = scaler.transform(df_knn_imp)

X_knn_imp_test = scaler.transform(X_test)
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



imp = IterativeImputer(max_iter = 10, random_state = 42)

imp.fit(X)

X_iter_imp = imp.transform(X)

X_iter_imp = pd.DataFrame(X_iter_imp, columns = X.columns)





clf = RandomForestClassifier()

clf.fit(X_iter_imp, y)

plt.figure()

importance = clf.feature_importances_

print(X_iter_imp.columns)

print(clf.feature_importances_)

importance = pd.DataFrame(importance, index=X_iter_imp.columns, columns=["Importance"])

importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))



scaler = preprocessing.StandardScaler().fit(X_iter_imp)

X_iter_imp_train = scaler.transform(X_iter_imp)

X_iter_imp_test = scaler.transform(X_test)
# 

def fit_ml_algo(algo, X_train, y_train, X_test, y_test, cv):

    model = algo.fit(X_train, y_train)

    test_prediction = model.predict(X_test)

    test_probs = model.predict_proba(X_test)[:,1]

    train_accuracy = model.score(X_train, y_train)*100

    test_accuracy = model.score(X_test, y_test)*100

    train_prediction = model_selection.cross_val_predict(algo, X_train, y_train, cv = 10, n_jobs = -1)

    acc_cv = metrics.accuracy_score(y_train, train_prediction)*100

    model_scores = model_selection.cross_val_score(LogisticRegression(), X_train, y_train, cv = 10, n_jobs = -1)



    print("Cross Validation accuracy: (%0.2f) %0.4f (+/- %0.4f)" % (acc_cv, model_scores.mean(), model_scores.std() * 2))

    print('Model Test Accuracy: %0.2f   Model Train Accuracy: %0.2f'%(test_accuracy, train_accuracy))

    print(metrics.classification_report(y_test, test_prediction))

    print("Confusion matrix")

    print(metrics.confusion_matrix(y_test,test_prediction))

    

    return train_prediction, test_prediction, test_probs

    

    

# calculate the fpr and tpr for all thresholds of the classification

def plot_roc_curve(y_test, preds):

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([-0.01, 1.01])

    plt.ylim([-0.01, 1.01])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

    

# Adding gridsearch report creating code

def report(results, n_top = 5):

    for i in range(1, n_top +1 ):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
c = [0.01, 0.1, 1, 5, 10]

param_grid = [{'C': c, 'penalty': ['l2'], 'solver': ['liblinear', 'newton-cg','saga','lbfgs']}, {'C': c, 'penalty': ['l1'], 'solver': ['liblinear', 'saga']}]



lg_grid = model_selection.GridSearchCV(LogisticRegression(), param_grid, n_jobs = -1)

lg_grid.fit(X_train_wo_msng, y_train_wo_msng)



report(lg_grid.cv_results_)
train_prediction_1, test_prediction_1, test_probs_1 = fit_ml_algo(LogisticRegression(C = 0.1, penalty = 'l1', solver = 'liblinear'), X_train_without_missing_tf, y_train_without_missing, X_test_tf, y_test, cv = 10)
plot_roc_curve(y_test, test_probs_1)
c = [0.01, 0.1, 1, 5, 10]

param_grid = [{'C': c, 'penalty': ['l2'], 'solver': ['liblinear', 'newton-cg','saga','lbfgs']}, {'C': c, 'penalty': ['l1'], 'solver': ['liblinear', 'saga']}]



lg_grid = model_selection.GridSearchCV(LogisticRegression(), param_grid, n_jobs = -1)

lg_grid.fit(X_knn_imp_train, y)



report(lg_grid.cv_results_)
train_prediction_2, test_prediction_2, test_probs_2 = fit_ml_algo(LogisticRegression(C = 0.1, penalty = 'l1', solver = 'liblinear'), X_knn_imp_train, y, X_knn_imp_test, y_test, cv = 10)
plot_roc_curve(y_test, test_probs_2)
c = [0.01, 0.1, 1, 5, 10]

param_grid = [{'C': c, 'penalty': ['l2'], 'solver': ['liblinear', 'newton-cg','saga','lbfgs']}, {'C': c, 'penalty': ['l1'], 'solver': ['liblinear', 'saga']}]



lg_grid = model_selection.GridSearchCV(LogisticRegression(), param_grid, n_jobs = -1)

lg_grid.fit(X_iter_imp_train, y)



report(lg_grid.cv_results_)
train_prediction_3, test_prediction_3, test_probs_3 = fit_ml_algo(LogisticRegression(C = 0.1, penalty = 'l1', solver = 'saga'), X_iter_imp_train, y, X_iter_imp_test, y_test, cv = 10)
plot_roc_curve(y_test, test_probs_3)
#gridsearch params

c = [0.01, 0.1, 1, 5, 10]

param_grid = [{'C': c, 'kernel': ['linear', 'poly','rbf','sigmoid']}]
svc_grid = model_selection.GridSearchCV(SVC(), param_grid, n_jobs = -1)

svc_grid.fit(X_train_wo_msng, y_train_wo_msng)

report(svc_grid.cv_results_)
train_prediction, test_prediction, test_probs = fit_ml_algo(SVC(C =5, kernel = 'linear', probability = True), X_train_wo_msng, y_train_wo_msng, X_test_tf, y_test, cv = 10)
plot_roc_curve(y_test, test_probs)
svc_grid = model_selection.GridSearchCV(SVC(), param_grid, n_jobs = -1)

svc_grid.fit(X_knn_imp_train, y)

report(svc_grid.cv_results_)
train_prediction, test_prediction, test_probs = fit_ml_algo(SVC(C = 0.01, kernel = 'linear', probability = True), X_knn_imp_train, y, X_knn_imp_test, y_test, cv = 10)
plot_roc_curve(y_test, test_probs)
svc_grid = model_selection.GridSearchCV(SVC(), param_grid, n_jobs = -1)

svc_grid.fit(X_iter_imp_train, y)

report(svc_grid.cv_results_)
train_prediction, test_prediction, test_probs = fit_ml_algo(SVC(C = 0.01, kernel = 'linear', probability = True), X_iter_imp_train, y, X_iter_imp_test, y_test, cv = 10)
plot_roc_curve(y_test, test_probs)
params = {'min_child_weight': [1, 5, 10], 'gamma': [0.5, 1, 1.5, 2, 5], 'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0], 'max_depth': [3, 4, 5]}

estimator = XGBClassifier(objective= 'binary:logistic', nthread=4, seed=42)

xg_grid = model_selection.GridSearchCV(estimator=estimator, param_grid=params, scoring = 'roc_auc', n_jobs = 10, cv = 10, verbose=True)
xg_grid.fit(X_wo_missing_train, y_wo_missing_train)
best_estimator = xg_grid.best_estimator_

report(xg_grid.cv_results_)
train_prediction, test_prediction, test_probs = fit_ml_algo(best_estimator, X_wo_missing_train, y_wo_missing_train, X_wo_missing_test_with_na, y_test, cv = 10)
plot_roc_curve(y_test, test_probs)
xg_grid.fit(X_knn_imp_train, y)
best_estimator = xg_grid.best_estimator_

report(xg_grid.cv_results_)
train_prediction, test_prediction, test_probs = fit_ml_algo(best_estimator, X_knn_imp_train, y, X_knn_imp_test, y_test, cv = 10)
xg_grid.fit(X_iter_imp_train, y)
best_estimator = xg_grid.best_estimator_

report(xg_grid.cv_results_)
train_prediction, test_prediction, test_probs = fit_ml_algo(best_estimator, X_iter_imp_train, y, X_iter_imp_test, y_test, cv = 10)