import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats as stats

import matplotlib.pyplot as plt

from random import sample

from sklearn.utils import resample

from imblearn import under_sampling,over_sampling

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.model_selection import train_test_split,learning_curve,StratifiedKFold

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.dummy import DummyClassifier

from statsmodels.stats import stattools

import statsmodels.graphics.tsaplots as smgt

from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_auc_score

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
def plot_categoric_feature(feature_name):

    dftemp = dfdata.groupby([feature_name,'churn'],as_index = False).agg({'customerid':'count','monthlycharges':'mean'})

    dftemp['feature_rate'] = dftemp.apply(lambda row: row['customerid'] / dftemp[dftemp[feature_name] == row[feature_name]]['customerid'].sum() ,axis = 1)

    dftemp['rate'] = dftemp['customerid'] / dftemp['customerid'].sum()

    fig,ax = plt.subplots(1,3,figsize=(20,3))

    sns.barplot(y=feature_name, x = 'feature_rate', color = 'darkorange',data = dftemp[dftemp['churn'] == 'Yes'],ax = ax[0])

    ax[0].set(title = 'churn rate in feature ' + feature_name, xlabel = '')

    sns.barplot(y=feature_name, x = 'rate', hue='churn',data = dftemp,ax = ax[1])

    ax[1].set(title = feature_name + ' rate in whole dataset',xlabel = '')

    sns.barplot(y=feature_name, x = 'monthlycharges', hue='churn',data = dftemp,ax = ax[2])

    ax[2].set(title = feature_name + ' monthlycharges',xlabel = '')

    plt.show()
def plot_learning_curve(estimator, X_train, y_train):

    kfold = StratifiedKFold(n_splits = 5)

    train_size,train_scores,test_scores = learning_curve(estimator,X_train,y_train,train_sizes = np.linspace(0.05,1,20),cv = kfold)

    train_scores_mean = np.mean(train_scores,axis=1)

    train_scores_std = np.std(train_scores,axis=1)

    test_scores_mean = np.mean(test_scores,axis=1)

    test_scores_std = np.std(test_scores,axis=1)



    sns.lineplot(x=train_size, y=train_scores_mean, c='r', label='train')

    plt.fill_between(x=train_size, y1=train_scores_mean+train_scores_std, y2=train_scores_mean-train_scores_std, alpha=0.1, color='r')

    sns.lineplot(x=train_size,y=test_scores_mean,c='b',label='test')

    plt.fill_between(x=train_size, y1=test_scores_mean+test_scores_std, y2=test_scores_mean-test_scores_std, alpha=0.1, color='b')

    plt.legend(loc='best')

    plt.title("Learning Curve")
def plot_confusion_matrix(estimator,y,y_pred):

    cm = confusion_matrix(y, y_pred, labels = [0,1] )

    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = ["No", "Yes"] , yticklabels = ["No", "Yes"],cbar = False)

    plt.title("Confusion Matrix")

    plt.ylabel('Actual')

    plt.xlabel('Prediction')
def plot_empty_confusion_matrix():

    plt.text(0.45, .6, "TN", size=100, horizontalalignment='right')

    plt.text(0.45, .1, "FN", size=100, horizontalalignment='right')

    plt.text(.95, .6, "FP", size=100, horizontalalignment='right')

    plt.text(.95, 0.1, "TP", size=100, horizontalalignment='right')

    plt.xticks([.25, .75], ["predicted negative", "predicted positive"], size=15)

    plt.yticks([.25, .75], ["positive class", "negative class"], size=15)

    plt.plot([.5, .5], [0, 1], '--', c='k')

    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)

    plt.ylim(0, 1)
def print_clf_result(estimator,X_train,y_train,X_test = None, y_test = None):

    y_pred = estimator.predict(X_train)

    acc = round(accuracy_score(y_train, y_pred),2)

    print("Train Accuracy : ", acc)

    if X_test is None:

        y_pred_test = estimator.predict(X_train)

        plt.figure(figsize=(18,5))

        plt.subplot(121)

        plot_learning_curve(estimator,X_train,y_train)

        plt.subplot(122)

        plot_confusion_matrix(estimator,y_train,y_pred)   

    else:

        test_acc = round(estimator.score(X_test,y_test),2)

        print("Classification Accuracy :", test_acc)

        y_pred_test = estimator.predict(X_test)

        plt.figure(figsize=(18,5))

        plt.subplot(121)

        plot_learning_curve(estimator,X_train,y_train)

        plt.subplot(122)

        plot_confusion_matrix(estimator,y_test,y_pred_test)

    plt.show()
def make_clf_data(n_points, n_centers = 2, random_state = 42):

    n_features=2

    rnd_gen = np.random.RandomState(random_state)

    feature_names = ['feature' + str(x+1) for x in range(n_features)]

    X = pd.DataFrame(columns = feature_names)

    for center in range(n_centers):

        X = X.append(pd.DataFrame(rnd_gen.normal(loc=5*center, size=(n_points, n_features)), columns=feature_names),ignore_index=True)

    X['target'] = (X['feature1'] > 2)

    n_changes = int(n_points * 0.2)

    list_true = sample(X[X['target'] == True].index.to_list(), n_changes)

    X.loc[list_true,'target'] = False

    n_changes = int(n_points * 0.1)

    list_false = sample(X[X['target'] == False].index.to_list(),n_changes)

    X.loc[list_false,'target'] = True

    X = X.sample(frac=1).reset_index(drop=True)

    return X
def plot_decision_boundary(estimator,X,title = None):

    xx = np.linspace(-3, 9, 100)

    yy = np.linspace(-3, 9, 100)

    X1, X2 = np.meshgrid(xx, yy)

    X_grid = np.c_[X1.ravel(), X2.ravel()]

    decision_values = estimator.decision_function(X_grid)

    sns.scatterplot(x='feature1', y='feature2', hue='target', data = X)

    plt.contour(X1, X2, decision_values.reshape(X1.shape), colors="black",levels = 0)

    if title is not None:

        plt.title(title)

dfdata = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

dfdata.columns = dfdata.columns.str.lower()

dfdata.shape
dfdata.describe()
dfdata.duplicated().sum()
dfdata.isnull().sum()
dfdata.info()
dfdata.nunique()
dfdata['seniorcitizen'] = dfdata['seniorcitizen'].map({0:'No',1:'Yes'})
dfdata['totalcharges_new'] = pd.to_numeric(dfdata['totalcharges'],errors = 'coerce')

dfdata[dfdata['totalcharges_new'].isnull() == True][['totalcharges','totalcharges_new']]
dfdata['totalcharges_new'].fillna(0,inplace = True)

dfdata['totalcharges'] = dfdata['totalcharges_new']

dfdata.drop('totalcharges_new',axis=1,inplace=True)
dfdata.head()
print(round(100 * dfdata['churn'].value_counts() / dfdata.shape[0],0))

plt.figure(figsize=(10,2))

sns.countplot(y='churn',data = dfdata)

plt.show()
numeric_features = dfdata.columns[dfdata.dtypes != 'object'].values.tolist()

categoric_features = dfdata.columns[dfdata.dtypes == 'object'].values.tolist()

categoric_features.remove('customerid')

categoric_features.remove('churn')

print("Categoric features : ",categoric_features)

print("Numeric features : ",numeric_features)
fig,ax = plt.subplots(1,3,figsize=(18,5))



sns.distplot(dfdata[dfdata['churn'] == 'No']['tenure'],label = 'No',ax=ax[0])

sns.distplot(dfdata[dfdata['churn'] == 'Yes']['tenure'],label = 'Yes',ax=ax[0])

ax[0].set_title('Tenure')

ax[0].legend()



sns.distplot(dfdata[dfdata['churn'] == 'No']['monthlycharges'],label = 'No',ax=ax[1])

sns.distplot(dfdata[dfdata['churn'] == 'Yes']['monthlycharges'],label = 'Yes',ax=ax[1])

ax[1].set_title('Monthly Charges')

ax[1].legend()



sns.distplot(dfdata[dfdata['churn'] == 'No']['totalcharges'],label = 'No',ax=ax[2])

sns.distplot(dfdata[dfdata['churn'] == 'Yes']['totalcharges'],label = 'Yes',ax=ax[2])

ax[2].set_title('Total Charges')

ax[2].legend()



plt.show()
sns.pairplot(dfdata[numeric_features + ['churn']],hue = 'churn', diag_kind = 'kde')

plt.show()
bins = range(12,75,12)

dfdata['tenure_bin'] = np.digitize(dfdata['tenure'],bins,right = True)

dfdata['tenure_bin'] = dfdata['tenure_bin'].astype('category')

categoric_features.append('tenure_bin')

plot_categoric_feature('tenure_bin')
dfdata['meancharges'] = dfdata['totalcharges'] / dfdata['tenure']

numeric_features.append('meancharges')

fig,ax = plt.subplots(1,2,figsize=(18,4))

sns.scatterplot(x='monthlycharges',y='meancharges',data = dfdata,ax = ax[0])

sns.boxplot(x = 'tenure_bin',y = 'monthlycharges',data= dfdata,ax = ax[1])

plt.show()
dfdata['comparemean'] = dfdata['meancharges'] > dfdata['monthlycharges']

dfdata['comparemean'] = np.where(dfdata['meancharges'] == dfdata['monthlycharges'],'Equal',dfdata['comparemean'])

dfdata['comparemean'] = dfdata['comparemean'].astype('category')

plot_categoric_feature('comparemean')
dfdata[categoric_features].nunique()
plot_categoric_feature('gender')
plot_categoric_feature('seniorcitizen')
plot_categoric_feature('partner')
plot_categoric_feature('dependents')
plot_categoric_feature('phoneservice')
pd.crosstab(dfdata['phoneservice'],dfdata['multiplelines'])
plot_categoric_feature('multiplelines')
plot_categoric_feature('internetservice')
plot_categoric_feature('onlinesecurity')
plot_categoric_feature('onlinebackup')
plot_categoric_feature('deviceprotection')
plot_categoric_feature('techsupport')
plot_categoric_feature('streamingtv')
plot_categoric_feature('streamingmovies')
plot_categoric_feature('contract')
plot_categoric_feature('paperlessbilling')
plot_categoric_feature('paymentmethod')
dfdata['internet_fiber'] = np.where(dfdata['internetservice'] == 'Fiber optic','Yes','No')

dfdata['monthly_contract'] = np.where(dfdata['contract'] == 'Month-to-month','Yes','No')

dfdata['electronic_payment'] = np.where(dfdata['paymentmethod'] == 'Electronic check','Yes','No')

categoric_features.extend(['internet_fiber','monthly_contract','electronic_payment'])
dfdata['internet'] = np.where(dfdata['internetservice'] == 'No','No','Yes')

dfdata['num_services'] = (dfdata[['internet','onlinesecurity','onlinebackup','deviceprotection','techsupport','streamingtv','streamingmovies']] == 'Yes').sum(axis=1)

dfdata['num_services'] = dfdata['num_services'].astype('category')

plot_categoric_feature('num_services')

categoric_features.append('internet')

dfdata['num_services'] = dfdata['num_services'].astype('int')

numeric_features.append('num_services')

dfdata['monthly_mean_diff'] = (dfdata['monthlycharges'] - dfdata['monthlycharges'].mean()) / dfdata['monthlycharges'].mean()

numeric_features.append('monthly_mean_diff')

services_list = ['internetservice','onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport','streamingtv', 'streamingmovies']

for service in services_list:

    colname = service + '_mean_diff'

    dfdata[colname] = dfdata['monthlycharges'] / dfdata.groupby(service)['monthlycharges'].transform('mean')

    numeric_features.append(colname)
le = LabelEncoder()

encoded_features = []

for feature in categoric_features:

    colname = 'le_' + feature

    dfdata[colname] = le.fit_transform(dfdata[feature])

    encoded_features.append(colname)
plt.figure(figsize=(12,6))

sns.heatmap(dfdata[numeric_features].corr(),annot = True,fmt  ='.1g')

plt.show()
drop_list = ['meancharges','totalcharges','monthly_mean_diff','onlinebackup_mean_diff','deviceprotection_mean_diff','techsupport_mean_diff','streamingmovies_mean_diff']
plt.figure(figsize=(12,6))

sns.heatmap(dfdata[numeric_features].drop(drop_list,axis=1).corr(),annot = True,fmt  ='.1g')

plt.show()
numeric_features = [x for x in numeric_features if x not in drop_list]
dfdata.columns
plt.figure(figsize=(18,12))

sns.heatmap(dfdata[encoded_features].corr(),annot = True,fmt  ='.1g')

plt.show()
drop_list = ['le_phoneservice','le_contract','le_internet','le_tenure_bin']
plt.figure(figsize=(18,12))

sns.heatmap(dfdata[encoded_features].drop(drop_list, axis=1).corr(),annot = True,fmt  ='.1g')

plt.show()
categoric_features = [x for x in encoded_features if x not in drop_list and categoric_features]
[categoric_features + numeric_features]
dftemp = make_clf_data(100)

X = dftemp.drop('target',axis=1)

y = dftemp['target']

log_reg = LogisticRegression()

log_reg.fit(X, y)

print(np.round(log_reg.intercept_, 2), np.round(log_reg.coef_, 2))

print_clf_result(log_reg, X, y)
plot_decision_boundary(log_reg,dftemp)
c_list = [0.01,0.1,1,10]

plt.figure(figsize=(18,5))

for index, c_value in enumerate(c_list):

    title = 'C : ' + str(c_value)

    log_reg = LogisticRegression(C = c_value).fit(X,y)

    plt.subplot(1,len(c_list),index+1)

    plot_decision_boundary(log_reg, dftemp, title)

plt.show()
X = dfdata[numeric_features + categoric_features]

y = dfdata['churn'].map({'No':0,'Yes':1})

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y)
scaler = StandardScaler().fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
log_reg = LogisticRegression()

log_reg.fit(X_train_scaled, y_train)

print(np.round(log_reg.intercept_, 2), np.round(log_reg.coef_, 2))

print_clf_result(log_reg, X_train_scaled, y_train, X_test_scaled, y_test)
c_list = [0.001,0.01,0.1,1,10,100,1000]

accuracy_list = []

coef_list = []

fig,ax = plt.subplots(1,2,figsize=(18,5))

for index, c_value in enumerate(c_list):

    log_reg = LogisticRegression(C = c_value).fit(X_train_scaled,y_train)

    y_pred = log_reg.predict(X_train_scaled)

    accuracy_list.append(log_reg.score(X_train_scaled,y_train))

    ax[0].plot(np.array(log_reg.coef_).ravel(),label = 'c:'+str(c_value))

    ax[0].legend()

    ax[0].set_title('Coefficients')

ax[1].plot(accuracy_list)

ax[1].set_title('Accuracy')

plt.xticks(range(len(c_list)),c_list)

plt.show()
c_list = [0.001,0.01,0.05,0.1,1,10]

score = []

n_zero_coefs = []

for c_value in c_list:

    log_reg = LogisticRegression(C=c_value, penalty='l1', solver='liblinear').fit(X_train,y_train)

    coef = np.round(log_reg.coef_,4)

    n_zero_coefs.append(len(coef[coef == 0]))

    score.append(round(log_reg.score(X_train, y_train),2))

    

dftemp = pd.DataFrame(zip(c_list, n_zero_coefs, score), columns = ['alpha','zero_coef','score'])

dftemp
estimator = LogisticRegression(C=0.05, penalty='l1', solver='liblinear')

log_reg = estimator.fit(X_train_scaled, y_train)

rfe = RFE(estimator,n_features_to_select=3).fit(X_train_scaled, y_train)

dftemp = pd.DataFrame(zip(X_train_scaled.columns.values,rfe.ranking_,log_reg.coef_.ravel()),columns = ['feature','rank','coef'])

dftemp.sort_values('rank', ascending=True, inplace=True)

dftemp.reset_index(inplace=True,drop=True)

dftemp.tail(10)
X_temp = X_train_scaled.drop(dftemp.iloc[-8:,0].values,axis = 1)

log_reg = LogisticRegression()

log_reg.fit(X_temp, y_train)

print(np.round(log_reg.intercept_, 2), np.round(log_reg.coef_, 2))

print_clf_result(log_reg, X_temp, y_train)
X.drop(dftemp.iloc[-8:,0].values,axis = 1,inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X,y)
scaler = StandardScaler().fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
svc = LinearSVC(C=0.05)

svc.fit(X_train_scaled,y_train)

print(np.round(svc.intercept_, 2), np.round(svc.coef_, 2))

print_clf_result(svc, X_train_scaled, y_train,X_test_scaled,y_test)
svc = SVC(C=1, gamma=0.1, kernel = 'rbf')

svc.fit(X_train_scaled,y_train)

print_clf_result(svc, X_train_scaled, y_train,X_test_scaled,y_test)
sgd = SGDClassifier(loss='hinge', alpha=0.05, eta0=0.01)

sgd.fit(X_train_scaled, y_train)

print_clf_result(sgd, X_train_scaled, y_train, X_test_scaled, y_test)
gnb = GaussianNB()

gnb.fit(X_train, y_train)

print_clf_result(gnb, X_train, y_train, X_test, y_test)
mnb = MultinomialNB(alpha=0.05)

mnb.fit(X_train, y_train)

print_clf_result(gnb, X_train, y_train, X_test, y_test)
bnb = BernoulliNB(alpha=0.05)

bnb.fit(X_train, y_train)

print_clf_result(bnb, X_train, y_train, X_test, y_test)
dt = DecisionTreeClassifier(max_depth=5, random_state=12)

dt.fit(X_train_scaled, y_train)

print_clf_result(dt, X_train_scaled, y_train, X_test_scaled, y_test)
dftemp = pd.DataFrame(zip(X.columns.values,dt.feature_importances_), columns = ['feature','importance'])

plt.figure(figsize=(18,5))

sns.barplot(x='importance', y='feature', data=dftemp)

plt.show()
dftemp = pd.DataFrame(np.round(100 * log_reg.predict_proba(X_test_scaled),0),columns = ['prob0','prob1'])

dftemp['y'] = y_test.values

dftemp['y_pred'] = log_reg.predict(X_test_scaled)

dftemp['decision'] = np.round(log_reg.decision_function(X_test_scaled), 2)

dftemp['error'] = np.abs(dftemp['y'] - dftemp['y_pred'])

print("Min and max decision function values : ",round(np.min(log_reg.decision_function(X_test_scaled)),2),round(np.max(log_reg.decision_function(X_test_scaled)),2))

dftemp.head()

bins = range(0,100,5)

dftemp['prob1bin'] = np.digitize(dftemp['prob1'],bins,right=True)

sns.barplot(x='prob1bin',y='error',data=dftemp)

plt.show()
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train_scaled, y_train)

pred_most_frequent = dummy_majority.predict(X_test_scaled)

print("Test score: {:.2f}".format(dummy_majority.score(X_test_scaled, y_test)))
round(100 * dfdata['churn'].value_counts() / dfdata.shape[0], 0)
rus = under_sampling.RandomUnderSampler()

X_rus, y_rus = rus.fit_sample(X_train_scaled,y_train)
log_reg = LogisticRegression()

log_reg.fit(X_rus, y_rus)

print(np.round(log_reg.intercept_, 2), np.round(log_reg.coef_, 2))

print_clf_result(log_reg, X_rus, y_rus,X_test_scaled,y_test)
smote = over_sampling.SMOTE(sampling_strategy='minority')

X_sm, y_sm = smote.fit_sample(X_train_scaled, y_train)
log_reg = LogisticRegression()

log_reg.fit(X_sm, y_sm)

print(np.round(log_reg.intercept_, 2), np.round(log_reg.coef_, 2))

print_clf_result(log_reg, X_sm, y_sm,X_test_scaled,y_test)
svc = SVC(C=1, gamma=0.1, kernel = 'rbf', class_weight='balanced', probability=True)

svc.fit(X_train_scaled, y_train)

print_clf_result(svc, X_train_scaled, y_train, X_test_scaled, y_test)
plot_empty_confusion_matrix()
y_pred = log_reg.predict(X_test_scaled)

plt.figure(figsize=(2,2))

plot_confusion_matrix(log_reg, y_test, y_pred)

plt.show()

print(classification_report(y_test,y_pred))
y_pred_threshold = log_reg.decision_function(X_test_scaled) > -0.2

plt.figure(figsize=(2,2))

plot_confusion_matrix(log_reg, y_test, y_pred_threshold)

plt.show()

print(classification_report(y_test,y_pred_threshold))
y_pred_threshold = log_reg.predict_proba(X_test_scaled)[:,1] > 0.35

plt.figure(figsize=(2,2))

plot_confusion_matrix(log_reg, y_test, y_pred_threshold)

plt.show()

print(classification_report(y_test,y_pred_threshold))
aps_logreg = round(average_precision_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1]),2)

aps_svc = round(average_precision_score(y_test, svc.decision_function(X_test_scaled)),2)

print("Average Precision Scores (log reg and svc) : ", aps_logreg, aps_svc)

precision_lr, recall_lr, thresholds_lr = precision_recall_curve(y_test, log_reg.decision_function(X_test_scaled))

close_zero_lr = np.argmin(np.abs(thresholds_lr))

plt.plot(precision_lr[close_zero_lr], recall_lr[close_zero_lr], 'o', markersize=10, label="threshold zero logreg", fillstyle="none", c='k', mew=2)

plt.plot(precision_lr, recall_lr, label="log reg")



precision_svc, recall_svc, thresholds_svc = precision_recall_curve(y_test, svc.decision_function(X_test_scaled))

close_zero_svc = np.argmin(np.abs(thresholds_svc))

plt.plot(precision_svc[close_zero_svc], recall_svc[close_zero_svc], 'v', markersize=10, label="threshold zero svc", fillstyle="none", c='k', mew=2)

plt.plot(precision_svc, recall_svc, label="svc")

plt.legend()

plt.title('Precision Recall Curve')

plt.xlabel("Precision")

plt.ylabel("Recall")

plt.show()
plot_empty_confusion_matrix()
auc_logreg = roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1])

auc_svc = roc_auc_score(y_test, svc.decision_function(X_test_scaled))

print("AUC scores (logreg and svc) : ", round(auc_logreg,2), round(auc_svc,2))



fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, log_reg.decision_function(X_test_scaled))

plt.plot(fpr_lr, tpr_lr, label="ROC LogReg")

close_zero_lr = np.argmin(np.abs(thresholds_lr))

plt.plot(fpr_lr[close_zero_lr], tpr_lr[close_zero_lr], 'o', markersize=10,label="threshold zero", fillstyle="none", c='k', mew=2)



fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, svc.decision_function(X_test_scaled))

plt.plot(fpr_svc, tpr_svc, label="ROC SVC")

close_zero_svc = np.argmin(np.abs(thresholds_svc))

plt.plot(fpr_svc[close_zero_svc], tpr_svc[close_zero_svc], 'v', markersize=10, label="threshold zero", fillstyle="none", c='k', mew=2)



plt.title('ROC Curve')

plt.xlabel("FPR")

plt.ylabel("TPR (recall)")

plt.legend(loc=4)

plt.show()