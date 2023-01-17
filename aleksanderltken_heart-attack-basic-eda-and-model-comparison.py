########################## Import libraries ###################################

#loading dataset
import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#EDA
import pandas_profiling as pp
#! pip install -q scikit-plot
import scikitplot as skplt

# data preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# data splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# data modeling
from tpot import TPOTClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# data interpretation
from collections import Counter
########################## Load data ##########################################

df = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

# Quick check
print(df.info(), "\n","\n", "Head(5): ", "\n", df.head(), "\n","\n", "Describe: ", "\n", df.describe())
# Check for NaN values
df.isna().sum()
# Another option: df.isna().any()
########################## EDA ################################################

# Pandas profiling report
pp.ProfileReport(df).to_file('heart_attack.html')
# Gives an easy interpretation of the df
# Countplot for imbalance in target
sns.countplot(df['target'])
plt.show()

# Countplot for imbalance in gender
sns.countplot(df['sex'])
plt.show()
# Age distribution
fig, ax = plt.subplots(figsize=(10,5))
sns.distplot(df['age'])
plt.xlim([0,80])
# Chol distribution
fig, ax = plt.subplots(figsize=(10,5))
sns.distplot(df['chol'])
plt.xlim([0,600])
# Trestbps distribution
fig, ax = plt.subplots(figsize=(10,5))
sns.distplot(df['trestbps'])
plt.xlim([0,250])
# Pairplot
sns.pairplot(data=df)
plt.show()
#Pearson Correlation Heatmap
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
########################## Prepare data #######################################

# Seperating the target from the features
X = df.copy(deep = True)
y = X.pop('target')

# Train, test, split the data with test sze at 20% and random state = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, 
                                                    random_state = 0)

# Is there imbalance in the training dataset? No
print('Original dataset shape %s' % Counter(y_train))
# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
####################### Machine Learning models################################

#1
m1 = 'Logistic Regression'
LogReg = LogisticRegression()
model = LogReg.fit(X_train, y_train)
LogReg_predict = LogReg.predict(X_test)
LogReg_conf_matrix = confusion_matrix(y_test, LogReg_predict)
LogReg_acc_score = accuracy_score(y_test, LogReg_predict)
print("confussion matrix", "\n", LogReg_conf_matrix, "\n", 
      "Accuracy of Logistic Regression:",LogReg_acc_score*100)
#2
m2 = 'Naive Bayes'
nb = GaussianNB()
nb.fit(X_train,y_train)
nbpred = nb.predict(X_test)
nb_conf_matrix = confusion_matrix(y_test, nbpred)
nb_acc_score = accuracy_score(y_test, nbpred)
print("confussion matrix", "\n", nb_conf_matrix, "\n",
      "Accuracy of Naive Bayes:",nb_acc_score*100)
#3
m3 = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=500, random_state=12, max_depth=10)
rf.fit(X_train,y_train)
rf_predict = rf.predict(X_test)
rf_conf_matrix = confusion_matrix(y_test, rf_predict)
rf_acc_score = accuracy_score(y_test, rf_predict)
print("confussion matrix", "\n", rf_conf_matrix,"\n", 
      "Accuracy of Random Forest:",rf_acc_score*100)


###
import scikitplot as skplt
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                    'restecg', 'thalach', 'exang', 'oldpeak',
                    'slope', 'ca', 'thal']
skplt.estimators.plot_feature_importances(rf, feature_names)
plt.show()
#4
m4 = 'AdaBoostClassifier'
param_grid2 = [{'n_estimators': list(range(1,20)), 'learning_rate': list(np.arange(0.1,2,0.1))}] # params to try in the grid search
clfAB = AdaBoostClassifier()
ad = GridSearchCV(clfAB, param_grid2, cv=5, verbose=1, return_train_score = True, n_jobs = -1)
ad.fit(X_train,y_train)
ad_predict = ad.predict(X_test)
ad_conf_matrix = confusion_matrix(y_test, ad_predict)
ad_acc_score = accuracy_score(y_test, ad_predict)
print("confussion matrix", "\n", ad_conf_matrix,"\n", 
      "Accuracy of AdaBoost:",ad_acc_score*100)

###
print(ad.best_params_)
#how should we expect this to do based on the validation scores?
print('''best score = {:.2f}'''.format(ad.best_score_))
###
#5
m5 = 'DecisionTreeClassifier'
param_grid1 = [{'max_depth': list(range(1,9))}] # params to try in the grid search
clfDT = DecisionTreeClassifier()
dt = GridSearchCV(clfDT, param_grid1, cv=5, verbose=1, return_train_score = True)
dt.fit(X_train,y_train)
dt_predict = dt.predict(X_test)
dt_conf_matrix = confusion_matrix(y_test, dt_predict)
dt_acc_score = accuracy_score(y_test, dt_predict)
print("confussion matrix", "\n", dt_conf_matrix,"\n", 
      "Accuracy of Decision Tree:",dt_acc_score*100)

###
print(dt.best_params_)
#how should we expect this to do based on the validation scores?
print('''best score = {:.2f}'''.format(dt.best_score_))
###

#6
m6 = 'TPOT'
TPOT = TPOTClassifier(generations=7, population_size=60, verbosity=2, n_jobs = -1)
TPOT.fit(X_train,y_train)
TPOT_predict = TPOT.predict(X_test)
TPOT_conf_matrix = confusion_matrix(y_test, TPOT_predict)
TPOT_acc_score = accuracy_score(y_test, TPOT_predict)
print("confusion matrix", "\n", TPOT_conf_matrix,"\n", 
      "Accuracy of TPOT:",TPOT_acc_score*100)

#7
m7 = 'SupportVectorMachineClassifier'
param_grid3 = [{'kernel': ['rbf','sigmoid'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]
svc = GridSearchCV(SVC(), param_grid3, cv=5,return_train_score = True)
svc.fit(X_train,y_train)
svc_predict = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predict)
svc_acc_score = accuracy_score(y_test, svc_predict)
print("confusion matrix", "\n", svc_conf_matrix,"\n", 
      "Accuracy of Support Vector Machine:",svc_acc_score*100)

#8
m8 = 'ElasticNet'
en_alpha = 0.1
en_l1ratio = 0.5
en = ElasticNet(alpha = en_alpha, l1_ratio = en_l1ratio)
en.fit(X_train,y_train)
en_predict = en.predict(X_test)
en_predict = np.where(en_predict>0.5,1,0)
en_conf_matrix = confusion_matrix(y_test, en_predict)
en_acc_score = accuracy_score(y_test, en_predict)
print("confusion matrix", "\n", en_conf_matrix,"\n", 
      "Accuracy of Elastic Net:",en_acc_score*100)

############################### Model results #################################

# Barplot of the accuracy score
colors = ["orange", "green", "magenta", "red", "blue" , "grey", "yellow", "purple"]
acc = [LogReg_acc_score,nb_acc_score,rf_acc_score,ad_acc_score,dt_acc_score,
       TPOT_acc_score,svc_acc_score,en_acc_score]
m = [m1,m2,m3,m4,m5,m6,m7,m8]
plt.figure(figsize=(18,4))
plt.yticks(np.arange(0,100,10))
plt.title("barplot Represent Accuracy of different models")
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot( y=acc,x=m, palette=colors)
plt.show()
# Accuracy score for the models
modelscore = [["Accuracy of Logistic Regression:",LogReg_acc_score*100],
              ["Accuracy of Naive Bayes:",nb_acc_score*100],
              ["Accuracy of Random Forest:",rf_acc_score*100],
              ["Accuracy of AdaBoost:",ad_acc_score*100],
              ["Accuracy of Decision Tree:",dt_acc_score*100],
              ["Accuracy of TPOT:",TPOT_acc_score*100],
              ["Accuracy of Support Vector Machine:",svc_acc_score*100],
              ["Accuracy of Elastic Net:",en_acc_score*100]]
for m,ms in modelscore:
    print(m,ms)

#The model with the best accuracy score in our case turned out to be the
# adaboost model with a score of 86,88%. Lets have a closer look at the
# confusion matrixes
# Confusion matrix for the models
confusion_matrix = [["Confusion matrix for Logistic Regression:", "\n",LogReg_conf_matrix],
                    ["Confusion matrix for Naive Bayes:", "\n",nb_conf_matrix],
                    ["Confusion matrix for Random Forest:", "\n",rf_conf_matrix],
                    ["Confusion matrix for AdaBoost:", "\n",ad_conf_matrix],
                    ["Confusion matrix for Decision Tree:", "\n",dt_conf_matrix],
                    ["Confusion matrix for TPOT:", "\n",TPOT_conf_matrix],
                    ["Confusion matrix for Support Vector Machine:", "\n",svc_conf_matrix],
                    ["Confusion matrix for Elastic Net:", "\n", en_conf_matrix]]
for m,tab,cm in confusion_matrix:
    print(m,tab,cm)

# My interpretation of the models:
# I want my model to be make as many correct predicions as possible. It is also
# important to be aware of where the model fails. For the Adaboost model (w/high accuracy):
# It had 30 TP and 23 TN. There were 4 instances in the FP. This is not too bad
# as we predicted heart attack and it turns out they dont got it. For FN,there 
# is 4 instances. This means we predicted no chance for heart attack but
# they still got it. We want to minize this as much as possible.


# 1. One could make changes to the threshold, and test if it would make any 
# difference on a valuation dataset.

# 2. Choose another model: We could for example choose the support vector
# machine model which had a lower accuracy score, but fewer FN (more FP).


# Visualization of confusion matrix
y_pred_model = [[ad_predict, 'AdaBoost'], [svc_predict, 'Support Vector Machine']]
for y_pred_i, m_name in y_pred_model:
    skplt.metrics.plot_confusion_matrix(y_test,y_pred_i, figsize=(4,4), title=(m_name))
    plt.show()

# One could also try to balance out the small imbalance in the dataset to see if
# it would make any improvement. An option could be using SMOTE,which will use a
# syntethic upsampling method. We could also use class_weight as a way to deal
# with imbalance.

