# Import necessary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize

from imblearn.over_sampling import SMOTE 



from sklearn import metrics

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc
df1 = pd.read_csv('bank-full.csv')
df1.shape
df1.info()
df1.isna().count()
# change the object datatype to category



for i in df1.columns:

    if df1[i].dtype == 'object':

        df1[i] = pd.Categorical(df1[i])
df1.info()
# check whether duplicates present in the data



dups = df1[df1.duplicated()]

dups
df1.describe().T
print('Skewness of Age        : ', df1["age"].skew())

print('Skewness of Balance    : ', df1["balance"].skew())

print('Skewness of Day        : ', df1["day"].skew())

print('Skewness of Duration   : ', df1["duration"].skew())

print('Skewness of Campaign   : ', df1["campaign"].skew())

print('Skewness of Pdays      : ', df1["pdays"].skew())

print('Skewness of Previous   : ', df1["previous"].skew())
# Hist plots to find the distribution of continuous variables



plt.figure(figsize = (16,12))

plt.subplot(3,3,1)

sns.distplot(df1['age'])

plt.xlabel("Age")

plt.subplot(3,3,2)

sns.distplot(df1['balance'])

plt.xlabel("Balance")

plt.subplot(3,3,3)

sns.distplot(df1['duration'])

plt.xlabel("Duration")

plt.show()
plt.figure(figsize = (16,12))

plt.subplot(3,3,1)

sns.distplot(df1['campaign'])

plt.xlabel("Campaign")

plt.subplot(3,3,2)

sns.distplot(df1['pdays'])

plt.xlabel("PDays")

plt.subplot(3,3,3)

sns.distplot(df1['previous'])

plt.xlabel("Previous")

plt.show()
plt.figure(figsize = (16,12))

plt.subplot(3,3,1)

sns.distplot(df1['day'])

plt.xlabel("Day")

plt.show()
#Count plot to check the distribution of categorical variables

plt.figure(figsize=(24,17))

plt.subplot(3,3,1)

ch1 = sns.countplot(df1.job,hue=df1.Target)

ch1.set_xticklabels(ch1.get_xticklabels(), rotation=45)

plt.xlabel("Job")



plt.subplot(3,3,2)

sns.countplot(df1.marital,hue=df1.Target)

plt.xlabel("Marital")

plt.show()
plt.figure(figsize=(24,17))

plt.subplot(3,3,1)

sns.countplot(df1.education,hue=df1.Target)

plt.xlabel("Education")



plt.subplot(3,3,2)

sns.countplot(df1.default,hue=df1.Target)

plt.xlabel("Default")

plt.show()
plt.figure(figsize=(24,17))

plt.subplot(3,3,1)

sns.countplot(df1.housing,hue=df1.Target)

plt.xlabel("Housing Loan")



plt.subplot(3,3,2)

ch1 = sns.countplot(df1.loan,hue=df1.Target)

plt.xlabel("Personal Loan")

plt.show()
plt.figure(figsize=(24,17))

plt.subplot(3,3,1)

sns.countplot(df1.contact,hue=df1.Target)

plt.xlabel("Communication Type")



plt.subplot(3,3,2)

ch1 = sns.countplot(df1.day,hue=df1.Target)

plt.xlabel("Last contact day of the month")

plt.show()
plt.figure(figsize=(24,17))

plt.subplot(3,3,1)

sns.countplot(df1.month,hue=df1.Target)

plt.xlabel("Last contact month of the year")



plt.subplot(3,3,2)

ch1 = sns.countplot(df1.poutcome,hue=df1.Target)

plt.xlabel("Outcome of Previous marketing campaign")

plt.show()
# violin plot to check the 

plt.figure(figsize = (25,10))

sns.violinplot(x= df1.job, y = df1.balance, hue = df1.Target)
plt.figure(figsize=(14,6))

sns.boxplot(df1.balance)

plt.show()



plt.figure(figsize=(14,6))

ch1 = sns.scatterplot(x= df1.job, y = df1.balance, hue = df1.Target)

plt.xlabel("Job")

plt.ylabel("Balance")

plt.show()
sns.pairplot(df1)
corr1 = df1.corr()

plt.figure(figsize=(14,7))

hm1 = sns.heatmap(corr1,annot=True, fmt = '.2f')

bottom, top = hm1.get_ylim()

hm1.set_ylim(bottom + 0.5, top - 0.5)
lbl = LabelEncoder()

df1["Target"]  = lbl.fit_transform(df1["Target"])

df1["default"] = lbl.fit_transform(df1["default"])

df1["housing"] = lbl.fit_transform(df1["housing"])

df1["loan"]    = lbl.fit_transform(df1["loan"])



df1['education'] = df1['education'].map({'primary': 1, 'secondary': 2, 'tertiary': 3, 'unknown': -1})

df1['job'] = df1['job'].map({'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5, 'retired': 6,

                            'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10, 'unemployed': 11, 

                            'unknown': -1})

df1['contact'] = df1['contact'].map({'cellular': 1, 'telephone': 2, 'unknown': -1})

df1['month'] = df1['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9,

                                'oct': 10, 'nov': 11, 'dec': 12})

df1['poutcome'] = df1['poutcome'].map({'failure': 1, 'other': 2, 'success': 3, 'unknown': -1})



dCols=["marital"]

df1 = pd.get_dummies(df1, columns = dCols)
df1.head(10)
df1['Target'].value_counts()
df1['Target'].value_counts(normalize=True)
sns.countplot(df1.Target)
# check the impact of duration on Target

z = df1["duration"].mean()



# created a new column dur_t which will have value "Low" for duration value less than its mean, and "High" for values greater

#  than mean. This is to find the influence of higher call duration on Target variable. 

df1["dur_t"] = np.where(df1["duration"] < z, "Low", "High") 

sns.countplot(df1.dur_t, hue=df1.Target)
df1.drop("dur_t", axis=1,inplace = True)

df1.head()
# change the datatype to int64 since numerical values are present in the dataset

df1['job'] = df1['job'].astype('int64') 

df1['education'] = df1['education'].astype('int64') 

df1['contact'] = df1['contact'].astype('int64') 

df1['month'] = df1['month'].astype('int64') 

df1['poutcome'] = df1['poutcome'].astype('int64') 
df1.info()
x1 = df1.drop("Target", axis = 1) # Duration is not dropped

y1 = df1["Target"]



xtrain1, xtest1, ytrain1, ytest1 = train_test_split(x1, y1, test_size = 0.30, random_state=1)
print("Training set contains {0:0.1f}% of data" .format((len(xtrain1)/len(df1))*100))

print("Test set contains {0:0.1f}% of data" .format((len(xtest1)/len(df1))*100))
DT1 = DecisionTreeClassifier(criterion = 'gini', random_state=1)

DT1.fit(xtrain1,ytrain1)

print("Decision Tree Training score : ", DT1.score(xtrain1,ytrain1))

print("Decision Tree Test score     : ", DT1.score(xtest1,ytest1))
print (pd.DataFrame(DT1.feature_importances_, columns = ["Feature Importances"], index = xtrain1.columns))
LR1 = LogisticRegression(solver='liblinear')

LR1.fit(xtrain1,ytrain1)

print("Logistic regression Training score :", LR1.score(xtrain1,ytrain1))

print("Logistic regression Test score     :", LR1.score(xtest1,ytest1))
# Confustion Matrix

predz = LR1.predict(xtest1)

cmz=metrics.confusion_matrix(ytest1, predz, labels=[1, 0])

df_cma = pd.DataFrame(cmz, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])



predy = DT1.predict(xtest1)

cmy=metrics.confusion_matrix(ytest1, predy, labels=[1, 0])

df_cmb = pd.DataFrame(cmy, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])



plt.figure(figsize = (16,14))

plt.subplot(3,3,1)

hm2 = sns.heatmap(df_cma, annot=True, fmt='g')

bottom, top = hm2.get_ylim()

hm2.set_ylim (bottom +0.5, top - 0.5)

plt.title("Logistic regression before SMOTE")



plt.subplot(3,3,3)

hm3 = sns.heatmap(df_cmb, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)

plt.title("Decision Tree before SMOTE")
print("Classification Report For DT1")

print("")

pred1 = DT1.predict(xtest1)

DT_log1 = metrics.classification_report(ytest1, pred1, labels = [1,0])

print(DT_log1)

print("-----------------------------------------------------")

print("Classification Report For LR1")

print("")

pred2 = LR1.predict(xtest1)

LR_log = metrics.classification_report(ytest1, pred2, labels = [1,0])

print(LR_log)
for idx, col_name in enumerate(xtrain1.columns):

    print("The coefficient for {} is {}".format(col_name, round(LR1.coef_[0][idx], 5)))
df1.drop("duration", axis=1,inplace = True)

df1.head()
x2 = df1.drop("Target", axis = 1)

y2 = df1["Target"]
# train and test datasets are prepared without duration



xtrain2, xtest2, ytrain2, ytest2 = train_test_split(x2, y2, test_size = 0.30, random_state=1)
# SMOTE has been applied only on Training sets



sm = SMOTE(random_state = 2)  

xtrain_sm, ytrain_sm = sm.fit_sample(xtrain2, ytrain2) 
print("Before SMOTE - xtrain shape : ", xtrain2.shape)

print("After SMOTE  - xtrain shape : ", xtrain_sm.shape)

print("")

print("Before SMOTE - ytrain shape : ", ytrain2.shape)

print("After  SMOTE - ytrain shape : ", ytrain_sm.shape)

print("")

print("Before SMOTE - ytrain distribution : ")

dfx = pd.DataFrame()

dfx['Target0'] = ytrain2

print( dfx['Target0'].value_counts())

print("")

dfy = pd.DataFrame()

dfy['Target1'] = ytrain_sm

print("After SMOTE  - ytrain distribution : ")

print(dfy['Target1'].value_counts())
# xtrain_sm, xtest2, ytrain_sm, ytest2
# Models built WITHOUT Scaled/Normalized data

print("Model scores run WITHOUT Scaled/Normalized data")

print("")



LR1 = LogisticRegression(solver='liblinear')

LR1.fit(xtrain_sm,ytrain_sm)

LR1_TR = LR1.score(xtrain_sm,ytrain_sm)

LR1_TS = LR1.score(xtest2,ytest2)

print("Logistic Regression Training score :", round(LR1_TR * 100, 2))

print("Logistic Regression Test score     :", round(LR1_TS * 100, 2))

print("")



KNN1 = KNeighborsClassifier(n_neighbors=235, weights='distance') 

# k = squareroot(trainingsample) -> round(math.sqrt(xtrain_sm.shape[0]))

KNN1.fit(xtrain_sm,ytrain_sm)

KNN1_TR = KNN1.score(xtrain_sm,ytrain_sm)

KNN1_TS = KNN1.score(xtest2,ytest2)

print("K-Nearest Neighbor Training score  :", round(KNN1_TR * 100, 2))

print("K-Nearest Neighbor Test score      :", round(KNN1_TS * 100, 2))

print("")



GNM1 = GaussianNB()

GNM1.fit(xtrain_sm,ytrain_sm)

GNM1_TR = GNM1.score(xtrain_sm,ytrain_sm)

GNM1_TS = GNM1.score(xtest2,ytest2)

print("Naive Bayes Training score         :", round(GNM1_TR * 100, 2))

print("Naive Bayes Test score             :", round(GNM1_TS * 100, 2))

print("")



#SVM1 = svm.SVC(gamma=0.025, C=3)   # Random gamma and C values are chosen here

#SVM1.fit(xtrain_sm,ytrain_sm)

#SVM1_TR = SVM1.score(xtrain_sm,ytrain_sm)

#SVM1_TS = SVM1.score(xtest2,ytest2)

#print("Support Vector Classifier Training score : ", round(SVM1_TR * 100, 2))

#print("Support Vector Classifier Test score     : ", round(SVM1_TS * 100, 2))

#print("")



DT1 = DecisionTreeClassifier(criterion = 'gini', random_state=1)

DT1.fit(xtrain_sm,ytrain_sm)

DT1_TR = DT1.score(xtrain_sm,ytrain_sm)

DT1_TS = DT1.score(xtest2,ytest2)

print("Decision Tree Training score       :", round(DT1_TR * 100, 2))

print("Decision Tree Test score           :", round(DT1_TS * 100, 2))
# Scale the training data



sc = StandardScaler()

xtrain_sm_sc = sc.fit_transform(xtrain_sm)

xtest_sm_sc = sc.transform(xtest2)
x_sc = sc.transform(x2)
ytest_sm = ytest2
#x_sm_sc = sc.fit_transform(x_sm)



# xtrain_sm_sc, xtest_sm_sc, ytrain_sm, ytest_sm
# Models built with Scaled data

print("Scaled data is passed as Input to Models")

print("")



LR2 = LogisticRegression(solver='liblinear')

LR2.fit(xtrain_sm_sc,ytrain_sm)

LR2_TR = LR2.score(xtrain_sm_sc,ytrain_sm)

LR2_TS = LR2.score(xtest_sm_sc,ytest_sm)

print("Logistic Regression Training score :", round(LR2_TR * 100, 2))

print("Logistic Regression Test score     :", round(LR2_TS * 100, 2))

print("")



KNN2 = KNeighborsClassifier(n_neighbors=235, weights='distance') 

# k = squareroot(trainingsample) -> round(math.sqrt(xtrain_sm.shape[0]))

KNN2.fit(xtrain_sm_sc,ytrain_sm)

KNN2_TR = KNN2.score(xtrain_sm_sc,ytrain_sm)

KNN2_TS = KNN2.score(xtest_sm_sc,ytest_sm)

print("K-Nearest Neighbor Training score  :", round(KNN2_TR * 100, 2))

print("K-Nearest Neighbor Test score      :", round(KNN2_TS * 100, 2))

print("")



GNM2 = GaussianNB()

GNM2.fit(xtrain_sm_sc,ytrain_sm)

GNM2_TR = GNM2.score(xtrain_sm_sc,ytrain_sm)

GNM2_TS = GNM2.score(xtest_sm_sc,ytest_sm)

print("Naive Bayes Training score         :", round(GNM2_TR * 100, 2))

print("Naive Bayes Test score             :", round(GNM2_TS * 100, 2))

print("")



DT2 = DecisionTreeClassifier(criterion = 'gini', random_state=1)

DT2.fit(xtrain_sm_sc,ytrain_sm)

DT2_TR = DT2.score(xtrain_sm_sc,ytrain_sm)

DT2_TS = DT2.score(xtest_sm_sc,ytest_sm)

print("Decision Tree Training score       :", round(DT2_TR * 100, 2))

print("Decision Tree Test score           :", round(DT2_TS * 100, 2))
# Normalise the data



xtrain_sm_nm = normalize(xtrain_sm)

xtest_sm_nm = normalize(xtest2)
# Models ran with Normalized data

print("Normalized data is passed as Input to Models")

print("")



LR3 = LogisticRegression(solver='liblinear')

LR3.fit(xtrain_sm_nm,ytrain_sm)

LR3_TR = LR3.score(xtrain_sm_nm,ytrain_sm)

LR3_TS = LR3.score(xtest_sm_nm,ytest_sm)

print("Logistic Regression Training score :", round(LR3_TR * 100, 2))

print("Logistic Regression Test score     :", round(LR3_TS * 100, 2))

print("")



KNN3 = KNeighborsClassifier(n_neighbors=235, weights='distance')

# k = squareroot(trainingsample) -> round(math.sqrt(xtrain_sm.shape[0]))

KNN3.fit(xtrain_sm_nm,ytrain_sm)

KNN3_TR = KNN3.score(xtrain_sm_nm,ytrain_sm)

KNN3_TS = KNN3.score(xtest_sm_nm,ytest_sm)

print("K-Nearest Neighbor Training score  :", round(KNN3_TR * 100, 2))

print("K-Nearest Neighbor Test score      :", round(KNN3_TS * 100, 2))

print("")



GNM3 = GaussianNB()

GNM3.fit(xtrain_sm_nm,ytrain_sm)

GNM3_TR = GNM3.score(xtrain_sm_nm,ytrain_sm)

GNM3_TS = GNM3.score(xtest_sm_nm,ytest_sm)

print("Naive Bayes Training score         :", round(GNM3_TR * 100, 2))

print("Naive Bayes Test score             :", round(GNM3_TS * 100, 2))

print("")



DT3 = DecisionTreeClassifier(criterion = 'gini', random_state=1)

DT3.fit(xtrain_sm_nm,ytrain_sm)

DT3_TR = DT3.score(xtrain_sm_nm,ytrain_sm)

DT3_TS = DT3.score(xtest_sm_nm,ytest_sm)

print("Decision Tree Training score       :", round(DT3_TR * 100, 2))

print("Decision Tree Test score           :", round(DT3_TS * 100, 2))
def parm_tune(mod, parm, nbr_iter, x, y):

    rdmsearch = RandomizedSearchCV(mod, param_distributions = parm, n_jobs = -1, n_iter = nbr_iter, cv = 7)

    rdmsearch.fit(x, y)

    rd_parms = rdmsearch.best_params_

    rd_score = rdmsearch.best_score_

    return rd_parms, rd_score
log_reg = LogisticRegression(solver = 'liblinear')

penalty1 = ['l1', 'l2']  

C1 = np.logspace(0, 1, 2, 3, 4)  

hyperparm1 = dict(C=C1, penalty=penalty1)



LR1_parms, LR1_score = parm_tune(log_reg, hyperparm1, 50, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", LR1_parms)

print("\nBest Score: \n", LR1_score)
log_reg = LogisticRegression(solver = 'liblinear', C = 1, penalty = 'l1')

log_reg.fit(xtrain_sm_sc, ytrain_sm)

LR3_TR = log_reg.score(xtrain_sm_sc,ytrain_sm)

LR3_TS = log_reg.score(xtest_sm_sc,ytest_sm)

print("Logistic Regression Training score with best liblinear parameters :", round(LR3_TR * 100, 2))

print("Logistic Regression Test score with best liblinear parameters     :", round(LR3_TS * 100, 2))
# Draw the confusion matrix for Logistic Regression

pred1 = log_reg.predict(xtest_sm_sc)

cm=metrics.confusion_matrix(ytest_sm, pred1, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])



plt.figure(figsize = (7,5))

hm2 = sns.heatmap(df_cm, annot=True, fmt='g')

bottom, top = hm2.get_ylim()

hm2.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 20 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=20)

cv_logistic = cross_val_score(log_reg, x_sc, y2, cv=kfold)

print("Cross Validation Score of Logistic Regression Model: ", round(cv_logistic.mean() * 100,2))
cv_log = cv_logistic.mean()

f1_log = metrics.f1_score(ytest_sm, pred1, average='micro')

tn_log, fp_log, fn_log, tp_log = metrics.confusion_matrix(ytest_sm, pred1).ravel()
print("Classification Report For Logistic Regression (BEFORE SMOTE)")

print("")

print(LR_log)

print("--------------------------------------------------------------")

print("Classification Report For Logistic Regression (AFTER SMOTE)")

print("")

LR2_log = metrics.classification_report(ytest_sm, pred1, labels = [1,0])

print(LR2_log)
print("K values as per formula sqrt(n) : ", round(math.sqrt(xtrain_sm.shape[0])))
# Tuning of KNN



KNC = KNeighborsClassifier(weights = 'distance')

neighb1 = (3,5,7,147,149,171,173,231,233,235,237,239,241) 

hyperparm2 = dict(n_neighbors=neighb1)



KNC_parms, KNC_score = parm_tune(KNC, hyperparm2, 20, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", KNC_parms)

print("\nBest Score: \n", KNC_score)
# Running the model again with n_neighbors=3

KNN3 = KNeighborsClassifier(n_neighbors=7,weights = 'distance')

KNN3.fit(xtrain_sm_sc, ytrain_sm)

KNN_TR = KNN3.score(xtrain_sm_sc,ytrain_sm)

KNN_TS = KNN3.score(xtest_sm_sc,ytest_sm)

print("KNN Model Training score with best possible k value : ", round(KNN_TR * 100, 2))

print("KNN Model Test score with best possible k value     : ", round(KNN_TS * 100, 2))
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 20 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=20)

cv_knn = cross_val_score(KNN3, x_sc, y2, cv=kfold)

print("Cross Validation Score of KNN Model: ", round(cv_knn.mean() * 100,2))
# Confustion Matrix

pred2 = KNN3.predict(xtest_sm_sc)



cm1=metrics.confusion_matrix(ytest_sm, pred2, labels=[1, 0])



df_cm = pd.DataFrame(cm1, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm2 = sns.heatmap(df_cm, annot=True, fmt='g')

bottom, top = hm2.get_ylim()

hm2.set_ylim (bottom +0.5, top - 0.5)
cv_knn1 = cv_knn.mean()

f1_knn = metrics.f1_score(ytest_sm, pred2, average='micro')

tn_knn, fp_knn, fn_knn, tp_knn = metrics.confusion_matrix(ytest_sm, pred2).ravel()
print("Classification Report For KNN")

print("")

CR_KNN = metrics.classification_report(ytest_sm, pred2, labels = [1,0])

print(CR_KNN)
GNM3 = GaussianNB()

GNM3.fit(xtrain_sm_sc,ytrain_sm)

GNM_TR = GNM3.score(xtrain_sm_sc,ytrain_sm)

GNM_TS = GNM3.score(xtest_sm_sc,ytest_sm)

print("Naive Bayes Training score : ", round(GNM_TR * 100, 2))

print("Naive Bayes Test score     : ", round(GNM_TS * 100, 2))
pred3 = GNM3.predict(xtest_sm_sc)



cm2=metrics.confusion_matrix(ytest_sm, pred3, labels=[1, 0])



df_cm_gnm = pd.DataFrame(cm2, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm3 = sns.heatmap(df_cm_gnm, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 20 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=20)

cv_gnm = cross_val_score(GNM3, x_sc, y2, cv=kfold)

print("Cross Validation Score of Gaussian Naive Bayes Model: ", round(cv_gnm.mean() * 100,2))
cv_gnm1 = cv_gnm.mean()

f1_gnm = metrics.f1_score(ytest_sm, pred3, average='micro')

tn_gnm, fp_gnm, fn_gnm, tp_gnm = metrics.confusion_matrix(ytest_sm, pred3).ravel()
print("Classification Report For Gaussian Naive Bayes")

print("")

CR_NB = metrics.classification_report(ytest_sm, pred3, labels = [1,0])

print(CR_NB)
DT3 = DecisionTreeClassifier(random_state=1)

DT3.fit(xtrain_sm_sc,ytrain_sm)

print("DecisionTree Classifier Test score with default parameters :", DT3.score(xtest_sm_sc,ytest_sm))
DT3
#Tuning of Decision Tree Classifier



DTC = DecisionTreeClassifier(random_state=1)

hyperparm3 = {'criterion':['gini','entropy'],'max_depth': np.arange(3, 16), 'max_features': np.arange(4,17)}



DT1_parms, DT1_score = parm_tune(DTC, hyperparm3, 20, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", DT1_parms)

print("\nBest Score: \n", DT1_score)
DT4 = DecisionTreeClassifier(criterion = 'entropy',max_depth = 14, max_features = 16, random_state=1)

DT4.fit(xtrain_sm_sc,ytrain_sm)

DT4_TR = DT4.score(xtrain_sm_sc,ytrain_sm)

DT4_TS = DT4.score(xtest_sm_sc,ytest_sm)

print("Decistion Tree Training score with best hyperparameters : ", round(DT4_TR * 100, 2))

print("Decistion Tree Test score with best hyperparameters     : ", round(DT4_TS * 100, 2))
pred4 = DT4.predict(xtest_sm_sc)



cm3=metrics.confusion_matrix(ytest_sm, pred4, labels=[1, 0])



df_cm = pd.DataFrame(cm3, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm3 = sns.heatmap(df_cm, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 20 folds, and run the model 100 times. 



kfold = StratifiedKFold(n_splits=100)

cv_dt = cross_val_score(DT4, x_sc, y2, cv=kfold)

print("Cross Validation Score of Decision Tree Model: ", round(cv_dt.mean() * 100,2))
cv_dt1 = cv_dt.mean() 

f1_dt = metrics.f1_score(ytest_sm, pred4, average='micro')

tn_dt, fp_dt, fn_dt, tp_dt = metrics.confusion_matrix(ytest_sm, pred4).ravel()
print("Classification Report For Decision Tree classifier")

print("")

CR_DT = metrics.classification_report(ytest_sm, pred4, labels = [1,0])

print(CR_DT)
# with default hyperparameters

RFCL1 = RandomForestClassifier(random_state = 1)

RFCL1.fit(xtrain_sm_sc,ytrain_sm)

RFCL1_TR = RFCL1.score(xtrain_sm_sc,ytrain_sm)

RFCL1_TS = RFCL1.score(xtest_sm_sc,ytest_sm)

print("Random Forest Classifier Training score with default parameters : ", round(RFCL1_TR * 100, 2))

print("Random Forest Classifier Test score with default parameters     : ", round(RFCL1_TS * 100, 2))
RFC = RandomForestClassifier(random_state = 1)

hyperparm4 = {'n_estimators': np.arange(49,54), 'criterion':['gini','entropy'],'max_features': np.arange(5,8)}

# various Parameters range were tried to get score better than default parameters score



RF1_parms, RF1_score = parm_tune(RFC, hyperparm4, 20, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", RF1_parms)

print("\nBest Score: \n", RF1_score)
RFCL2 = RandomForestClassifier(n_estimators = 50, criterion = 'gini', max_features = 7, random_state=1)

RFCL2.fit(xtrain_sm_sc,ytrain_sm)

RFCL2_TR = RFCL2.score(xtrain_sm_sc,ytrain_sm)

RFCL2_TS = RFCL2.score(xtest_sm_sc,ytest_sm)

print("Random Forest Classifier Training score with best hyperparameters : ", round(RFCL2_TR * 100, 2))

print("Random Forest Classifier Test score with best hyperparameters     : ", round(RFCL2_TS * 100, 2))
pred6 = RFCL2.predict(xtest_sm_sc)



cm3=metrics.confusion_matrix(ytest_sm, pred6, labels=[1, 0])



df_cm = pd.DataFrame(cm3, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm3 = sns.heatmap(df_cm, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 20 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=50)

cv_rfcl = cross_val_score(RFCL2, x_sc, y2, cv=kfold)

print("Cross Validation Score of Random Forest Model: ", round(cv_rfcl.mean() * 100,2))
cv_rfcl1 = cv_rfcl.mean()

f1_rfcl = metrics.f1_score(ytest_sm, pred6, average='micro')

tn_rfcl, fp_rfcl, fn_rfcl, tp_rfcl = metrics.confusion_matrix(ytest_sm, pred6).ravel()
print("Classification Report For Random Forest classifier with best hyper parameters")

print("")

CR_RF2 = metrics.classification_report(ytest_sm, pred6, labels = [1,0])

print(CR_RF2)
BC1 = BaggingClassifier(max_samples =.7, n_jobs = -1)

BC1.fit(xtrain_sm_sc,ytrain_sm)

BC1_TR = BC1.score(xtrain_sm_sc,ytrain_sm)

BC1_TS = BC1.score(xtest_sm_sc,ytest_sm)

print("Bagging Classifier Training score with default parameters : ", round(BC1_TR * 100, 2))

print("Bagging Classifier Test score with default parameters     : ", round(BC1_TS * 100, 2))
BC2 = BaggingClassifier(max_samples =.7, n_jobs = -1)

hyperparm5 = {'n_estimators':[50, 100], 'max_features' : np.arange(8,10)}

# various Parameters range were tried to get score better than default parameters score



BC1_parms, BC1_score = parm_tune(BC2, hyperparm5, 20, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", BC1_parms)

print("\nBest Score: \n", BC1_score)
BC3 = BaggingClassifier(n_estimators = 100,max_features = 9, max_samples =.7, n_jobs = -1)

BC3.fit(xtrain_sm_sc,ytrain_sm)

BC3_TR = BC3.score(xtrain_sm_sc,ytrain_sm)

BC3_TS = BC3.score(xtest_sm_sc,ytest_sm)

print("Bagging Classifier Training score with best hyper parameters) : ", round(BC3_TR * 100, 2))

print("Bagging Classifier Test score with best hyper parameters)     : ", round(BC3_TS * 100, 2))
pred7 = BC3.predict(xtest_sm_sc)



cm3=metrics.confusion_matrix(ytest_sm, pred7, labels=[1, 0])



df_cm = pd.DataFrame(cm3, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm3 = sns.heatmap(df_cm, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 20 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=50)

cv_bc = cross_val_score(BC3, x_sc, y2, cv=kfold)

print("Cross Validation Score of Bagging Classifier Model: ", round(cv_bc.mean() * 100,2))
cv_bc1 = cv_bc.mean()

f1_bc = metrics.f1_score(ytest_sm, pred7, average='micro')

tn_bc, fp_bc, fn_bc, tp_bc = metrics.confusion_matrix(ytest_sm, pred7).ravel()
print("Classification Report For Bagging classifier ")

print("")

CR_BC = metrics.classification_report(ytest_sm, pred7, labels = [1,0])

print(CR_BC)
BC1_GNB = BaggingClassifier(base_estimator = GNM3, max_samples =.7, n_jobs = -1)

BC1_GNB.fit(xtrain_sm_sc,ytrain_sm)

BC1_GNB_TR = BC1_GNB.score(xtrain_sm_sc,ytrain_sm)

BC1_GNB_TS = BC1_GNB.score(xtest_sm_sc,ytest_sm)

print("Bagging Classifier Training score with default parameters (with underlying GNB) : ", round(BC1_GNB_TR * 100, 2))

print("Bagging Classifier Test score with default parameters (with underlying GNB)     : ", round(BC1_GNB_TS * 100, 2))
BC2_GNB = BaggingClassifier(base_estimator = GNM3, max_samples =.7, n_jobs = -1)

hyperparm6 = {'n_estimators':[40, 50], 'max_features' : np.arange(5,7)}

# various Parameters range were tried to get score better than default parameters score



BC2_GNB_parms, BC2_GNB_score = parm_tune(BC2_GNB, hyperparm6, 10, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", BC2_GNB_parms)
BC3_GNB = BaggingClassifier(base_estimator = GNM3, n_estimators = 40, max_features = 6, max_samples =.7, n_jobs = -1)

BC3_GNB.fit(xtrain_sm_sc,ytrain_sm)

BC3_GNB_TR = BC3_GNB.score(xtrain_sm_sc,ytrain_sm)

BC3_GNB_TS = BC3_GNB.score(xtest_sm_sc,ytest_sm)

print("Bagging Classifier Training score with best hyper parameters (with underlying GNB) : ", round(BC3_GNB_TR * 100, 2))

print("Bagging Classifier Test score with best hyper parameters (with underlying GNB)     : ", round(BC3_GNB_TS * 100, 2))
pred8 = BC3_GNB.predict(xtest_sm_sc)



cm3=metrics.confusion_matrix(ytest_sm, pred8, labels=[1, 0])



df_cm_bcgnm = pd.DataFrame(cm3, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize = (20,15))

plt.subplot(3,3,1)

hm2 = sns.heatmap(df_cm_bcgnm, annot=True, fmt='g')

bottom, top = hm2.get_ylim()

hm2.set_ylim (bottom +0.5, top - 0.5)

plt.title("Bagging Classifier with Gaussian Naive Bayes")



plt.subplot(3,3,3)

hm3 = sns.heatmap(df_cm_gnm, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)

plt.title("Gaussian Naive Bayes")
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 20 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=50)

cv_bcgnb = cross_val_score(BC3_GNB, x_sc, y2, cv=kfold)

print("Cross Validation Score of Bagging Classifier (with underlying Gaussian Naive Bayes): ", round(cv_bcgnb.mean() * 100,2))
cv_bcgn = cv_bcgnb.mean()

f1_bcgn = metrics.f1_score(ytest_sm, pred8, average='micro')

tn_bcgn, fp_bcgn, fn_bcgn, tp_bcgn = metrics.confusion_matrix(ytest_sm, pred8).ravel()
print("Classification Report For Bagging classifier( with GNB) ")

print("")

CR_BC_GNM = metrics.classification_report(ytest_sm, pred8, labels = [1,0])

print(CR_BC_GNM)
AC2 = AdaBoostClassifier(learning_rate=0.1)

AC2.fit(xtrain_sm_sc,ytrain_sm)

AC2_TR = AC2.score(xtrain_sm_sc,ytrain_sm)

AC2_TS = AC2.score(xtest_sm_sc,ytest_sm)

print("Adaboost Classifier Training score with default parameters : ", round(AC2_TR * 100, 2))

print("Adaboost Classifier Test score with default parameters     : ", round(AC2_TS * 100, 2))
# Tuning of AdaBoost Classifier



hyperparm7 = {'n_estimators':[25, 75, 100]}

AC1_parms, AC1_score = parm_tune(AC2, hyperparm7, 10, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", AC1_parms)
AC3 = AdaBoostClassifier(n_estimators = 100, learning_rate=0.1)

AC3.fit(xtrain_sm_sc,ytrain_sm)

AC3_TR = AC3.score(xtrain_sm_sc,ytrain_sm)

AC3_TS = AC3.score(xtest_sm_sc,ytest_sm)

print("Adaboost Classifier Training score with best parameters : ", round(AC3_TR * 100, 2))

print("Adaboost Classifier Test score with best parameters     : ", round(AC3_TS * 100, 2))
pred9 = AC3.predict(xtest_sm_sc)



cm3=metrics.confusion_matrix(ytest_sm, pred9, labels=[1, 0])



df_ac = pd.DataFrame(cm3, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm3 = sns.heatmap(df_ac, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 50 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=50)

cv_ac = cross_val_score(AC3, x_sc, y2, cv=kfold)

print("Cross Validation Score of Adaptive Boosting Classifier : ", round(cv_ac.mean() * 100,2))
cv_ac1 = cv_ac.mean() 

f1_ac = metrics.f1_score(ytest_sm, pred9, average='micro')

tn_ac, fp_ac, fn_ac, tp_ac = metrics.confusion_matrix(ytest_sm, pred9).ravel()
print("Classification Report For AdaBoost classifier ")

print("")

CR_AC = metrics.classification_report(ytest_sm, pred9, labels = [1,0])

print(CR_AC)
AC1_GNB = AdaBoostClassifier(base_estimator = GNM3,learning_rate=0.1)

AC1_GNB.fit(xtrain_sm_sc,ytrain_sm)

AC1_GNB_TR = AC1_GNB.score(xtrain_sm_sc,ytrain_sm)

AC1_GNB_TS = AC1_GNB.score(xtest_sm_sc,ytest_sm)

print("Adaboost Classifier Training score with default parameters (with underlying GNB) : ", round(AC1_GNB_TR * 100, 2))

print("Adaboost Classifier Test score with default parameters (with underlying GNB)     : ", round(AC1_GNB_TS * 100, 2))
hyperparm7 = {'n_estimators':[25, 75, 100]}

# various Parameters range were tried to get score better than default parameters score



AC1_GNB_parms, AC1_GNB_score = parm_tune(AC1_GNB, hyperparm7, 10, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", AC1_GNB_parms)
AC2_GNB = AdaBoostClassifier(base_estimator = GNM3,n_estimators = 100, learning_rate=0.1)

AC2_GNB.fit(xtrain_sm_sc,ytrain_sm)

AC2_GNB_TR = AC2_GNB.score(xtrain_sm_sc,ytrain_sm)

AC2_GNB_TS = AC2_GNB.score(xtest_sm_sc,ytest_sm)

print("Adaboost Classifier Training score with default parameters (with underlying GNB) : ", round(AC2_GNB_TR * 100, 2))

print("Adaboost Classifier Test score with default parameters (with underlying GNB)     : ", round(AC2_GNB_TS * 100, 2))
pred10 = AC2_GNB.predict(xtest_sm_sc)



cm3=metrics.confusion_matrix(ytest_sm, pred10, labels=[1, 0])



df_acgnb = pd.DataFrame(cm3, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm3 = sns.heatmap(df_acgnb, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 50 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=50)

cv_acgnb = cross_val_score(AC2_GNB, x_sc, y2, cv=kfold)

print("Cross Validation Score of Adaptive Boosting Classifier (With GNB) : ", round(cv_acgnb.mean() * 100,2))
cv_acgn = cv_acgnb.mean()

f1_acgn = metrics.f1_score(ytest_sm, pred10, average='micro')

tn_acgn, fp_acgn, fn_acgn, tp_acgn = metrics.confusion_matrix(ytest_sm, pred10).ravel()
print("Classification Report For AdaBoost classifier (with GNB) ")

print("")

CR_AC_GNB = metrics.classification_report(ytest_sm, pred10, labels = [1,0])

print(CR_AC_GNB)
GBC1 = GradientBoostingClassifier(learning_rate = 0.1)

GBC1.fit(xtrain_sm_sc,ytrain_sm)

GBC1_TR = GBC1.score(xtrain_sm_sc,ytrain_sm)

GBC1_TS = GBC1.score(xtest_sm_sc,ytest_sm)

print("Gradient Boost Classifier Training score with default parameters  : ", round(GBC1_TR * 100, 2))

print("Gradient Boost Classifier Test score with default parameters      : ", round(GBC1_TS * 100, 2))
hyperparm8 = {'n_estimators':[25, 50, 75], 'max_depth':np.arange(3, 7), 'max_features': np.arange(8,14)}





GBC1_parms, GBC1_score = parm_tune(GBC1, hyperparm8, 10, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", GBC1_parms)
GBC2 = GradientBoostingClassifier(n_estimators =75, max_depth = 5, max_features = 11, learning_rate = 0.1)

GBC2.fit(xtrain_sm_sc,ytrain_sm)

GBC2_TR = GBC2.score(xtrain_sm_sc,ytrain_sm)

GBC2_TS = GBC2.score(xtest_sm_sc,ytest_sm)

print("Gradient Boost Classifier Training score with best parameters  : ", round(GBC2_TR * 100, 2))

print("Gradient Boost Classifier Test score with best parameters      : ", round(GBC2_TS * 100, 2))
pred11 = GBC2.predict(xtest_sm_sc)



cm3=metrics.confusion_matrix(ytest_sm, pred11, labels=[1, 0])



df_gbc2 = pd.DataFrame(cm3, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm3 = sns.heatmap(df_gbc2, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 50 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=50)

cv_gc = cross_val_score(GBC2, x_sc, y2, cv=kfold)

print("Cross Validation Score of Gradient Boosting Classifier : ", round(cv_gc.mean() * 100,2))
cv_gc1 = cv_gc.mean()

f1_gc = metrics.f1_score(ytest_sm, pred11, average='micro')

tn_gc, fp_gc, fn_gc, tp_gc = metrics.confusion_matrix(ytest_sm, pred11).ravel()
print("Classification Report For Gradient Boosting classifier")

print("")

CR_GC = metrics.classification_report(ytest_sm, pred11, labels = [1,0])

print(CR_GC)
XGB1 = XGBClassifier(learning_rate =0.01, n_jobs = -1)

XGB1.fit(xtrain_sm_sc,ytrain_sm)

XGB1_TR = XGB1.score(xtrain_sm_sc,ytrain_sm)

XGB1_TS = XGB1.score(xtest_sm_sc,ytest_sm)

print("XGBoost Classifier Training score with default parameters  : ", round(XGB1_TR * 100, 2))

print("XGBoost Classifier Test score with default parameters      : ", round(XGB1_TS * 100, 2))
XGB1
hyperparm9 = {'n_estimators':[25, 50, 75], 'max_depth':np.arange(4, 8), 'gamma': np.arange(2,5)}





XGB1_parms, XGB1_score = parm_tune(XGB1, hyperparm9, 10, xtrain_sm_sc, ytrain_sm)

print("Best Parameters Identified: \n", XGB1_parms)
XGB2 = XGBClassifier(n_estimators = 50, max_depth = 7, gamma = 4, learning_rate =0.01, n_jobs = -1)

XGB2.fit(xtrain_sm_sc,ytrain_sm)

XGB2_TR = XGB2.score(xtrain_sm_sc,ytrain_sm)

XGB2_TS = XGB2.score(xtest_sm_sc,ytest_sm)

print("XGBoost Classifier Training score with default parameters  : ", round(XGB2_TR * 100, 2))

print("XGBoost Classifier Test score with default parameters      : ", round(XGB2_TS * 100, 2))
pred12 = XGB2.predict(xtest_sm_sc)



cm3=metrics.confusion_matrix(ytest_sm, pred12, labels=[1, 0])

df_xgb2 = pd.DataFrame(cm3, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

hm3 = sns.heatmap(df_xgb2, annot=True, fmt='g')

bottom, top = hm3.get_ylim()

hm3.set_ylim (bottom +0.5, top - 0.5)
# StratifiedKfold is used to maintain the original data ratio in each folds. 

# Below code will split the data in 50 folds, and run the model 20 times. 



kfold = StratifiedKFold(n_splits=50)

cv_xgb = cross_val_score(XGB2, x_sc, y2, cv=kfold)

print("Cross Validation Score of XGB Classifier : ", round(cv_xgb.mean() * 100,2))
cv_xgb1 = cv_xgb.mean()

f1_xgb = metrics.f1_score(ytest_sm, pred12, average='micro')

tn_xgb, fp_xgb, fn_xgb, tp_xgb = metrics.confusion_matrix(ytest_sm, pred12).ravel()
print("Classification Report For XGB classifier")

print("")

CR_XGB = metrics.classification_report(ytest_sm, pred12, labels = [1,0])

print(CR_XGB)
# fpr, tpr and thresholds are caluclated for the all the models



lr_probs = log_reg.predict_proba(xtest_sm_sc)

fpr1, tpr1, thres1 = roc_curve(ytest_sm,lr_probs[:,1])

roc_auc1 = auc(fpr1,tpr1)

print("Logistic Regression - Area under the ROC curve : ", roc_auc1)



knn_probs = KNN3.predict_proba(xtest_sm_sc)

fpr2, tpr2, thres2 = roc_curve(ytest_sm,knn_probs[:,1])

roc_auc2 = auc(fpr2,tpr2)

print("KNN - Area under the ROC curve                 : ", roc_auc2)



nb_probs = GNM3.predict_proba(xtest_sm_sc)

fpr3, tpr3, thres3 = roc_curve(ytest_sm,nb_probs[:,1])

roc_auc3 = auc(fpr3,tpr3)

print("Naive Bayes - Area under the ROC curve         : ", roc_auc3)



dt_probs = DT4.predict_proba(xtest_sm_sc)

fpr4, tpr4, thres4 = roc_curve(ytest_sm,dt_probs[:,1])

roc_auc4 = auc(fpr4,tpr4)

print("Decision Tree - Area under the ROC curve       : ", roc_auc4)



rfcl_probs = RFCL2.predict_proba(xtest_sm_sc)

fpr5, tpr5, thres5 = roc_curve(ytest_sm,rfcl_probs[:,1])

roc_auc5 = auc(fpr5,tpr5)

print("Random Forest - Area under the ROC curve       : ", roc_auc5)



bc_probs = BC3.predict_proba(xtest_sm_sc)

fpr6, tpr6, thres6 = roc_curve(ytest_sm,bc_probs[:,1])

roc_auc6 = auc(fpr6,tpr6)

print("Bagging Classifier - Area under the ROC curve  : ", roc_auc6)



bcgnb_probs = BC3_GNB.predict_proba(xtest_sm_sc)

fpr7, tpr7, thres7 = roc_curve(ytest_sm,bcgnb_probs[:,1])

roc_auc7 = auc(fpr7,tpr7)

print("Bagging Classifier (With GNB) - Area under the ROC curve  : ", roc_auc7)



ac_probs = AC3.predict_proba(xtest_sm_sc)

fpr8, tpr8, thres8 = roc_curve(ytest_sm,ac_probs[:,1])

roc_auc8 = auc(fpr8,tpr8)

print("AdaBoost Classifier - Area under the ROC curve            : ", roc_auc8)

  

acgnb_probs = AC2_GNB.predict_proba(xtest_sm_sc)

fpr9, tpr9, thres9 = roc_curve(ytest_sm,acgnb_probs[:,1])

roc_auc9 = auc(fpr9,tpr9)

print("AdaBoost Classifier (With GNB) - Area under the ROC curve : ", roc_auc9)



gbc_probs = GBC2.predict_proba(xtest_sm_sc)

fpr10, tpr10, thres10 = roc_curve(ytest_sm,gbc_probs[:,1])

roc_auc10 = auc(fpr10,tpr10)

print("GradientBoost Classifier - Area under the ROC curve       : ", roc_auc10)



xgb_probs = XGB2.predict_proba(xtest_sm_sc)

fpr11, tpr11, thres11 = roc_curve(ytest_sm,xgb_probs[:,1])

roc_auc11 = auc(fpr11,tpr11)

print("XGBoost Classifier - Area under the ROC curve             : ", roc_auc11)
# Plot the ROC Plot

plt.clf()

plt.figure(figsize=(10,7))

plt.plot(fpr1,tpr1, label='Logistic Regression (area = %0.2f)' % roc_auc1)

plt.plot(fpr2,tpr2, label='KNN (area = %0.2f)' % roc_auc2)

plt.plot(fpr3,tpr3, label='Gaussian NB (area = %0.2f)' % roc_auc3)

plt.plot(fpr4,tpr4, label='Decision Tree (area = %0.2f)' % roc_auc4)

plt.plot(fpr5,tpr5, label='Random Forest (area = %0.2f)' % roc_auc5)

plt.plot(fpr6,tpr6, label='Bagging Classifier (area = %0.2f)' % roc_auc6)

plt.plot(fpr7,tpr7, label='Bagging Cla (With GNB)  (area = %0.2f)' % roc_auc7)

plt.plot(fpr8,tpr8, label='AdaBoost Classifier (area = %0.2f)' % roc_auc8)

plt.plot(fpr9,tpr9, label='AdaBoost Cla (With GNB) (area = %0.2f)' % roc_auc9)

plt.plot(fpr10,tpr10, label='GradientBoost Classifier (area = %0.2f)' % roc_auc10)

plt.plot(fpr11,tpr11, label='XGBoost Classifier (area = %0.2f)' % roc_auc11)

#plt.plot(fpr1,tpr1)

plt.plot([0,1],[0,1],'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel("False Postive Rate",fontsize=16)

plt.ylabel("True Postive Rate",fontsize=16)

plt.title("ROC Plot To Compare Models",fontsize=16)

plt.legend(loc="lower right")

plt.show()
# Loading the values in a dataframe



mod_dict = {'Model': ['Log_Reg', 'KNN', 'Gaussian NB', 'Decision Tree', 'Random Forest', 'Bagging', 'Bagging GNB', 'AdaBoost', 

                   'AdaBoost GNB', 'Gradient Boost','XGBoost'], 

     'Train_Score':[LR3_TR, KNN_TR, GNM_TR, DT4_TR, RFCL2_TR, BC3_TR, BC3_GNB_TR, AC3_TR, AC2_GNB_TR, GBC2_TR,XGB2_TR], 

     'Test_Score':[LR3_TS, KNN_TS, GNM_TS, DT4_TS, RFCL2_TS, BC3_TS, BC3_GNB_TS, AC3_TS, AC2_GNB_TS, GBC2_TS, XGB2_TS],

     'Cross_Val_Score': [cv_log, cv_knn1, cv_gnm1, cv_dt1, cv_rfcl1, cv_bc1, cv_bcgn, cv_ac1, cv_acgn, cv_gc1, cv_xgb1], 

     'F1_score':[f1_log, f1_knn, f1_gnm, f1_dt, f1_rfcl, f1_bc, f1_bcgn, f1_ac, f1_acgn, f1_gc, f1_xgb]}

     

df_mod = pd.DataFrame(mod_dict)



df_mod = df_mod.set_index('Model')

df_mod
# Plot the bar chart



ax = df_mod.plot(kind='bar', rot=0, figsize = (27,14), fontsize = 16,colormap='Paired')

ax.legend(bbox_to_anchor=(1, 1), prop={'size': 15})

ax.set_ylabel("Value", fontsize=30)

ax.set_xlabel("Models", fontsize=30)

plt.show()
# Loading the values in a dataframe



mod_dict2 = {'Model': ['Log_Reg', 'KNN', 'Gaussian NB', 'Decision Tree', 'Random Forest', 'Bagging', 'Bagging GNB', 'AdaBoost', 

                   'AdaBoost GNB', 'Gradient Boost','XGBoost'], 

     'True_Positive':[tp_log, tp_knn, tp_gnm, tp_dt, tp_rfcl, tp_bc, tp_bcgn, tp_ac, tp_acgn, tp_gc, tp_xgb], 

     'False_Negative':[fn_log, fn_knn, fn_gnm, fn_dt, fn_rfcl, fn_bc, fn_bcgn, fn_ac, fn_acgn, fn_gc, fn_xgb],

     'True_Negative': [tn_log, tn_knn, tn_gnm, tn_dt, tn_rfcl, tn_bc, tn_bcgn, tn_ac, tn_acgn, tn_gc, tn_xgb], 

     'False_Positive':[fp_log, fp_knn, fp_gnm, fp_dt, fp_rfcl, fp_bc, fp_bcgn, fp_ac, fp_acgn, fp_gc, fp_xgb]}

     

df_mod2 = pd.DataFrame(mod_dict2)



df_mod2 = df_mod2.set_index('Model')

df_mod2
# Plot the bar chart



ax2 = df_mod2.plot(kind='bar', rot=0, figsize = (27,14), fontsize = 16,colormap='Paired')

ax2.legend(bbox_to_anchor=(1, 1), prop={'size': 15})

ax2.set_ylabel("Value", fontsize=30)

ax2.set_xlabel("Models", fontsize=30)

plt.show()