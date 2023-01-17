import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
pima=pd.read_csv('../input/diabetes.csv')
pima.head()
pima.info()
plt.figure(figsize=(8,5 ))
sns.heatmap(pima.isnull(),yticklabels=False,cbar=False,cmap='viridis')
pima['Outcome'].value_counts()
sns.pairplot(pima,hue='Outcome')
pima.describe()
corrmat= pima.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corrmat, vmax=.8, square=True, annot= True)
pima.columns
pima.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))

# Set up the matplotlib figure
f, axes = plt.subplots(nrows=4,ncols=2, figsize=(15, 20))

# Graph pregnancies
sns.distplot(pima.Pregnancies, kde=False, color="g", ax=axes[0][0]).set_title('Pregnanices')
axes[0][0].set_ylabel('Count')

# Graph Glucose
sns.distplot(pima.Glucose, kde=False, color="r", ax=axes[0][1]).set_title('Glucose')
axes[0][1].set_ylabel('Count')

# Graph Blood Pressure
sns.distplot(pima.BloodPressure, kde=False, color="b", ax=axes[1][0]).set_title('Blood Pressure')
axes[1][0].set_ylabel('Count')

# Graph Skin Thickness
sns.distplot(pima.SkinThickness, kde=False, color="g", ax=axes[1][1]).set_title('Skin Thickness')
axes[1][1].set_ylabel('Count')

# Graph Insulin
sns.distplot(pima.Insulin, kde=False, color="r", ax=axes[2][0]).set_title('Insulin')
axes[2][0].set_ylabel('Count')

# Graph BMI
sns.distplot(pima.BMI, kde=False, color="b", ax=axes[2][1]).set_title('BMI')
axes[2][1].set_ylabel('Count')

# Graph Diabetes Pedigree function
sns.distplot(pima.DiabetesPedigreeFunction, kde=False, color="g", ax=axes[3][0]).set_title('DiabetesPedigreeFunction')
axes[3][0].set_ylabel('Count')

# Graph Age
sns.distplot(pima.Age, kde=False, color="r", ax=axes[3][1]).set_title('Age')
axes[3][1].set_ylabel('Count')

#Removing outliers 
pima_new=pima
pima_new.info()
# Removing Outliers in the data based on Box Plots 
pima_new = pima_new[pima_new["Pregnancies"] <13]
pima_new = pima_new[(pima_new["Glucose"] > 30)]
#pima_new = pima_new[(pima_new['BloodPressure'] > 26) & (pima_new['BloodPressure'] <105) ]
#pima_new = pima_new[pima_new['Insulin'] < 300]
pima_new = pima_new[pima_new['BMI'] > 10]  
pima_new =pima_new[pima_new['BMI'] <50]
pima_new = pima_new[pima_new['DiabetesPedigreeFunction'] < 1.2]
pima_new = pima_new[pima_new['Age'] < 65]
pima_new.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'Pregnancies'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'Pregnancies'] , color='r',shade=True, label='Yes')
ax.set(xlabel='Pregnancies', ylabel='Frequency')
plt.title('Pregnancies vs Yes or No')
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'Glucose'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'Glucose'] , color='r',shade=True, label='Yes')
ax.set(xlabel='Glucose', ylabel='Frequency')
plt.title('Glucose vs Yes or No')
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'BMI'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'BMI'] , color='r',shade=True, label='Yes')
ax.set(xlabel='BMI', ylabel='Frequency')
plt.title('BMI vs Yes or No')
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'DiabetesPedigreeFunction'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'DiabetesPedigreeFunction'] , color='r',shade=True, label='Yes')
ax.set(xlabel='DiabetesPedigreeFunction', ylabel='Frequency')
plt.title('DiabetesPedigreeFunction vs Yes or No')
# Kernel Density Plot
fig = plt.figure(figsize=(15,4),)
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 0),'Age'] , color='b',shade=True,label='No')
ax=sns.kdeplot(pima_new.loc[(pima_new['Outcome'] == 1),'Age'] , color='r',shade=True, label='Yes')
ax.set(xlabel='Age', ylabel='Frequency')
plt.title('Age vs Yes or No')
# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pima_new.drop('Outcome',axis=1), 
                                                    pima_new['Outcome'], test_size=0.30, 
                                                    random_state=123)
#Feature Scaling
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
# Cross Validation function
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn import metrics
#Defining a Cross validation function
#n_folds = 10
def classification_cv(model):
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    scoring = 'accuracy'
    results = model_selection.cross_val_score(model, X_train_transformed, y_train, cv=kfold, scoring=scoring)
    return(print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std())))
#Feature importance
#decision tree classifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12,6)

# Create train and test splits

dtree = tree.DecisionTreeClassifier(
    #max_depth=3,
    class_weight="balanced",
    min_weight_fraction_leaf=0.01
    )
dtree = dtree.fit(X_train_transformed,y_train)

## plot the importances ##
importances = dtree.feature_importances_
feat_names = pima_new.drop(['Outcome'],axis=1).columns


indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='blue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()
# Creating base rate model
def base_rate_model(X) :
    y = np.zeros(X.shape[0])
    return y
# Check accuracy of base rate model-- same as percentage of majority class
y_base_rate = base_rate_model(X_test_transformed)
from sklearn.metrics import accuracy_score
print ("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

print ("---Base Model---")
base_roc_auc = roc_auc_score(y_test, y_base_rate)
print ("Base Rate AUC = %2.2f" % base_roc_auc)
print(classification_report(y_test, y_base_rate))
print ("---Confusion Matrix---")
print(confusion_matrix(y_test, y_base_rate))
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression(penalty='l2', C=0.3,class_weight = "balanced")
#Cross validation Holdout method for learning
Logistic_regression_cv=classification_cv(logis)
Logistic_regression_cv
logis.fit(X_train_transformed, y_train)
print ("\n\n ---Logistic Model---")
logit_roc_auc = roc_auc_score(y_test, logis.predict(X_test_transformed))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print(classification_report(y_test, logis.predict(X_test_transformed)))
print(confusion_matrix(y_test, logis.predict(X_test_transformed)))
print ("Logistic regression accuracy is %2.2f" % accuracy_score(y_test, logis.predict(X_test_transformed) ))
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# x_train,x_test,y_train,y_test = cross_validation.train_test_split(x,y,test_size=0.4,random_state=0)
depth = []
for i in range(3,9):
    dtree = tree.DecisionTreeClassifier(max_depth=i,criterion='gini',class_weight="balanced",min_weight_fraction_leaf=0.01)
    # Perform 7-fold cross validation 
    scores = cross_val_score(estimator=dtree, X=X_train_transformed, y=y_train, cv=7, n_jobs=4)
    depth.append((i,scores.mean()))
print(depth)
dtree = tree.DecisionTreeClassifier(max_depth=4,criterion='gini',class_weight="balanced",min_weight_fraction_leaf=0.01)
dtree.fit(X_train_transformed, y_train)
print ("\n\n ---Decision Tree  Model---")
dtree_roc_auc = roc_auc_score(y_test, dtree.predict(X_test_transformed))
print ("Decision tree AUC = %2.2f" % dtree_roc_auc)
print(classification_report(y_test, dtree.predict(X_test_transformed)))
print(confusion_matrix(y_test, dtree.predict(X_test_transformed)))
print ("Decision Tree accuracy is %2.2f" % accuracy_score(y_test, dtree.predict(X_test_transformed) ))
from sklearn.ensemble import RandomForestClassifier
# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100, 
    max_depth=6, 
    min_samples_split=10, 
    class_weight="balanced",
    random_state=100
    )
#cross validation Random Forest
classification_cv(rf)

rf.fit(X_train_transformed, y_train)
print ("\n\n ---Random Forest  Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test_transformed))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test_transformed)))
print(confusion_matrix(y_test, rf.predict(X_test_transformed)))
print ("Random Forest is %2.2f" % accuracy_score(y_test, rf.predict(X_test_transformed) ))
# Ada Boost
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=4),n_estimators=400, learning_rate=0.1,random_state=100)

classification_cv(ada)
# evaluation ADA boost
ada.fit(X_train_transformed, y_train)
print ("\n\n ---AdaBoost Model---")
ada_roc_auc = roc_auc_score(y_test, ada.predict(X_test_transformed))
print ("AdaBoost AUC = %2.2f" % ada_roc_auc)
print(classification_report(y_test, ada.predict(X_test_transformed)))
confusion_matrix(y_test, ada.predict(X_test_transformed))
print(confusion_matrix(y_test, ada.predict(X_test_transformed)))
print ("ADA boost is %2.2f" % accuracy_score(y_test, ada.predict(X_test_transformed)))
from sklearn.svm import SVC
clf = SVC(kernel="linear", C=0.35, probability=True, random_state=100)
classification_cv(clf)
clf.fit(X_train_transformed, y_train)
print ("\n\n ---SVM---")
svc_roc_auc = roc_auc_score(y_test, clf.predict(X_test_transformed))
print ("SVM AUC = %2.2f" % svc_roc_auc)
print(classification_report(y_test, clf.predict(X_test_transformed)))
print(confusion_matrix(y_test, clf.predict(X_test_transformed)))
print ("SVM accuracy is %2.2f" % accuracy_score(y_test, clf.predict(X_test_transformed) ))
# Gradient Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=400,learning_rate=0.1,random_state=100,max_features=4 )
gbc.fit(X_train_transformed, y_train)
print ("\n\n ---GBC---")
gbc_roc_auc = roc_auc_score(y_test, gbc.predict(X_test_transformed))
print ("GBC AUC = %2.2f" % gbc_roc_auc)
print(classification_report(y_test, gbc.predict(X_test_transformed)))
print(confusion_matrix(y_test, gbc.predict(X_test_transformed)))
print ("GBC accuracy is %2.2f" % accuracy_score(y_test, gbc.predict(X_test_transformed) ))
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('lr', logis), ('svc', clf),('tree',dtree)], voting='soft', weights=[1,1,1])
eclf.fit(X_train_transformed, y_train)

print ("\n\n ---ECLF---")
eclf_roc_auc = roc_auc_score(y_test, eclf.predict(X_test_transformed))
print ("ECLF AUC = %2.2f" % eclf_roc_auc)
print(classification_report(y_test, eclf.predict(X_test_transformed)))
print(confusion_matrix(y_test, eclf.predict(X_test_transformed)))
print ("ECLF accuracy is %2.2f" % accuracy_score(y_test, eclf.predict(X_test_transformed) ))
# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, logis.predict_proba(X_test_transformed)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test_transformed)[:,1])
dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(X_test_transformed)[:,1])
ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada.predict_proba(X_test_transformed)[:,1])
svc_fpr, svc_tpr, svc_thresholds = roc_curve(y_test, clf.predict_proba(X_test_transformed)[:,1])
eclf_fpr, eclf_tpr, eclf_thresholds = roc_curve(y_test, eclf.predict_proba(X_test_transformed)[:,1])



plt.figure(figsize=(15, 10))

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot Decision Tree ROC
plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dtree_roc_auc)

# Plot AdaBoost ROC
plt.plot(ada_fpr, ada_tpr, label='AdaBoost (area = %0.2f)' % ada_roc_auc)

# Plot SVM classifier ROC
plt.plot(svc_fpr, svc_tpr, label='SVM (area = %0.2f)' % svc_roc_auc)

# Plot Ensemble classifier ROC
plt.plot(eclf_fpr, eclf_tpr, label='Ensemble (area = %0.2f)' % eclf_roc_auc)



# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
pima.info()
# Train test split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(pima.drop('Outcome',axis=1), 
                                                    pima['Outcome'], test_size=0.30, 
                                                    random_state=123)
# Cross validation function
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'
results = model_selection.cross_val_score(rf, X_train_1, y_train_1, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
rf.fit(X_train_1, y_train_1)
print ("\n\n ---Random Forest  Model---")
rf_roc_auc = roc_auc_score(y_test_1, rf.predict(X_test_1))
print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test_1, rf.predict(X_test_1)))
print(confusion_matrix(y_test_1, rf.predict(X_test_1)))
print ("Random Forest is %2.2f" % accuracy_score(y_test_1, rf.predict(X_test_1) ))
# Cross validation function
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'
results = model_selection.cross_val_score(gbc, X_train_1, y_train_1, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
gbc.fit(X_train_1, y_train_1)
print ("\n\n ---Gradient Boosting Model---")
gbc_roc_auc = roc_auc_score(y_test_1, gbc.predict(X_test_1))
print ("Gradient Boosting AUC = %2.2f" % gbc_roc_auc)
print(classification_report(y_test_1, gbc.predict(X_test_1)))
print(confusion_matrix(y_test_1, gbc.predict(X_test_1)))
print ("Gradient Boosting Accuracy is %2.2f" % accuracy_score(y_test_1, gbc.predict(X_test_1) ))
eclf_1 = VotingClassifier(estimators=[('rf', rf), ('gbc', gbc)], voting='soft', weights=[1,2])
eclf_1.fit(X_train_1, y_train_1)

print ("\n\n ---Ensembled Gradient Boost and Random Forest---")
eclf_1_roc_auc = roc_auc_score(y_test_1, eclf_1.predict(X_test_1))
print (" AUC = %2.2f" % eclf_1_roc_auc)
print(classification_report(y_test_1, eclf_1.predict(X_test_1)))
print(confusion_matrix(y_test_1, eclf_1.predict(X_test_1)))
print (" accuracy is %2.2f" % accuracy_score(y_test_1, eclf_1.predict(X_test_1) ))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test_1, rf.predict_proba(X_test_1)[:,1])
gbc_fpr, gbc_tpr, gbc_thresholds = roc_curve(y_test_1, gbc.predict_proba(X_test_1)[:,1])
eclf_1_fpr, eclf_1_tpr, eclf_1_thresholds = roc_curve(y_test_1, eclf_1.predict_proba(X_test_1)[:,1])

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot Ensembled GBC ROC
plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting(area = %0.2f)' % gbc_roc_auc)

# Plot Ensembled GBC and RF ROC
plt.plot(eclf_1_fpr, eclf_1_tpr, label='Ensembled (area = %0.2f)' % eclf_1_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

