#Importing the necessary libraries 



import pandas as pd

import numpy as np

import seaborn as sns



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc, roc_auc_score



import matplotlib.pyplot as plt

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()



%matplotlib inline
df = pd.read_csv('../input/Ensemble Techniques.csv')   #Read the datafile as dataframe
df_shape = df.shape                                          #View the shape of the dataset

print('Data has Rows {}, Columns {}'.format(*df_shape)) 
df.info()                                                    #Understand the various attributes of the dataset
df.apply(lambda x: sum(x.isnull()))              #Check if the dataset has any missing values
df.describe().T                                    #Get the 5-point summary of data
# Getting the distribution of 'age' variable

fig, ax = plt.subplots()

fig.set_size_inches(20,8)



sns.countplot(df['age'])

plt.title('Age Distribution')

plt.show()



# Trying to understand the client's minimum and maximum age 

print('Maximum Age of client:', df['age'].max())

print('Minimum Age of client:', df['age'].min())

print('Minimum Age of client:', df['age'].mean())

print('\n')

# Let's see if there too much variance in the variable

print('The Mean is:', df['age'].mean())

print('The Standard Deviation is:', df['age'].std())

print('The Coefficient of variation is:', round(df['age'].std()*100 / df['age'].mean()))
plt.figure(figsize=(10, 8))



df.plot.scatter('age', 'balance')

plt.title('Relationship between Age and Balance variable')



plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(10,8)



sns.countplot(df['housing'], hue=df['default'], palette="ch:.25")



plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(20,8)



sns.countplot(df['day'])

plt.title('Day of last contact')



plt.show()
plt.figure(figsize=(10, 8))



df.plot.scatter('age', 'duration')

plt.title('Relationship between Age and Duration variable')



plt.show()
plt.figure(figsize=(20,8)) 

sns.countplot(df['month'], hue=df['Target'], palette="Set3")

plt.title('fig 1: Distribution of success rate for the year')



plt.figure(figsize=(20,10)) 

sns.countplot(df['job'], hue=df['Target'])

plt.title('fig 2: Understanding Employment data')



plt.show()
plt.figure(figsize=(10, 8))



sns.distplot(df['duration'])

plt.show()
plt.figure(figsize=(15,8))



sns.scatterplot(x=df['campaign'], y=df['duration'], hue=df['Target'])



plt.show()
plt.figure(figsize= (25,25))

plt.subplot(3,3,1)

sns.countplot(df.education, hue=df['default'], color='red', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Education')

plt.title('Education vs Default')



plt.subplot(3,3,2)

sns.countplot(df.marital, hue=df['default'], color='red', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Marital')

plt.title('Marital Status vs Defaults')



plt.subplot(3,3,3)

sns.countplot(df.loan, hue=df['default'], color='red', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Loan')

plt.title('Loan vs Default')



plt.subplot(3,3,4)

sns.countplot(df.housing, hue=df['default'], color='red', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Housing')

plt.title('Housing vs Default')



plt.show()
plt.figure(figsize= (25,25))

plt.subplot(3,3,1)

sns.countplot(df.education, hue=df['Target'], color='red', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Education')

plt.title('Fig 1: Education vs Target')



plt.subplot(3,3,2)

sns.countplot(df.marital, hue=df['Target'], color='red', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Marital')

plt.title('Fig 2:Marital Status vs Target')



plt.subplot(3,3,3)

sns.countplot(df.loan, hue=df['Target'], color='red', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Loan')

plt.title('Fig 3: Loan vs Target')



plt.subplot(3,3,4)

sns.countplot(df.housing, hue=df['Target'], color='red', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Housing')

plt.title('Fig 4: Housing vs Target')



plt.show()
plt.figure(figsize=(20,20))

plt.subplot(3,3,1)

sns.countplot(df.education, hue=df['loan'], color='olive', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Education')

plt.title('Education vs Loan')



plt.subplot(3,3,2)

sns.countplot(df.marital, hue=df['loan'], color='olive', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Marital Status')

plt.title('Marital Status vs Loan')



plt.subplot(3,3,3)

sns.countplot(df.housing, hue=df['loan'], color='olive', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Housing')

plt.title('Housing vs Loan')



plt.show()
plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

sns.countplot(df.marital, color='deepskyblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Marital')

plt.title('Marital Data')



plt.subplot(3,3,2)

sns.countplot(df.education, color='deepskyblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Education')

plt.title('Education Data')



plt.show()
sns.countplot(df['Target'], palette='dark')



print('Total number of Targets:', df['Target'].count())

print('Number of Target said yes:', df[df['Target'] == 'yes'] ['age'].count())

print('Number of Target said no:', df[df['Target'] == 'no'] ['age'].count())
plt.figure(figsize=(10,8))



df['Target'].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', colors='cyan')

plt.title('Subcriptions Data')

plt.ylabel('Conditions of loans')

plt.show()
from sklearn.preprocessing import LabelEncoder      # Converting the labels into numeric form so as make it machine-learing form



job_enc = LabelEncoder()

df['job'] = job_enc.fit_transform(df['job']) 



marital_enc = LabelEncoder()

df['marital'] = marital_enc.fit_transform(df['marital'])



education_enc = LabelEncoder()

df['education'] = education_enc.fit_transform(df['education']) 



default_enc = LabelEncoder()

df['default'] = default_enc.fit_transform(df['default'])



housing_enc = LabelEncoder()

df['housing'] = housing_enc.fit_transform(df['housing'])



df['contact'] = df['contact'].fillna(df['contact'].mode())



contact_enc = LabelEncoder()

df['contact'] = contact_enc.fit_transform(df['contact'])



loan_enc = LabelEncoder()

df['loan'] = loan_enc.fit_transform(df['loan'])



Target_enc = LabelEncoder()

df['Target'] = Target_enc.fit_transform(df['Target']) 
#Assinging the value to each month in an year so it makes us easy to analyze



df.month[df.month == 'jan'] = 1

df.month[df.month == 'feb'] = 2

df.month[df.month == 'mar'] = 3

df.month[df.month == 'apr'] = 4

df.month[df.month == 'may'] = 5

df.month[df.month == 'jun'] = 6

df.month[df.month == 'jul'] = 7

df.month[df.month == 'aug'] = 8

df.month[df.month == 'sep'] = 9

df.month[df.month == 'oct'] = 10

df.month[df.month == 'nov'] = 11

df.month[df.month == 'dec'] = 12



df = df
#Even here we are assigning the value to each status of he 'poutcome' variable



df.poutcome[df.poutcome == 'failure'] = 0

df.poutcome[df.poutcome == 'success'] = 1

df.poutcome[df.poutcome == 'unknown'] = 'other'

df.poutcome[df.poutcome == 'other'] = 2
plt.figure(figsize=(20,20))



plt.subplot(3,3,1)

sns.boxplot(df.education) 



plt.subplot(3,3,2)

sns.boxplot(df.balance) 



plt.subplot(3,3,3)

sns.boxplot(df.day) 



plt.subplot(3,3,4)

sns.boxplot(df.duration)



plt.subplot(3,3,5)

sns.boxplot(df.campaign) 



plt.subplot(3,3,6)

sns.boxplot(df.pdays)



plt.subplot(3,3,7)

sns.boxplot(df.previous)
plt.subplots(figsize=(12,12))                  #Checking the correlation metrics between independent and dependent variables

sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')

plt.show()
X = df.drop(['Target'], axis=1)                         #Creating variable 'X' for independent features

y = df['Target']                                          #Creating variable 'y' for independent features
#Before we proceed in removing outliers we need to make sure that all our attributes are in integer format 



#df.info()                              
df = df.astype({"poutcome":'int32'})         #Converting 'poutcome' attribute to integer

df = df.astype({"month":'int32'})            #Converting the 'month' attribute to integer
# It first computes the zscore for each value in the column, relative to the column mean and standard deviation

# Then it consideres the absolute z-score, only if it is below the threshold

# axis =1 applying to each row, all columns satisfy the constraint



from scipy import stats               #Importing the 'stats' as I need zscore librariy to set the threshold 

df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]    
from sklearn.preprocessing import StandardScaler
df_std = StandardScaler()
X = df_std.fit_transform(X)
#from sklearn.model_selection import StratifiedKFold



#skf = StratifiedKFold(n_splits=6)

#skf.get_n_splits(X, y)



#for train_index, test_index in skf.split(X, y):

    #print("TRAIN:", train_index, "TEST:", test_index)

    #X_train, X_test = X[train_index], X[test_index]

    #y_train, y_test = y[train_index], y[test_index]
from sklearn.model_selection import RepeatedKFold



rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=None)



for train_index, test_index in rkf.split(X):

    print("Train:", train_index, "Validation:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]
from sklearn import svm



svm = svm.SVC(probability=True, gamma='auto')

svm_model = svm.fit(X_train, y_train)



svm_pred = svm_model.predict(X_test)
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)



print('Accuracy', metrics.accuracy_score(y_test, svm_pred))

print('F1 core', metrics.f1_score(y_test, svm_pred))

print('\n')                                                                                                                

print('Scoring on Training Dataset', svm.score(X_train, y_train))

print('Scoring on Test Dataset', svm.score(X_test, y_test))

print('\n')                                                                                                                

print('Confusion matrix \n',confusion_matrix(y_test, svm_pred))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, svm_model.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, svm_model.predict(X)))



SVM = (cross_val_score(svm_model, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
from sklearn.naive_bayes import GaussianNB



naive_model = GaussianNB()

NB_model = naive_model.fit(X_train, y_train)

NB_predict = naive_model.predict(X_test)



print('Accuracy Score for Naive Bayes Classifier is:', metrics.accuracy_score(y_test, NB_predict))

print('F1 Score for Naive Bayes Classifier is:', metrics.f1_score(y_test, NB_predict))

print('\n')                                                                                                                

print('Scoring on Training Data:', naive_model.score(X_train, y_train)) 

print('Scoring on Test Data:', naive_model.score(X_test, y_test)) 

print('\n')                                                                                                                                                                                                                               

print('Confusion matrix', '\n', confusion_matrix(y_test, NB_predict))

print('\n')                                                                                                                



false_positive_rate, true_positive_rate, thresholds = roc_curve(y, NB_model.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, NB_model.predict(X)))



Gausian = (cross_val_score(naive_model, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
from sklearn.neighbors import KNeighborsClassifier
error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',

        markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate') 
#FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1



knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)

print('**Accuracy for KNN classifier is:', metrics.accuracy_score(y_test, pred))

print('**F1_score for KNN classifier is:', metrics.f1_score(y_test, pred))
#Now with K=5



KNN_model = KNeighborsClassifier(n_neighbors=32, weights = 'distance')

knn_model = KNN_model.fit(X_train, y_train)

knn_predict = KNN_model.predict(X_test)



print('**Accuracy for KNN classifier is:', metrics.accuracy_score(y_test, knn_predict))

print('**F1_score for KNN classifier is:', metrics.f1_score(y_test, knn_predict))

print('\n')

print('Scoring on Training Dataset is:', KNN_model.score(X_train, y_train))

print('Scoring on Test Dataset is:', KNN_model.score(X_test, y_test))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, knn_model.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, knn_model.predict(X)))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, knn_predict))



knn = (cross_val_score(KNN_model, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
from sklearn.linear_model import LogisticRegression



logit_model = LogisticRegression()

LG_model = logit_model.fit(X_train, y_train)



logit_predict = logit_model.predict(X_test)



print('Accuracy for Logistic Regression model is:', metrics.accuracy_score(y_test, logit_predict))

print('F1_score for Logistic Regression model is:', metrics.f1_score(y_test, logit_predict))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, logit_predict))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, LG_model.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, LG_model.predict(X)))

print('\n')

print('Scoring on Training Data:', logit_model.score(X_train, y_train)) 

print('Scoring on Test Data:', logit_model.score(X_test, y_test)) 



LogitRegression = (cross_val_score(logit_model, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb_model = xgb.fit(X_train, y_train)

xgb_predict = xgb.predict(X_test)



print('**Scoring on Training Data:', xgb.score(X_train, y_train)) 

print('**Scoring on Test Data:',xgb.score(X_test, y_test))

print('\n')

print('**Accuracy for XGBoost classifier is:', metrics.accuracy_score(y_test, xgb_predict))

print('**F1_score for XGBoost classifier is:', metrics.f1_score(y_test, xgb_predict))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, xgb_predict))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, xgb_model.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, xgb_model.predict(X)))



XGBoost = (cross_val_score(xgb, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
from sklearn.ensemble import VotingClassifier



v_clf = VotingClassifier(estimators=[('svm',svm_model),('NB',NB_model),('knn',knn_model),('Logistic',LG_model),

                                    ('xgb',xgb_model)], voting='hard')



vclf_model = v_clf.fit(X_train, y_train)



pred_vclf =v_clf.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, pred_vclf))

print('F1_Score:', metrics.f1_score(y_test, pred_vclf))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, pred_vclf))

print('\n')

print('Training Score:', vclf_model.score(X_train, y_train))

print('Testing Score:', vclf_model.score(X_test, y_test))
#Invoking the Decision Tree classifier function using 'Entropy' method of finding the split columns. 

#Other option could be gini index and also restricting the depth of the tree to 5 (just a random number)



from sklearn.tree import DecisionTreeClassifier



model_entropy = DecisionTreeClassifier(criterion='entropy')

entropy_model = model_entropy.fit(X_train, y_train)
DT_predict = model_entropy.predict(X_test)



print('**Accuracy for Decision Tree classifier is:', metrics.accuracy_score(y_test, DT_predict))

print('**F1_score for Decision Tree classifier is:', metrics.f1_score(y_test, DT_predict))

print('\n')

print('Confusion matrix', confusion_matrix(y_test, DT_predict))

print('\n')

print('**Scoring on Training Data is:', model_entropy.score(X_train, y_train))

print('**Scoring on Test Data is:', model_entropy.score(X_test, y_test))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, entropy_model.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, entropy_model.predict(X)))



DecisionTree = (cross_val_score(model_entropy, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
clf_pruned = DecisionTreeClassifier(criterion = 'entropy', random_state=100, max_depth = 4, min_samples_leaf=5)

pruned_clf = clf_pruned.fit(X_train, y_train)   #Fitting the training to the model 
clf_pruned.score(X_train, y_train)

clf_pruned.score(X_test, y_test)



dtentropy_predict = clf_pruned.predict(X_test)



print('**Accuracy for Decision Tree with Entropy is:', metrics.accuracy_score(y_test, dtentropy_predict))

print('**F1_score for Decision Tree with Entropy is:', metrics.f1_score(y_test, dtentropy_predict))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, dtentropy_predict))

print('\n')

print('Training Score is:', clf_pruned.score(X_train, y_train))

print('Test Score is:', clf_pruned.score(X_test, y_test))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, pruned_clf.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, pruned_clf.predict(X)))



DecisionTree_Entropy = (cross_val_score(clf_pruned, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
import pydotplus

from sklearn.tree import export_graphviz

from sklearn import tree

import collections

import matplotlib.image as mpimg

from IPython.display import Image
data_feature_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',

       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',

       'previous', 'poutcome', ]
data = tree.export_graphviz(clf_pruned, feature_names=data_feature_names, out_file=None,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(data)



colors = ('turquoise', 'orange')

edges = collections.defaultdict(list)



for edge in graph.get_edge_list():

    edges[edge.get_source()].append(int(edge.get_destination()))

    

for edge in edges:

    edges[edge].sort()

    for i in range(2):

        dest = graph.get_node(str(edges[edge][i]))[0]

        dest.set_fillcolor(colors[i])

        

graph.write_png('tree.png')

Image(graph.create_png())
feat_importance = clf_pruned.tree_.compute_feature_importances(normalize=False)



feat_imp_dict = dict(zip(df, clf_pruned.feature_importances_))

feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')

feat_imp.sort_values(by=0, ascending=False)
from sklearn.ensemble import RandomForestClassifier

rfcl = RandomForestClassifier(n_estimators = 100)

rfcl = rfcl.fit(X_train, y_train)



rf_predict = rfcl.predict(X_test)



print('Accuracy Score is:', metrics.accuracy_score(y_test, rf_predict))

print('F1 Score is:', metrics.f1_score(y_test, rf_predict))

print('\n')



print('Confusion matrix', '\n', confusion_matrix(y_test, rf_predict))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, rfcl.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, rfcl.predict(X)))

print('\n')

print('Training Score is:', rfcl.score(X_train, y_train))

print('Test Score is:', rfcl.score(X_test, y_test)) 



RandomForest = (cross_val_score(rfcl, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
from sklearn.ensemble import AdaBoostClassifier



abcl = AdaBoostClassifier( n_estimators= 100, learning_rate=0.1, random_state=22)

abcl = abcl.fit(X_train, y_train)



pred_AB =abcl.predict(X_test)



print('Accuracy Score is:', metrics.accuracy_score(y_test, pred_AB))

print('F1 Score is:', metrics.f1_score(y_test, pred_AB))

print('\n')

print('Training Score is:',abcl.score(X_train, y_train)) 

print('Test Score is:',abcl.score(X_test, y_test)) 

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, abcl.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, abcl.predict(X)))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, pred_AB))



Adaboost = (cross_val_score(abcl, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
from sklearn.ensemble import BaggingClassifier           #Importing the necessary libraries 



bgcl = BaggingClassifier(n_estimators=100, max_samples= .7, bootstrap = True, oob_score=True, random_state=22)

bgcl = bgcl.fit(X_train, y_train)



pred_BG =bgcl.predict(X_test)



print('Accuracy Score is:', metrics.accuracy_score(y_test, pred_BG))

print('F1 Score is:', metrics.f1_score(y_test, pred_BG))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, pred_BG))

print('\n')

print('Training Score is:', bgcl.score(X_train, y_train))

print('Test Score:', bgcl.score(X_test, y_test))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, bgcl.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, bgcl.predict(X)))



Bagging = (cross_val_score(bgcl, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
from sklearn.ensemble import GradientBoostingClassifier

gbcl = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, random_state=22)

gbcl = gbcl.fit(X_train, y_train)



pred_GB =gbcl.predict(X_test)



print('Accuracy:', metrics.accuracy_score(y_test, pred_GB))

print('F1_Score:', metrics.f1_score(y_test, pred_GB))

print('\n')

print('Confusion matrix', '\n', confusion_matrix(y_test, pred_GB))

print('\n')

print('Training Score:', gbcl.score(X_train, y_train))

print('Testing Score:', gbcl.score(X_test, y_test))

print('\n')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y, gbcl.predict_proba(X)[:,1])

print('AUC Score',auc(false_positive_rate, true_positive_rate))

print('roc_auc_score',roc_auc_score(y, gbcl.predict(X)))



GradientBoost = (cross_val_score(gbcl, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy').mean())
plt.figure(figsize=(20,15))



probs = svm_model.predict_proba(X_test)                              #ROC Curve for svm 

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,1)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic SVM')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



probs = NB_model.predict_proba(X_test)                               #ROC Curve for Naive Bayes 

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,2)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic Naive Bayes')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



probs = KNN_model.predict_proba(X_test)                              #ROC Curve for KNN 

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,3)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic KNN')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()



plt.figure(figsize=(20,15))

probs = logit_model.predict_proba(X_test)                            #ROC Curve for Logistic Regression

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,4)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic Logistic Regression')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



probs = xgb.predict_proba(X_test)                                    #ROC Curve for XGBoost

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,5)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic XGBoost')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
probs = model_entropy.predict_proba(X_test)                        #ROC Curve for Decision Tree with Entropy 

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.figure(figsize=(20,15))

plt.subplot(3,3,1)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic Decision Tree Entropy')

plt.legend(loc = 'lower right') 

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



probs = clf_pruned.predict_proba(X_test)                          #ROC Curve for Decision Tree Purned

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,2)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic Decision Tree Pruned')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



probs = rfcl.predict_proba(X_test)                                 #ROC Curve for Random Forest 

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,3)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic Random Forest')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



probs = abcl.predict_proba(X_test)                                 #ROC Curve for AdaBoost 

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,4)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic AdaBoost')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



probs = bgcl.predict_proba(X_test)                                 #ROC Curve for Bagging

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,5)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic Bagging')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



probs = gbcl.predict_proba(X_test)                                 #ROC Curve for GradientBoost

preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)

roc_auc = metrics.auc(fpr, tpr)



plt.subplot(3,3,6)

plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.title('Receiver Operating Characteristic Gradient Boosting')

plt.legend(loc = 'lower right')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()

models = pd.DataFrame({'Models': ['Gausian', 'knn', 'LogitRegression', 'GradientBoost', 'Bagging',

                                 'Adaboost', 'XGBoost', 'DecisionTree', 'DecisionTree_Entropy', 'RandomForest'],

                       'Score': [Gausian, knn, LogitRegression, GradientBoost, Bagging, Adaboost, XGBoost, DecisionTree, 

                                 RandomForest, DecisionTree_Entropy]})

models.sort_values(by='Score', ascending = False)