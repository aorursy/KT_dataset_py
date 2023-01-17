# Importing Data Science Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

%matplotlib inline

from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.neighbors  import KNeighborsClassifier

from sklearn import svm,tree
# Reading Csv file

df = pd.read_csv('../input/carInsurance_train.csv',index_col = 'Id')
# Top rows

df.head()
# Shape of dataframe

df.shape
# Columns in dataset

df.columns
# Statistics of numerical columns

df.describe()
# Datatypes of columns in dataset

df.dtypes
# Statistics of categorical features

df.describe(include=['O'])
# Plotting Balance field as a Boxplot using Seaborn

sns.boxplot(x='Balance',data=df,palette='hls');
# Maximum value in Balance field

df.Balance.max()
# Looking at the particular maximum value in the dataframe

df[df['Balance'] == 98417]
# Dropping the index value corresponding to the outlier

df_new = df.drop(df.index[1742]);
#checking for missing values using isnull() method

df_new.isnull().sum()
# Using frontfill to fill the missing values in Job and Education fields

df_new['Job'] = df_new['Job'].fillna(method ='pad')

df_new['Education'] = df_new['Education'].fillna(method ='pad')
# Using none to fill Nan values in Communication and Outcome fields

df_new['Communication'] = df_new['Communication'].fillna('none')

df_new['Outcome'] = df_new['Outcome'].fillna('none')
#Looks like all missing values have been imputed

df_new.isnull().sum()
#Setting up correlation for our dataframe and passing it to seaborn heatmap function

sns.set(style="white")

corr = df_new.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr,annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
# Plotting paired fields of intrest using Seaborn pairplot

df_sub = ['Age','Balance','HHInsurance', 'CarLoan','NoOfContacts','DaysPassed','PrevAttempts','CarInsurance']

sns.pairplot(df_new[df_sub],hue='CarInsurance',size=1.5);
#Uses multiple x and y variables to form pair grid of categorical values passed

g = sns.PairGrid(df_new,

                 x_vars=["Education","Marital", "Job"],

                 y_vars=["CarInsurance", "Balance"],

                 aspect=.75, size=6)

plt.xticks(rotation=90)

g.map(sns.barplot, palette="pastel");
#Seaborn violin plot for LastContactMonth and CarInsurance fields

sns.violinplot(x="LastContactMonth",y='CarInsurance',data=df_new);
#Count of CarInsurance against Outcome i.e previous campaign outcome

sns.countplot(x="Outcome",hue='CarInsurance',data=df_new);
#Qcut splits both the attribute into 5 buckets

df_new['AgeBinned'] = pd.qcut(df_new['Age'], 5 , labels = False)

df_new['BalanceBinned'] = pd.qcut(df_new['Balance'], 5,labels = False)
#Converting CallStart and CallEnd to datetime datatype

df_new['CallStart'] = pd.to_datetime(df_new['CallStart'] )

df_new['CallEnd'] = pd.to_datetime(df_new['CallEnd'] )

#Subtracting both the Start and End times to arrive at the actual CallTime

df_new['CallTime'] = (df_new['CallEnd'] - df_new['CallStart']).dt.total_seconds()

#Binning the CallTime

df_new['CallTimeBinned'] = pd.qcut(df_new['CallTime'], 5,labels = False)
#Dropping the original columns of the binned, just to make things easy

df_new.drop(['Age','Balance','CallStart','CallEnd','CallTime'],axis = 1,inplace = True)
# Using get_dummies function to assign binary values to each value in the categorical column

Job = pd.get_dummies(data = df_new['Job'],prefix = "Job")

Marital= pd.get_dummies(data = df_new['Marital'],prefix = "Marital")

Education= pd.get_dummies(data = df_new['Education'],prefix="Education")

Communication = pd.get_dummies(data = df_new['Communication'],prefix = "Communication")

LastContactMonth = pd.get_dummies(data = df_new['LastContactMonth'],prefix= "LastContactMonth")

Outcome = pd.get_dummies(data = df_new['Outcome'],prefix = "Outcome")
# Dropping the categorical columns which have been assigned dummies

df_new.drop(['Job','Marital','Education','Communication','LastContactMonth','Outcome'],axis=1,inplace=True)
#Concatenating the dataframe with the categorical dummy columns

df = pd.concat([df_new,Job,Marital,Education,Communication,LastContactMonth,Outcome],axis=1)
# The dataframe has some new additions resulting from the categorical dummies added

df.columns
# Dropping the Target for X

X= df.drop(['CarInsurance'],axis=1).values

# Including only the Target for y

y=df['CarInsurance'].values

#Splitting the Training and Testing data having 20% of Test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42, stratify = y)
#The code for the below matrix is taken from sklearn documentation

#Defining the confusion matrix function

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    

   

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    





    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

        

        

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

#Using Success and Failure for 0 and 1    

class_names = ['Success','Failure']
# Defining the kNNClassifier with 6 neighbors

knn = KNeighborsClassifier(n_neighbors = 6)

#Fitting the classifier to the training set

knn.fit(X_train,y_train)

print ("kNN Accuracy is %2.2f" % accuracy_score(y_test, knn.predict(X_test)))

#The cross validation score is obtained for kNN using 10 folds

score_knn = cross_val_score(knn, X, y, cv=10).mean()

print("Cross Validation Score = %2.2f" % score_knn)

y_pred= knn.predict(X_test)

print(classification_report(y_test, y_pred))

#Defining the confusion matrix

cm = confusion_matrix(y_test,y_pred)

#Plotting the confusion matrix

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
#Logistic Regression Classifier

LR = LogisticRegression()

LR.fit(X_train,y_train)

print ("Logistic Accuracy is %2.2f" % accuracy_score(y_test, LR.predict(X_test)))

score_LR = cross_val_score(LR, X, y, cv=10).mean()

print("Cross Validation Score = %2.2f" % score_LR)

y_pred = LR.predict(X_test)

print(classification_report(y_test, y_pred))

# Confusion matrix for LR

cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
#SVM Classifier

SVM = svm.SVC()

SVM.fit(X_train, y_train)

print ("SVM Accuracy is %2.2f" % accuracy_score(y_test, SVM.predict(X_test)))

score_svm = cross_val_score(SVM, X, y, cv=10).mean()

print("Cross Validation Score = %2.2f" % score_svm)

y_pred = SVM.predict(X_test)

print(classification_report(y_test,y_pred))

#Confusion matrix for SVM

cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# Decision Tree Classifier

DT = tree.DecisionTreeClassifier(random_state = 0,class_weight="balanced",

    min_weight_fraction_leaf=0.01)

DT = DT.fit(X_train,y_train)

print ("Decision Tree Accuracy is %2.2f" % accuracy_score(y_test, DT.predict(X_test)))

score_DT = cross_val_score(DT, X, y, cv=10).mean()

print("Cross Validation Score = %2.2f" % score_DT)

y_pred = DT.predict(X_test)

print(classification_report(y_test, y_pred))

# Confusion Matrix for Decision Tree

cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
#Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=10,class_weight="balanced")

rfc.fit(X_train, y_train)

print ("Random Forest Accuracy is %2.2f" % accuracy_score(y_test, rfc.predict(X_test)))

score_rfc = cross_val_score(rfc, X, y, cv=10).mean()

print("Cross Validation Score = %2.2f" % score_rfc)

y_pred = rfc.predict(X_test)

print(classification_report(y_test,y_pred ))

#Confusion Matrix for Random Forest

cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
#AdaBoost Classifier

ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)

ada.fit(X_train,y_train)

print ("AdaBoost Accuracy= %2.2f" % accuracy_score(y_test,ada.predict(X_test)))

score_ada = cross_val_score(ada, X, y, cv=10).mean()

print("Cross Validation Score = %2.2f" % score_ada)

y_pred = ada.predict(X_test)

print(classification_report(y_test,y_pred ))

#Confusion Marix for AdaBoost

cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
#XGBoost Classifier

xgb = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.01)

xgb.fit(X_train,y_train)

print ("GradientBoost Accuracy= %2.2f" % accuracy_score(y_test,xgb.predict(X_test)))

score_xgb = cross_val_score(xgb, X, y, cv=10).mean()

print("Cross Validation Score = %2.2f" % score_ada)

y_pred = xgb.predict(X_test) 

print(classification_report(y_test,y_pred))

#Confusion Matrix for XGBoost Classifier

cm_xg = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm_xg, classes=class_names, title='Confusion matrix')
#Obtaining False Positive Rate, True Positive Rate and Threshold for all classifiers

fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(X_test)[:,1])

LR_fpr, LR_tpr, thresholds = roc_curve(y_test, LR.predict_proba(X_test)[:,1])

#SVM_fpr, SVM_tpr, thresholds = roc_curve(y_test, SVM.predict_proba(X_test)[:,1])

DT_fpr, DT_tpr, thresholds = roc_curve(y_test, DT.predict_proba(X_test)[:,1])

rfc_fpr, rfc_tpr, thresholds = roc_curve(y_test, rfc.predict_proba(X_test)[:,1])

ada_fpr, ada_tpr, thresholds = roc_curve(y_test, ada.predict_proba(X_test)[:,1])

xgb_fpr, xgb_tpr, thresholds = roc_curve(y_test, xgb.predict_proba(X_test)[:,1])

#PLotting ROC Curves for all classifiers

plt.plot(fpr, tpr, label='KNN' )

plt.plot(LR_fpr, LR_tpr, label='Logistic Regression')

#plt.plot(SVM_fpr, SVM_tpr, label='SVM')

plt.plot(DT_fpr, DT_tpr, label='Decision Tree')

plt.plot(rfc_fpr, rfc_tpr, label='Random Forest')

plt.plot(ada_fpr, ada_tpr, label='AdaBoost')

plt.plot(xgb_fpr, xgb_tpr, label='GradientBoosting')

# Plot Base Rate ROC

plt.plot([0,1],[0,1],label='Base Rate')



plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Graph')

plt.legend(loc="lower right")

plt.show()
# Using Recursive Feature Elimination Function and fitting it in a Logistic Regression Model

modell = LogisticRegression()

rfe = RFE(modell, 5)

rfe = rfe.fit(X_train,y_train)

# Displays the feature rank

rfe.ranking_
# Using ExtraTreesClassifier model function

model = ExtraTreesClassifier()

model.fit(X_train, y_train)

# Printing important features in the model

print(model.feature_importances_)

importances = model.feature_importances_

feat_names = df.drop(['CarInsurance'],axis=1).columns



# Displaying the feature importances as a chart by sorting it in the order of importances

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")

plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)

plt.xlim([-1, len(indices)])

plt.show()
rfc = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=10,class_weight="balanced")

y_proba = cross_val_predict(rfc, X, y, cv=10, n_jobs=-1, method='predict_proba')

results = pd.DataFrame({'y': y, 'y_proba': y_proba[:,1]})

results = results.sort_values(by='y_proba', ascending=False).reset_index(drop=True)

results.index = results.index + 1

results.index = results.index / len(results.index) * 100
sns.set_style('darkgrid')

pred = results

pred['Lift Curve'] = pred.y.cumsum() / pred.y.sum() * 100

pred['Baseline'] = pred.index

base_rate = y.sum() / len(y) * 100

pred[['Lift Curve', 'Baseline']].plot(style=['-', '--', '--'])

pd.Series(data=[0, 100, 100], index=[0, base_rate, 100]).plot(style='--')

plt.title('Cumulative Gains')

plt.xlabel('% of Customers Contacted')

plt.ylabel("% of Positive Results")

plt.legend(['Lift Curve', 'Baseline', 'Ideal']);