# Importing the required Libraries.

import pandas as pd

import numpy as np

import sys

import os

import time

#ignore warnings

import warnings

warnings.filterwarnings('ignore')
#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import model_selection

from sklearn import metrics



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

sns.set_style('white')



from sklearn.model_selection import cross_val_score
df = pd.read_csv('../input/indianliver/indian_liver_patient.csv')

df.describe()
print(df.columns)

print('*'*50)

for i in df.columns :

    print(i)

    print(df[i].describe())

    print('*'*50)
df.info()
df[df['Albumin_and_Globulin_Ratio'].isnull()]
# Re-naming the columns

df =  df.rename(columns={'Dataset':'Liver_disease','Alamine_Aminotransferase':'Alanine_Aminotransferase'}, inplace=False)
# Renaming Done

df.describe()
# Dropping Null Values

df = df.dropna()

# Changing the values in "Liver_Disease" column 

df['Liver_disease'] = df['Liver_disease'] - 1 

# Converting Gender column into categorical data 

LabelEncoder = LabelEncoder()

df['Is_male'] = LabelEncoder.fit_transform(df['Gender'])

df = df.drop(columns='Gender')
X = df[['Age', 'Total_Bilirubin', 

        'Direct_Bilirubin',

        'Alkaline_Phosphotase',

        'Alanine_Aminotransferase', 'Aspartate_Aminotransferase',

       'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Is_male']]

y = df['Liver_disease']
# Validate each class to understand if the dataset is imbalanced.



print ('Total Unhealthy Livers :  {} and its percentage is {} %'.format(df.Liver_disease.value_counts()[0], round(df.Liver_disease.value_counts()[0]/df.Liver_disease.value_counts().sum()*100,2)) )

print ('Total Healthy Livers :  {} and its percentage is {} %'.format(df.Liver_disease.value_counts()[1], round(df.Liver_disease.value_counts()[1]/df.Liver_disease.value_counts().sum()*100,2)) )
df.skew(axis = 0, skipna = True) 
# Plotting the box plots 

plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x = X['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (years)')



plt.subplot(232)

plt.boxplot(X['Total_Bilirubin'], showmeans = True, meanline = True)

plt.title('Total Bilirubin Boxplot')

plt.ylabel('Total Bilirubin (mg/dL)')



plt.subplot(233)

plt.boxplot(X['Direct_Bilirubin'], showmeans = True, meanline = True)

plt.title('Direct Bilirubin Boxplot')

plt.ylabel('Direct Bilirubin (mg/dL)')



plt.subplot(234)

plt.hist(x = [X[y==1]['Is_male'], X[y ==0]['Is_male']], 

         stacked=True, color = ['g','r'],label = ['Healthy','Patient'])

plt.title('Gender Histogram by patients')

plt.xlabel('Gender [0 - female : 1 - male]')

plt.ylabel('# of people')

plt.legend()



plt.subplot(235)

plt.boxplot(x = X['Alkaline_Phosphotase'], showmeans = True, meanline = True)

plt.title('Alkaline Phosphotase')

plt.ylabel('Alkaline Phosphotase (International Units /Litre)')



plt.subplot(236)

plt.boxplot(X['Alanine_Aminotransferase'], showmeans = True, meanline = True)

plt.title('Alanine Aminotransferase Boxplot')

plt.ylabel('Alanine Aminotransferase (units/L)')
plt.figure(figsize=[16,12])

plt.subplot(231)

plt.boxplot(X['Aspartate_Aminotransferase'], showmeans = True, meanline = True)

plt.title('Aspartate Aminotransferase Boxplot')

plt.ylabel('Aspartate_Aminotransferase (units/L)')





plt.subplot(232)

plt.boxplot(X['Total_Protiens'], showmeans = True, meanline = True)

plt.title('Total Protiens Boxplot')

plt.ylabel('Total Protiens (g/dL)')



plt.subplot(233)

plt.boxplot(X['Albumin'], showmeans = True, meanline = True)

plt.title('Albumin Boxplot')

plt.ylabel('Albumin (g/dL)')
fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(y = 'Alanine_Aminotransferase', x = 'Liver_disease', data=df, ax = saxis[0,0])

sns.pointplot(y = 'Total_Bilirubin', x = 'Liver_disease', data=df, ax = saxis[0,1])

sns.pointplot(y = 'Direct_Bilirubin', x = 'Liver_disease', data=df, ax = saxis[0,2])





sns.barplot(y = 'Alkaline_Phosphotase', x = 'Liver_disease', data=df, ax = saxis[1,0])

sns.barplot(y = 'Aspartate_Aminotransferase', x = 'Liver_disease', data=df, ax = saxis[1,1])

sns.boxplot(y = 'Total_Protiens', x = 'Liver_disease', data=df, ax = saxis[1,2])
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(df)
from sklearn import preprocessing

X_scaler = preprocessing.normalize(X)
# Splitting the data 

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_scaler, y, random_state = 0)



print("Train Shape: {}".format(X_train.shape))

print("Test Shape: {}".format(X_test.shape))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
# Use score method to get accuracy of model

score = lr.score(X_test, y_test)

print("Score of the model is - ",score)

print("Report card of this model - ")

print(metrics.classification_report(y_test, y_pred, digits=3))

print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred))
from sklearn.metrics import roc_auc_score

test_roc_auc = roc_auc_score(y_test, y_pred)



# Print test_roc_auc

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
cm1 = metrics.confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9,9))

sns.heatmap(cm1, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15)
# Naives Bayes

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

y_pred_nb = nb.predict(X_test)
score = nb.score(X_test, y_test)

print("Score of the model is - ",score)

print("Report card of this model - ")

print(metrics.classification_report(y_test, y_pred_nb, digits=3))

print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred_nb))
from sklearn.metrics import roc_auc_score

test_roc_auc = roc_auc_score(y_test, y_pred_nb)



# Print test_roc_auc

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
cm2 = metrics.confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(9,9))

sns.heatmap(cm2, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Wistia');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sg = SGDClassifier()

sg.fit(X_train,y_train)

y_pred_sg = sg.predict(X_test)
score = sg.score(X_test, y_test)

print("Score of the model is - ",score)

print("Report card of this model - ")

print(metrics.classification_report(y_test, y_pred_sg, digits=3))

print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred_sg))
from sklearn.metrics import roc_auc_score

test_roc_auc = roc_auc_score(y_test, y_pred_sg)



# Print test_roc_auc

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
cm3 = metrics.confusion_matrix(y_test, y_pred_sg)

plt.figure(figsize=(9,9))

sns.heatmap(cm3, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Greens');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15)
# KNN Model

from sklearn.neighbors import KNeighborsClassifier

hist = []

for i in range(1,10):

    clf = KNeighborsClassifier(n_neighbors=i)

    cross_val = cross_val_score(clf, X_scaler, y, cv=5)

    hist.append(np.mean(cross_val))

plt.plot(hist)

plt.title('Cross Validations score for KNeighborsClassifier')

plt.xlabel('n_neighbors')

plt.ylabel('Accuracy')

plt.grid()

plt.show()
knn = KNeighborsClassifier(n_neighbors = 7)

knn.fit(X_train,y_train)

y_pred_knn = knn.predict(X_test)
score = knn.score(X_test, y_test)

print("Score of the model is - ",score)

print("Report card of this model - ")

print(metrics.classification_report(y_test, y_pred_knn, digits=3))

print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred_knn))
from sklearn.metrics import roc_auc_score

test_roc_auc = roc_auc_score(y_test, y_pred_knn)

# Print test_roc_auc

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
cm4 = metrics.confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(9,9))

sns.heatmap(cm4, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Accent');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth = None , random_state = 1 , max_features = None, min_samples_leaf =20)

dtree.fit(X_train,y_train)

y_pred_dtree = dtree.predict(X_test)
score = dtree.score(X_test, y_test)

print("Score of the model is - ",score)

print("Report card of this model - ")

print(metrics.classification_report(y_test, y_pred_dtree, digits=3))

print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred_dtree))
from sklearn.metrics import roc_auc_score

test_roc_auc = roc_auc_score(y_test, y_pred_dtree)



# Print test_roc_auc

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
cm5 = metrics.confusion_matrix(y_test, y_pred_dtree)

plt.figure(figsize=(9,9))

sns.heatmap(cm5, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'viridis');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15)
# Random Forest 

from sklearn.ensemble import RandomForestClassifier



hist1 = []

for i in range(1,10):

    clf = RandomForestClassifier(n_estimators=80, max_depth=i, random_state=0)

    cross_val = cross_val_score(clf, X_train, y_train, cv=5)

    hist1.append(np.mean(cross_val))

plt.plot(hist1)

plt.title('Cross Validations score for RandomForestClassifier')

plt.xlabel('Max_depth')

plt.ylabel('Accuracy')

plt.grid()
ran_for = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=0)

ran_for.fit(X_train,y_train)

y_pred_ran = ran_for.predict(X_test)
score = ran_for.score(X_test, y_test)

print("Score of the model is - ",score)

print("Report card of this model - ")

print(metrics.classification_report(y_test, y_pred_ran, digits=3))

print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred_ran))
from sklearn.metrics import roc_auc_score

test_roc_auc = roc_auc_score(y_test, y_pred_ran)



# Print test_roc_auc

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
cm6 = metrics.confusion_matrix(y_test, y_pred_ran)

plt.figure(figsize=(9,9))

sns.heatmap(cm6, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'viridis');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15)
# Support Vector machine Model

from sklearn.svm import SVC

grid = [0.00001, 0.0001, 0.001, 0.01, 0.1]

hist = []

for val in grid:

    clf = SVC(gamma=val)

    cross_val = cross_val_score(clf, X, y, cv=5)

    hist.append(np.mean(cross_val))

plt.plot([str(i) for i in grid], hist)

plt.title('Cross Validations score for SVC')

plt.xlabel('gamma')

plt.ylabel('Accuracy')

plt.grid()

plt.show()

svm = SVC(kernel= "linear",C=0.025, random_state = 0 , gamma=0.01)

svm.fit(X_train,y_train)

y_pred_svm = svm.predict(X_test)
score = svm.score(X_test, y_test)

print("Score of the model is - ",score)

print("Report card of this model - ")

print(metrics.classification_report(y_test, y_pred_svm, digits=3))

print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred_svm))
from sklearn.metrics import roc_auc_score

test_roc_auc = roc_auc_score(y_test, y_pred_svm)



# Print test_roc_auc

print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
cm7 = metrics.confusion_matrix(y_test, y_pred_ran)

plt.figure(figsize=(9,9))

sns.heatmap(cm7, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Accent_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15)
#print the true and predicted values

dictionary = {'Actual values': y_test, 'Predicted values': y_pred_dtree}

pd.DataFrame.from_dict(dictionary)