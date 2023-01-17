import pandas as pd

import numpy as np



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
# data = pd.read_csv('Data/train.csv',index_col='ID_code')

data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')



# data.set_index('ID_code',inplace=True)

data.drop('ID_code',axis=1,inplace=True)
data = data.sample(100)
fig, axs = plt.subplots(1,10,figsize=(30,5))

fig.subplots_adjust(hspace = .5, wspace=.5)

axs = axs.ravel()



counter=0

for i in data.columns[1:10]:

    

    axs[counter].set_title(i)

#     axs[counter].hist(data[i])

    sns.distplot(data[i],ax=axs[counter])

    axs[counter].set_ylabel("Distribution")

#     plt.xticks(rotation=90)

    axs[counter].set_xticklabels(data[i].unique(),rotation=90)

    counter+=1  
fig, axs = plt.subplots(1,10,figsize=(30,5))

fig.subplots_adjust(hspace = .5, wspace=.5)

axs = axs.ravel()

counter=0



for i in data.columns[1:10]:



    

    data_pivot = data[['target']]

    data_pivot[i] = pd.qcut(data[i],q=10,duplicates='drop')

    data_plot = pd.pivot_table(data_pivot,'target',i)



    plt.title(i)

    sns.barplot(data_plot.index,'target',data=data_plot,ax=axs[counter])

    plt.ylabel("Percentage of Approval")

    plt.xticks(rotation=90)



    counter+=1   
from sklearn.preprocessing import MinMaxScaler



X = data.drop('target',axis=1)

y = data.target



mms = MinMaxScaler()

X = pd.DataFrame(mms.fit_transform(X),columns=X.columns)



data_plot = pd.concat([X,y],axis=1)
fig, axs = plt.subplots(1,10,figsize=(30,5))

fig.subplots_adjust(hspace = .5, wspace=.5)

axs = axs.ravel()



counter=0

for i in data.columns[1:10]:

    

    axs[counter].set_title(i)

#     axs[counter].hist(data[i])

    sns.distplot(data_plot[i],ax=axs[counter])

    axs[counter].set_ylabel("Distribution")

#     plt.xticks(rotation=90)

    axs[counter].set_xticklabels(data_plot[i].unique(),rotation=90)

    counter+=1  
# data = pd.read_csv('Data/train.csv',index_col='ID_code')

data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')



# data.set_index('ID_code',inplace=True)

data.drop('ID_code',axis=1,inplace=True)
X = data.drop('target',axis=1)

y = data.target
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression



logreg=LogisticRegression(max_iter=300)

logreg.fit(X_train,y_train.values.ravel())

y_pred = logreg.predict(X_test)



print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



plt.figure(figsize = (5,4))

sns.set(font_scale=1.4)



sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.preprocessing import MinMaxScaler



X = data.drop('target',axis=1)

y = data.target



mms = MinMaxScaler()

X = pd.DataFrame(mms.fit_transform(X),columns=X.columns)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression



logreg=LogisticRegression(max_iter=300)

logreg.fit(X_train,y_train.values.ravel())

y_pred = logreg.predict(X_test)



print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



plt.figure(figsize = (5,4))

sns.set(font_scale=1.4)



sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
data.target.value_counts()
from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline

from collections import Counter





under = RandomUnderSampler(sampling_strategy=0.25)

over = SMOTE(sampling_strategy=0.5)



X = data.drop('target',axis=1)

y = data.target

X,y = under.fit_resample(X,y)
y.value_counts()
X,y = over.fit_resample(X,y)
y.value_counts()
# summarize the new class distribution

counter = Counter(y)

print(counter)

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression



logreg2=LogisticRegression(max_iter=300)

logreg2.fit(X_train,y_train.values.ravel())

y_pred = logreg2.predict(X_test)



print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg2.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



plt.figure(figsize = (5,4))

sns.set(font_scale=1.4)



sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.preprocessing import MinMaxScaler



X = data.drop('target',axis=1)

y = data.target



mms = MinMaxScaler()

X = pd.DataFrame(mms.fit_transform(X),columns=X.columns)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)
import statsmodels.api  as sm



logit_model2=sm.Logit(y_train,X_train)

result=logit_model2.fit()

print(result.summary2())
X_train_negd = X_train.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                             'var_38','var_39','var_41','var_73','var_98','var_61',

                             'var_100','var_103','var_136','var_153','var_158',

                             'var_161',

                             ],axis=1)
X_test_negd = X_test.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                            'var_38','var_39','var_41','var_73','var_98','var_61',

                            'var_100','var_103','var_136','var_153','var_158',

                            'var_161',

                             ],axis=1)
from sklearn.linear_model import LogisticRegression



logreg=LogisticRegression(max_iter=300)

logreg.fit(X_train_negd,y_train.values.ravel())

y_pred = logreg.predict(X_test_negd)



print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test_negd, y_test)))
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



plt.figure(figsize = (5,4))

sns.set(font_scale=1.4)



sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
X_train_negd = X_train.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                             'var_38','var_39','var_41','var_73','var_98','var_61',

                             'var_100','var_103','var_136','var_153','var_158',

                             'var_161','var_183',

                             'var_16','var_37','var_46','var_65','var_69','var_79',

                             'var_96','var_117','var_124','var_126','var_185','var_189',

                             'var_61','var_60','var_47'                             

                             ],axis=1)
X_test_negd = X_test.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                            'var_38','var_39','var_41','var_73','var_98','var_61',

                            'var_100','var_103','var_136','var_153','var_158',

                            'var_161','var_183',

                           'var_16','var_37','var_46','var_65','var_69','var_79',

                             'var_96','var_117','var_124','var_126','var_185','var_189',

                           'var_61','var_60','var_47'

                             ],axis=1)
from sklearn.linear_model import LogisticRegression



logreg=LogisticRegression(max_iter=300)

logreg.fit(X_train_negd,y_train.values.ravel())

y_pred = logreg.predict(X_test_negd)



print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test_negd, y_test)))
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



plt.figure(figsize = (5,4))

sns.set(font_scale=1.4)



sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Recursive Feature Elimination

from sklearn.feature_selection import RFE



logreg = LogisticRegression(max_iter=4000)



# Select Best X Features

rfe = RFE(logreg, n_features_to_select=None)

rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)

print(rfe.ranking_)
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_test_rfe = X_test[X_train.columns[rfe.support_]]
from sklearn.linear_model import LogisticRegression



logreg=LogisticRegression(max_iter=300)

logreg.fit(X_train_rfe,y_train.values.ravel())

y_pred = logreg.predict(X_test_rfe)



print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(logreg.score(X_test_rfe, y_test)))
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



plt.figure(figsize = (5,4))

sns.set(font_scale=1.4)



sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
ratio = y[y==1].shape[0]/y[y==0].shape[0]

index_1 = y[y==1].index

index_0 = y[y==0].index
from numpy import random



index_0_sel = random.choice(index_1, size = 20000)

index_1_sel = random.choice(index_0, size = int(20000*ratio))

index_sel = np.concatenate((index_0_sel,index_1_sel))





X_sel = X.iloc[index_sel]

y_sel = y.iloc[index_sel]
X_train,X_test,y_train,y_test = train_test_split(X_sel, y_sel, test_size=0.33, random_state=42)
X_train = X_train.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                             'var_38','var_39','var_41','var_73','var_98','var_61',

                             'var_100','var_103','var_136','var_153','var_158',

                             'var_161','var_183',

                             'var_16','var_37','var_46','var_65','var_69','var_79',

                             'var_96','var_117','var_124','var_126','var_185','var_189',

                             'var_61','var_60','var_47'                             

                             ],axis=1)
X_test = X_test.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                            'var_38','var_39','var_41','var_73','var_98','var_61',

                            'var_100','var_103','var_136','var_153','var_158',

                            'var_161','var_183',

                           'var_16','var_37','var_46','var_65','var_69','var_79',

                             'var_96','var_117','var_124','var_126','var_185','var_189',

                           'var_61','var_60','var_47'

                             ],axis=1)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

## Logistic Regression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, penalty = 'l2')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
## SVM (Linear)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'linear', probability= True)

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## SVM (rbf)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Naive Bayes

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Decision Tree

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

classifier.fit(X_train, y_train)



#Predicting the best set result

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Gini (n=100)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'gini')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Gini (n=100)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Gini (n=200)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 200,

                                    criterion = 'gini')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Gini (n=200)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Gini (n=300)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 300,

                                    criterion = 'gini')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Gini (n=300)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Entropy (n=100)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'entropy')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Entropy (n=100)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Entropy (n=200)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 200,

                                    criterion = 'entropy')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Entropy (n=200)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest Entropy (n=300)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 300,

                                    criterion = 'entropy')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Entropy (n=300)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## K-Nearest Neighbors (K-NN)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p= 2)

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
results
X = data.drop('target',axis=1)

y = data.target
mms = MinMaxScaler()

X = pd.DataFrame(mms.fit_transform(X),columns=X.columns)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = X_train.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                             'var_38','var_39','var_41','var_73','var_98','var_61',

                             'var_100','var_103','var_136','var_153','var_158',

                             'var_161','var_183',

                             'var_16','var_37','var_46','var_65','var_69','var_79',

                             'var_96','var_117','var_124','var_126','var_185','var_189',

                             'var_61','var_60','var_47'                             

                             ],axis=1)
X_test = X_test.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                            'var_38','var_39','var_41','var_73','var_98','var_61',

                            'var_100','var_103','var_136','var_153','var_158',

                            'var_161','var_183',

                           'var_16','var_37','var_46','var_65','var_69','var_79',

                             'var_96','var_117','var_124','var_126','var_185','var_189',

                           'var_61','var_60','var_47'

                             ],axis=1)
## Naive Bayes

from sklearn.naive_bayes import GaussianNB



classifier = GaussianNB()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print('Accuracy of Naive bayes classifier on test set: {:.5f}'.format(accuracy_score(y_test,y_pred)))
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))



plt.figure(figsize = (5,4))

sns.set(font_scale=1.4)



sns.heatmap(df_cm, annot=True, fmt='g', cmap="Blues")

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv",index_col='ID_code')



test = test.drop(['var_7','var_10','var_17','var_29','var_30','var_27',

                             'var_38','var_39','var_41','var_73','var_98','var_61',

                             'var_100','var_103','var_136','var_153','var_158',

                             'var_161','var_183',

                             'var_16','var_37','var_46','var_65','var_69','var_79',

                             'var_96','var_117','var_124','var_126','var_185','var_189',

                             'var_61','var_60','var_47'                             

                             ],axis=1)



sol = pd.DataFrame(test.index)



mms = MinMaxScaler()

test = pd.DataFrame(mms.fit_transform(test),columns=test.columns)



sol['target']=classifier.predict(test)



sol.set_index('ID_code',inplace=True)



sol.to_csv('submission_NB')