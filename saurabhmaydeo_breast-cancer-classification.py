# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")



# Any results you write to the current directory are saved as output.
data.head()
data.info()
pd.set_option('float_format', '{:f}'.format)

data.describe().T

# Since Unnamed: 32 column doesn't have any value we are dropping it

data.drop(['Unnamed: 32', 'id'],axis = 1 ,inplace = True)
data_quality_report = pd.DataFrame(columns = ['feature','count', 'missing %','unique values' ,'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max', 'IQR'])



i = 0

for f, ser in data._get_numeric_data().iteritems():

    

    Q1 = ser.quantile(0.25)

    Q3 = ser.quantile(0.75)

    

    data_quality_report.at[i, :] = [f, ser.count(), (ser.isnull().sum()/ser.size)*100, ser.unique().size, ser.mean(), ser.std(), ser.min(), Q1, ser.median(), Q3, ser.max(), Q3 - Q1]

    i = i + 1

data_quality_report
data.isnull().sum().sum()
data['diagnosis'].value_counts().plot(kind = 'bar')
corr = data.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

f, ax = plt.subplots(figsize=(21, 19))

sns.heatmap(corr, cmap=cmap, center=0,annot = True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5});
corr.head()
#It is taking a lot of time to execute this function as there are 30 features. If you are running this on TPU or something then uncomment and run



#ns.pairplot(data);

class_mapping = {label:idx for idx,label in enumerate(np.unique(data['diagnosis']))}

data['diagnosis'] = data['diagnosis'].map(class_mapping)

data['diagnosis'].value_counts()
#data = data[selected_columns]

len(data.columns)

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

# find best scored k features

select_feature = SelectKBest(chi2, k=10).fit(data.drop('diagnosis',axis = 1 ), data['diagnosis'])



print('Score list:', select_feature.scores_)

print('Feature list:', data.columns)

select_feature
select_feature.transform(data.drop('diagnosis',axis = 1))
selected_columns = np.array(data.drop('diagnosis',axis = 1).columns)[select_feature.get_support()]

selected_columns
ABT = pd.DataFrame(select_feature.transform(data.drop('diagnosis',axis = 1)),columns=selected_columns)

y = data.diagnosis
ABT.head()
ABT.describe().T
compare = pd.DataFrame(index=['RandomForest', 'SVM', 'LogisticRegression', 'kNN', 'Naive Bayes'], 

                      columns=['Accuracy', 'f1 score', 'Precision', 'Recall'])

compare
X_train, X_test, y_train, y_test = train_test_split(ABT, y, test_size=0.2, random_state=0)

X_train, X_v, y_train, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
rf = RandomForestClassifier(bootstrap= False, criterion='entropy', max_features = 'sqrt', min_samples_leaf =1, n_estimators= 300)      

rf = rf.fit(X_train,y_train)



y_pred = rf.predict(X_v)

print(f'Accuracy on validation set is {accuracy_score(y_v,y_pred)}')

print(f'f1 score on validation set is {f1_score(y_true=y_v, y_pred=y_pred)}')

print(f'Precision on validation set is {precision_score(y_true=y_v, y_pred=y_pred)}')

print(f'Recall on validation set is {recall_score(y_true=y_v, y_pred=y_pred)}')

confmat = confusion_matrix(y_v,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

y_pred = rf.predict(X_test)



accuracy = accuracy_score(y_test,y_pred)

f1 = f1_score(y_true=y_test, y_pred=y_pred)

precision = precision_score(y_true=y_test, y_pred=y_pred)

recall = recall_score(y_true=y_test, y_pred=y_pred)





compare.at['RandomForest', :] = (accuracy, f1, precision, recall)



print(f'Accuracy on test set is {accuracy}')

print(f'f1 score on test set is {f1}')

print(f'Precision on test set is {precision}')

print(f'Recall on test set is {recall}')





confmat = confusion_matrix(y_test,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()

NormalizedABT = min_max_scaler.fit_transform(ABT)

NormalizedABT=pd.DataFrame(NormalizedABT, columns=selected_columns)

X_train, X_test, y_train, y_test = train_test_split(NormalizedABT, y, test_size=0.2, random_state=0)

X_train, X_v, y_train, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

NormalizedABT.describe().T

from sklearn import svm



clf = svm.SVC().fit(X_train, y_train)

y_pred = clf.predict(X_v)



print(f'Accuracy on validation set is {accuracy_score(y_v,y_pred)}')

print(f'f1 score on validation set is {f1_score(y_true=y_v, y_pred=y_pred)}')

print(f'Precision on validation set is {precision_score(y_true=y_v, y_pred=y_pred)}')

print(f'Recall on validation set is {recall_score(y_true=y_v, y_pred=y_pred)}')



confmat = confusion_matrix(y_v,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")



y_pred = clf.predict(X_test)



accuracy = accuracy_score(y_test,y_pred)

f1 = f1_score(y_true=y_test, y_pred=y_pred)

precision = precision_score(y_true=y_test, y_pred=y_pred)

recall = recall_score(y_true=y_test, y_pred=y_pred)



compare.at['SVM', :] = (accuracy, f1, precision, recall)



print(f'Accuracy on test set is {accuracy}')

print(f'f1 score on test set is {f1}')

print(f'Precision on test set is {precision}')

print(f'Recall on test set is {recall}')



confmat = confusion_matrix(y_test,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_v)



print(f'Accuracy on validation set is {accuracy_score(y_v,y_pred)}')

print(f'f1 score on validation set is {f1_score(y_true=y_v, y_pred=y_pred)}')

print(f'Precision on validation set is {precision_score(y_true=y_v, y_pred=y_pred)}')

print(f'Recall on validation set is {recall_score(y_true=y_v, y_pred=y_pred)}')



confmat = confusion_matrix(y_v,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

y_pred = clf.predict(X_test)



accuracy = accuracy_score(y_test,y_pred)

f1 = f1_score(y_true=y_test, y_pred=y_pred)

precision = precision_score(y_true=y_test, y_pred=y_pred)

recall = recall_score(y_true=y_test, y_pred=y_pred)



compare.at['LogisticRegression', :] = (accuracy, f1, precision, recall)



print(f'Accuracy on test set is {accuracy}')

print(f'f1 score on test set is {f1}')

print(f'Precision on test set is {precision}')

print(f'Recall on test set is {recall}')





confmat = confusion_matrix(y_test,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=19).fit(X_train, y_train)



y_pred = clf.predict(X_v)



print(f'Accuracy on validation set is {accuracy_score(y_v,y_pred)}')

print(f'f1 score on validation set is {f1_score(y_true=y_v, y_pred=y_pred)}')

print(f'Precision on validation set is {precision_score(y_true=y_v, y_pred=y_pred)}')

print(f'Recall on validation set is {recall_score(y_true=y_v, y_pred=y_pred)}')



confmat = confusion_matrix(y_v,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

y_pred = clf.predict(X_test)



accuracy = accuracy_score(y_test,y_pred)

f1 = f1_score(y_true=y_test, y_pred=y_pred)

precision = precision_score(y_true=y_test, y_pred=y_pred)

recall = recall_score(y_true=y_test, y_pred=y_pred)



compare.at['kNN', :] = (accuracy, f1, precision, recall)



print(f'Accuracy on test set is {accuracy}')

print(f'f1 score on test set is {f1}')

print(f'Precision on test set is {precision}')

print(f'Recall on test set is {recall}')



confmat = confusion_matrix(y_test,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

from sklearn.naive_bayes import GaussianNB



clf = GaussianNB().fit(X_train, y_train)



y_pred = clf.predict(X_v)



print(f'Accuracy on validation set is {accuracy_score(y_v,y_pred)}')

print(f'f1 score on validation set is {f1_score(y_true=y_v, y_pred=y_pred)}')

print(f'Precision on validation set is {precision_score(y_true=y_v, y_pred=y_pred)}')

print(f'Recall on validation set is {recall_score(y_true=y_v, y_pred=y_pred)}')



confmat = confusion_matrix(y_v,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

y_pred = clf.predict(X_test)



accuracy = accuracy_score(y_test,y_pred)

f1 = f1_score(y_true=y_test, y_pred=y_pred)

precision = precision_score(y_true=y_test, y_pred=y_pred)

recall = recall_score(y_true=y_test, y_pred=y_pred)



compare.at['Naive Bayes', :] = (accuracy, f1, precision, recall)



print(f'Accuracy on test set is {accuracy}')

print(f'f1 score on test set is {f1}')

print(f'Precision on test set is {precision}')

print(f'Recall on test set is {recall}')



confmat = confusion_matrix(y_test,y_pred)

sns.heatmap(confmat,annot=True,fmt="d")

compare