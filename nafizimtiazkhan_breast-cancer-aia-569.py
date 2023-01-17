!pip install --upgrade pip

%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

import numpy as np

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

! pip install -q scikit-plot

import scikitplot as skplt

import numpy as np

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



df = pd.read_csv('../input/instances-569/569.csv')



df.head(100)
df.drop(columns=['ID number'])
df["Diagnosis"] = df["Diagnosis"].astype('category')

df['Diagnosis']=df['Diagnosis'].cat.codes

df["Diagnosis"]=df["Diagnosis"].astype('float')

df.head()
df = df.replace([np.inf, -np.inf], np.nan)

df = df.replace('?', np.nan)

df = df.fillna(method='ffill')
df = df.fillna(df.mean())

df.to_csv('df_test.csv', index=False, encoding='utf-8')

df.tail()
from imblearn.over_sampling import SMOTE



# for reproducibility purposes

seed = 100

# SMOTE number of neighbors

k = 1



#df = pd.read_csv('df_imbalanced.csv', encoding='utf-8', engine='python')

# make a new df made of all the columns, except the target class

X = df.loc[:, df.columns != 'Diagnosis']

y = df.Diagnosis

sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=seed)

X_res, y_res = sm.fit_resample(X, y)



# plt.title('base')

# plt.xlabel('x')

# plt.ylabel('y')

# plt.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=y_res,

#            s=25, edgecolor='k', cmap=plt.cm.coolwarm)

# plt.show()



df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)

# rename the columns

#df.columns = ['e1/do']+['e2/do']+['fu/fy']+['fmx/fndt']+['type']

df.to_csv('df_smoted.csv', index=False, encoding='utf-8')

df.tail()
print(df.columns)

print(df.info())

df = df.replace([np.inf, -np.inf], np.nan)

df = df.fillna(df.mean())

data=df

print(data.head())

print(data.shape)
print(data.info())

features = list(data.columns.values)

print(features)
corr = data.corr() 

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.0) | (corr <= -0.0)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
# Plottinf correlation above or below 0.5

corr = data.corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
print(data.head())

X = data.loc[:, data.columns != 'Diagnosis']

y=data['Diagnosis']

print(X.head())

print(y.head())
evaluation = pd.DataFrame({'Model': [],

                           'Accuracy(train)':[],

                           'Precision(train)':[],

                           'Recall(train)':[],

                           'F1_score(train)':[],

                           'Accuracy(test)':[],

                           'Precision(test)':[],

                           'Recalll(test)':[],

                           'F1_score(test)':[]})



evaluation2 = pd.DataFrame({'Model': [],

                           'Test':[],

                           '1':[],

                           '2':[],

                           '3':[],

                           '4':[],

                           '5':[],

                           '6':[],

                           '7':[],

                           '8':[],

                           '9':[],

                           '10':[],

                           'Mean':[]})




from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn import svm

clf =svm.SVC(kernel='rbf',degree=10)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['SVM',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)



p=y_train

q=y_test

y_train=y_train.replace([2,4], ["benign","malignant"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([2,4], ["benign","malignant"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([2,4], ["benign","malignant"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([2,4], ["benign","malignant"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn.ensemble import RandomForestClassifier

clf =RandomForestClassifier()

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Random Forest',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)





features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
p=y_train

q=y_test

y_train=y_train.replace([2,4], ["benign","malignant"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([2,4], ["benign","malignant"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([2,4], ["benign","malignant"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([2,4], ["benign","malignant"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn.neighbors import KNeighborsClassifier

clf =KNeighborsClassifier(n_neighbors=1)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['KNN',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)

p=y_train

q=y_test

y_train=y_train.replace([2,4], ["benign","malignant"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([2,4], ["benign","malignant"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([2,4], ["benign","malignant"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([2,4], ["benign","malignant"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn.naive_bayes import GaussianNB

clf =GaussianNB()

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Naive Bayes',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)

p=y_train

q=y_test

y_train=y_train.replace([2,4], ["benign","malignant"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([2,4], ["benign","malignant"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([2,4], ["benign","malignant"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([2,4], ["benign","malignant"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
evaluation.to_csv('eval.csv')