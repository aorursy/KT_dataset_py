# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.model_selection import GridSearchCV

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier

import os

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

from sklearn.utils import shuffle

print(os.listdir("../input"))

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv')

df.head()
x_testFINAL=pd.read_csv('../input/test.csv')

sub=pd.read_csv('../input/sample_submission.csv')

x_testFINAL.head()
print(df.groupby('target').describe())

print("shape of dataset::" +str(df.shape))
df=shuffle(df)

df_test=df.iloc[:12000,:]

df_train=df.iloc[12000:,:]
df_train.shape
import seaborn as sns

x_train=df_train.iloc[:,2:]

y_train=df_train['target']

sns.boxplot(x=x_train['var_2'])
from scipy import stats

import numpy as np

z = np.abs(stats.zscore(df_train.iloc[:,1:]))

threshold = 3

df_train_rm = df_train[(z < threshold).all(axis=1)]

print('df_train_rm shape :'+str(df_train_rm.shape))

print('df_train shape :'+str(df_train.shape))
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
x_train1=df_train_rm.iloc[:,2:]

y_train1=df_train_rm.iloc[:,1]

min_max_scaler =MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(x_train1)

x_train1 = pd.DataFrame(np_scaled)



pca = PCA(n_components=2)

x_plot1 = pca.fit_transform(x_train1)

plot_2d_space(x_plot1,y_train1, 'after removing outlier dataset (2 PCA components)')

sns.boxplot(x=x_train1.iloc[:,5])
x_train=df_train.iloc[:,2:]

y_train=df_train['target']

min_max_scaler =MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(x_train)

x_train = pd.DataFrame(np_scaled)



pca = PCA(n_components=2)

x_plot= pca.fit_transform(x_train)

plot_2d_space(x_plot,y_train, 'before removing outlier dataset (2 PCA components)')

sns.boxplot(x=x_train.iloc[:,5])
# Class count

count_class_0, count_class_1 = df_train_rm.target.value_counts()



# Divide by class

df_train_rm_class_0 = df_train_rm[df_train_rm['target'] == 0]

df_train_rm_class_1 = df_train_rm[df_train_rm['target'] == 1]

frac=.5

df_train_rm_class_1_over = df_train_rm_class_1.sample(int(count_class_0*frac), replace=True)

df_test_over = pd.concat([df_train_rm_class_0, df_train_rm_class_1_over], axis=0)

df_test_over=shuffle(df_test_over)

print('Random over-sampling:')

print(df_test_over.target.value_counts())



df_test_over.target.value_counts().plot(kind='bar', title='Count (target)'); 
df_ov=df_test_over.iloc[:,2:]

min_max_scaler =MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df_ov)

df_normov = pd.DataFrame(np_scaled)

df_normov.head()
X_test=df_test.iloc[:,1:]

y_test=df_test['target']
X=df_normov

Y=df_test_over['target']

df_testnorm=df_test.iloc[:,2:]

min_max_scaler =MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df_testnorm)

df_testnorm = pd.DataFrame(np_scaled)



X_test=df_testnorm

y_test=df_test['target']
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
model = XGBClassifier(n_estimators=300,max_dapth=10)

model.fit(X, Y)

y_pred_xgb = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred_xgb)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
#y_predoverrfc = clf.predict(X)

#accuracy = accuracy_score(Y, y_predoverrfc)

#print("Accuracy: %.2f%%" % (accuracy * 100.0))
#clf_ada= AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),n_estimators=100)

#clf_ada.fit(X,Y)

#y_predoverada = clf_ada.predict(X_test)

#accuracy = accuracy_score(y_test, y_predoverada)

#print("Accuracy: %.2f%%" % (accuracy * 100.0))
bag=BaggingClassifier(DecisionTreeClassifier(),max_samples=.3,max_features=1.0,n_estimators=300)

bag.fit(X,Y)

y_predoverbag = bag.predict(X_test)

accuracy = accuracy_score(y_test, y_predoverbag)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
svm=SVC()
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred_xgb)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
mysat0=conf_mat[0,0]/(conf_mat[0,0]+conf_mat[0,1])

print('acc_0::'+str(mysat0))

mysat1=conf_mat[1,1]/(conf_mat[1,0]+conf_mat[1,1])

print('acc_1::'+str(mysat1))
df_ov=x_testFINAL.iloc[:,1:]

min_max_scaler =MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(df_ov)

x_test_final = pd.DataFrame(np_scaled)
y_predFINAL = pd.DataFrame(model.predict(x_test_final))

y_predexp=y_predFINAL

sub['target']=y_predexp.iloc[:,:]

sub.to_csv('submission.csv', index=False)