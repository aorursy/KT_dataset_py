# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/datamaestro2020/astro_train.csv")

test=pd.read_csv("/kaggle/input/datamaestro2020/astro_test.csv")

sub=pd.read_csv("/kaggle/input/datamaestro2020/sample_submission.csv")
train.head()
train.columns
import matplotlib.pyplot as plt

visible_band = ['dered_u', 'dered_g', 'dered_i', 'dered_r', 'dered_z']

ax=[]

for i,band in enumerate(visible_band):

    fig1 = plt.figure()

    ax.append(fig1.add_subplot(3,2,i+1))

    ax[i].hist(train[band])

    ax[i].set_title(band)

plt.show()
train['dered_i'].describe()

features =['dered_u', 'dered_g', 'dered_i', 'dered_r', 'dered_z','#ra', 'dec','class']

for feature in features:

    total_objects = sum(train[feature]!=-9999)

    print("The number of objects for feature: %s is %d"%(feature,total_objects))
train['ug'] = train['dered_u'] - train['dered_g']

train['gr'] = train['dered_g'] - train['dered_r']

train['ri'] = train['dered_r'] - train['dered_i']

train['iz'] = train['dered_i'] - train['dered_z']
del train['dered_u']

del train['dered_g']

del train['dered_r']

del train['dered_i']

del train["dered_z"]
test['ug'] = test['dered_u'] - test['dered_g']

test['gr'] = test['dered_g'] - test['dered_r']

test['ri'] = test['dered_r'] - test['dered_i']

test['iz'] = test['dered_i'] - test['dered_z']
del test['dered_u']

del test['dered_g']

del test['dered_r']

del test['dered_i']

del test["dered_z"]
train["class"].value_counts()
train.shape
train.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

sns.heatmap(train.corr(),annot=True)
train["skyVersion"].value_counts()
train["rerun"].value_counts()
del train["skyVersion"]

del train["rerun"]
del test["skyVersion"]

del test["rerun"]
del train["id"]

del test["id"]
#del train["run"]

#del test["run"]
#del train["#ra"]

#del test["#ra"]
#del train["dec"]

#del test["dec"]
#del train["camCol"]

#del test["camCol"]
#train["run"]=train["run"].replace([752,745],[1,0])

#train["camCol"]=train["camCol"].replace([2,3,1],[3,2,1])
#test["run"]=test["run"].replace([752,745],[1,0])

#test["camCol"]=test["camCol"].replace([2,3,1],[3,2,1])
from sklearn.decomposition import PCA





pca = PCA(n_components=3)





#Training data

pc = pca.fit_transform(train[['camCol','field','obj','#ra','dec']])

train = pd.concat((train, pd.DataFrame(pc)), axis=1)

train.rename({0: 'pca_1', 1: 'pca_2', 2: 'pca_3'}, axis=1, inplace = True)

train.drop(['camCol','field','obj','#ra', 'dec'], axis = 1, inplace=True)



#Testing data

pc1 = pca.fit_transform(test[['camCol', 'field','obj','#ra','dec']])

test = pd.concat((test, pd.DataFrame(pc1)), axis=1)

test.rename({0: 'pca_1', 1: 'pca_2',2: 'pca_3'}, axis=1, inplace = True)

test.drop(['camCol', 'field','obj','#ra', 'dec'], axis = 1, inplace=True)



'''pca = PCA(n_components=5)





#Training data

pc = pca.fit_transform(train[['run', '#ra', 'dec', 'camCol', 'field','obj','extinction_r']])

train = pd.concat((train, pd.DataFrame(pc)), axis=1)

train.rename({0: 'pca_1', 1: 'pca_2', 2: 'pca_3',3: 'pca_4',4:'pca_5'}, axis=1, inplace = True)

train.drop(['run', '#ra', 'dec', 'camCol', 'field','obj','extinction_r'], axis = 1, inplace=True)



#Testing data

pc1 = pca.fit_transform(test[['run', '#ra', 'dec', 'camCol', 'field','obj','extinction_r']])

test = pd.concat((test, pd.DataFrame(pc1)), axis=1)

test.rename({0: 'pca_1', 1: 'pca_2', 2: 'pca_3',3: 'pca_4',4:'pca_5'}, axis=1, inplace = True)

test.drop(['run', '#ra', 'dec', 'camCol', 'field','obj','extinction_r'], axis = 1, inplace=True)'''



'''#Training data

pca_error = PCA(n_components=2)

pc_er = pca_error.fit_transform(train[['err_i', 'err_z', 'err_r']])

train = pd.concat((train, pd.DataFrame(pc_er)), axis=1)

train.rename({0: 'err_1', 1: 'err_2'}, axis=1, inplace = True)

train.drop(['err_i', 'err_z', 'err_r'], axis = 1, inplace=True)



#Testing data

pca_error1 = PCA(n_components=2)

pc_er1 = pca_error1.fit_transform(test[['err_i', 'err_z', 'err_r']])

test = pd.concat((test, pd.DataFrame(pc_er1)), axis=1)

test.rename({0: 'err_1', 1: 'err_2'}, axis=1, inplace = True)

test.drop(['err_i', 'err_z', 'err_r'], axis = 1, inplace=True)'''
train.head(1)
import seaborn as sns

#sns.heatmap(train.corr())

corr=train.corr()

corr.style.background_gradient()
y=train["class"]

del train["class"]

x=train
print(train.shape , test.shape , x.shape)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.linear_model import LassoCV

reg = LassoCV()

reg.fit(x_train,y_train)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(x_train,y_train))

coef = pd.Series(reg.coef_, index = x_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")



imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8,8)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

scaler = StandardScaler()

x_train1 = scaler.fit_transform(x_train)

x_test1 = scaler.transform(x_test)
from sklearn import linear_model

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

classifier = RandomForestClassifier( n_estimators = 600,

                                    random_state = 42)

classifier.fit(x_train1, y_train)
pred = classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix as cm

print(cm(pred,y_test))
from sklearn.metrics import f1_score as f1

print(f1(pred,y_test,average=None))
from sklearn import metrics 

print("Acc :",metrics.accuracy_score(y_test,pred))
test = scaler.transform(test)

pred1=classifier.predict(test)
sub["class"]=pred1
sub.head()
sub.to_csv('sub1.csv',index=False)

from IPython.display import FileLink

FileLink(r'sub1.csv')