# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
import re
data = pd.read_csv('/kaggle/input/income-classification/income_evaluation.csv')
data.head()
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',

             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']



data.columns = col_names

data.columns
data.shape
data.info()
print("%18s %10s   %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))

for col in data.columns:

    dtyp = data[col].dtype

    uniq = data[col].nunique()

    na = data[col].isna().sum()

    print("%18s %10s %10s %10s" % (col, dtyp, uniq, na))

    
categorical_feature = [feature for feature in data.columns if data[feature].dtype == 'O']

print("len of categorical_feature:",len(categorical_feature))
data[categorical_feature].head()
# there is an extra space before each value of categorical column so correct it.

for col in categorical_feature:

    data[col] = data[col].str.strip()
data['workclass'].value_counts()
data['workclass'] = np.where(data['workclass']=='?',np.NaN,data['workclass'])
plt.figure(figsize=(8,8))

data['workclass'].value_counts().plot(kind='bar')

plt.xticks(rotation=45)

plt.show()
data['marital_status'].value_counts()

# it is good
plt.figure(figsize=(8,8))

sns.countplot(x='income',hue='marital_status',data=data)

plt.show()
data['occupation'].value_counts()
# same problem in occupation is there

data['occupation'] = np.where(data['occupation']=='?',np.NaN, data['occupation'])
plt.figure(figsize=(8,8))

data['occupation'].value_counts().sort_values().plot(kind='bar')

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(x='income',hue='occupation',data=data,palette='Dark2')

plt.legend(loc='best')

plt.show()
# relationship

data['relationship'].value_counts()
plt.figure(figsize=(8,8))

sns.countplot(x='income',hue='relationship',data=data,palette='Paired')

plt.legend(loc='best')

plt.show()
# RACE

data['race'].value_counts()
# let's see income with respect to sex

plt.figure(figsize=(8,6))

sns.countplot(x='income',hue='race',data=data,palette='Set1')

plt.title("Frequency dist of Income W.R.T to Race")

plt.show()
data.sex.value_counts()
# let's see income with respect to sex

plt.figure(figsize=(8,8))

sns.countplot(x='income',hue='sex',data=data,palette='magma')

plt.show()
data['native_country'].value_counts()
data['native_country'].replace('?',np.NaN,inplace=True) 
# Education

plt.figure(figsize=(8,8))

#data['education'].value_counts().plot(kind='bar')

sns.countplot(x='income',hue='education',data=data,palette='Dark2')

plt.show()
# Education

plt.figure(figsize=(8,8))

#data['education'].value_counts().plot(kind='bar')

sns.countplot(x='income',hue='education_num',data=data,palette='Dark2')

plt.show()
data['education'].value_counts().sort_values()
data.groupby('education')['education_num'].unique().sort_values()

#here is a encoded number for each education.
numerical_feature = [feature for feature in data.columns if data[feature].dtype != 'O']

print("len of numerical_feature:",len(numerical_feature))
data[numerical_feature].head()
sns.set_color_codes()

sns.distplot(data['age'],hist=False)

plt.show()
# Income wrt age

plt.figure(figsize=(8,6))

sns.boxplot(x='income',y='age',data=data)

plt.show()
# Income wrt age

plt.figure(figsize=(10,8))

ax = sns.boxplot(x='income',y='age',hue='sex',data=data)

ax.set_title("visualize Income wrt sex and age")

ax.legend(loc='best')

plt.show()
plt.figure(figsize=(10,8))

ax = sns.boxplot(x='income',y='hours_per_week',hue='sex',data=data)

ax.set_title("visualization of income wrt hours work done per week and sex")

plt.show()
data['income'].value_counts()
plt.figure(figsize=(6,6))

data['income'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',shadow=True,startangle=15)

plt.xlabel("Income_Share")

plt.show()
data[categorical_feature].isnull().sum()
for col in categorical_feature:

    if(data[col].isnull().mean() > 0):

        print(col, ' ', data[col].isnull().mean())
data['workclass'].mode()[0]
data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)

data['occupation'].fillna(data['occupation'].mode()[0], inplace=True)

data['native_country'].fillna(data['native_country'].mode()[0], inplace=True)
data.head(3)
categorical_feature
from sklearn.preprocessing import LabelEncoder
cols = ['sex','income']

for col in cols:

    data[col] = pd.get_dummies(data[col],drop_first=True) 
data.head(3)   # 0=F/1=M & 0=<=50k
categorical = [feature for feature in data.columns if data[feature].dtype == 'O']

categorical
data.head(3)
data['workclass'].unique()
#data.groupby('workclass')['income'].mean().sort_values().index
for col in categorical:

    labels = data.groupby(col)['income'].mean().sort_values().index

    print(col,'\n',labels,'\n')
for col in categorical:

    labels = data.groupby(col)['income'].mean().sort_values().index

    mapping_dict = {k: i for i, k in enumerate(labels, 0)}

    # apply encoding to our data

    data[col] = data[col].map(mapping_dict)
data.head(3)
plt.figure(figsize=(8,8))

sns.distplot(data['age'])

plt.show()
import scipy.stats as stat

import pylab   # for probability plot
def plot_data(df,variable):

    plt.figure(figsize=(10,8))

    plt.subplot(1,2,1)

    df[variable].hist()

    plt.subplot(1,2,2)

    stat.probplot(df[variable],dist='norm',plot=pylab)

    plt.show()
plot_data(data,'age')
plot_data(data,'hours_per_week')
data.corr()['income'][:].sort_values(ascending=False)
from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier
data.head(3)
x = data.drop('income',axis=1)

y = data['income']
from sklearn.preprocessing import MinMaxScaler



minmax = MinMaxScaler()



X = minmax.fit_transform(x)
extra_tree = ExtraTreesClassifier(n_estimators=5, criterion='entropy', max_features=3)



extra_tree.fit(X,y)
feature_imp = extra_tree.feature_importances_

feature_imp
# Normalizing the individual importances 

#feature_importance_normalized = np.std([tree.feature_importances_ for tree in 

 #                                       extra_tree.estimators_], 

  #                                      axis = 0)

#feature_importance_normalized 
'''

plt.figure(figsize=(8,8))

plt.barh(x.columns,feature_importance_normalized )

plt.ylabel('feature labels')

plt.xlabel('target')

plt.show()

'''
plt.figure(figsize=(11,9))

feat_imp = pd.Series(extra_tree.feature_importances_, index=x.columns)

feat_imp.nlargest(20).plot(kind = 'barh')

plt.show()
from sklearn.model_selection import train_test_split



from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



#metrics

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
svc_clf = SVC()

knn_clf = KNeighborsClassifier()

dt_clf = DecisionTreeClassifier()

rf_clf = RandomForestClassifier()
for clf in (svc_clf, knn_clf, dt_clf, rf_clf):

    clf.fit(x_train, y_train)

    

    y_pred = clf.predict(x_test)

    

    print(clf.__class__.__name__, accuracy_score(y_test,y_pred))
rf_clf.fit(x_train,y_train)



ypred = rf_clf.predict(x_test)



print("accuracy:",accuracy_score(y_test,ypred))

print("\nclassification_report:\n",classification_report(y_test,y_pred))
# VotingClassifier

from sklearn.ensemble import VotingClassifier



vot_clf = VotingClassifier([('svc',svc_clf),('knn',knn_clf),('dt',dt_clf),('rf',rf_clf)],voting='hard')



vot_clf.fit(x_train,y_train)



mypred = vot_clf.predict(x_test)



print("accuracy:",accuracy_score(y_test,mypred))
#The number of trees in the forest

n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]

#measure quality of split

criterion = ["gini", "entropy"]

# The maximum depth of the tree.

max_depth = [int(x) for x in np.linspace(6,30,5)]

# The minimum number of samples required to split an internal node:

min_samples_split = [2,7,10,15]

#The minimum number of samples required to be at a leaf node.

min_samples_leaf = [2,5,7]

# The number of features to consider when looking for the best split:

max_features = ['sqrt','auto']



random_grid = dict(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,

                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,

                  max_features=max_features)

print(random_grid)
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(rf_clf, param_distributions=random_grid, cv=5,n_iter=15, random_state=42,n_jobs=-1,verbose=1)



rf_random.fit(x_train,y_train)
rf_random.best_params_
rf_random.best_score_
rf_clf = RandomForestClassifier(n_estimators= 1000,min_samples_split= 2,

                                min_samples_leaf= 2,max_features= 'auto',max_depth= 18,

                               criterion= 'entropy')



rf_clf.fit(x_train, y_train)



pred = rf_clf.predict(x_test)



print("accuracy:",accuracy_score(y_test,y_pred))

print("\nclassification_report:",classification_report(y_test,y_pred))
import pickle



file = open('incomeclf.pkl','wb')

pickle.dump(rf_clf,file)
clf = pickle.load(open('incomeclf.pkl','rb'))
predi = clf.predict(x_test)

print(accuracy_score(y_test,predi))