# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as matplot



from matplotlib import pyplot as plt

from matplotlib.pyplot import show

from sklearn import metrics

from sklearn.model_selection import train_test_split

from scipy.stats import zscore

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from mlxtend.feature_selection import SequentialFeatureSelector as sfs





pd.set_option("display.max_rows",None)

pd.set_option("display.max_columns",None)
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

sub_df=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
train_df.dtypes
test_df.head()
print(train_df.shape)

print(test_df.shape)

# Checking for null values

train_df.isnull().sum()
test_df.isnull().sum()
train_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)

train_df[train_df['Embarked'].isnull()]
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0],inplace=True)
train_df['Age'].fillna(train_df['Age'].median(),inplace=True) # imputing with mdeian as there are some outlier, since only few people have travelled whose age is close to eighty. using mean will not be suitable
test_df['Age'].fillna(test_df['Age'].median(),inplace=True) # imputing with mdeian as there are some outlier, since only few people have travelled whose age is close to eighty. using mean will not be suitable
train_df.drop(['Name'],axis=1,inplace=True)

test_df.drop(['Name'],axis=1,inplace=True)
train_df.describe().transpose()
#sns.pairplot(train_df,hue='Survived')


total = float(len(train_df))

ax = sns.countplot(train_df['Survived']) # for Seaborn version 0.7 and more

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

show()
total = float(len(train_df))

ax = sns.countplot(train_df['Pclass']) # for Seaborn version 0.7 and more

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

show()
sns.distplot(train_df['Age'])
total = float(len(train_df))

ax = sns.countplot(train_df['Sex']) # for Seaborn version 0.7 and more

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

show()
total = float(len(train_df))

ax = sns.countplot(train_df['Embarked']) # for Seaborn version 0.7 and more

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

show()
total = float(len(train_df))

ax = sns.countplot(train_df['SibSp']+train_df['Parch']) # for Seaborn version 0.7 and more

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

show()
sns.distplot(train_df['Fare'])
train_df.head()
sns.countplot(x='Survived',data=train_df)

total = float(len(train_df))

ax = sns.countplot(x='Survived',hue='Pclass',data=train_df) # for Seaborn version 0.7 and more

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

show()
p_cont1 = pd.crosstab(train_df['Pclass'], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Pclass'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Pclass',  data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Pclass', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate Passenger class wise", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
p_cont1 = pd.crosstab(train_df['Sex'], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Sex'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Sex',  data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Sex', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate Genderwise", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
p_cont1 = pd.crosstab(train_df['Embarked'], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Embarked'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Embarked',  data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Embarked', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate based on Embarked", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
train_df[train_df['Fare']==0]
def personclassifier(x):

    age, sex = x

    if age <=12:

        return 'Child'

    if age > 12 and age <=19:

        return 'Teenager'

    if age >19 and age <=30:

        return 'Young Adult'

    if age >30 and age <= 45:

        return 'Mid aged Adult'

    if age >45:

        return 'Old Adults'

    

    

    
def singletraveller(x):

    single  = x

    if single ==0:

        return 'Yes'

    else:

        return 'No'   
train_df['Single_Traveller'] = train_df['SibSp']+train_df['Parch']
train_df['Single_Traveller'] = train_df['Single_Traveller'].apply(singletraveller)
test_df['Single_Traveller'] = test_df['SibSp']+test_df['Parch']
test_df['Single_Traveller'] = test_df['Single_Traveller'].apply(singletraveller)
train_df['Person'] = train_df[['Age','Sex']].apply(personclassifier,axis=1)
test_df['Person'] = test_df[['Age','Sex']].apply(personclassifier,axis=1)
p_cont1 = pd.crosstab(train_df['Person'], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Person'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Person',  data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Person', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate Age Group wise", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
p_cont1 = pd.crosstab([train_df['Person'],train_df['Sex']], train_df['Survived'], normalize='index') * 100

p_cont1
p_cont1 = pd.crosstab([train_df['Person'],train_df['Sex']], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Person'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Person', hue='Sex', data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Person', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate By Age Group by Gender ", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
p_cont1 = pd.crosstab([train_df['Person'],train_df['Pclass']], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Person'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Person', hue='Pclass', data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Person', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate by age group by class", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
p_cont1 = pd.crosstab([train_df['Person'],train_df['Pclass']], train_df['Survived'], normalize='index') * 100

p_cont1
test_df.isnull().sum()
test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)
p_cont1 = pd.crosstab(train_df['Single_Traveller'], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Single_Traveller'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Single_Traveller', data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Single_Traveller', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate by Single Traveller", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
p_cont1 = pd.crosstab([train_df['Single_Traveller'],train_df['Sex']], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Single_Traveller'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Single_Traveller', hue='Sex', data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Single_Traveller', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate by Single traveller by gender", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
p_cont1 = pd.crosstab([train_df['Single_Traveller'],train_df['Sex']], train_df['Survived'], normalize='index') * 100

p_cont1
p_cont1 = pd.crosstab([train_df['Pclass'],train_df['Single_Traveller']], train_df['Survived'], normalize='index') * 100

p_cont1 = p_cont1.reset_index()

p_cont1.rename(columns={0:'Dead', 1:'Survived'}, inplace=True)





listv = []

for var in train_df['Pclass'].unique():

    listv.append(var)



plt.figure(figsize=(20,6))



ax1 = plt.subplot(121)



g1= sns.countplot(x='Pclass', hue='Single_Traveller', data=train_df,order=listv,palette="Set1")

gt = g1.twinx()

gt = sns.pointplot(x='Pclass', y='Survived', data=p_cont1, color='black', legend=False,order=listv)

gt.set_ylabel("% of Survived", fontsize=12)



g1.set_title("Survival Rate by class by single traveller", fontsize=14)

g1.set_ylabel("Count", fontsize=12)





 



plt.show()
p_cont1 = pd.crosstab([train_df['Pclass'],train_df['Single_Traveller']], train_df['Survived'], normalize='index') * 100

p_cont1
train_df.head()
train_df.drop('Ticket',axis=1,inplace=True)

test_df.drop('Ticket',axis=1,inplace=True)

#train_df.drop(['SibSp','Parch'],axis=1,inplace=True)

#test_df.drop(['SibSp','Parch'],axis=1,inplace=True)
col = []

for c in train_df.columns:

    if train_df[c].dtypes=='object':

        col.append(c)

        



train_df_dummies = pd.get_dummies(train_df , columns=col, drop_first=True)
corr_mat = train_df_dummies.corr()

train_df_dummies.corr()
# Getting the columns that are having multi collinearity

# Creating a dataframe with correlated column, the correlation value and the source column to which it is correlated

# Filtering only those that are correlated more than 96%

multi_col_df = pd.DataFrame(columns=['corr_col','corr_val','source_col'])

for i in corr_mat:

    temp_df = pd.DataFrame(corr_mat[corr_mat[i]>0.9][i])

    temp_df = temp_df.reset_index()

    temp_df['source_col'] = i

    temp_df.columns = ['corr_col','corr_val','source_col']

    multi_col_df = pd.concat((multi_col_df,temp_df),axis=0)
multi_col_df
X = train_df_dummies.drop(['Survived','PassengerId'],axis=1)

X_id = train_df_dummies['PassengerId']

y = train_df_dummies['Survived']
X_trainval, X_test, y_trainval, y_test = train_test_split(X,y, test_size=0.20,random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval, test_size=0.20,random_state=1)
X_trainval_z =X_trainval.apply(zscore)

X_train_z =X_train.apply(zscore)

X_val_z =X_val.apply(zscore)

X_test_z =X_test.apply(zscore)

X_z = X.apply(zscore)
# Grid Search based on Max_features, Min_Samples_Split and Max_Depth

param_grid = [

{

'n_neighbors': list(range(1,50)),

'algorithm': ['auto', 'ball_tree', 'kd_tree','brute'],

'leaf_size': [10,15,20,30],

'n_jobs': [-1], 

'weights' : ['uniform','distance']

}

]



import multiprocessing

gs = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring='accuracy', n_jobs=multiprocessing.cpu_count(),cv=3)

gs.fit(X_train_z, y_train)
gs.best_estimator_
gs.best_score_
knn_clfr = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',

                     metric_params=None, n_jobs=-1, n_neighbors=6, p=2,

                     weights='uniform')
sfs1 = sfs(knn_clfr, k_features=8, forward=False, scoring='accuracy', cv=10,n_jobs=-1)
sfs1 = sfs1.fit(X_train_z.values, y_train.values)
X_train_z.head()
sfs1.get_metric_dict()
columnList = list(X_train_z.columns)

feat_cols = list(sfs1.k_feature_idx_)

print(feat_cols)
subsetColumnList = [columnList[i] for i in feat_cols] 

print(subsetColumnList)
train_df_dummies.dtypes
train_df_dummies_knn = train_df_dummies.drop(['Age','Parch','Embarked_Q','Single_Traveller_Yes'],axis=1)

train_df_dummies_knn.head()
X_knn = train_df_dummies_knn.drop(['PassengerId','Survived'],axis=1)

X_id_knn = train_df_dummies_knn['PassengerId']

y_knn = train_df_dummies_knn['Survived']
from imblearn.combine import SMOTETomek

smk = SMOTETomek(random_state=1)

X_res,y_res=smk.fit_sample(X_knn,y_knn)
X_trainval_knn, X_test_knn, y_trainval_knn, y_test_knn = train_test_split(X_res,y_res, test_size=0.20,random_state=1)

X_train_knn, X_val_knn, y_train_knn, y_val_knn = train_test_split(X_trainval_knn,y_trainval_knn, test_size=0.20,random_state=1)
X_trainval_z_knn =X_trainval_knn.apply(zscore)

X_train_z_knn =X_train_knn.apply(zscore)

X_val_z_knn =X_val_knn.apply(zscore)

X_test_z_knn =X_test_knn.apply(zscore)

X_z_knn = X_res.apply(zscore)
import multiprocessing

gs = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring='accuracy', n_jobs=multiprocessing.cpu_count(),cv=10)

gs.fit(X_train_z_knn, y_train_knn)
gs.best_estimator_
gs.best_score_
knnclfr = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',

                     metric_params=None, n_jobs=-1, n_neighbors=6, p=2,

                     weights='uniform')
knnclfr.fit(X_train_z_knn, y_train_knn)
y_predict_knn = knnclfr.predict(X_val_z_knn)
print(knnclfr.score(X_train_z_knn,y_train_knn))

print(knnclfr.score(X_val_z_knn,y_val_knn))

print(metrics.classification_report(y_val_knn,y_predict_knn))

print(metrics.confusion_matrix(y_val_knn,y_predict_knn))

knnclfr.fit(X_trainval_z_knn, y_trainval_knn)
y_predict_knn = knnclfr.predict(X_test_z_knn)
print(knnclfr.score(X_trainval_z_knn, y_trainval_knn))

print(knnclfr.score(X_test_z_knn,y_test_knn))

print(metrics.classification_report(y_test_knn,y_predict_knn))

print(metrics.confusion_matrix(y_test_knn,y_predict_knn))

knnclfr.fit(X_z_knn,y_res)
test_df.head()
test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)
col = []

for c in test_df.columns:

    if test_df[c].dtypes=='object':

        col.append(c)

        



test_df_dummies = pd.get_dummies(test_df , columns=col, drop_first=True)
test_df_dummies.head()
test_df_dummies_knn = test_df_dummies.drop(['Age','Parch','Embarked_Q','Single_Traveller_Yes'],axis=1,inplace=True)

test_df_dummies_knn
X_test_knn = test_df_dummies.drop(['PassengerId'],axis=1)

X_test_id_knn = test_df_dummies['PassengerId']

X_test_knn =X_test_knn.apply(zscore)

y_predict = knnclfr.predict(X_test_knn)
final_pred_df = pd.DataFrame(y_predict)

final_pred_df.columns = ['Survived']

X_test_id_knn = pd.DataFrame(X_test_id_knn)

final_pred = X_test_id_knn.merge(final_pred_df, left_index=True, right_index=True)

final_pred.shape

final_pred.to_csv('csv_to_submit2602-2.csv', index = False)
from sklearn import preprocessing



lab_enc = preprocessing.LabelEncoder()

lab_enc = preprocessing.LabelEncoder()
col = []

for c in train_df.columns:

    if train_df[c].dtypes=='object':

        train_df[c] = lab_enc.fit_transform(train_df[c])

        print("Column {} has been encoded".format(c))
col = []

for c in test_df.columns:

    if test_df[c].dtypes=='object':

        test_df[c] = lab_enc.fit_transform(test_df[c])

        print("Column {} has been encoded".format(c))
X = train_df.drop(['Survived','PassengerId'],axis=1)

X_id = train_df['PassengerId']

y = train_df['Survived']
from imblearn.combine import SMOTETomek

smk = SMOTETomek(random_state=1)

X_res,y_res=smk.fit_sample(X,y)
X_trainval, X_test, y_trainval, y_test = train_test_split(X_res,y_res, test_size=0.20,random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval, test_size=0.20,random_state=1)
X_trainval_z =X_trainval.apply(zscore)

X_train_z =X_train.apply(zscore)

X_val_z =X_val.apply(zscore)

X_test_z =X_test.apply(zscore)

X_z = X_res.apply(zscore)
# Grid Search based on Max_features, Min_Samples_Split and Max_Depth

param_grid = [

{

'n_neighbors': list(range(1,50)),

'algorithm': ['auto', 'ball_tree', 'kd_tree','brute'],

'leaf_size': [10,15,20,30],

'n_jobs': [-1], 

'weights' : ['uniform','distance']

}

]
import multiprocessing

gs = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring='accuracy', n_jobs=multiprocessing.cpu_count(),cv=3)

gs.fit(X_train_z, y_train)
gs.best_estimator_
gs.best_score_
knn_clfr = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',

                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,

                     weights='uniform')
train_df.shape
sfs1 = sfs(knn_clfr, k_features=6, forward=True, scoring='accuracy', cv=10,n_jobs=-1)
sfs1 = sfs1.fit(X_train_z.values, y_train.values)
sfs1.get_metric_dict()
columnList = list(X_train_z.columns)

feat_cols = list(sfs1.k_feature_idx_)

print(feat_cols)
subsetColumnList = [columnList[i] for i in feat_cols] 

print(subsetColumnList)
train_df.head(1)
train_df_bkp = train_df.copy()

test_df_bkp = test_df.copy()
X_knn = train_df[['Pclass', 'Sex', 'Age', 'Fare', 'Single_Traveller', 'Person']]

X_id_knn = train_df['PassengerId']

y_knn = train_df['Survived']
from imblearn.combine import SMOTETomek

smk = SMOTETomek(random_state=1)

X_res,y_res=smk.fit_sample(X_knn,y_knn)
X_trainval_knn, X_test_knn, y_trainval_knn, y_test_knn = train_test_split(X_res,y_res, test_size=0.20,random_state=1)

X_train_knn, X_val_knn, y_train_knn, y_val_knn = train_test_split(X_trainval_knn,y_trainval_knn, test_size=0.20,random_state=1)
X_trainval_z_knn =X_trainval_knn.apply(zscore)

X_train_z_knn =X_train_knn.apply(zscore)

X_val_z_knn =X_val_knn.apply(zscore)

X_test_z_knn =X_test_knn.apply(zscore)

X_z_knn = X_res.apply(zscore)