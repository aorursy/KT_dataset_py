import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#import missingno

%matplotlib inline
master_train_df = pd.read_csv('../input/titanic/train.csv')

master_test_df = pd.read_csv('../input/titanic/test.csv')
gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
master_train_df.info()
master_test_df.info()
master_train_df.head()
plt.figure(figsize=(20,20))

master_test_df.hist()
master_test_df.head()
train_eda = pd.DataFrame()

train_eda = master_train_df
#createing two more dataframe. 



df_bin = pd.DataFrame()  # for discreated continous numbers e.g. 1-10, 10-20, 20-30....

df_conti = pd.DataFrame()   #for continous e.g. number 1-100
train_eda.isnull().sum()
train_eda.dtypes
plt.figure(figsize=(20,1))

sns.countplot(data=train_eda,y='Survived')

print(train_eda.Survived.value_counts())

plt.title("Survior Ratio")
df_bin['Survived'] = train_eda['Survived']

df_conti['Survived'] = train_eda['Survived']
sns.distplot(train_eda.Pclass)

print(train_eda.Pclass.value_counts())
df_bin['Pclass'] = train_eda['Pclass']

df_conti['Pclass'] = train_eda['Pclass']
df_conti.head()
plt.figure(figsize=(20,1))

sns.countplot(data=train_eda,y='Sex')

print(train_eda.Sex.value_counts())

plt.title("Male Female Ratio")
df_bin['Sex'] = train_eda['Sex']

df_bin['Sex'] = np.where(df_bin['Sex']=='female',1,0)

df_conti['Sex'] = train_eda['Sex']
train_eda.Sex.head()
df_bin.head()
df_bin.Sex.value_counts()
plt.figure(figsize=(10,6))

sns.distplot(df_bin.loc[df_bin['Sex'] == 1]['Survived'], kde_kws={'label':'Female'})

sns.distplot(df_bin.loc[df_bin['Sex'] == 0]['Survived'],kde_kws={'label':'Male'})

#0=Male , 1= Female
#Missing values

train_eda.Age.isnull().sum()
df_bin['SibSp'] = train_eda['SibSp']

df_conti['SibSp'] = train_eda['SibSp']
plt.figure(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(data=df_bin, y = 'SibSp')

plt.title('no of siblings or spouse on board')

plt.subplot(1, 2, 2)

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['SibSp'], kde_kws={'label':'Survived'})

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['SibSp'],kde_kws={'label':'Not Survived'})
df_bin['Parch'] = train_eda['Parch']

df_conti['Parch'] = train_eda['Parch']
plt.figure(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(data=df_bin, y = 'Parch')

plt.title('no of Parents on board')

plt.subplot(1, 2, 2)

sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Parch'], kde_kws={'label':'Survived'})

sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Parch'],kde_kws={'label':'Not Survived'})
#train_eda['Ticket_class'] = train_eda.groupby(train_eda['Pclass'])['Ticket']
df_bin['Fare'] = pd.cut(train_eda['Fare'],5)

df_conti['Fare'] = train_eda['Fare']
df_bin['Fare'].min()
plt.figure(figsize=(20,10))

plt.subplot(1, 2, 1)

sns.countplot(data=df_bin, y = 'Fare')

plt.subplot(1, 2, 2)

#sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Fare'], kde_kws={'label':'Survived'})

#sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Fare'],kde_kws={'label':'Not Survived'},kde=True, hist=True, hist_kws={"range": [-0.512,512]})
train_eda.Embarked.value_counts()
df_bin['Embarked'] = train_eda['Embarked']

df_conti['Embarked'] = train_eda['Embarked']
plt.figure(figsize=(20,6))

plt.subplot(1, 2, 1)

sns.countplot(data=df_bin, y = 'Embarked')
df_bin = df_bin.dropna(subset=['Embarked'])

df_conti = df_conti.dropna(subset=['Embarked'])
df_bin.head()
df_conti.dtypes


df_embarked_one_hot = pd.get_dummies(df_conti['Embarked'], 

                                     prefix='embarked')



df_sex_one_hot = pd.get_dummies(df_conti['Sex'], 

                                prefix='sex')



df_plcass_one_hot = pd.get_dummies(df_conti['Pclass'], 

                                   prefix='pclass')
df_conti_encod = pd.concat([df_conti,df_embarked_one_hot,df_sex_one_hot,df_plcass_one_hot],axis=1)
df_conti_encod = df_conti_encod.drop(['Pclass', 'Sex', 'Embarked'], axis=1)
df_conti_encod.head()
model_df = pd.DataFrame()

model_df = df_conti_encod
X = model_df.drop('Survived',axis=1)

y = model_df['Survived']
X.shape
y.shape
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(X,y)
y_pred = cross_val_predict(LogisticRegression(),X,y,cv=10,n_jobs=-1)
reg_score = round(lr.score(X, y) * 100, 2)

reg_cv_score = round(accuracy_score(y,y_pred)*100,2)
print('Accuracy: ', reg_score)

print('Accuracy Cv : ', reg_cv_score)
knn = KNeighborsClassifier()
knn.fit(X,y)
k_pred = cross_val_predict(KNeighborsClassifier(),X,y,cv=10,n_jobs=-1)
knn_score = round(knn.score(X, y) * 100, 2)

knn_cv_score = round(accuracy_score(y,k_pred)*100,2)
print('Accuracy: ', knn_score)

print('Accuracy Cv : ', knn_cv_score)
rfc = RandomForestClassifier()
rfc.fit(X,y)
rfc_pred = cross_val_predict(RandomForestClassifier(),X,y,cv=10,n_jobs=-1)
rfc_score = round(rfc.score(X, y) * 100, 2)

rfc_cv_score = round(accuracy_score(y,rfc_pred)*100,2)
print('Accuracy: ', rfc_score)

print('Accuracy Cv : ', rfc_cv_score)
gbt = GradientBoostingClassifier()

gbt.fit(X,y)
gbt_pred = cross_val_predict(GradientBoostingClassifier(),X,y,cv=10,n_jobs=-1)
gbt_score = round(gbt.score(X, y) * 100, 2)

gbt_cv_score = round(accuracy_score(y,gbt_pred)*100,2)
print('Accuracy: ', gbt_score)

print('Accuracy Cv 10 Folds: ', gbt_cv_score)
cv_model = pd.DataFrame({'Model':['LogisticRegression','KNN','RandomForest','GradientBoost'],

                      'Score':[reg_cv_score,knn_cv_score,rfc_cv_score,gbt_cv_score]})

cv_model.sort_values(by='Score',ascending=False)
feature_imp = pd.DataFrame({'imp':gbt.feature_importances_,'col':X.columns})

sorted_df = feature_imp.sort_values(by=['imp','col'],ascending=False)

sorted_df.plot(kind='barh',x='col',y='imp',color=(0.2, 0.4, 0.6, 0.6),edgecolor='blue')
X.head()
y.shape
test_col=X.columns

test_col
master_test_df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']].head()
dummy_test_df = pd.DataFrame()

dummy_test_df = master_test_df
test_embarked_dummies = pd.get_dummies(dummy_test_df['Embarked'],prefix='embarked')

test_sex_dummies = pd.get_dummies(dummy_test_df['Sex'],prefix='sex')

test_pclass_dummies = pd.get_dummies(dummy_test_df['Pclass'],prefix='pclass')
dummy_test_df = pd.concat([dummy_test_df,test_embarked_dummies,test_sex_dummies,test_pclass_dummies],axis=1)
dummy_test_df.head()
test_col = X.columns

test_col
#dummy_test_df.fillna(subset=['Fare'],inplace=True)

#master_test_df.mean()

dummy_test_df.Fare.isnull().sum()
dummy_test_df.Fare.fillna(dummy_test_df.Fare.mean(),inplace=True)
predict = gbt.predict(dummy_test_df[test_col])
predict[:20]
dummy_test_df.columns
submit_df = pd.DataFrame()
submit_df['PassengerId'] = dummy_test_df['PassengerId']

submit_df['Survived'] = predict
submit_df.head()
gender_submission.head()
if len(submit_df) == len(dummy_test_df):

    print("Submission dataframe is the same length as test ({} rows).".format(len(submit_df)))

else:

    print("Dataframes mismatched, won't be able to submit to Kaggle.")
submit_df.to_csv('submission.csv',index=False)

print("Submission Ready")
check_submission = pd.read_csv('submission.csv')
check_submission.info()