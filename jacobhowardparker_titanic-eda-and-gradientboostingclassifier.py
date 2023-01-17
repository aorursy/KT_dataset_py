

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_train=pd.read_csv('../input/titanic/train.csv')

df_test=pd.read_csv('../input/titanic/test.csv')

print(df_train.columns)

print(df_train.shape)

df_train.head(5)
#We check our test data is as expected. 

print(df_test.columns)

print(df_test.shape)
#Dropping passengerId and Ticket. We store the id feature as it is needed for submission. 

id_train=df_train['PassengerId']

id_test=df_test['PassengerId']

df_train.drop('PassengerId', axis=1, inplace=True)

df_test.drop('PassengerId', axis=1, inplace=True)

df_train.drop('Ticket', axis=1, inplace=True)

df_test.drop('Ticket', axis=1, inplace=True)
#merging data

y=df_train.Survived

df_all=pd.concat([df_train, df_test], sort=False).reset_index(drop=True)

df_all.drop('Survived',axis=1, inplace=True)

df_all.describe()
#null values

missing_values=df_all.isnull().sum()

missing_values.sort_values(ascending=False, inplace=True)

missing_values
df_all.drop('Cabin', axis=1, inplace=True)

df_all['Age']=df_all['Age'].fillna(df_all['Age'].median())

df_all['Fare']=df_all['Fare'].fillna(df_all['Fare'].median())

df_all['Embarked']=df_all['Embarked'].fillna(df_all['Embarked'].mode()[0])
#splitting back to train and test for EDA of train

df_train=df_all[:891]

df_test=df_all[891:]

df_train['Survived']=y
#our current list of features post changes above

df_train.columns
sns.countplot('Survived', data=df_train)

print('Percentage dead: ', round(100*(df_train.Survived==0).sum()/df_train.shape[0], 1))
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

sns.countplot('Sex', hue='Survived', data=df_train)

plt.title('Survival by Sex')

plt.subplot(1,2,2)

sns.countplot('Pclass', hue='Survived', data=df_train)

plt.title('Survival by Pclass')
sns.catplot('Pclass', 'Survived', hue='Sex', data=df_train, kind='point')

plt.title('Sex vs Pclass in Predicting Survival')
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)

g=sns.countplot('SibSp', hue='Survived',data=df_train)

plt.title('Num Siblings/Spouses Survival Rates')

plt.legend(loc='upper right')

plt.subplot(1,3,2)

sns.countplot('Parch', hue='Survived', data=df_train)

plt.title('Num parents/Children Survival Rates')

plt.legend(loc='upper right')

plt.subplot(1,3,3)

sns.countplot('Embarked', hue='Survived', data=df_train)

plt.title('Location Embarked Survival Rates')

plt.legend(loc='upper right')
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

sns.countplot('Embarked', hue='Pclass', data=df_train)

plt.title('Embarked vs Pclass')

plt.legend(loc='upper right')

plt.subplot(1,2,2)

sns.countplot('Embarked', hue='Sex', data=df_train)

plt.title('Embarked vs Sex')

plt.legend(loc='upper right')
plt.figure(figsize=(16,8))

plt.subplot(1,2,1)

plt.hist(x=[df_train[df_train['Survived']==1].Age, df_train[df_train['Survived']==0].Age], stacked=True, color=['orange','blue'],label=['Survivors', 'Non-survivors'],bins=20)

plt.title('Distribution of survivors and non-survivors by age')

plt.legend(loc='best')

plt.subplot(1,2,2)

plt.hist(x=[df_train[df_train['Survived']==1].Fare, df_train[df_train['Survived']==0].Fare], stacked=True, color=['orange','blue'],label=['Survivors', 'Non-survivors'],bins=20)

plt.title('Distribution of survivors and non-survivors by fare')

plt.legend(loc='best')
fig =plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

sns.violinplot('Pclass', 'Age', hue='Survived', data=df_train, split=True)

plt.title('Pclass vs Age Survival')

plt.subplot(1,2,2)

sns.violinplot('Sex', 'Age', hue='Survived', data=df_train, split=True)

plt.title('Sex vs Age Survival')
fig =plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

sns.violinplot('Pclass', 'Fare', hue='Survived', data=df_train, split=True)

plt.title('Pclass vs Age Survival')

plt.subplot(1,2,2)

sns.violinplot('Sex', 'Fare', hue='Survived', data=df_train, split=True)

plt.title('Sex vs Fare Survival')
df_all=pd.concat([df_train, df_test], sort=False).reset_index(drop=True)

df_all.drop('Survived', axis=1, inplace=True)
df_all['Title']=df_all.Name.str.extract('([A-Za-z]+)\.')

df_all['Title'].unique()
plt.figure(figsize=(16,5))

sns.countplot('Title', data=df_all)
#group titles into 3 categories

def title_converter(title):

    if title=='Mr' or title=='Mrs':

        return 'Common old'

    elif title=='Miss' or title=='Master':

        return 'Common young'

    else: 

        return 'Rare'

df_all['Title']=df_all['Title'].apply(title_converter)

sns.countplot('Title', data=df_all)

plt.title('Title Type Count')
df_all.drop('Name', axis=1, inplace=True)
df_all['Family_size']=df_all['Parch']+df_all['SibSp']+1 # the 1 is to include the person themselves

sns.countplot('Family_size', data=df_all)

plt.title('Family size count')
df_all.drop(['SibSp','Parch'], axis=1, inplace=True)
df_train=df_all[:891]

df_test=df_all[891:]

df_train['Survived']=y
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)

sns.countplot('Title', hue='Survived', data=df_train)

plt.title('Survived by Title')



plt.subplot(1,3,2)

sns.violinplot('Title', 'Age', hue='Survived',data=df_train, split=True)

plt.title('Title and Age')



plt.subplot(1,3,3)

sns.violinplot('Title','Fare', hue='Survived', data=df_train, split=True)

plt.title('Title and Fare')



           

sns.catplot('Title', 'Survived', hue='Pclass', data=df_train, kind='point')

plt.title('Title and Pclass')



plt.figure(figsize=(18,15))

plt.subplot(3,1,1)

sns.countplot('Family_size', hue='Survived', data=df_train)

plt.title('Survived by Family_size')



plt.subplot(3,1,2)

sns.violinplot('Family_size', 'Age', hue='Survived',data=df_train, split=True)

plt.title('Family_size and Age')



plt.subplot(3,1,3)

sns.violinplot('Family_size','Fare', hue='Survived', data=df_train, split=True)

plt.title('Family_size and Fare')



sns.catplot('Family_size', 'Survived', hue='Pclass', data=df_train, kind='point')

plt.title('Family_size and Pclass')



sns.catplot('Family_size', 'Survived', hue='Sex', data=df_train, kind='point')

plt.title('Family_size and Sex')
for col in ['Fare', 'Age']:

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)

    sns.distplot(df_all[col], fit=stats.norm, hist_kws=dict(edgecolor='w', linewidth=2))

    plt.subplot(1,2,2)

    stats.probplot(df_all[col],plot=plt)

    plt.title('ProbPlot %s' % col)

    print('Skewness of '+col+ ' ' +str(round(stats.skew(df_all[col]),3)))

    print('Kurtosis of '+col+ ' ' +str(round(stats.kurtosis(df_all[col]),3)))
df_all['Fare']=np.log1p(df_all['Fare'])



plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

sns.distplot(df_all['Fare'], fit=stats.norm, hist_kws=dict(edgecolor='w', linewidth=2))

plt.subplot(1,2,2)

stats.probplot(df_all['Fare'],plot=plt)



print('Skewness: ' +str(round(stats.skew(df_all['Fare']),3)))

print('Kurtosis:'  +str(round(stats.kurtosis(df_all['Fare']),3)))
df_all['Age']=(df_all['Age']-df_all['Age'].mean())/df_all['Age'].std()
df_all.head()
df_all=pd.get_dummies(df_all, drop_first=True)

df_all.head()
corr_mat=df_all.corr()

fig=plt.figure(figsize=(20,10))

sns.heatmap(corr_mat, vmax=1, center=0, cmap='bwr', annot=True)
X_train=df_all[:891]

X_test=df_all[891:]

y_train=y
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

from sklearn.svm import LinearSVC, SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
def scores(model):

    kfold=KFold(5, shuffle=True, random_state=23).get_n_splits(X_train.values)

    scores=cross_val_score(model, X_train, y_train, cv=kfold) #leave scoring as default=None to use built in scorer. 

    return (scores.mean(), scores.std())



def optimise_model(model, parameters):

    kfold=KFold(5, shuffle=True, random_state=23).get_n_splits(X_train.values)

    model_grid=GridSearchCV(model, parameters, cv=kfold)

    model_grid.fit(X_train, y_train)

    print('Best estimator: ', model_grid.best_estimator_)

    print('Score: ', model_grid.best_score_)

    print('Sd: ', model_grid.cv_results_['std_test_score'][model_grid.best_index_])

    #return model_grid

    return model_grid
models=[LinearSVC(),SVC(),LogisticRegression(), KNeighborsClassifier(), \

        RandomForestClassifier(random_state=2), AdaBoostClassifier(random_state=3), \

        GradientBoostingClassifier(random_state=5), XGBClassifier(random_state=7)]

base_scores_list=[]

for model in models: 

    score=scores(model)

    base_scores_list.append([model.__class__.__name__,score[0],score[1]])

base_scores_df=pd.DataFrame(data=base_scores_list, columns=['Model', 'Mean_score','Std'])

base_scores_df.sort_values(by='Mean_score', ascending=False, inplace=True)

base_scores_df
sns.barplot(x='Mean_score',y='Model',data=base_scores_df)
#model=GradientBoostingClassifier(random_state=5)

#parameters={'learning_rate':np.arange(0.01,0.1,0.01),'n_estimators': np.arange(500,2000,500) , 'max_depth':np.arange(3,7,2)}

#gb_grid=optimise_model(model, parameters)
gb1=GradientBoostingClassifier(learning_rate=0.01, n_estimators=1500, random_state=5)

print(scores(gb1))
#parameters={'max_depth':[2,3], 'loss':['deviance','exponential'],'criterion':['friedman_mse','mse','mae'],'tol':[0.0001,0.00001]}

#gb_grid2=optimise_model(gb1, parameters)

#DID NOT IMPROVE FIT
gb1.fit(X_train, y_train)

predictions=gb1.predict(X_test)

submission=pd.DataFrame({'PassengerId':id_test, 'Survived':predictions})

submission.to_csv('submission.csv',index=False)