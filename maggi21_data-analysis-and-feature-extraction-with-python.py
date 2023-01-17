import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
def draw_missing_data_table(df):
    total=df.isnull().sum().sort_values(ascending=False)
    percent=(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
    return missing_data
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None, n_jobs=1,train_sizes=np.linspace(.1,1.0,5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores=learning_curve(estimator,X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean=np.mean(train_scores, axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1, color="g")
    plt.plot(train_sizes,train_scores_mean,'o-', color="r",label="Training Score")
    plt.plot(train_sizes,test_scores_mean, 'o-',color="g",label="Validation Score")
    plt.legend(loc="best")
    return plt
def plot_validation_curve(estimator, title, X, y,param_name, param_range, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1,1.0,5)):
    train_scores, test_scores=validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean=np.mean(train_scores, axis=1)
    train_std=np.std(train_scores, axis=1)
    test_mean=np.mean(test_scores, axis=1)
    test_std=np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r',marker='o', markersize=5, label='Training Score')
    plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s',markersize=5, label='ValidationScore')
    plt.fill_between(param_range,test_mean+test_std, test_mean-test_std, alpha=0.15, color='g')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('Parameter')
    plt.ylabel('Score')
    plt.ylim(ylim)
df=pd.read_csv('../input/train.csv')
df_raw=df.copy()
df.head()
df.describe()
draw_missing_data_table(df)
df.drop('Cabin',axis=1,inplace=True)
df.head()
value=1000
df['Age'].fillna(1000,inplace=True)
df['Age'].max()
df.drop(df[pd.isnull(df['Embarked'])].index,inplace=True)
df[pd.isnull(df['Embarked'])]
df.dtypes
df.drop('PassengerId',axis=1,inplace=True)
df.head()
df['Sex']=pd.Categorical(df['Sex'])
df['Embarked']=pd.Categorical(df['Embarked'])
df['FamilySize']=df['SibSp']+df['Parch']
df.head()
df.drop('SibSp',axis=1,inplace=True)
df.drop('Parch',axis=1, inplace=True)
df.head()
df.drop('Name', axis=1, inplace=True)
df.drop('Ticket', axis=1, inplace=True)
df.head()
df=pd.get_dummies(df, drop_first=True)
df.head()
X=df[df.loc[:,df.columns!='Survived'].columns]
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)
print('Inputs: \n',X_train.head())
print('Outputs: \n', y_train.head())

logreg=LogisticRegression()
logreg.fit(X_train,y_train)
scores=cross_val_score(logreg, X_train, y_train, cv=10)
print('CV Accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))
title='Learning Curve(logreg)'
cv=10
plot_learning_curve(logreg, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);
title='Validation Curve(logreg)'
param_name='C'
param_range=[0.001,0.01,0.1,1.0,10.0,100.0]
cv=10
plot_validation_curve(estimator=logreg, title=title, X=X_train, y=y_train, param_name=param_name, ylim=(0.5,1.01), param_range=param_range);
#chubby approach
df=df_raw.copy()
df.head()
df['FamilySize']=df['SibSp']+df['Parch']
df.drop('SibSp', axis=1, inplace=True)
df.drop('Parch', axis=1, inplace=True)
df.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
df_raw['Name'].unique()[:10]
df['Title']=0
for i in df:
    df['Title']=df_raw['Name'].str.extract('([A-Za-z]+)\.', expand=False)
df.head()
df['Title'].unique()
plt.figure(figsize=(15,5))
sns.barplot(x=df['Title'], y=df_raw['Age']);
df_raw['Title']=df['Title']
means=df_raw.groupby('Title')['Age'].mean()
means.head()
map_means=means.to_dict()
map_means
idx_nan_age=df.loc[np.isnan(df['Age'])].index
df.loc[idx_nan_age,'Age'].loc[idx_nan_age]=df['Title'].loc[idx_nan_age].map(map_means)
df.head()
df['Imputed']=0
df.at[idx_nan_age.values,'Imputed']=1
df.head()
sns.barplot(df['Pclass'],df['Survived']);
df.groupby(['Title'])['PassengerId'].count()
titles_dict={'Capt':'Other', 
            'Major': 'Other',
             'Jonkheer':'Other',
             'Don':'Other',
             'Sir':'Other',
             'Dr':'Other',
             'Rev':'Other',
             'Countess':'Other',
             'Dona':'Other',
             'Mme':'Mrs',
             'Mlle':'Miss',
             'Ms':'Miss',
             'Mr':'Mr',
             'Mrs':'Mrs',
             'Miss':'Miss',
             'Master':'Master',
             'Lady':'Other'
            }
df['Title']=df['Title'].map(titles_dict)
df['Title'].head()
df['Title']=pd.Categorical(df['Title'])
df.dtypes
sns.barplot(x='Title',y='Survived',data=df);
df['Sex']=pd.Categorical(df['Sex'])
df['Age']=pd.cut(df['Age'], bins=[0,12,50,200], labels=['Child','Adult','Elder'])
df.groupby(['Embarked']).mean()
df.drop('PassengerId', axis=1, inplace=True)
df.dtypes
df['Embarked']=pd.Categorical(df['Embarked'])
df.dtypes
df=pd.get_dummies(df)
df.head()
df.drop(['Sex_female','Age_Adult','Embarked_C','Title_Master'], axis=1, inplace=True)
df.head()
from sklearn.model_selection import train_test_split
X=df[df.loc[:,df.columns!='Survived'].columns]
y=df['Survived']
X_train, X_test, y_train,y_test=train_test_split(X, y, test_size=.2, random_state=0)
from scipy.stats import boxcox
X_train_transformed=X_train.copy()
X_train_transformed['Fare']=boxcox(X_train_transformed['Fare']+1)[0]
X_test_transformed=X_test.copy()
X_test_transformed['Fare']=boxcox(X_test_transformed['Fare']+1)[0]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_transformed_scaled=scaler.fit_transform(X_train_transformed)
X_test_transformed_scaled=scaler.transform(X_test_transformed)
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2).fit(X_train_transformed)
X_train_poly=poly.transform(X_train_transformed_scaled)
X_test_poly=poly.transform(X_test_transformed_scaled)
print(poly.get_feature_names())
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
logreg=LogisticRegression(C=1)
logreg.fit(X_train, y_train)
scores=cross_val_score(logreg, X_train, y_train, cv=10)
print('CV Accuracy (original): %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
highest_score=np.mean(scores)
for i in range(1, X_train_poly.shape[1]+1, 1):
    select=SelectKBest(score_func=chi2, k=i)
    select.fit(X_train_poly, y_train)
    X_train_poly_selected=select.transform(X_train_poly)
    logreg.fit(X_train_poly_selected, y_train)
    scores=cross_val_score(logreg, X_train_poly_selected, y_train, cv=10)
    print('CV Accuracy (number of features= %i): %.3f +/- %.3f' %(i, np.mean(scores), np.std(scores)))
    if(np.mean(scores)>highest_score):
        highest_score=np.mean(scores)
        std=np.std(scores)
        k_features_highest_score=i
    elif np.mean(scores)==highest_score:
        if np.std(scores)< std:
            highest_score=np.mean(scores)
            std=np.std(scores)
            k_features_highest_score=i
print('Number of features with highest score: %i' % k_features_highest_score)
select=SelectKBest(score_func=chi2, k=k_features_highest_score)
select.fit(X_train_poly, y_train)
X_train_poly_selected=select.transform(X_train_poly)
logreg=LogisticRegression(C=1)
logreg.fit(X_train_poly_selected, y_train)

scores=cross_val_score(logreg, X_train_poly_selected, y_train, cv=10)
print('CV Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
title='Learning Curve(logreg)'
cv=10
plot_learning_curve(logreg, title, X_train_poly_selected, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);
title='Validation'
param_name='C'
param_range=[.001,.01,.1,1,10,100]
cv=10
plot_validation_curve(estimator=logreg,title=title, X=X_train_poly_selected, y=y_train, param_name=param_name, ylim=(0.5,1.01), param_range=param_range);
df=pd.read_csv('C:\\Users\pragy\Desktop\downloads\\titanic\\test (1).csv')
df_raw=df.copy()
df['FamilySize'] = df['SibSp'] + df['Parch']
df.drop('SibSp',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

df['Title']=0
for i in df:
    df['Title']=df_raw['Name'].str.extract('([A-Za-z]+)\.', expand=False)  
df_raw['Title'] = df['Title']  
means = df_raw.groupby('Title')['Age'].mean()
map_means = means.to_dict()
idx_nan_age = df.loc[np.isnan(df['Age'])].index
df.loc[idx_nan_age, 'Age'] = df['Title'].loc[idx_nan_age].map(map_means)
df['Title'] = df['Title'].map(titles_dict)
df['Title'] = pd.Categorical(df['Title'])

df['Imputed'] = 0
df.at[idx_nan_age.values, 'Imputed'] = 1

df['Age'] = pd.cut(df['Age'], bins=[0, 12, 50, 200], labels=['Child','Adult','Elder'])

## 2.3
passenger_id = df['PassengerId'].values
df.drop('PassengerId', axis=1, inplace=True)
df['Embarked'] = pd.Categorical(df['Embarked'])
df = pd.get_dummies(df)
df.drop(['Sex_female',
         'Age_Adult',
         'Embarked_C',
         'Title_Master'], axis=1, inplace=True)

df = df.fillna(df.mean())  # There is one missing value in 'Fare'

X = df[df.loc[:, df.columns != 'Survived'].columns]

X_transformed = X.copy()
X_transformed['Fare'] = boxcox(X_transformed['Fare'] + 1)[0]

scaler = MinMaxScaler()
X_transformed_scaled = scaler.fit_transform(X_transformed)

poly = PolynomialFeatures(degree=2).fit(X_transformed)
X_poly = poly.transform(X_transformed_scaled)

X_poly_selected = select.transform(X_poly)

predictions=logreg.predict(X_poly_selected)
submission = pd.DataFrame({ 'PassengerId': passenger_id,
                            'Survived': predictions})
submission.to_csv("C:\\Users\pragy\Desktop\downloads\\titanic\\submission.csv", index=False)








 
















