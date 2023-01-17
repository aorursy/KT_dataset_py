import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
df= pd.read_csv('./data/train.csv')
df.head(100)
df.dtypes
for column in df : 
    print('{} : {}'.format(column,df[column].isnull().values.any()))
total=0
count=0
for value in df['Age']: 
    if not np.isnan(value): 
        total+=value
        count+=1
avg = total / count 
df['Age'].fillna(avg,inplace=True)
def avgMapper(x): 
    if x <15 : 
        return 0
    elif x >= 15 and x<=avg: 
        return 1
    else :
        return 2
df['AgeCategory'] = df['Age'].apply(avgMapper)
df['Cabin'].unique()
df['Cabin'].fillna('X',inplace=True)
df['Cabin']=df['Cabin'].str.split(' ')
df['CountCabin']=df['Cabin'].apply(len)
df['Cabin']=df['Cabin'].apply(lambda x : x[-1][0])
def mapper(df,column):
    value_types = list(df[column].unique())
    x = {}
    for i in range(len(value_types)): 
        x[value_types[i]]=i
    return x
#I decided to write a mapper function because, I will be using it again and I don't like having duplicate codes. 
df['CabinCategories']=df['Cabin'].map(mapper(df,'Cabin'))
df['EmbarkCategories']=df['Embarked'].map(mapper(df,'Embarked'))
df['FamilySize'] = df['SibSp'] + df['Parch']
sns.barplot(x='Cabin',y='Fare',data=df)
sns.barplot(x='Cabin',y='CountCabin',data=df)
df['FarePerson']=df['Fare']/(df['FamilySize']+1)
df['HasSibSp'] = df['SibSp'].apply(lambda x : 1 if x>0 else 0)
df['GotParch'] = df['Parch']
df['GotParch']=df['GotParch'].apply(lambda x : 1 if x >0 else 0)
df.head()
train_df=df.drop(['Name','PassengerId','Ticket','Cabin','Embarked'],axis=1)
train_df['Sex'] = train_df['Sex'].apply(lambda x : 0 if x=='male' else 1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_df.drop(['Survived'],axis=1),train_df['Survived'],test_size=0.20, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(x_train, y_train)
from sklearn import metrics
prediction_test = model.predict(x_test)
# Print the prediction accuracy

print (metrics.accuracy_score(y_test, prediction_test))
weights = pd.Series(model.coef_[0],
                 index=train_df.drop(['Survived'],axis=1).columns.values)
print (weights.sort_values(ascending = False)[:].plot(kind='bar'))
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(x_train, y_train)
prediction_test = model_rf.predict(x_test)
prediction_test = model_rf.predict(x_test)
print (metrics.accuracy_score(y_test, prediction_test))
print (metrics.recall_score(y_test, prediction_test))
print (metrics.precision_score(y_test, prediction_test))
weights = pd.Series(model_rf.feature_importances_,
                 index=train_df.drop(['Survived'],axis=1).columns.values)
print (weights.sort_values(ascending = False)[:].plot(kind='bar'))
train_df2=train_df[['Sex','Age','Fare','FarePerson','Pclass','FamilySize','CabinCategories','AgeCategory','EmbarkCategories','SibSp','Survived']]
from sklearn.model_selection import train_test_split

x_train2, x_test2, y_train2, y_test2 = train_test_split(train_df2.drop(['Survived'],axis=1),train_df2['Survived'],test_size=0.20, random_state=0)
from sklearn.ensemble import RandomForestClassifier
model_rf2 = RandomForestClassifier()
model_rf2.fit(x_train2, y_train2)
prediction_test = model_rf2.predict(x_test2)
print (metrics.accuracy_score(y_test2, prediction_test))
print (metrics.recall_score(y_test2, prediction_test))
print (metrics.precision_score(y_test2, prediction_test))
weights = pd.Series(model_rf2.feature_importances_,
                 index=train_df2.drop(['Survived'],axis=1).columns.values)
print (weights.sort_values(ascending = False)[:].plot(kind='bar'))
from sklearn.naive_bayes import GaussianNB
model_gnb = GaussianNB()
model_gnb.fit(x_train, y_train)
model_gnb2=GaussianNB()
model_gnb2.fit(x_train2,y_train2)
prediction_test = model_gnb.predict(x_test)
print(metrics.accuracy_score(y_test,prediction_test))
prediction_test=model_gnb2.predict(x_test2)
print(metrics.accuracy_score(y_test2,prediction_test))
print(model_rf2.get_params())
from sklearn.model_selection import RandomizedSearchCV# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
model_random=RandomForestClassifier()
rf_random=RandomizedSearchCV(estimator=model_random, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(x_train2,y_train2)
rf_random.best_params_
prediction_test = rf_random.best_estimator_.predict(x_test2)
#prediction_test=.predict(x_test2)
print(metrics.accuracy_score(y_test2,prediction_test))
params = {
    'bootstrap':[True],
    'max_depth':[60,70,80],
    'max_features':['auto',3,2,4],
    'min_samples_leaf':[3,4,5],
    'min_samples_split':[8,10,12],
    'n_estimators':[200,400,800]
}
from sklearn.model_selection import GridSearchCV
grid_rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator=grid_rf,param_grid=params,cv=3,n_jobs=-1,verbose=2)
grid_search.fit(x_train2,y_train2)
grid_search.best_params_
best_grid=grid_search.best_estimator_.predict(x_test2)
print(metrics.accuracy_score(y_test2,best_grid))

from sklearn.model_selection import StratifiedKFold
grid_rf = RandomForestClassifier()
k=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
grid_search = GridSearchCV(estimator=grid_rf,param_grid=params,cv=k,n_jobs=-1,verbose=2)
grid_search.fit(x_train2,y_train2)
grid_search.best_params_
best_grid=grid_search.best_estimator_.predict(x_test2)
print(metrics.accuracy_score(y_test2,best_grid))
test_df= pd.read_csv('./data/test.csv')
total=0
count=0
for value in test_df['Age']: 
    if not np.isnan(value): 
        total+=value
        count+=1
avg = total / count 
test_df['Age'].fillna(avg,inplace=True)
def avgMapper(x): 
    if x <15 : 
        return 0
    elif x >= 15 and x<=avg: 
        return 1
    else :
        return 2
test_df['AgeCategory'] = test_df['Age'].apply(avgMapper)
test_df['Cabin'].fillna('X',inplace=True)
test_df['Cabin']=test_df['Cabin'].str.split(' ')
test_df['CountCabin']=test_df['Cabin'].apply(len)
test_df['Cabin']=test_df['Cabin'].apply(lambda x : x[-1][0])
def mapper(df,column):
    value_types = list(df[column].unique())
    x = {}
    for i in range(len(value_types)): 
        x[value_types[i]]=i
    return x
#I decided to write a mapper function because, I will be using it again and I don't like having duplicate codes. 
test_df['CabinCategories']=test_df['Cabin'].map(mapper(test_df,'Cabin'))
test_df['EmbarkCategories']=test_df['Embarked'].map(mapper(test_df,'Embarked'))
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
test_df['FarePerson']=test_df['Fare']/(test_df['FamilySize']+1)
test_df['HasSibSp'] = test_df['SibSp'].apply(lambda x : 1 if x>0 else 0)
test_df['GotParch'] = test_df['Parch']
test_df['GotParch']=test_df['GotParch'].apply(lambda x : 1 if x >0 else 0)
test_df['Sex'] = test_df['Sex'].apply(lambda x : 0 if x=='male' else 1)
test_fin=test_df[['Sex','Age','Fare','FarePerson','Pclass','FamilySize','CabinCategories','AgeCategory','EmbarkCategories','SibSp']]
test_fin.dtypes
for column in test_fin : 
    print('{} : {}'.format(column,test_fin[column].isnull().values.any()))
test_fin['Age']=test_fin['Age'].astype(np.int64)
test_fin['Fare']=test_fin['Fare'].astype(np.int64)
test_fin['FarePerson']=test_fin['FarePerson'].astype(np.int64)
total=0
count=0
for value in test_df['Fare']: 
    if not np.isnan(value): 
        total+=value
        count+=1
avg = total / count 
test_df['Fare'].fillna(avg,inplace=True)
grid_test=grid_search.best_estimator_.predict(test_fin)
grid_test
Ids=test_df['PassengerId']
print(type(grid_test))
submission=pd.DataFrame(data=Ids)
submission['Survived']=grid_test
submission.to_csv('sub.csv',index=False)
random_test=rf_random.best_estimator_.predict(test_fin)
sub2=pd.DataFrame(data=Ids)
sub2['Survived']=random_test
sub2.to_csv('random.csv',index=False)