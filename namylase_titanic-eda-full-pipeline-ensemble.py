import pandas as pd
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette=sns.color_palette('Set2',9))

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, GridSearchCV,RandomizedSearchCV

from xgboost import XGBRFClassifier,XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier

from scipy.stats import uniform,randint
# !pip install seaborn --upgrade
# import os
# os._exit(00)
print(sns.__version__)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_path='/kaggle/input/titanic/train.csv'
test_path='/kaggle/input/titanic/test.csv'
titanic_train=pd.read_csv(train_path)
titanic_test=pd.read_csv(test_path)
titanic_train.head(10)
titanic_test.head()
titanic_train.info()
titanic_test.info()
def plot_mutihist(num_row,num_col,hist_columns):

    f,axs=plt.subplots(num_row,num_col,squeeze=True,figsize=(20,10))
    for i in range(len(hist_columns)%num_col,num_col):
        axs[num_row-1,i].remove()
    for i in range(len(hist_columns)):
        sns.histplot(data=titanic_train,x=hist_columns[i], ax=axs[i//num_col,i%num_col])
    f.tight_layout()
titanic_hist_columns=['Pclass','Sex','Age','SibSp','Parch','Fare']
plot_mutihist(num_row=2,num_col=4,hist_columns=titanic_hist_columns)
def plot_countplot_hue(num_row,num_col,count_columns):

    f,axs=plt.subplots(num_row,num_col,squeeze=True,figsize=(15,10))
    for i in range(len(count_columns)%num_col,num_col):
        axs[num_row-1,i].remove()
    for i in range(len(count_columns)):
        sns.countplot(data=titanic_train,x=count_columns[i],hue='Survived',ax=axs[i//num_col,i%num_col])
titanic_count_columns=['Pclass','Sex','SibSp','Parch']
plot_countplot_hue(num_row=2,num_col=4,count_columns=titanic_count_columns)
titanic_heatmap_columns=['Survived','Age','SibSp','Parch','Fare','Pclass']
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(titanic_train[titanic_heatmap_columns].corr(), cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}) 
sns.catplot(x='Sex',y='Survived',kind='bar',hue='Pclass',data=titanic_train)
f,axs=plt.subplots(1,4,squeeze=True,figsize=(20,5))
sns.barplot(x='SibSp',y='Survived',data=titanic_train,ax=axs[0])
sns.barplot(x='Parch',y='Survived',data=titanic_train,ax=axs[1])

titanic_train['Family']=titanic_train['SibSp']+titanic_train['Parch']+1
sns.barplot(x='Family',y='Survived',data=titanic_train,ax=axs[2])
def groupfamily(x):
    if x==1:
        return 1
    elif (2<=x)&(x<=4):
        return 2
    elif (5<=x)&(x<=6):
        return 3
    else:
        return 4
titanic_train['Family_group']=titanic_train['Family'].apply(groupfamily)

sns.barplot(x='Family_group',y='Survived',data=titanic_train,ax=axs[3])
AGE_BINS=[-1,5,17,20,24,28,32,40,48,100]
titanic_train['Age_cat']=pd.cut(titanic_train['Age'],AGE_BINS,labels=[0,1,2,3,4,5,6,7,8])

f,axs=plt.subplots(1,3,squeeze=True,figsize=(22,7))

sns.histplot(data=titanic_train, x="Age", bins='auto',kde=True, hue="Survived",ax=axs[0])
axs[0].plot(AGE_BINS, np.ones(len(AGE_BINS)) ,'rv')

sns.barplot(x='Age_cat',y='Survived',data=titanic_train,ax=axs[1])

sns.countplot(data=titanic_train, x="Age_cat", hue="Survived",ax=axs[2])
FARE_BINS=[-1,7.35, 7.82, 8, 10, 13, 23,30, 45, 80, 150, 1000]
titanic_train['Fare_cat']=pd.cut(titanic_train['Fare'],bins=FARE_BINS,labels=[0,1,2,3,4,5,6,7,8,9,10])

f,axs=plt.subplots(1,3,squeeze=True,figsize=(22,7))

sns.histplot(data=titanic_train, x="Fare",bins='auto',kde=True, hue="Survived",ax=axs[0])
axs[0].plot(FARE_BINS,np.ones(len(FARE_BINS)),'rv')
axs[0].set_ylim(0,70)
axs[0].set_xlim(0,500)

sns.barplot(x='Fare_cat',y='Survived',data=titanic_train,ax=axs[1])

sns.countplot(data=titanic_train, x="Fare_cat", hue="Survived",ax=axs[2])
sns.countplot(data=titanic_train, x="Embarked", hue="Survived")
titanic_train['Cabin_t']=titanic_train['Cabin'].str.get(i=0)
titanic_train['Cabin_t'].fillna('X',inplace=True)
f,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
sns.countplot(data=titanic_train, x="Cabin_t", hue="Survived",ax=ax1)
sns.countplot(data=titanic_train, x="Cabin_t", hue="Pclass",ax=ax2)
ax1.set_ylim(0,50)
ax2.set_ylim(0,50)
titanic_train['Cabin_t'] = titanic_train['Cabin_t'].replace(['A', 'B', 'C','T'], 'ABCT')
titanic_train['Cabin_t'] = titanic_train['Cabin_t'].replace(['D', 'E'], 'DE')
titanic_train['Cabin_t'] = titanic_train['Cabin_t'].replace(['F', 'G'], 'FG')
f,ax1=plt.subplots(figsize=(12,5))
sns.countplot(data=titanic_train, x="Cabin_t", hue="Pclass",ax=ax1)
ax1.set_ylim(0,200)
titanic_train['Name_m']=titanic_train['Name'].str.split(pat=', ',n=1,expand=True)[1].str.split(pat='.',n=1,expand=True)[0]


def Name_transform(x):
    if x=='Mr':
        return 'Mr'
    elif x=='Mrs':
        return 'Mrs'
    elif x=='Miss':
        return 'Miss'
    else:
        return 'etc'
titanic_train['Name_M']=titanic_train['Name_m'].apply(Name_transform)
f,ax1=plt.subplots(1,1,squeeze=True,figsize=(10,5))
sns.countplot(data=titanic_train, x="Name_M", hue="Survived",ax=ax1)
titanic_train['Ticket_Freq']=titanic_train.groupby('Ticket')['Ticket'].transform('count')
titanic_train['Ticket_Freq']
sns.countplot(data=titanic_train, x="Ticket_Freq", hue="Survived")
titanic_train=pd.read_csv(train_path)
titanic_test=pd.read_csv(test_path)
target=titanic_train['Survived']
Id=titanic_test[['PassengerId']]
class NullTransformer(BaseEstimator,TransformerMixin):

    def fit(self,df,y=None):
        return self
    def transform(self,df):
        
    ##fill Embarked with most frequently variable 'S'    
        df['Embarked'].fillna('S',inplace=True)  
        
    ##fill Cabin with 'X'
        df['Cabin'].fillna('X',inplace=True)     
        
    ##fill Fare with median value of same Pclass
        missing_Fare_index=list(df['Fare'][df['Fare'].isnull()].index)     
        
        for index in missing_Fare_index:
            if df['Pclass'][index]==1:
                df['Fare'].iloc[index]=df.groupby('Pclass')['Fare'].median()[1]
            elif df['Pclass'][index]==2:
                df['Fare'].iloc[index]=df.groupby('Pclass')['Fare'].median()[2]
            elif df['Pclass'][index]==3:
                df['Fare'].iloc[index]=df.groupby('Pclass')['Fare'].median()[3]
            
    ##fill Age with median value of same Pclass    
        index_NaN_age = list(df["Age"][df["Age"].isnull()].index)             

        for index in index_NaN_age :
            if df['Pclass'][index]==1:
                df['Age'].iloc[index]=df.groupby('Pclass')['Age'].median()[1]
            elif df['Pclass'][index]==2:
                df['Age'].iloc[index]=df.groupby('Pclass')['Age'].median()[2]
            elif df['Pclass'][index]==3:
                df['Age'].iloc[index]=df.groupby('Pclass')['Age'].median()[3]
                
        return df
class FeatureExtraction(BaseEstimator,TransformerMixin):

    def fit(self,df,y=None):
        return self
    def transform(self,df):
        
    ## create Family Features   
        df['Family']=df['SibSp']+df['Parch']+1  
        def groupfamily(x):
            if x==1:
                return 1
            elif (2<=x)&(x<=4):
                return 2
            elif (5<=x)&(x<=6):
                return 3
            else:
                return 4
        df['Family']=df['Family'].apply(groupfamily)
        
        df.drop(['SibSp','Parch','PassengerId'],inplace=True,axis=1)
        if 'Survived' in list(df.keys()):
            df.drop(['Survived'],inplace=True,axis=1)
                     
    ## extract Ticket Frequency and prefix
        df['Ticket_freq']=df.groupby('Ticket')['Ticket'].transform('count')
        df.drop(['Ticket'],inplace=True,axis=1)
        
    ## extract Cabin prefix
        df['Cabin']=df['Cabin'].str.get(i=0)                            
        df['Cabin'] = df['Cabin'].replace(['A', 'B', 'C','T'], 'ABCT')
        df['Cabin'] = df['Cabin'].replace(['D', 'E'], 'DE')
        df['Cabin'] = df['Cabin'].replace(['F', 'G'], 'FG')
        
    ## extract Name title
        df['Name']=df['Name'].str.split(pat=', ',n=1,expand=True)[1].str.split(pat='.',n=1,expand=True)[0]      
        def Name_transform(x):
            if x=='Mr':
                return 'Mr'
            elif x=='Mrs':
                return 'Mrs'
            elif x=='Miss':
                return 'Miss'
            else:
                return 'etc'
        df['Name']=df['Name'].apply(Name_transform)     
        
     ## Binning Age, Fare
        AGE_BINS=[-1,5,17,20,24,28,32,40,48,100]
        df['Age_cat']=pd.cut(df['Age'],AGE_BINS,labels=[0.,1.,2.,3.,4.,5.,6.,7.,8.])
        
        FARE_BINS=[-1,7.35, 7.82, 8, 10, 13, 23,30, 45, 80, 150, 1000]
        df['Fare_cat']=pd.cut(df['Fare'],bins=FARE_BINS,labels=[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
        df.drop(['Age','Fare'],inplace=True,axis=1)
        
    ## Fare to log scale 
       # df["Fare"] = df["Fare"].map(lambda x: np.log(x) if x>0 else 0)
    
        return df
attribs=['Pclass','Name','Sex','Age','Ticket','Fare','Cabin','Embarked','SibSp','Parch']
num_attribs=['Pclass','Age_cat','Fare_cat',"Ticket_freq"]
cat_attribs=['Name','Sex','Cabin','Embarked','Family']

pipeline1=Pipeline([
        ('NT',NullTransformer()),
        ('FE',FeatureExtraction())
])

train=pipeline1.fit_transform(titanic_train)
test=pipeline1.transform(titanic_test)

train=pd.get_dummies(train,columns=cat_attribs)
test=pd.get_dummies(test,columns=cat_attribs)

train=train.to_numpy()
test=test.to_numpy()
kfolds=StratifiedKFold(n_splits=3)    

scores=[]

##Base models
svc=SVC(random_state=42)
rdf=RandomForestClassifier(random_state=42)
ada=AdaBoostClassifier(random_state=42)
log=LogisticRegression(random_state=42)
grb=GradientBoostingClassifier(random_state=42)
dct=DecisionTreeClassifier(random_state=42)
ext=ExtraTreesClassifier(random_state=42)
xgb=XGBRFClassifier(random_state=42)
xgbc=XGBClassifier(random_state=42)

models=[svc,rdf,ada,log,grb,dct,ext,xgb,xgbc]

for model in models:
    scores.append(cross_val_score(model,train,target,cv=kfolds,n_jobs=-1))
    scores_mean=np.mean(scores,axis=1)
columns_name=[]
for model in models:
    columns_name.append(model.__class__.__name__)

scores_df=pd.DataFrame(np.array(scores).T,columns=columns_name)
scores_mean_tosort=pd.Series(np.array(scores_mean).T,index=columns_name)
scores_mean_sorted=scores_mean_tosort.sort_values(ascending=False)

f, ax1=plt.subplots(figsize=(10,5))
sns.barplot(data=scores_df,orient='h',order=list(scores_mean_sorted.index))
scores_mean_sorted
def grid_search(estimator,param,X,y):

    grid=GridSearchCV(estimator, param, n_jobs=-1, cv=kfolds, return_train_score=True)
    grid.fit(X,y)

    grid_results=grid.cv_results_
    #for mean_score,param in zip(grid_results['mean_test_score'],grid_results['params']):
        #print(mean_score,param)
    
    print('\nBestParams and Score : \n',grid.best_params_,'\n',grid.best_score_)
    
    return grid.best_estimator_ , grid, estimator
def random_grid_search(estimator,param_d,X,y,n):

    grid=RandomizedSearchCV(estimator, param_d, n_jobs=-1, cv=kfolds, return_train_score=True, n_iter=n)
    grid.fit(X,y)

    grid_results=grid.cv_results_
    #for mean_score,param in zip(grid_results['mean_test_score'],grid_results['params']):
        #print(mean_score,param)
    
    print('\nBestParams and Score : \n',grid.best_params_,'\n',grid.best_score_)
    
    return grid.best_estimator_ , grid, estimator
param_grid = [
    {'C':[0.6, 0.8,1,1.15,1.2,1.23,1.4], 'kernel':['rbf'], 'gamma':[0.1] }   
]
svc_best, svc_grid, svc=grid_search(SVC(random_state=42, probability=True),param_grid,X=train,y=target)
param_grid = [
    {'C': uniform(0.9,0.3), 'kernel':['rbf'], 'gamma':[0.1] }   
]
svc_best, svc_grid, svc =random_grid_search(SVC(random_state=42, probability=True),param_grid,X=train,y=target,n=20)
param_grid = [
    {'learning_rate':[0.1], 'n_estimators':[30,50,100,200,240], 'max_depth':[2,3,4] }   
]
gbc_best, gbc_grid, gbc =grid_search(GradientBoostingClassifier(random_state=42),param_grid,X=train,y=target)
param_grid = [
    {'learning_rate':[0.1], 'n_estimators': randint(80,110), 'max_depth':[3] }   
]
gbc_best, gbc_grid, gbc =random_grid_search(GradientBoostingClassifier(random_state=42),param_grid,X=train,y=target,n=20)
param_grid = [
    {'penalty':['l1','l2'],'C':[0.1,0.8,0.9,1,1.1,1.2,1.3,5], 'tol':[1e-4,1e-3],'solver':['liblinear']}   
]
log_best, log_grid, log =grid_search(LogisticRegression(random_state=42),param_grid,X=train,y=target)
param_grid = [
    {'penalty':['l1'],'C': uniform(0.7,0.3), 'tol':[1e-4],'solver':['liblinear']}   
]
log_best, log_grid, log =random_grid_search(LogisticRegression(random_state=42),param_grid,X=train,y=target,n=20)
param_grid = [
    {'n_estimators':[30,50,70,100,200,300],'max_depth':[2,3,4,5], 'gamma':[0.1]}   
]
xgb_best, xgb_grid, xgb=grid_search(XGBRFClassifier(random_state=42),param_grid,X=train,y=target)
param_grid = [
    {'n_estimators':randint(20,60),'max_depth':[4], 'gamma':[0.1]}   
]
xgb_best, xgb_grid, xgb =random_grid_search(XGBRFClassifier(random_state=42),param_grid,X=train,y=target,n=20)
param_grid = [
    {'n_estimators':[30,50,100,200],'max_depth':[2,3,4], 'gamma':[0.1]}   
]
xgbc_best, xgbc_grid, xgbc =grid_search(XGBClassifier(random_state=42),param_grid,X=train,y=target)
param_grid = [
    {'n_estimators': randint(70,120),'max_depth':[3], 'gamma':[0.1]}   
]
xgbc_best, xgbc_grid, xgbc =random_grid_search(XGBClassifier(random_state=42),param_grid,X=train,y=target,n=20)
param_grid = [
    {'n_estimators': [30,50,100,200,300],'learning_rate':[0.9,1.0,1.1,1.2,1.3]}   
]
ada_best, ada_grid, ada =grid_search(AdaBoostClassifier(random_state=42),param_grid,X=train,y=target)
param_grid = [
    {'n_estimators': randint(20,50),'learning_rate':[1.3]}   
]
ada_best, ada_grid, ada =random_grid_search(AdaBoostClassifier(random_state=42),param_grid,X=train,y=target,n=20)
param_grid = [
    {'n_estimators': [30,50,100,200,300],'max_depth':[2,3,4,5], 'criterion':['gini','entropy']}   
]
rdf_best, rdf_grid, rdf =grid_search(RandomForestClassifier(random_state=42),param_grid,X=train,y=target)
param_grid = [
    {'n_estimators': randint(200,400),'max_depth':[5], 'criterion':['gini']}   
]
rdf_best, rdf_grid, rdf =random_grid_search(RandomForestClassifier(random_state=42),param_grid,X=train,y=target,n=20)
param_grid = [
    {'n_estimators': [30,50,100,200,300],'max_depth':[2,3,4,5], 'criterion':['gini','entropy']}   
]
ext_best, ext_grid, ext =grid_search(ExtraTreesClassifier(random_state=42),param_grid,X=train,y=target)
param_grid = [
    {'n_estimators': randint(80,150),'max_depth':[5], 'criterion':['gini']}   
]
ext_best, ext_grid, ext =random_grid_search(ExtraTreesClassifier(random_state=42),param_grid,X=train,y=target,n=20)
models=[
        ('svc', svc_best),
        ('log', log_best), 
        ('xgb',xgb_best),
        ('gbc', gbc_best),
        ('xgbc', xgbc_best),
        ('ada', xgbc_best),
        ('rdf',rdf_best),
        ('ext',ext_best)
    
]

vot_hard = VotingClassifier(estimators=models, voting='hard', n_jobs=-1)

vot_hard.fit(train,target)

vot_soft = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)

vot_soft.fit(train,target)

stack=StackingClassifier(estimators=models,cv=kfolds,n_jobs=-1,stack_method='predict_proba')

stack.fit(train,target)

prediction_hard=vot_hard.predict(test)
prediction_soft=vot_soft.predict(test)
prediction_stack=stack.predict(test)

kfolds=StratifiedKFold(n_splits=3)    

scores=[]

models=[vot_hard,vot_soft,stack]

for model in models:
    scores.append(cross_val_score(model,train,target,cv=kfolds))
    scores_mean=np.mean(scores,axis=1)
scores_mean
Survived_hard = pd.Series(prediction_hard, name="Survived")
result_hard = pd.concat([Id,Survived_hard],axis=1)
result_hard.to_csv("submission_hard.csv",index=False)

Survived_soft = pd.Series(prediction_soft, name="Survived")
result_soft = pd.concat([Id,Survived_soft],axis=1)
result_soft.to_csv("submission_soft.csv",index=False)

Survived_stack = pd.Series(prediction_stack, name="Survived")
result_stack = pd.concat([Id,Survived_stack],axis=1)
result_stack.to_csv("submission_stack.csv",index=False)
from sklearn.decomposition import PCA


pca=PCA(n_components=0.879)
train_pca=pca.fit_transform(train,target)
test_pca=pca.transform(test)

svc_pca=SVC(C=1.0, kernel='rbf', gamma=0.1,  probability=True)




cv_pca=cross_val_score(svc_pca,train_pca,target,n_jobs=-1, cv=kfolds)
svc_pca.fit(train_pca,target)
prediction_pca=svc_pca.predict(test_pca)

Survived_pca = pd.Series(prediction_pca, name="Survived")
result_pca = pd.concat([Id,Survived_pca],axis=1)
result_pca.to_csv("submission_pca.csv",index=False)

"""
train
test
target
=SVC()
param_grid_pca=[
    {
        'kpca':['linear','poly','rbf']
    }
]
"""
cv_pca