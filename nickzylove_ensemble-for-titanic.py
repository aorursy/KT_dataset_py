import numpy as np 
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.plotly as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tools
init_notebook_mode(connected=True)
import cufflinks as cf
cf.set_config_file(offline=True, theme='ggplot')
from IPython.display import Image, display

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

# Store our passenger ID for easy access
PassengerId_test = test['PassengerId']

print('The shape of the training data:', train.shape)
print('The shape of the testing data:', test.shape)
print('The features in the data:',train.columns.values)
train.head()
train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
survive_counts = train['Survived'].value_counts().reset_index().replace([0,1],['Dead','Survived'])
survive_counts.iplot(kind='pie',labels='index',values='Survived',pull=.02,title='Survived Vs Die')
dft = pd.crosstab(train.Sex,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='Sex: Survived Vs Dead')
dft = pd.crosstab(train.Pclass,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='Pclass: Survived Vs Dead')
dft = pd.crosstab([train.Sex,train.Pclass],train.Survived).rename(columns={0:'Dead',1:'Survived'})
dft.iplot(kind='bar',title='Pclass: Survived Vs Dead',barmode='stack')
sns.factorplot('Pclass','Survived',hue='Sex',data=train)
plt.show()
print('Oldest Passenger was of:',train.Age.max(),'Years')
print('Youngest Passenger was of:',train.Age.min(),'Years')
print('Average Age on the ship:',train.Age.mean(),'Years')
Age_female_survived = train[(train.Sex=='female') & (train.Survived==1)].Age
Age_female_dead = train[(train.Sex=='female') & (train.Survived==0)].Age
Age_male_survived = train[(train.Sex=='male') & (train.Survived==1)].Age
Age_male_dead = train[(train.Sex=='male') & (train.Survived==0)].Age

fig = tools.make_subplots(rows=1, cols=2,subplot_titles=('Female', 'Male'))

survived_female = go.Histogram(
    name='Survived_female',
    x=Age_female_survived
)
fig.append_trace(survived_female, 1, 1)
dead_female = go.Histogram(
    name='Dead_female',
    x=Age_female_dead
)
fig.append_trace(dead_female, 1, 1)
fig.layout.xaxis1.update({'title':'Age'})

survived_male = go.Histogram(
    name='Survived_male',
    x=Age_male_survived
)
dead_male = go.Histogram(
    name='Dead_male',
    x=Age_male_dead
)
fig.append_trace(survived_male,1,2)
fig.append_trace(dead_male,1,2)
fig.layout.xaxis2.update({'title':'Age'})
fig.layout.update({'barmode':'stack'})
iplot(fig)
full_data = [train, test]  # Both training and testing data need processing
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Intial, containing the titles of passenger names
for dataset in full_data:
    dataset['Initial'] = dataset.Name.apply(get_title)
# The initials need processing
for dataset in full_data:
    dataset['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                             'Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs',
                             'Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
pd.crosstab(train.Sex,train.Initial).style.background_gradient(cmap='summer_r')
train.groupby('Initial')['Age'].mean() #lets check the average age by Initials
for dataset in full_data:
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Master'),'Age']=5
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Miss'),'Age']=22
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Mr'),'Age']=33
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Mrs'),'Age']=36
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Other'),'Age']=46
train.Age.isnull().any() #So no null values left finally 
Age_female_survived = train[(train.Sex=='female') & (train.Survived==1)].Age
Age_female_dead = train[(train.Sex=='female') & (train.Survived==0)].Age
Age_male_survived = train[(train.Sex=='male') & (train.Survived==1)].Age
Age_male_dead = train[(train.Sex=='male') & (train.Survived==0)].Age

age_data = [Age_female_survived, Age_female_dead,Age_male_survived,Age_male_dead]
age_groups = ['Survived_female', 'Dead_female','Survived_male','Dead_male']
fig = ff.create_distplot(age_data, age_groups, bin_size=3)
iplot(fig)
sns.factorplot('Pclass','Survived',hue='Initial',data=train)
plt.show()
dft = pd.crosstab(train.Embarked,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='Embarked: Survived Vs Dead')
sns.factorplot('Embarked','Survived',data=train)
sns.factorplot('Embarked','Survived',hue='Sex',col='Pclass',data=train)
plt.show()
for dataset in full_data:
    dataset['Embarked'].fillna('S',inplace=True)
train.Embarked.isnull().any()# Finally No NaN values
dft = pd.crosstab(train.SibSp,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='SibSp: Survived Vs Dead')
sns.factorplot('SibSp','Survived',data=train)
sns.factorplot('SibSp','Survived',hue='Sex',col='Pclass',data=train)
plt.show()
dft = pd.crosstab(train.Parch,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='Parch: Survived Vs Dead')
sns.factorplot('Parch','Survived',data=train)
sns.factorplot('Parch','Survived',hue='Sex',col='Pclass',data=train)
plt.show()
print('Highest Fare was:',train['Fare'].max())
print('Lowest Fare was:',train['Fare'].min())
print('Average Fare was:',train['Fare'].mean())
fare_pc1 = train[(train.Pclass==1)].Fare
fare_pc2 = train[(train.Pclass==2)].Fare
fare_pc3 = train[(train.Pclass==3)].Fare

fig = tools.make_subplots(rows=1, cols=3,subplot_titles=('Pclass 1', 'Pclass 2', 'Pclass 3'))

p1_fare = ff.create_distplot([fare_pc1], ['Pclass 1'], bin_size=30)
fig.append_trace(p1_fare.data[0], 1, 1)
fig.append_trace(p1_fare.data[1], 1, 1)

p2_fare = ff.create_distplot([fare_pc2], ['Pclass 2'], bin_size=5)
fig.append_trace(p2_fare.data[0], 1, 2)
fig.append_trace(p2_fare.data[1], 1, 2)

p3_fare = ff.create_distplot([fare_pc3], ['Pclass 3'], bin_size=5)
fig.append_trace(p3_fare.data[0], 1, 3)
fig.append_trace(p3_fare.data[1], 1, 3)
fig.layout.update({'showlegend':False})

iplot(fig)
fare_survived = train[train.Survived==1].Fare
fare_dead = train[train.Survived==0].Fare

survived_fare = go.Histogram(
    name='Survived',
    x=fare_survived
)

dead_fare = go.Histogram(
    name='Dead',
    x=fare_dead
)
layout = go.Layout(title='Fare: Survived Vs Dead',barmode='stack')
fig = go.Figure(data=[survived_fare, dead_fare],layout=layout)
iplot(fig)
# Feature engineering steps taken from Sina
# build fare band
for dataset in full_data:
    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['Fare_Range']=pd.qcut(train['Fare'],4) # to check the value for dividing fare
for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Create new feature FamilySize as a combination of SibSp and Parch
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Build age band
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
    dataset['Age'] = dataset['Age'].astype(int)
    
    # Mapping character values to integers    
    dataset['Sex'].replace(['male','female'],[0,1],inplace=True)
    dataset['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
    dataset['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']
for dataset in full_data:
    dataset.drop(drop_elements,axis=1,inplace=True)
train.drop(['Fare_Range'],axis=1,inplace=True)
sns.factorplot('Age','Survived',data=train,col='Pclass')
plt.show()
sns.factorplot('Fare','Survived',data=train,hue='Sex')
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('FamilySize','Survived',data=train,ax=ax[0])
ax[0].set_title('FamilySize vs Survived')
sns.factorplot('IsAlone','Survived',data=train,ax=ax[1])
ax[1].set_title('IsAlone vs Survived')
plt.close(2)
plt.close(3)
plt.show()
sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
for dataset in full_data:
    dataset.drop(['SibSp'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes
from sklearn.svm import SVC #support vector classifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
import xgboost as xgb  # xgboost is not a classifier in sklean, therefore needs better attention
SEED = 0 # for reproducibility
## Prepares training and testing input and target data
training,val=train_test_split(train,test_size=0.3,random_state=SEED,stratify=train['Survived'])
Xtrain=training[training.columns[1:]]
ytrain=training[training.columns[:1]]
Xval=val[val.columns[1:]]
yval=val[val.columns[:1]]

# Some useful parameters which will come in handy later on
ntrain = Xtrain.shape[0]
nval = Xval.shape[0]
# build a set of base learners
def base_learners():
    """Construct a list of base learners"""
    lr = LogisticRegression(random_state=SEED)
    svc = SVC(kernel='linear', C=0.1, gamma=0.1, probability=True,random_state=SEED)
    knn = KNeighborsClassifier()
    nb = GaussianNB()
    nn = MLPClassifier((80,10),random_state=SEED)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    ab = AdaBoostClassifier(n_estimators=100, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    
    models = {
        'Logistic Regression': lr,
        'SVM': svc,
        'KNN': knn,
        'Naive Bayes': nb,
        'Neural Network': nn,
        'Random Forest': rf,
        'Extra Trees': et,
        'AdaBoosting': ab,
        'GradientBoosting': gb
    }
    
    return models

def train_predict(models, Xtrain, ytrain, Xtest):
    """Fit models"""
    P = np.zeros((nval,len(models)))
    P = pd.DataFrame(P)
    
    print("Fitting models")
    for i, (name, model) in enumerate(models.items()):
        print("%s..."%name, end=' ', flush=False)
        model.fit(Xtrain,ytrain)
        P.iloc[:,i]=model.predict(Xtest)
        print('done')
    print('Fitting done!')   
    P.columns = models.keys()

    return P 

def score_models(P, y):
    "Obtain accuracy scores of models"
    acc = []
    print("Scoring models")
    for m in P.columns:
        acc.append(metrics.accuracy_score(y, P.loc[:,m]))
    print('Done!')
    
    acc = pd.Series(data=acc,index=P.columns,name='Accuracy')
    return acc
    
models = base_learners()
P = train_predict(models, Xtrain, ytrain, Xval)
acc = score_models(P, yval) 
iplot(ff.create_table(acc.reset_index()))
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction

SEED = 0 # for reproducibility
Xtrain = train[train.columns[1:]]
ytrain = train[train.columns[:1]]
Xtest = test
kf = KFold(n_splits=5, random_state=SEED) # k=5, split the data into 5 equal part

def trainCV(models, X, y):
    """Fit models using cross validation"""   
    
    print("Fitting models")
    acc, std = [], []
    acc_all = []
    for i, (name, model) in enumerate(models.items()):
        print("%s..."%name, end=' ', flush=False)
        cv_result = cross_val_score(model,X,y,cv=kf,scoring='accuracy')
        acc.append(cv_result.mean())
        std.append(cv_result.std())
        acc_all.append(cv_result)
        print('done')
    print('Fitting done!') 
    acc=pd.DataFrame({'CV Mean':acc,'Std':std},index=models.keys())
    acc_all= pd.DataFrame(acc_all,index=models.keys()).T
    return acc, acc_all
    
models = base_learners()
acc, acc_all = trainCV(models, Xtrain, ytrain)
iplot(ff.create_table(acc.reset_index()))
acc_all.iplot(kind='box')
from sklearn.model_selection import GridSearchCV
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(Xtrain,ytrain)
print(gd.best_score_)
print(gd.best_estimator_)
n_estimators=[100,200,300,400,500,600]
max_depth=[2,3,4,5,6]
min_samples_leaf=[1,2,3]
hype_param={'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf}
grid=GridSearchCV(estimator=GradientBoostingClassifier(random_state=SEED),param_grid=hype_param,verbose=True)
grid.fit(Xtrain,ytrain)
print(grid.best_score_)
print(grid.best_estimator_)
# build a set of base learners
def base_learners():
    """Construct a list of base learners"""
    lr = LogisticRegression(random_state=SEED)
    svc = SVC(kernel='linear', C=0.1, gamma=0.1, probability=True,random_state=SEED)
    knn = KNeighborsClassifier()
    nb = GaussianNB()
    nn = MLPClassifier((80,10),random_state=SEED)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    ab = AdaBoostClassifier(n_estimators=100, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    
    models = {
        'Logistic Regression': lr,
        'SVM': svc,
        'KNN': knn,
        'Naive Bayes': nb,
        'Neural Network': nn,
        'Random Forest': rf,
        'Extra Trees': et,
        'AdaBoosting': ab,
        'GradientBoosting': gb
    }
    
    return models
from sklearn.ensemble import VotingClassifier
models = base_learners()
ensemble_voting=VotingClassifier(estimators=list(zip(models.keys(),models.values())), 
                       voting='soft')
scores=cross_val_score(ensemble_voting,Xtrain,ytrain, cv = 10,scoring = "accuracy")
print('The cross validated score is',scores.mean())
from sklearn.ensemble import BaggingClassifier
model =BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
scores=cross_val_score(model,Xtrain,ytrain,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',scores.mean())
xgboost=xgb.XGBClassifier(n_estimators=900,learning_rate=0.1)
scores=cross_val_score(xgboost,Xtrain,ytrain,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',scores.mean())
base_models = base_learners()
meta_learner = xgb.XGBClassifier(n_estimators= 2000,
                                 max_depth= 4,
                                 min_child_weight= 2,
                                 gamma=0.9,                        
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 objective= 'binary:logistic',
                                 nthread= -1,
                                 scale_pos_weight=1)
## Put the training and testing data here for better coherence
Xtrain = train[train.columns[1:]]
ytrain = train[train.columns[:1]]
Xtest = test

def train_base_learners(models, X, y):
    """Fit base models"""

    print("Fitting models")
    for i, (name, model) in enumerate(models.items()):
        print("%s..."%name, end=' ', flush=False)
        model.fit(X,y)
        print('done')
    print('Fitting done!')   
def pred_base_learners(models, X):
    "Generate a prediction matrix"
    P = np.zeros((X.shape[0],len(models)))
    P = pd.DataFrame(P)
    
    print("Generating base learner predictions")
    for i, (name, m) in enumerate(models.items()):
        print("%s..."%name, end=' ', flush=False)
        P.iloc[:,i] = m.predict_proba(X)[:,1]
        print('done')
        
    P.columns = models.keys()
    
    return P
train_base_learners(base_models, Xtrain,ytrain)
P = pred_base_learners(base_models, Xtest)

sns.heatmap(P.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
P_label = P.apply(lambda x: 1*(x>=0.5))
sns.heatmap(P_label.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
feature_importance = pd.DataFrame({'Random Forest': base_models['Random Forest'].feature_importances_,
                                   'Extra Trees': base_models['Extra Trees'].feature_importances_,
                                   'GradientBoosting': base_models['GradientBoosting'].feature_importances_,
                                   'AdaBoosting': base_models['AdaBoosting'].feature_importances_},
                                  index=Xtrain.columns.values)
display(feature_importance)
fig = tools.make_subplots(rows=2, cols=2,subplot_titles=(
                        'AdaBoosting Feature Importance', 
                        'Extra Trees Feature Importance', 
                        'GradientBoosting Feature Importance',
                        'Random Forest Feature Importance'))

bars = feature_importance.iplot(kind='barh',subplots=True,asFigure=True)
fig.append_trace(bars.data[0], 1, 1)
fig.append_trace(bars.data[1], 1, 2)
fig.append_trace(bars.data[2], 2, 1)
fig.append_trace(bars.data[3], 2, 2)
fig.layout.update({'showlegend':False})
iplot(fig)
from sklearn.base import clone

## Some useful parameters
ntrain = train.shape[0]
ntest  = test.shape[0]
kf = KFold(n_splits=5, random_state=SEED) # k=5, split the data into 10 equal part

def stacking(base_learners, meta_learners, X, y):
    """Training routine for stacking"""
    print('Fitting final base learners...')
    train_base_learners(base_learners, X, y)
    print('done')
    
    ## geneerate input of meta learner
    print('Generate cross-validated predictions...')
    cv_pred, cv_y = [], []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        fold_Xtrain, fold_ytrain = X[train_index,:], y[train_index]
        fold_Xtest,  fold_ytest = X[test_index,:], y[test_index]
        
        # inner loops: step 3 and 4
        fold_base_learners = {name:clone(model) for name, model in base_learners.items()}
        
        train_base_learners(fold_base_learners, fold_Xtrain, fold_ytrain)
        fold_P_base = pred_base_learners(fold_base_learners, fold_Xtest)
        
        cv_pred.append(fold_P_base)
        cv_y.append(fold_ytest)
    print('CV prediction done')
    
    cv_pred = np.vstack(cv_pred)
    cv_y = np.vstack(cv_y)
    
    print('Fitting meta learner...', end='')
    meta_learner.fit(cv_pred, cv_y)
    print('done')
    
    return base_learners, meta_learner
def ensemble_predict(base_learners,meta_learner,X):
    """Generate prediction from ensemble"""
    
    P_base = pred_base_learners(base_learners, X)
    return P_base.values, meta_learner.predict(P_base.values)
cv_base_learners, cv_meta_learner= stacking(base_learners(), meta_learner,Xtrain.values, ytrain.values)
P_base, P_ensemble = ensemble_predict(cv_base_learners, meta_learner, Xtest)
p_ens[:10]
p_ens1[:10]
from mlens.ensemble import SuperLearner

val_train, val_test = train_test_split(train,test_size=0.3,random_state=SEED,stratify=train['Survived'])
val_Xtrain=val_train[val_train.columns[1:]]
val_ytrain=val_train[val_train.columns[:1]]
val_Xtest=val[val_test.columns[1:]]
val_ytest=val[val_test.columns[:1]]
# Instantiate the ensemble with 10 folds
super_learner = SuperLearner(folds=10,random_state=SEED,verbose=2,backend='multiprocessing')
# Add the base learners and the meta learner
super_learner.add(list(base_learners().values()),proba=True)
super_learner.add_meta(LogisticRegression(), proba=True)

# Train the ensemble
super_learner.fit(val_Xtrain,val_ytrain)
# predict the test set
p_ens = super_learner.predict(val_Xtest)[:,1]
p_ens_label = 1*(p_ens>=0.5)
print('The acccuracy of super learner:',metrics.accuracy_score(p_ens_label, val_ytest))
# Generate Submission File 
Submission = pd.DataFrame({ 'PassengerId': PassengerId_test,
                            'Survived': P_ensemble })
Submission.to_csv("Submission.csv", index=False)
