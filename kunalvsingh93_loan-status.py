import pandas as pd
import numpy as np
import seaborn as sns
train_set=pd.read_csv('../input/loan-statusclassifiaction/train_loan1.csv')
train_set
test_set=pd.read_csv('../input/test-data/test_loan1.csv')
test_set
train_set.isnull().sum()
test_set.isnull().sum()
train_set['Gender'].value_counts()
train_set['Married'].value_counts()
train_set['Dependents'].value_counts()
train_set['Dependents']=train_set['Dependents'].str.replace('+','')
train_set['Dependents'].value_counts()
test_set['Dependents'].value_counts()
test_set['Dependents']=test_set['Dependents'].str.replace('+','')
train_set['Education'].value_counts()
def a(x):
    if x=='Graduate':
        x=2
    else:
        x=1
    return x
train_set['Education']=train_set['Education'].apply(a)
train_set['Education'].value_counts()
test_set['Education']=test_set['Education'].apply(a)
test_set['Education'].value_counts()
train_set['Self_Employed'].value_counts()
train_set['ApplicantIncome'].value_counts()
train_set['CoapplicantIncome'].value_counts()
train_set['CoapplicantIncome'].min()
train_set['CoapplicantIncome'].max()
train_set['ApplicantIncome'].min()
train_set['ApplicantIncome'].max()
sns.distplot(train_set['ApplicantIncome'])
sns.distplot(train_set['CoapplicantIncome'])
train_set['LoanAmount'].value_counts()
train_set['LoanAmount'].min()
train_set['LoanAmount'].max()
train_set['Credit_History'].value_counts()
def a(x):
    if x==1.0:
        x='Yes'
    else:
        x='No'
    return x
train_set['Credit_History']=train_set['Credit_History'].apply(a)
train_set['Credit_History'].value_counts()
test_set['Credit_History']=test_set['Credit_History'].apply(a)
test_set['Credit_History'].value_counts()
train_set['Property_Area'].value_counts()
train_set['Loan_Status'].value_counts()
def a(x):
    if x=='Y':
        x=1
    else:
        x=0
    return x
train_set['Loan_Status']=train_set['Loan_Status'].apply(a)
train_set['Loan_Status'].value_counts()
train_set['Dependents']=pd.to_numeric(train_set['Dependents'],errors='coerce')
test_set['Dependents']=pd.to_numeric(test_set['Dependents'],errors='coerce')
train_set['Loan_Amount_Term'].value_counts()
train_set.skew()
import scipy.stats as st
l=['ApplicantIncome','CoapplicantIncome','LoanAmount', 'Loan_Amount_Term']
l
for i in l:
    train_set[i]=st.boxcox(train_set[i]+1)[0]
train_set.skew()
for i in l:
        test_set[i]=st.boxcox(test_set[i]+1)[0]
test_set.skew()
train_set.info()
test_set.info()
train_set.isnull().sum()[train_set.isnull().sum()!=0]
test_set.isnull().sum()[test_set.isnull().sum()!=0]
pd.DataFrame(test_set.groupby(['Dependents','Credit_History','Gender','Married'])['Education'].value_counts())
a=test_set.loc[test_set['Gender'].isnull()]
a
pd.DataFrame(a.groupby(['Dependents','Credit_History','Married'])['Education'].value_counts())
test_set.loc[(test_set['Married']=='No')
             &(test_set['Dependents']==0)
             &(test_set['Credit_History']=='Yes')
             &(test_set['Education']==2)
             &(test_set['Gender'].isnull()), 'Gender']='Male'
test_set['Gender']=test_set['Gender'].fillna('Male')
train_set.loc[(train_set['Married']=='Yes')
             &(train_set['Dependents']==3)
             &(train_set['Credit_History']=='Yes')
             &(train_set['Loan_Status']==1)
             &(train_set['Gender'].isnull()), 'Gender']='Male'
train_set.loc[(train_set['Married']=='Yes')
             &(train_set['Dependents']==2)
             &(train_set['Credit_History']=='No')
             &(train_set['Loan_Status']==0)
             &(train_set['Gender'].isnull()), 'Gender']='Male'
train_set.loc[(train_set['Married']=='Yes')
             &(train_set['Property_Area']==1)
             &(train_set['Credit_History']=='Yes')
             &(train_set['Education']==2)
             &(train_set['Gender'].isnull()), 'Gender']='Male'
train_set.loc[(train_set['Married']=='Yes')
             &(train_set['Property_Area']==2)
             &(train_set['Credit_History']=='No')
             &(train_set['Education']==2)
             &(train_set['Gender'].isnull()), 'Gender']='Male'
train_set.loc[(train_set['Married']=='Yes')
             &(train_set['Property_Area']==2)
             &(train_set['Credit_History']=='Yes')
             &(train_set['Education']==2)
             &(train_set['Gender'].isnull()), 'Gender']='Male'
train_set.loc[(train_set['Married']=='Yes')
             &(train_set['Property_Area']==3)
             &(train_set['Credit_History']=='Yes')
             &(train_set['Education']==2)
             &(train_set['Gender'].isnull()), 'Gender']='Male'
train_set['Gender']=train_set['Gender'].fillna('Male')
train_set.isnull().sum()[train_set.isnull().sum()!=0]
b=train_set.loc[train_set['Married'].isnull()]
b
pd.DataFrame(b.groupby(['Property_Area','Credit_History','Gender'])['Education'].value_counts())
pd.DataFrame(train_set.groupby(['Property_Area','Credit_History','Gender','Married'])['Education'].value_counts())
train_set['Married']=train_set['Married'].fillna('Yes')
test_set.isnull().sum()[train_set.isnull().sum()!=0]
c=test_set.loc[test_set['Dependents'].isnull()]
c
pd.DataFrame(c.groupby(['Property_Area','Gender'])['Credit_History'].value_counts())
pd.DataFrame(test_set.groupby(['Property_Area','Gender','Credit_History'])['Dependents'].value_counts())
train_set.loc[(train_set['Married']=='Yes')
             &(train_set['Loan_Status']==1)
             &(train_set['Gender']=='Male')
             &(train_set['Dependents'].isnull()), 'Dependents']=0
train_set.loc[(train_set['Married']=='Yes')
             &(train_set['Loan_Status']==0)
             &(train_set['Gender']=='Male')
             &(train_set['Dependents'].isnull()), 'Dependents']=0
train_set['Dependents']=train_set['Dependents'].fillna(0)
test_set['Dependents']=test_set['Dependents'].fillna(0)
test_set['Married']=test_set['Married'].fillna(0)
d=test_set.loc[test_set['Self_Employed'].isnull()]
d
pd.DataFrame(d.groupby(['Property_Area','Gender'])['Credit_History'].value_counts())
pd.DataFrame(test_set.groupby(['Property_Area','Gender','Credit_History'])['Self_Employed'].value_counts())
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']==1)
             &(train_set['Gender']=='Male')
             &(train_set['Self_Employed'].isnull()), 'Self_Employed']='No'
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']==3)
             &(train_set['Gender']=='Male')
             &(train_set['Self_Employed'].isnull()), 'Self_Employed']='No'
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']==2)
             &(train_set['Gender']=='Female')
             &(train_set['Self_Employed'].isnull()), 'Self_Employed']='No'
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']==2)
             &(train_set['Gender']=='Male')
             &(train_set['Self_Employed'].isnull()), 'Self_Employed']='No'
train_set['Self_Employed']=train_set['Self_Employed'].fillna('No')
test_set['Self_Employed']=test_set['Self_Employed'].fillna('No')
f=test_set.loc[test_set['LoanAmount'].isnull()]
f
pd.DataFrame(f.groupby(['Education','Dependents'])['Property_Area'].value_counts())
pd.DataFrame(test_set.groupby(['Education','Dependents','Property_Area'])['LoanAmount'].mean())
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']=='Urban')
             &(train_set['Gender']=='Male')
             &(train_set['Loan_Status']==1)
             &(train_set['Education']==2)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=149.81250
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']=='Rural')
             &(train_set['Gender']=='Male')
             &(train_set['Loan_Status']==1)
             &(train_set['Education']==1)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=115.77778
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']=='Urban')
             &(train_set['Gender']=='Male')
             &(train_set['Loan_Status']==0)
             &(train_set['Education']==1)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=118.250
train_set.loc[(train_set['Credit_History']=='No')
             &(train_set['Property_Area']=='Rural')
             &(train_set['Gender']=='Male')
             &(train_set['Loan_Status']==0)
             &(train_set['Education']==2)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=148.722
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']=='Semiurban')
             &(train_set['Gender']=='Female')
             &(train_set['Loan_Status']==1)
             &(train_set['Education']==2)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=149.57
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']=='Semiurban')
             &(train_set['Gender']=='Male')
             &(train_set['Loan_Status']==1)
             &(train_set['Education']==2)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=146.06
train_set.loc[(train_set['Credit_History']=='Yes')
             &(train_set['Property_Area']=='Urban')
             &(train_set['Gender']=='Male')
             &(train_set['Loan_Status']==1)
             &(train_set['Education']==2)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=149.81250
train_set.loc[(train_set['Credit_History']=='No')
             &(train_set['Property_Area']=='Urban')
             &(train_set['Gender']=='Male')
             &(train_set['Loan_Status']==0)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=142.521739
train_set.loc[(train_set['Property_Area']=='Urban')
             &(train_set['Loan_Status']==0)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=139.49
train_set.loc[(train_set['Property_Area']=='Rural')
             &(train_set['Loan_Status']==0)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=158.4477
train_set.loc[(train_set['Property_Area']=='Rural')
             &(train_set['Loan_Status']==1)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=147.6656
train_set.loc[(train_set['Property_Area']=='Semiurban')
             &(train_set['Loan_Status']==0)
             &(train_set['LoanAmount'].isnull()), 'LoanAmount']=154.566
test_set.loc[(test_set['Property_Area']=='Rural')
             &(test_set['Dependents']==0)
             &(test_set['Education']==1)
             &(test_set['LoanAmount'].isnull()), 'LoanAmount']=127.0000
test_set.loc[(test_set['Property_Area']=='Semiurban')
             &(test_set['Dependents']==0)
             &(test_set['Education']==2)
             &(test_set['LoanAmount'].isnull()), 'LoanAmount']=132.54
test_set.loc[(test_set['Property_Area']=='Semiurban')
             &(test_set['Dependents']==1)
             &(test_set['Education']==2)
             &(test_set['LoanAmount'].isnull()), 'LoanAmount']=157.56
test_set.loc[(test_set['Property_Area']=='Urban')
             &(test_set['Dependents']==1)
             &(test_set['Education']==2)
             &(test_set['LoanAmount'].isnull()), 'LoanAmount']=134.5000
train_set['LoanAmount']=train_set['LoanAmount'].fillna(train_set['LoanAmount'].mean())
test_set['LoanAmount']=test_set['LoanAmount'].fillna(test_set['LoanAmount'].mean())
test_set.isnull().sum()[test_set.isnull().sum()!=0]
train_set.isnull().sum()[train_set.isnull().sum()!=0]
g=test_set.loc[test_set['Loan_Amount_Term'].isnull()]
g
pd.DataFrame(g.groupby(['Married'])['Property_Area'].value_counts())
pd.DataFrame(train_set.groupby(['Married','Property_Area'])['Loan_Amount_Term'].mean())
train_set.loc[(train_set['Property_Area']=='Rural')
             &(train_set['Married']=='No')
             &(train_set['Loan_Status']==0)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=362.5
train_set.loc[(train_set['Property_Area']=='Urban')
             &(train_set['Married']=='No')
             &(train_set['Loan_Status']==0)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=351.724
train_set.loc[(train_set['Property_Area']=='Urban')
             &(train_set['Married']=='Yes')
             &(train_set['Loan_Status']==0)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=334.05
train_set.loc[(train_set['Property_Area']=='Rural')
             &(train_set['Married']=='Yes')
             &(train_set['Loan_Status']==0)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=335.44
train_set.loc[(train_set['Property_Area']=='Semiurban')
             &(train_set['Married']=='Yes')
             &(train_set['Loan_Status']==0)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=349.65
train_set.loc[(train_set['Property_Area']=='Urban')
             &(train_set['Married']=='No')
             &(train_set['Loan_Status']==1)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=337.89
train_set.loc[(train_set['Property_Area']=='Rural')
             &(train_set['Married']=='No')
             &(train_set['Loan_Status']==1)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=363.24
train_set.loc[(train_set['Property_Area']=='Semiurban')
             &(train_set['Married']=='Yes')
             &(train_set['Loan_Status']==1)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=345.81
train_set.loc[(train_set['Property_Area']=='Urban')
             &(train_set['Married']=='Yes')
             &(train_set['Loan_Status']==1)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=324.131
train_set.loc[(train_set['Property_Area']=='Rural')
             &(train_set['Married']=='Yes')
             &(train_set['Loan_Status']==1)
             &(train_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=336.67
train_set['Loan_Amount_Term']=train_set['Loan_Amount_Term'].fillna(train_set['Loan_Amount_Term'].mean())
train_set.isnull().sum()[train_set.isnull().sum()!=0]
test_set.loc[(test_set['Property_Area']=='Urban')
             &(test_set['Married']=='No')
             &(test_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=343.82
test_set.loc[(test_set['Property_Area']=='Rural')
             &(test_set['Married']=='No')
             &(test_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=362.94
test_set.loc[(test_set['Property_Area']=='Urban')
             &(test_set['Married']=='Yes')
             &(test_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=327.06
test_set.loc[(test_set['Property_Area']=='Semiurban')
             &(test_set['Married']=='Yes')
             &(test_set['Loan_Amount_Term'].isnull()), 'Loan_Amount_Term']=346.57
test_set['Loan_Amount_Term']=test_set['Loan_Amount_Term'].fillna(test_set['Loan_Amount_Term'].mean())
test_set.isnull().sum()[test_set.isnull().sum()!=0]
train_set.head()
train_set=pd.get_dummies(train_set,columns=['Gender','Married','Self_Employed','Credit_History','Property_Area'])
train_set.shape
test_set=pd.get_dummies(test_set,columns=['Gender','Married','Self_Employed','Credit_History','Property_Area'])
test_set.shape
train=train_set.drop('Loan_ID',axis=1)
test=test_set.drop('Loan_ID',axis=1)
train.info()
X=train.drop('Loan_Status',axis=1)
y=train['Loan_Status']
X.info()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
#from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from tpot import TPOTClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)
pipeline_lr=Pipeline([('scalar1',StandardScaler()),
                     ('lr',LogisticRegression())])
pipeline_dt=Pipeline([('scaler2',StandardScaler()),
                     ('dt',DecisionTreeClassifier())])
pipeline_rf=Pipeline([('scalar3',StandardScaler()),
                     ('rfc',RandomForestClassifier())])
pipeline_knn=Pipeline([('scalar4',StandardScaler()),
                     ('knn',KNN())])
pipeline_xgbc=Pipeline([('scalar5',StandardScaler()),
                     ('xgboost',XGBClassifier())])
pipeline_lgbc=Pipeline([('scalar6',StandardScaler()),
                     ('lgbc',lgb.LGBMClassifier())])
pipeline_ada=Pipeline([('scalar7',StandardScaler()),
                     ('adaboost',AdaBoostClassifier())])
pipeline_sgdc=Pipeline([('scalar8',StandardScaler()),
                     ('sgradient',SGDClassifier())])
pipeline_nb=Pipeline([('scalar9',StandardScaler()),
                     ('nb',GaussianNB())])
pipeline_extratree=Pipeline([('scalar10',StandardScaler()),
                     ('extratree',ExtraTreesClassifier())])
pipeline_svc=Pipeline([('scalar11',StandardScaler()),
                     ('svc',SVC())])
pipeline_gbc=Pipeline([('scalar12',StandardScaler()),
                     ('GBC',GradientBoostingClassifier())])
pipelines=[pipeline_lr,pipeline_dt,pipeline_rf,pipeline_knn,pipeline_xgbc,pipeline_lgbc,pipeline_ada,pipeline_sgdc,pipeline_nb,pipeline_extratree,pipeline_svc,pipeline_gbc]
best_accuracy=0.0
best_classifier=0
best_pipeline=""
pipe_dict={0:'Logistic Regression',1:'Random Forest',2:'Decision Tree',3:'KNN',4:'XGBC',5:'LGBC',6:'ADA',7:'SGDC',8:'NB',9:'ExtraTree',10:'SVC',11:'GBC'}
sns.countplot(train_set['Loan_Status'])
from imblearn.over_sampling import SMOTE
smote = SMOTE('auto')
X_sm, y_sm = smote.fit_sample(X_train,y_train)
print(X_sm.shape, y_sm.shape)
for i in pipelines:
    i.fit(X_sm,y_sm)
    predictions=i.predict(X_test)
    print('Classification Report : \n',(classification_report(y_test,predictions)))
for i,model in enumerate(pipelines): print('{} Train Accuracy {}'.format(pipe_dict[i],model.score(X_sm,y_sm)))
for i,model in enumerate(pipelines): print('{} Test Accuracy {}'.format(pipe_dict[i],model.score(X_test,y_test)))
# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
r_param = [{"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(0, 4, 10)},
              
               {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_leaf":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]},
            {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2'],
                 "classifier__C": np.logspace(0, 4, 10),
                 "classifier__solver":['newton-cg','saga','sag','liblinear'] ##This solvers don't allow L1 penalty
                 },
           
{"classifier": [DecisionTreeClassifier()],
 "classifier__criterion":['gini','entropy'], 
 "classifier__max_depth":[5,8,15,25,30,None],
 "classifier__min_samples_leaf":[1,2,5,10,15,100],
 "classifier__max_leaf_nodes": [2, 5,10]},
           
{'classifier':[lgb.LGBMClassifier()], 
 'classifier__n_estimators':np.arange(50,250,5),
 'classifier__max_depth':np.arange(2,15,5), 
 'classifier__num_leaves':np.arange(2,60,5)},
           
           {'classifier':[XGBClassifier()],
            "classifier__learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
            "classifier__max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15], 
            "classifier__min_child_weight" : [ 1, 3, 5, 7 ], 
            "classifier__gamma" : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ], 
            "classifier__colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]},
           
           {'classifier':[SGDClassifier()], ##Stocasticated Gradient decent
       # "classifier__C": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0],
        "classifier__penalty": ['l2']},
           
            {"classifier":[GradientBoostingClassifier()],
        "classifier__learning_rate":np.arange(0.05,0.5,0.01),
        "classifier__n_estimators":np.arange(50,250,5),
        'classifier__max_depth':np.arange(4,15,5),
        #'classifier__num_leaves':np.arange(4,60,5),
        "classifier__min_samples_leaf":[1,2,5,10,15,100],
        "classifier__max_leaf_nodes": [2, 5,10]}

      ]
           
           
import sklearn.metrics
scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average = 'micro')
rsearch = RandomizedSearchCV(pipe, r_param, cv=5, verbose=0,n_jobs=-1,random_state=0,scoring=scorer)
rsearch.fit(X_train,y_train)
print(rsearch.best_estimator_)
GBC=GradientBoostingClassifier(ccp_alpha=0.0,
                                            criterion='friedman_mse', init=None,
                                            learning_rate=0.22000000000000003,
                                            loss='deviance', max_depth=4,
                                            max_features=None, max_leaf_nodes=2,
                                            min_impurity_decrease=0.0,
                                            min_impurity_split=None,
                                            min_samples_leaf=1,
                                            min_samples_split=2,
                                            min_weight_fraction_leaf=0.0,
                                            n_estimators=110,
                                            n_iter_no_change=None,
                                            presort='deprecated',
                                            random_state=None, subsample=1.0,
                                            tol=0.0001, validation_fraction=0.1,
                                            verbose=0, warm_start=False)
GBC=RandomizedSearchCV(estimator=RandomForestClassifier(),
                   param_distributions=[{'n_estimators': [5, 10,20,50,100],
                               'max_depth':[5, 10, 15,50,None],
                               'min_samples_leaf':[1,2,5,10, 50, 100],
                               'min_samples_split': [2,4,6,10,20,100,200]}])
GBC_model=GBC.fit(X_train,y_train)
Score=GBC_model.score(X_train,y_train)
Score
GBC_predictions=GBC_model.predict(test)
GBC_predictions.shape
sub = pd.DataFrame(GBC_predictions)
def a(x):
    if x==1:
        x='Y'
    else:
        x='N'
    return x
sub=sub.rename(columns={0:'Loan_Status'})
sub
sub['Loan_Status']=sub['Loan_Status'].apply(a)
att=pd.DataFrame(test_set['Loan_ID'])
att
final=pd.concat([att,sub],axis=1)
final
final.to_csv('Submission_loan_status5.csv',index=False)