import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_set=pd.read_csv('../input/banking-modelclassification/train_loan.csv')
train_set
test_set=pd.read_csv('../input/banking-modelclassification/test_loan.csv')
test_set
train_set.shape,test_set.shape
train_set['Data']='train'
test_set['Data']='test'
test_set['Interest_Rate']=np.nan
combined=pd.concat([train_set,test_set],ignore_index=True,sort=False)
combined
combined.shape
combined.info()
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.head()
combined['Loan_Amount_Requested'].value_counts()
combined['Loan_Amount_Requested']=combined['Loan_Amount_Requested'].str.replace(',','')
combined.info()
combined['Loan_Amount_Requested'].unique()
combined['Loan_Amount_Requested']=pd.to_numeric(combined['Loan_Amount_Requested'],errors='coerce')
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.info()
combined.head()
combined['Length_Employed']=combined['Length_Employed'].str.replace('<','')
combined['Length_Employed'].value_counts()
combined['Length_Employed']=combined['Length_Employed'].str.replace('+','')
combined['Length_Employed']=combined['Length_Employed'].str.replace('years','')
combined['Length_Employed'].value_counts()
combined['Length_Employed']=combined['Length_Employed'].str.replace('year','')
combined['Length_Employed'].value_counts()
combined['Length_Employed']=pd.to_numeric(combined['Length_Employed'],errors='coerce')
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.info()
combined['Home_Owner'].value_counts()
combined.head()
combined['Purpose_Of_Loan'].value_counts()
a=combined[combined['Home_Owner'].isnull()]
a
a.loc[a.Purpose_Of_Loan=='home_improvement']
combined.loc[combined.Purpose_Of_Loan=='home_improvement','Home_Owner']='Own'
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.loc[combined.Purpose_Of_Loan=='house','Home_Owner']='Rent'
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.head()
pd.DataFrame(combined.groupby('Length_Employed')['Annual_Income'].mean())
combined.Length_Employed=combined.Length_Employed.transform(lambda x:x.fillna(x.mean()))
combined.Annual_Income=combined.groupby('Length_Employed')['Annual_Income'].transform(lambda x:x.fillna(x.mean()))
combined.isnull().sum()[combined.isnull().sum()!=0]
combined.Home_Owner.value_counts()
combined.Home_Owner=combined.Home_Owner.fillna(combined.Home_Owner.mode()[0])
combined.Home_Owner.value_counts()
combined.Home_Owner=combined.Home_Owner.replace(['None','Other'],['Mortgage']*2)
combined.Home_Owner.value_counts()
combined.info()
combined.Income_Verified.value_counts()
combined.Income_Verified=combined.Income_Verified.replace(['VERIFIED - income source','VERIFIED - income'],['Verified']*2)
combined.Income_Verified=combined.Income_Verified.replace('not verified','Not_Verified')
combined.Income_Verified.value_counts()
combined.Purpose_Of_Loan.value_counts()
combined.Gender.value_counts()
combined.Months_Since_Deliquency.value_counts()
plt.figure(figsize=(10,8))
ax=sns.heatmap(combined.corr(),annot=True,linewidths=.5,fmt='.1f')
plt.show()
combined.head()
combined.drop(['Months_Since_Deliquency','Number_Open_Accounts'],axis=1,inplace=True)
combined.columns
df=combined.drop('Loan_ID',axis=1)
df.head()
df.info()
from sklearn.feature_extraction import FeatureHasher
fh=FeatureHasher(n_features=6,input_type='string')
hashed_feature=fh.fit_transform(df['Purpose_Of_Loan'])
hashed_feature=hashed_feature.toarray()
fh=pd.DataFrame(hashed_feature)
df=pd.get_dummies(df,columns=['Home_Owner','Income_Verified','Gender'],drop_first=True)
dff=pd.concat([df,fh],axis=1,sort=False)
dff.shape
dff.drop('Purpose_Of_Loan',axis=1,inplace=True)
dff.head()
dff.shape
dff.head(50)
dff.skew()
dff.columns
l=['Inquiries_Last_6Mo','Annual_Income']
for i in l:
    sns.distplot(dff[i])
    plt.show()
for i in l:
    sns.boxplot(dff[i])
    plt.show()
import scipy.stats as st
for i in l:
    dff[i]=list(st.boxcox(combined[i]+1)[0])
dff.skew()
for i in l:
    sns.boxplot(dff[i])
    plt.show()
train=dff.loc[dff['Data']=='train']
train.shape
test=dff.loc[dff['Data']=='test']
test.shape
train=train.drop('Data',axis=1)
test=test.drop(['Data','Interest_Rate'],axis=1)
train.shape,test.shape
train_set.shape,test_set.shape
X=train.drop('Interest_Rate',axis=1)
y=train['Interest_Rate']
sns.countplot(train['Interest_Rate'])
train['Interest_Rate'].value_counts()
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
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=0)
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
pipelines=[pipeline_lr,pipeline_dt,pipeline_rf,pipeline_knn,pipeline_xgbc,pipeline_lgbc,pipeline_ada,pipeline_sgdc,pipeline_nb,pipeline_extratree]
best_accuracy=0.0
best_classifier=0
best_pipeline=""
pipe_dict={0:'Logistic Regression',1:'Random Forest',2:'Decision Tree',3:'KNN',4:'XGBC',5:'LGBC',6:'ADA',7:'SGDC',8:'NB',9:'ExtraTree'}
for i in pipelines:
    i.fit(X_train,y_train)
    predictions=i.predict(X_test)
    print('Classification Report : \n',(classification_report(y_test,predictions)))
for i,model in enumerate(pipelines):
    print('{} Test Accuracy {}'.format(pipe_dict[i],model.score(X_test,y_test)))
for i,model in enumerate(pipelines):
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_classifier=i
        best_pipeline=model
print("Classifier with best accuracy:{}".format(pipe_dict[best_classifier]))
# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])

# Create dictionary with candidate learning algorithms and their hyperparameters
r_param = [ {"classifier": [RandomForestClassifier()],
             "classifier__n_estimators": [10,15,20],
             "classifier__max_depth":[15,25,30],
             "classifier__min_samples_leaf":[5,10,15,100],
             "classifier__max_leaf_nodes": [5,10]},
           
            {"classifier":[GradientBoostingClassifier()],
            "classifier__learning_rate":np.arange(0.05,0.5,0.1),
            "classifier__n_estimators":np.arange(5,10,20),
            'classifier__max_depth':np.arange(4,15),
            "classifier__min_samples_leaf":[10,15,100],
            "classifier__max_leaf_nodes": [5,10]}

      ]
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
import sklearn.metrics
scorer = sklearn.metrics.make_scorer(sklearn.metrics.f1_score, average = 'micro')
rsearch = RandomizedSearchCV(pipe, r_param, cv=5, verbose=0,n_jobs=-1,random_state=0,scoring=scorer)
rsearch.fit(X_train,y_train)
print(rsearch.best_estimator_)
print("The mean accuracy of the model is through randomized search is :",rsearch.score(X_test,y_test))
GBC=GradientBoostingClassifier(learning_rate=0.42000000000000004,
                                            max_depth=4, max_leaf_nodes=10,
                                            n_estimators=110)
GBC_model=GBC.fit(X_train,y_train)
accuracy_score=GBC.score(X_test,y_test)
accuracy_score
pred=GBC.predict(X_test)
f1_score=f1_score(pred,y_test,average='weighted')
f1_score
GBC_predictions=GBC.predict(test)
GBC_predictions.shape
test.shape
Submission_gbc = pd.DataFrame(GBC_predictions)
Submission_gbc.to_csv('Submission_loan_gbc.csv',index=False)