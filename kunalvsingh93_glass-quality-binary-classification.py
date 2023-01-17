import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
training_set=pd.read_csv('../input/glass-quality-classification/Glass_train.csv')
testing_set=pd.read_csv('../input/glass-quality-classification/Glass_Test.csv')
training_set.head()
training_set.shape
testing_set.head()
testing_set.shape
training_set['Data']='train'
testing_set['Data']='test'
testing_set['class']=np.nan
combined=pd.concat([training_set,testing_set],sort=False,ignore_index=True)
combined
combined['x_avg']=(combined['xmax']+combined['xmin'])/2
combined['y_avg']=(combined['ymax']+combined['ymin'])/2
combined.drop(['xmin','xmax','ymin','ymax','log_area'],axis=1,inplace=True)
combined
l=['max_luminosity',
       'thickness', 'pixel_area', 'x_avg',
       'y_avg']
for i in l:
    sns.distplot(combined[i])
    plt.show()
for i in l:
    sns.boxplot(combined[i])
    plt.show()
combined.skew()
for i in l:
    combined[i]=list(st.boxcox(combined[i]+1)[0])
combined.skew()
for i in l:
    sns.boxplot(combined[i])
    plt.show()
sns.countplot(combined['class'])
combined['class'].value_counts()
#Divide into test and train:
train = combined.loc[combined['Data']=="train"]
test = combined.loc[combined['Data']=="test"]
train.shape
test.shape
train.drop('Data',axis=1,inplace=True)
test.drop('Data',axis=1,inplace=True)
train.head()
test.head()
X=train.drop('class',axis=1)
y=train['class']
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
pipeline_svc=Pipeline([('scalar11',StandardScaler()),
                     ('svc',SVC())])
pipeline_gbc=Pipeline([('scalar12',StandardScaler()),
                     ('GBC',GradientBoostingClassifier())])
# Lets make the list of pipelines
pipelines=[pipeline_lr,pipeline_dt,pipeline_rf,pipeline_knn,pipeline_xgbc,pipeline_lgbc,pipeline_ada,pipeline_sgdc,pipeline_nb,pipeline_extratree,pipeline_svc,pipeline_gbc]
best_accuracy=0.0
best_classifier=0
best_pipeline=""
pipe_dict={0:'Logistic Regression',1:'Decision Tree',2:'Random Forest',3:'KNN',4:'XGBC',5:'LGBC',6:'ADA',7:'SGDC',8:'NB',9:'ExtraTree',10:'SVC',11:'GBC'}
smote = SMOTE('minority')
X_sm, y_sm = smote.fit_sample(X_train,y_train)
print(X_sm.shape, y_sm.shape)
for i in pipelines:
    i.fit(X_sm,y_sm)
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
r_param = [{"classifier": [LogisticRegression()],
            "classifier__penalty": ['l2','l1'],
            "classifier__C": np.logspace(0, 4, 10)},
           
           {"classifier": [LogisticRegression()],
           "classifier__penalty": ['l2'],
           "classifier__C": np.logspace(0, 4, 10),
           "classifier__solver":['newton-cg','saga','sag','liblinear']},
           
          {"classifier": [DecisionTreeClassifier()],
           "classifier__criterion":['gini','entropy'],
           "classifier__max_depth":[5,8,15,25,30,None],
           "classifier__min_samples_leaf":[1,2,5,10,15,100],
           "classifier__max_leaf_nodes": [2, 5,10]},
     
          {"classifier": [RandomForestClassifier()],
           "classifier__n_estimators": [10, 100, 1000],
           "classifier__max_depth":[5,8,15,25,30,None], 
           "classifier__min_samples_leaf":[1,2,5,10,15,100],
           "classifier__max_leaf_nodes": [2, 5,10]},
      
           {'classifier':[lgb.LGBMClassifier()],
            'classifier__n_estimators':np.arange(50,250,5),
            'classifier__max_depth':np.arange(2,15,5),
            'classifier__num_leaves':np.arange(2,60,5)},
           
           {'classifier':[XGBClassifier()],
            "classifier__learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
            "classifier__max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
            "classifier__min_child_weight" : [ 1, 3, 5, 7 ],
            "classifier__gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "classifier__colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]},
           
           {'classifier':[AdaBoostClassifier()],        
           "classifier__n_estimators": sp_randint(50,250), 
            'classifier__learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
            'classifier__algorithm': ['SAMME', 'SAMME.R']},
           
           {'classifier':[KNN()],
            "classifier__weights":['uniform','distance'],
            'classifier__n_neighbors':np.arange(1,40),
            'classifier__leaf_size':np.arange(2,40)},
           
           {'classifier':[SVC()],                      
           'classifier__gamma':np.logspace(-4,2,10000),
           'classifier__C':np.logspace(-2,2,10000)},
           
           {"classifier":[GradientBoostingClassifier()],
            "classifier__learning_rate":np.arange(0.05,0.5,0.01),
            "classifier__n_estimators":np.arange(50,250,5),
            'classifier__max_depth':np.arange(4,15,5),
            "classifier__min_samples_leaf":[1,2,5,10,15,100],
            "classifier__max_leaf_nodes": [2, 5,10]}
           
          ]
rsearch = RandomizedSearchCV(pipe, r_param, cv=5, verbose=0,n_jobs=-1,random_state=0)
best_model_r = rsearch.fit(X_sm,y_sm)
print(best_model_r.best_estimator_)
print("The mean accuracy of the model is through randomized search is :",best_model_r.score(X_test,y_test))
Final_model=GradientBoostingClassifier(learning_rate=0.24000000000000005,
                                            max_depth=9, max_leaf_nodes=10,
                                            min_samples_leaf=15,
                                            n_estimators=165)
model=Final_model.fit(X_sm,y_sm)
model
model.score(X_test,y_test)
test.drop('class',axis=1,inplace=True)
test.head()
predictions=model.predict(test)
predictions
predictions.shape
predictions_prob=model.predict_proba(test)
Submission = pd.DataFrame(predictions_prob)
Submission.to_csv('Submission_glass.csv',index=False)