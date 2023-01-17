import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df
df.isnull().sum()
df.info()
df['GRE Score'].unique()
df['TOEFL Score'].unique()
df['University Rating'].value_counts()
df['SOP'].unique()
df.columns
df.rename(columns={'LOR ':'LOR','Chance of Admit ':'Chance of Admit'},inplace=True)
df['LOR'].value_counts()
df['Research'].value_counts()
df.drop(columns=['Serial No.'],axis=1,inplace=True)
for i in df.columns:
    if i=='Research':
        continue
    else:
        
        plt.figure(figsize=(10,6))
        sns.distplot(df[i])
        plt.xlabel(i,fontsize=13)
        plt.title('Distribution of {}'.format(i),fontsize=15)
        plt.show()
        print('\n')

plt.figure(figsize=(10,6))
sns.scatterplot(df['GRE Score'],df['TOEFL Score'])
plt.xlabel('GRE Scores',fontsize=13)
plt.ylabel('TOFEL Scores',fontsize=13)
plt.title('Distribution of Gre and Tofel scores',fontsize=15)
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(df['GRE Score'],df['University Rating'])
plt.xlabel('GRE Scores',fontsize=13)
plt.ylabel('University ratings',fontsize=13)
plt.title('Distribution of Gre and University ratings',fontsize=15)
plt.show()
plt.figure(figsize=(15,6))
sns.scatterplot(df['GRE Score'],df['Chance of Admit'],hue=df['Research'])
plt.xlabel('GRE Scores',fontsize=13)
plt.ylabel('Chance of Admit',fontsize=13)
plt.title('GRE Scores VS Chance of Admit',fontsize=15)
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(df['GRE Score'],df['LOR'],hue=df['Research'])
plt.xlabel('GRE Scores',fontsize=13)
plt.ylabel('LOR',fontsize=13)
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(df['GRE Score'],df['CGPA'])
plt.xlabel('GRE score',fontsize=13)
plt.ylabel('CGPA',fontsize=13)
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(df['GRE Score'],df['CGPA'],hue=df['Research'])
plt.xlabel('GRE score',fontsize=13)
plt.ylabel('CGPA',fontsize=13)
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(df['GRE Score'],df['SOP'],hue=df['Research'])
plt.xlabel('GRE score',fontsize=13)
plt.ylabel('SOP',fontsize=13)
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(df['TOEFL Score'],df['CGPA'],hue=df['Research'])
plt.xlabel('TOEFL score',fontsize=13)
plt.ylabel('CGPA',fontsize=13)
plt.title('TOEFL score VS CGPA',fontsize=15)
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(df['CGPA'],df['Chance of Admit'],hue=df['Research'])
plt.xlabel('CGPA',fontsize=13)
plt.ylabel('Chance of Admit',fontsize=13)
plt.title('CGPA  VS Chance of Admit',fontsize=15)
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(df['GRE Score'],df['CGPA'])
plt.xlabel('GRE Score',fontsize=13)
plt.ylabel('CGPA',fontsize=13)
plt.title('GRE Score  VS CGPA',fontsize=15)
plt.show()
df.describe()
sns.pairplot(df)
plt.show()
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)
plt.show()
from sklearn.model_selection import train_test_split
X=df.drop(columns=['Chance of Admit'])
Y=df['Chance of Admit']
X.shape,Y.shape
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.23,random_state=23)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_sc=sc.fit_transform(X_train)
X_test_sc=sc.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

#metrics evaluation
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
pipeline_lr=Pipeline([('lr_regression',LinearRegression())])
pipeline_ls=Pipeline([('lasso_regression',Lasso())])
pipeline_rd=Pipeline([('Ridge_regression',Ridge())])
pipeline_dt=Pipeline([('DecisionTree_regression',DecisionTreeRegressor())])
pipeline_knn=Pipeline([('KNN',KNeighborsRegressor())])
pipeline_rf=Pipeline([('RandomForestRegressor',RandomForestRegressor())])
pipeline_ad=Pipeline([('Adaboosting',AdaBoostRegressor())])
pipeline_gb=Pipeline([('GradientBoosting',GradientBoostingRegressor())])
pipeline_xg=Pipeline([('Xgboost',XGBRegressor())])
pipeline_svm=Pipeline([('SVM',SVR())])
pipelines=[pipeline_lr,pipeline_ls,pipeline_rd,pipeline_dt,pipeline_knn,pipeline_rf,pipeline_ad,
          pipeline_gb,pipeline_xg,pipeline_svm]
for pipe in pipelines:
    pipe.fit(X_train,y_train)
pipe_dict={0:'Linear Regression',1: 'Lasso Regression', 2:'Ridge Regression',3:'DecisonTree',4: 'KNN',5:'RandomForestRegressor',6:'Adaboosting',7:'Gradientboosting',
          8:'Xgboost',9:'SVM'}
for i,model in enumerate(pipelines):
    print(pipe_dict[i])
    print('-'*20)
    print('cross_val_score_{} : {}'.format(pipe_dict[i],
                                           (cross_val_score(model,X_train,y_train,cv=5,scoring='explained_variance')).mean()))
    
    
    print('Mean_squared_error_{} : {}'.format(pipe_dict[i],mean_squared_error(y_test,model.predict(X_test))))
    
    print('RMSE_{}: {}'.format(pipe_dict[i],np.sqrt(mean_squared_error(y_test,model.predict(X_test)))))
    
    print('R2square_{} : {}'.format(pipe_dict[i],r2_score(y_test,model.predict(X_test))))
    print('\n')
best_r2square=0.0
best_regressor=0
best_pipeline=''
for i,model in enumerate(pipelines):
     if r2_score(y_test,model.predict(X_test))>best_r2square:
            best_r2square=r2_score(y_test,model.predict(X_test))
            
            best_pipeline=model
            best_regressor=i
print("Regressor with best r2square is {}".format(pipe_dict[best_regressor]))
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time
def gridrandomized(feature,clf,X,Y,parameters,scoring):
    if feature=='Random':
        search_obj=RandomizedSearchCV(estimator=clf,param_distributions=parameters,scoring=scoring,n_jobs=-1,cv=5)
    elif feature=='Grid':
        search_obj=GridSearchCV(estimator=clf,param_grid=parameters,scoring=scoring,n_jobs=-1,cv=5)
    
    start=time.time() 
    fit_obj=search_obj.fit(X,Y)
    end=time.time()
    print("The total time taken to execute {}".format(end-start))
    best_obj=fit_obj.best_estimator_
    print((best_obj))
    best_params=fit_obj.best_params_
    print((best_params))
    best_score=fit_obj.best_score_
    print((best_score))
    
pipe=Pipeline([('regressor',RandomForestRegressor())])
grid_param=[
            {
             'regressor':[Lasso()],
             'regressor__alpha':[x for x in [0.1,0.2,0.3,0.5,0.8,1,100]],
             'regressor__normalize':[False,True],
             'regressor__max_iter':[i for i in [1000,1300,1500,1800]]
            },
   
            {
             'regressor':[Ridge()],
             'regressor__alpha':[x for x in [0.1,0.2,0.3,0.5,0.8,1,100]],
             'regressor__normalize':[False,True],
             'regressor__max_iter':[i for i in [1000,1300,1500]]
             },
    
            {
             'regressor':[KNeighborsRegressor()],
             'regressor__n_neighbors':[x for x in range(5,40,3)],
             'regressor__leaf_size':[30,35,40,45],
             'regressor__weights':['uniform', 'distance'],
             'regressor__algorithm':['auto', 'ball_tree','kd_tree','brute'],
             'regressor__n_jobs':[-1]
            },
    
            {
             'regressor':[DecisionTreeRegressor()],
             'regressor__max_depth':[x for x in range(2,40,3)],
             'regressor__max_features':['auto', 'sqrt'],
             'regressor__min_samples_split': [x for x in [2,3,4,5,6,7,8,9,10,11,12]], 
             'regressor__min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]
            },
    
            {
             'regressor':[RandomForestRegressor()],
             'regressor__criterion':['gini','entropy'],
             'regressor__n_estimators':[10,15,20,25,30],
             'regressor__min_samples_leaf':[1,2,3],
             'regressor__min_samples_split':[3,4,5,6,7], 
             'regressor__n_jobs':[-1]
            },
    
            {
            'regressor':[AdaBoostRegressor()],
            'regressor__n_estimators':[50,100],
            'regressor__learning_rate':[0.01,0.05,0.1,0.3,1],
            'regressor__loss':['linear', 'square', 'exponential']
            },
   
            {
             'regressor':[GradientBoostingRegressor()],
             'regressor__learning_rate':[0.1,0.3,0.5,0.6],
             'regressor__n_estimators':[100,150,200],
             'regressor__min_samples_split':[2,3,4],
             'regressor__max_depth':[3,5,7,10],
             'regressor__max_features':['sqrt','auto'],
             'regressor__alpha':[0.9]
            },
    
            {
             'regressor':[XGBRegressor()],
             'regressor__max_depth':[3, 4, 5, 6, 8, 10, 12, 15],
             'regressor__min_child_weight':[1, 3, 5, 7],
             'regressor__gamma':[0.0, 0.1, 0.2 , 0.3, 0.4],
             'regressor__colsample_bytree':[0.3, 0.4, 0.5 , 0.7]},
    
            {
             'regressor':[SVR()],
             'regressor__kernel':['linear','rbf'],
             'regressor__C':[6,7,8,9,10,11,12]
            }
            ]
scorings='neg_mean_squared_error'
gridrandomized(feature='Grid',clf=pipe,X=X_train,Y=y_train,scoring=scorings,parameters=grid_param)
from sklearn.pipeline import make_pipeline
pipe=Pipeline([('regressor',Ridge())])
grid_param=[{
    'regressor':[Ridge()],
             'regressor__alpha':[x for x in [0.1,0.2,0.3,0.5,0.8,1,100]],
             'regressor__normalize':[False,True],
             'regressor__max_iter':[i for i in [1000,1300,1500]]
}]
grid_search=GridSearchCV(pipe,grid_param,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
best_model=grid_search.fit(X_train,y_train)
best_model.best_estimator_
rd=Ridge(alpha=0.5,max_iter=1000)
model=rd.fit(X_train,y_train)
print("The mse is {}".format(mean_squared_error(y_test,model.predict(X_test))))
print("The rmse is {}".format(np.sqrt(mean_squared_error(y_test,model.predict(X_test)))))
print("The r2square is {}".format(r2_score(y_test,model.predict(X_test))))
def ask():
    GRE=int(input("Enter GRE score: "))
    if GRE > 340:
        print("ERROR, Score should be less than 340")
        print("ENTER SCORES AGAIN")
        GRE=int(input("Enter GRE score again: "))
    else:
         pass
        
    TOEFL=int(input('Enter TOEFL score: '))
    if TOEFL>120:
        print("ERROR, Score should be less than 120")
        print("ENTER SCORES AGAIN")
        TOEFL=int(input("Enter TOEFL score again: "))
    else:
        pass
        
    University_Rating=float(input('Enter University_Rating'))
    if University_Rating>5:
        print("ERROR, Ratings should ranges from 0-5")
        print("ENTER RATINGS AGAIN")
        University_Rating=float(input("Enter ratings again: "))
    else:
        pass
        
    SOP=float(input('Enter SOP rating'))
    if SOP>5:
        print("ERROR, Ratings should ranges from 0-5")
        print("ENTER RATINGS AGAIN")
        SOP=float(input("Enter ratings again: "))
    else:
        pass
        
    LOR=float(input('Enter LOR rating'))
    if LOR>5:
        print("ERROR, Ratings should ranges from 0-5")
        print("ENTER RATINGS AGAIN")
        LOR=float(input("Enter ratings again: "))
    else:
        pass
        
    CGPA=float(input('Enter CGPA points'))
    if CGPA>10:
        print("ERROR, CGPA Should ranges from 0-10")
        print("ENTER CGPA AGAIN")
        CGPA=float(input("Enter cgpa again: "))
    else:
        pass
    
    Research=input('Yes or No: ')
    if Research=='Yes':
        Research=1
    elif Research=='No':
        Research=0
    else:
        print( 'enter correctly')
        Research=input('Yes or No')
    
    final=model.predict([[GRE,TOEFL,University_Rating,SOP,LOR,CGPA,Research]])
    return "The chance of admit is {}".format(final)
ask()
