# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/onlinenewspopularity/OnlineNewsPopularity.csv')
data.head()
data['url'][2000]
import nltk
import re
import string
from datetime import datetime
date=[]
date_original=[]

for i in range(data.shape[0]):
    x=re.findall(r'[0-9]{4}/[0-9]{2}/[0-9]{2}',data['url'][i])
    date.append(x)
    
for i in date:
    for r in i:
        date_original.append(r)
data['date']=date_original
data['date']= pd.to_datetime(data['date'])
fig=plt.figure(figsize=(10,10))
ax=fig.gca()
plt.plot(data['date'],data[' shares'])
plt.show()
plt.scatter(data[' timedelta'],data[' shares'],c='r')

data.shape[0]
for i in range(data.shape[0]):
    if data[' shares'][i] > 28000:
        data.drop(index=i,inplace=True)
        
data=data.reset_index(drop=True)
plt.scatter(data[' timedelta'],data[' shares'],c='r')
plt.plot(data[' num_imgs'],data[' shares'],'ro',label='Images')
plt.plot(data[' num_videos'],data[' shares'],'b^',label='Videos')
plt.legend()
sns.scatterplot(data[' n_tokens_title'],data[' shares'])
sns.scatterplot(data[' n_tokens_content'],data[' shares'])
plt.hist(data[' n_tokens_content'],alpha=0.5,color='b')
plt.hist(data[' shares'],alpha=0.5,color='g')
plt.legend()
from scipy.stats import norm
fig= plt.figure(figsize=(10,10))
ax=fig.gca()
ax.set_title("The 'Sharing' distribution of whole dataset")
sns.distplot(data[' shares'],ax=ax, fit=norm)
print("Skew:",data[' shares'].skew())
lifestyle_articles=data[data[' data_channel_is_lifestyle'] == 1][' shares'].sum()
entertainment_articles=data[data[' data_channel_is_entertainment'] == 1][' shares'].sum()
business_articles=data[data[' data_channel_is_bus'] == 1][' shares'].sum()
socialmedia_articles=data[data[' data_channel_is_socmed'] == 1][' shares'].sum()
technical_articles=data[data[' data_channel_is_tech'] == 1][' shares'].sum()
world_articles=data[data[' data_channel_is_world'] == 1][' shares'].sum()
articles_types=np.array([lifestyle_articles,entertainment_articles,business_articles,socialmedia_articles,technical_articles,world_articles],dtype=np.int64)
fig= plt.figure(figsize=(10,10))
ax=fig.gca()
ax.set_title('TOTAL SHARED ARTICLES OF EACH GENRE')
ax.set_ylabel('Number of Articles')
plt.bar(x=['lifestyle','entertainment','business','socialmedia','technical','world'],height=articles_types,color='rgbkymc')

articles_types
monday_articles=data[data[' weekday_is_monday'] == 1][' shares'].sum()
tuesday_articles=data[data[' weekday_is_tuesday'] == 1][' shares'].sum()
wednesday_articles=data[data[' weekday_is_wednesday'] == 1][' shares'].sum()
thursday_articles=data[data[' weekday_is_thursday'] == 1][' shares'].sum()
friday_articles=data[data[' weekday_is_friday'] == 1][' shares'].sum()
saturday_articles=data[data[' weekday_is_saturday'] == 1][' shares'].sum()
sunday_articles=data[data[' weekday_is_sunday'] == 1][' shares'].sum()
weekend_articles=data[data[' is_weekend'] == 1][' shares'].sum()
articles_publishing_days= np.array([monday_articles,tuesday_articles,wednesday_articles,thursday_articles,friday_articles,
                                    saturday_articles,sunday_articles,weekend_articles])
fig= plt.figure(figsize=(10,10))
ax=fig.gca()
ax.set_title('Total sharing of articles day-wise')
ax.set_ylabel('Number of Articles')
plt.bar(x=['monday','tuesday','wednesday','thursday','friday','saturday','sunday','weekend'],height=articles_publishing_days
        ,color='rgbkymc')

result=[]
days=[' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday',' weekday_is_thursday',' weekday_is_friday',
     ' weekday_is_saturday',' weekday_is_sunday',' is_weekend']
genre=[' data_channel_is_lifestyle',' data_channel_is_entertainment',' data_channel_is_bus',' data_channel_is_socmed',
       ' data_channel_is_tech',' data_channel_is_world']
for i in days:
    list1=[]
    for j in genre:
        list1.append(data.groupby([i,j])[' shares'].sum()[1][1])
    print('Best channel on {} has articles {} and channel is {}'.format(i,max(list1),genre[list1.index(max(list1))]))
Worst_min_shares=pd.DataFrame(data.groupby([' kw_min_min'],sort=True)[' shares'].sum())
Worst_max_shares=pd.DataFrame(data.groupby([' kw_max_min'],sort=True)[' shares'].sum())
Worst_avg_shares=pd.DataFrame(data.groupby([' kw_avg_min'],sort=True)[' shares'].sum())
Best_min_shares=pd.DataFrame(data.groupby([' kw_min_max'],sort=True)[' shares'].sum())
Best_max_shares=pd.DataFrame(data.groupby([' kw_max_max'],sort=True)[' shares'].sum())
Best_avg_shares=pd.DataFrame(data.groupby([' kw_avg_max'],sort=True)[' shares'].sum())
Normal_min_shares=pd.DataFrame(data.groupby([' kw_min_avg'],sort=True)[' shares'].sum())
Normal_max_shares=pd.DataFrame(data.groupby([' kw_max_avg'],sort=True)[' shares'].sum())
Normal_avg_shares=pd.DataFrame(data.groupby([' kw_avg_avg'],sort=True)[' shares'].sum())
Worst_min_shares.plot()
Lda_00=pd.DataFrame(data.groupby(by=[' LDA_00'])[' shares'].sum().sort_values(ascending=False)).reset_index()
Lda_01=pd.DataFrame(data.groupby(by=[' LDA_01'])[' shares'].sum().sort_values(ascending=False)).reset_index()
Lda_02=pd.DataFrame(data.groupby(by=[' LDA_02'])[' shares'].sum().sort_values(ascending=False)).reset_index()
Lda_03=pd.DataFrame(data.groupby(by=[' LDA_03'])[' shares'].sum().sort_values(ascending=False)).reset_index()
Lda_04=pd.DataFrame(data.groupby(by=[' LDA_04'])[' shares'].sum().sort_values(ascending=False)).reset_index()


## mean respective lda for > 50 shares
mean_lda_00=np.mean(Lda_00[Lda_00[' shares'] > 50])[0]
mean_lda_01=np.mean(Lda_01[Lda_01[' shares'] > 50])[0]
mean_lda_02=np.mean(Lda_02[Lda_02[' shares'] > 50])[0]
mean_lda_03=np.mean(Lda_03[Lda_03[' shares'] > 50])[0]
mean_lda_04=np.mean(Lda_04[Lda_04[' shares'] > 50])[0]
fig=plt.figure(figsize=(8,8))
ax=fig.gca()
plt.bar(x=['mean_lda_00','mean_lda_01','mean_lda_02','mean_lda_03','mean_lda_04'],
        height=[mean_lda_00,mean_lda_01,mean_lda_02,mean_lda_03,mean_lda_04])
sns.scatterplot(x=data[' global_subjectivity'],y=data[' shares']) # Subjectivity from 0.0-1.0
columns_group_3=[' global_sentiment_polarity', ' global_rate_positive_words',
       ' global_rate_negative_words', ' rate_positive_words',
       ' rate_negative_words', ' avg_positive_polarity',
       ' min_positive_polarity', ' max_positive_polarity',
       ' avg_negative_polarity', ' min_negative_polarity',
       ' max_negative_polarity', ' title_subjectivity',
       ' title_sentiment_polarity', ' abs_title_subjectivity',
       ' abs_title_sentiment_polarity', ' shares']
fig, ax = plt.subplots(figsize=(20,20))

sns.heatmap(data[columns_group_3].corr(),linewidth=1.0,ax=ax,square=True,annot=True)
from sklearn.decomposition import PCA
y= data[' shares']
pca_data=data.drop(labels=['url',' shares','date'],axis=1)
pca_data.head()
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
data_transformed= scaler.fit_transform(pca_data)
pca=PCA()
principal_comp=pd.DataFrame(pca.fit_transform(pca_data))
principal_comp.head()
pca.explained_variance_ratio_
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(pca_data, y,  
    test_size=test_size,random_state=23)

param_grid= {'n_estimators':[20,40],
            'max_depth':[10,20],
             'max_features':['auto',10,20],
             'bootstrap':[True,False],             
            }

## Initial result gave both extreme values as best parameters so run again by increasing limit
random_search= RandomizedSearchCV(RandomForestRegressor(),param_distributions=param_grid,
                                  cv=5,scoring='neg_mean_absolute_error',
                         verbose=1,n_jobs=-1)
randomsearch_result=random_search.fit(X_train,y_train)
best_paramters= randomsearch_result.best_params_
pd.DataFrame(randomsearch_result.cv_results_).sort_values('mean_test_score',ascending=False)
best_paramters
from sklearn.model_selection import cross_val_score
rf=RandomForestRegressor(n_estimators=40,max_depth=10,max_features=10)
scores=cross_val_score(rf,X_train,y_train,scoring='neg_mean_absolute_error',cv=5)
absolute_scores= -scores.mean()
## def display_scores(score):
   ## print("Mean:", score.mean())
   ## print("Standard deviation:", score.std())
rf.fit(X_train,y_train)
from sklearn.metrics import mean_absolute_error
y_pred=rf.predict(X_test)
test_score=mean_absolute_error(y_test,y_pred)
test_score
pd.DataFrame({'actual_train_mae_score':absolute_scores,
             'actual_test_mae_score':test_score},index=['Mean'])

### reversing the normalizing of target variable
df=pd.DataFrame(rf.feature_importances_,pca_data.columns).reset_index()
df.columns=['variables','score']
sorted_df=df.sort_values('score',ascending=False)
important_variables=sorted_df.iloc[1:15,:]
fig=plt.figure(figsize=(15,15))
ax=fig.gca()
plt.bar(x=important_variables.variables,height=important_variables.score,color='r')
plt.xticks(rotation=90)
plt.show()
pca_final_data=principal_comp[[0,1]]
X_train_pca,X_test_pca,y_train_pca,y_test_pca=train_test_split(pca_final_data,y,test_size=0.2,random_state=23)
param_grid_pca= {'n_estimators':[20,40],
            'max_depth':[10,20],
             'bootstrap':[True,False],             
            }
random_search_pca= RandomizedSearchCV(RandomForestRegressor(),param_distributions=param_grid_pca,
                                  cv=5,scoring='neg_mean_absolute_error',
                                  verbose=1,n_jobs=-1)
randomsearch_result_pca=random_search_pca.fit(X_train_pca,y_train_pca)
best_paramters_pca= randomsearch_result_pca.best_params_
pd.DataFrame(randomsearch_result_pca.cv_results_).sort_values('mean_test_score',ascending=False)
best_paramters_pca
rf_pca=RandomForestRegressor(n_estimators=20,max_depth=10)
scores_1=cross_val_score(rf_pca,X_train_pca,y_train_pca,scoring='neg_mean_absolute_error',cv=10)
absolute_scores_1=-scores_1.mean()

rf_pca.fit(X_train_pca,y_train_pca)
y_predict_pca= rf_pca.predict(X_test_pca)
test_score_pca= mean_absolute_error(y_test_pca,y_predict_pca)
pd.DataFrame({'train_mse_score':[absolute_scores_1],
             'test_mse_score':[test_score_pca]},index=['Mean'])
X_train.shape[0]
train_sizes=[500,800,1000,1250,2500,5000,10000,12000,16000,18000,20000]
from sklearn.model_selection import learning_curve
train_sizes,train_scores,validation_scores= learning_curve(rf,X=X_train,y=y_train,train_sizes=train_sizes,
                                             cv=3,scoring='neg_mean_absolute_error')
train_scores_mean= -train_scores.mean(axis=1)
validation_scores_mean=-validation_scores.mean(axis=1)
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('MAE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a random forest regression model', fontsize = 18, y = 1.03)
plt.legend()
train_sizes,train_scores_pca,validation_scores_pca= learning_curve(rf_pca,X=X_train_pca,y=y_train_pca,train_sizes=train_sizes,
                                             cv=3,scoring='neg_mean_absolute_error')
train_scores_mean_pca= -train_scores_pca.mean(axis=1)
validation_scores_mean_pca=-validation_scores_pca.mean(axis=1)
plt.plot(train_sizes, train_scores_mean_pca, label = 'Training error PCA')
plt.plot(train_sizes, validation_scores_mean_pca, label = 'Validation error PCA')
plt.ylabel('MAE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a random forest regression model', fontsize = 18, y = 1.03)
plt.legend()
from sklearn.ensemble import GradientBoostingRegressor
param_gradboost={'n_estimators':[100,150],
                'max_depth':[5,10],
                'learning_rate':[0.1,0.2]}
pca_data_gbr= principal_comp[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
X_train_gbr,X_test_gbr,y_train_gbr,y_test_gbr=train_test_split(pca_data_gbr,y,test_size=0.2,random_state=23)
grad_randomsearch= RandomizedSearchCV(GradientBoostingRegressor(),param_distributions=param_gradboost,cv=3,
                                      scoring='neg_mean_absolute_error',n_jobs=-1,verbose=1)
grad_fit=grad_randomsearch.fit(X_train_gbr,y_train_gbr)
best_param_grad= grad_fit.best_params_
pd.DataFrame(grad_fit.cv_results_)
best_param_grad
gbr= GradientBoostingRegressor(n_estimators=50,max_depth=5,learning_rate=0.1)
grad_result= gbr.fit(X_train_gbr,y_train_gbr)
scores_boosting= cross_val_score(gbr,X_train_gbr,y_train_gbr,scoring='neg_mean_absolute_error',cv=5)
absolute_scores_boosting= - scores_boosting.mean()
y_pred_gbr=gbr.predict(X_test_gbr)
test_score_gbr= mean_absolute_error(y_test_gbr,y_pred_gbr)
pd.DataFrame({'train_mae_score':[absolute_scores_boosting],
             'test_mae_score':[test_score_gbr]},index=['Mean'])
train_sizes,train_scores_gbr,validation_scores_gbr= learning_curve(gbr,X=X_train_pca,y=y_train_pca,train_sizes=train_sizes,
                                             cv=5,scoring='neg_mean_absolute_error')
train_scores_mean_gbr= -train_scores_gbr.mean(axis=1)
validation_scores_mean_gbr=-validation_scores_gbr.mean(axis=1)
plt.plot(train_sizes, train_scores_mean_gbr, label = 'Training error PCA')
plt.plot(train_sizes, validation_scores_mean_gbr, label = 'Validation error PCA')
plt.ylabel('MAE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a gradient boosting regression model', fontsize = 18, y = 1.03)
plt.legend()
gbr_original=gbr.fit(X_train,y_train)
df_1=pd.DataFrame(X_train.columns,gbr_original.feature_importances_).reset_index()
df_1.columns=['score','variables']
select_columns=df_1.sort_values('score',ascending=False)['variables']
feature_importance_df= pd.concat(objs=[df,df_1],axis=1)
feature_importance_df
feature_importance_df.columns=['Variables_rf','Score_rf','score_gb','Variables_gb']
feature_importance_df=feature_importance_df.sort_values('Score_rf',ascending=False).reset_index(drop=True)
np.sum(feature_importance_df['score_gb'][0:25])
select_columns=df_1.sort_values('score',ascending=False)['variables'][0:25]
X_train_select,X_test_select,y_train_select,y_test_select=train_test_split(data[select_columns.reset_index()['variables']],
                                                                           y,test_size=0.2,random_state=23)
grad_randomsearch_select= RandomizedSearchCV(GradientBoostingRegressor(),param_distributions=param_gradboost,cv=3,
                                      scoring='neg_mean_absolute_error',n_jobs=-1,verbose=1)
grad_fit_select=grad_randomsearch_select.fit(X_train_select,y_train_select)
best_param_grad_select= grad_fit_select.best_params_
pd.DataFrame(grad_fit_select.cv_results_)
best_param_grad_select
gbr_select= GradientBoostingRegressor(n_estimators=100,max_depth=5,learning_rate=0.1)
grad_result_select= gbr_select.fit(X_train_select,y_train_select)
scores_boosting_select= cross_val_score(gbr_select,X_train_select,y_train_select,scoring='neg_mean_absolute_error',cv=5)
absolute_scores_boosting_select= - scores_boosting_select.mean()
y_pred_gbr_select=gbr_select.predict(X_test_select)
test_score_gbr_select= mean_absolute_error(y_test_select,y_pred_gbr_select)
pd.DataFrame({'train_mae_score':[absolute_scores_boosting_select],
             'test_mae_score':[test_score_gbr_select]},index=['Mean'])
train_sizes,train_scores_select,validation_scores_select= learning_curve(gbr_select,X=X_train_select,
                                                                         y=y_train_select,train_sizes=train_sizes,
                                                                           cv=5,scoring='neg_mean_absolute_error')
train_scores_mean_select= -train_scores_select.mean(axis=1)
validation_scores_mean_select=-validation_scores_select.mean(axis=1)
plt.plot(train_sizes, train_scores_mean_select, label = 'Training error PCA')
plt.plot(train_sizes, validation_scores_mean_select, label = 'Validation error PCA')
plt.ylabel('MAE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a gradient boosting regression model', fontsize = 18, y = 1.03)
plt.legend()
Comparison_df= pd.DataFrame({'Training_Scores':[absolute_scores,absolute_scores_1,
                                                absolute_scores_boosting,absolute_scores_boosting_select],
                            'Test_Scores':[test_score,test_score_pca,test_score_gbr,test_score_gbr_select]},
                            index=['Rf','Rf_PCA','gbr','gbr_select'])
Comparison_df['Variance']=np.subtract(Comparison_df['Training_Scores'],Comparison_df['Test_Scores'])
Comparison_df=Comparison_df.sort_values('Training_Scores')
Comparison_df
data.head()
X1= rf.predict(pca_data)
X2=rf_pca.predict(pca_final_data)
X3=gbr.predict(pca_data_gbr)
X4=gbr_select.predict(data[select_columns.reset_index()['variables']])
data_stacking= pd.DataFrame({'Random_Forest':X1,
                            'Random_Forest_PCA':X2,
                            'GBR':X3,
                            "GBR_select":X4,
                            "Target":y})
data_stacking.head()
from sklearn.linear_model import LinearRegression
X_train_stack,X_test_stack,y_train_stack,y_test_stack= train_test_split(
    data_stacking[['Random_Forest','Random_Forest_PCA','GBR','GBR_select']],y,test_size=0.2)
lr=LinearRegression()
lr.fit(X_train_stack,y_train_stack)
training_score_lr= cross_val_score(lr,X_train_stack,y_train_stack,scoring='neg_mean_absolute_error',cv=20)
absolute_training_lr= -training_score_lr.mean()
y_predict_lr= lr.predict(X_test_stack)
test_score_lr= mean_absolute_error(y_test_stack,y_predict_lr)
train_sizes,train_scores_lr,validation_scores_lr= learning_curve(lr,X=X_train_stack,
                                                                         y=y_train_stack,train_sizes=train_sizes,
                                                                           cv=10,scoring='neg_mean_absolute_error')
train_scores_mean_lr= -train_scores_lr.mean(axis=1)
validation_scores_mean_lr=-validation_scores_lr.mean(axis=1)
plt.plot(train_sizes, train_scores_mean_lr, label = 'Training error PCA')
plt.plot(train_sizes, validation_scores_mean_lr, label = 'Validation error PCA')
plt.ylabel('MAE', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a linear regression model', fontsize = 18, y = 1.03)
plt.legend()