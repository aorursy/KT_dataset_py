import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import scipy
from collections import Counter
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import quantile_transform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import GammaRegressor
from sklearn.tree import DecisionTreeRegressor
import keras
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
import tensorflow as tf
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import uniform, truncnorm, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostRegressor
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
sales=pd.read_csv('../input/bigmart-sales-data/Train.csv')
sales.head(3)
sales_test=pd.read_csv('../input/bigmart-sales-data/Test.csv')
sales_test.head(3)
sales['Source']='train'
sales_test['Source']='test'
print('The number of rows present in the dataset',sales.shape[0])
print('The number of rows present in the test dataset',sales_test.shape[0])
print('The columns present in the dataset are',sales.columns.values)
print('Percentage of rows present in train data =',sales.shape[0]/(sales.shape[0]+sales_test.shape[0]))
print('Percentage of rows present in test data =',sales_test.shape[0]/(sales_test.shape[0]+sales.shape[0]))
print(sales.describe())
#Lets see how many retail shops are present and their distribution
outl=Counter(sales.loc[:,'Outlet_Identifier'].values)
print('Total number of stores present is =',len(list(outl.keys())))
plt.figure(figsize=(15,4))
sb.countplot(sales.loc[:,'Outlet_Identifier'].values,data=sales)
print('The number of unique product are',len(np.unique(sales.loc[:,'Item_Identifier'].values)))
#What about number of unique products present in the dataset
prod=Counter(sales.loc[:,'Item_Identifier'].values)
prod_val=list(prod.values())
ax=sb.countplot(prod_val,palette='rainbow')
ax.set_xlabel('Distribution of repeated products')
sb.distplot(sales.loc[:,'Item_MRP'].values)
from sklearn.preprocessing import quantile_transform
sb.distplot(quantile_transform(np.sin(5*np.sinh((np.sqrt(sales.loc[:,'Item_MRP'].values)))).reshape(-1,1),output_distribution='normal'),color='violet')
sb.distplot(sales.loc[:,'Item_Weight'].values,color='red')
#Which type of product are there in dataset and how about their distribution
plt.figure(figsize=(25,6))
ax1=plt.subplot(211)
sb.countplot(sales.loc[:,'Item_Type'].values,ax=ax1)
ax2=plt.subplot(212)
sb.countplot(sales.loc[:,'Item_Fat_Content'].values,ax=ax2)
#In which year most of the retail_outlets were opened
sb.countplot(sales.loc[:,'Outlet_Establishment_Year'],palette='cividis')
plt.figure(figsize=(15,6))
ax2=plt.subplot(311)
o=sb.countplot(sales.loc[:,'Outlet_Size'].values,ax=ax2,palette='rainbow',data=sales,hue='Outlet_Location_Type')
o.set_xlabel('Outlet_size')
ax3=plt.subplot(312)
sb.countplot(sales.loc[:,'Outlet_Location_Type'],ax=ax3,hue='Outlet_Type',data=sales)
ax4=plt.subplot(313)
sb.countplot(sales.loc[:,'Outlet_Type'],ax=ax4,palette='Oranges',data=sales,hue='Outlet_Size')
plt.subplots_adjust(top=2)
#Lets also understand which products each 'Outlet_Type' contains
plt.figure(figsize=(20,6))
sb.countplot(x='Outlet_Type',data=sales,hue='Item_Type')
categ=sales.select_dtypes(include='object').columns
categ
cont=sales.select_dtypes(include=['int','float'])
cor=cont.corr()
cor
sb.heatmap(cor)
plt.figure(figsize=(15,4))
sb.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',data=sales,palette='plasma')
sb.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=sales,palette='rainbow')
sb.boxplot(x='Item_Fat_Content',y='Item_Outlet_Sales',data=sales)
sb.boxplot(x='Outlet_Type',y='Item_MRP',data=sales)
#Regardless of Type of the store the price of the product remains same
plt.figure(figsize=(15,30))
ax=sales.groupby(by=['Outlet_Identifier','Item_Type'])['Item_Outlet_Sales'].median().plot.barh()
ax.set_xlabel('Mean Item sales')
sales=pd.concat((sales,sales_test),axis=0)
print('Total number of rows present in a combined dataset',sales.shape[0])
sales=sales.replace({'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'})
sales=sales.replace(to_replace='Supermarket Type3',value='Supermarket Type2')
plt.figure(figsize=(20,6))
sb.countplot(x='Outlet_Type',data=sales,hue='Item_Type')
plt.figure(figsize=(25,6))
count=Counter(sales['Item_Identifier'].apply(lambda x:x[:3]).sort_values())
sb.barplot(x=list(count.keys()),y=list(count.values()))
sales.loc[:,'Item_Brand']=sales['Item_Identifier'].apply(lambda x:x[:3])
brand_pivot=sales.pivot_table(index='Item_Brand',values='Item_MRP')
brand_pivot.reset_index(inplace=True)
brand_pivot.set_index('Item_Brand',inplace=True)
brand_pivot
for i,j in tqdm(sales.loc[:,['Item_Identifier','Item_Brand']].values):
    sales.loc[(sales.loc[:,'Item_Brand']==j)&(sales.loc[:,'Item_Identifier']==i),'Brand_Price_Diff']=brand_pivot.loc[j,'Item_MRP']-sales.loc[(sales.loc[:,'Item_Brand']==j)&(sales.loc[:,'Item_Identifier']==i),'Item_MRP']
sb.distplot(sales['Brand_Price_Diff'])
sales.loc[:,'New_itemtype']=sales.loc[:,'Item_Identifier'].apply(lambda x :x[0:2])
sales.replace({'FD':'Food','DR':'Drinks','NC':'Non_Consumables'},inplace=True)
sales.loc[(sales.loc[:,'Item_Type']=='Household')|(sales.loc[:,'Item_Type']=='Health and Hygiene'),'Item_Fat_Content']='Non-Edible'
plt.figure(figsize=(20,8))
sb.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='New_itemtype',data=sales)
#Lets see the Item_visibity vs Item sales with another dimension of Item_Type
plt.figure(figsize=(20,6))
sb.scatterplot(x='Item_Visibility',y='Item_Outlet_Sales',data=sales,hue='Item_Type')
z=sales.groupby(by=['Outlet_Identifier','New_itemtype'])['New_itemtype'].size().unstack()
z.reset_index(inplace=True)
z1=z.loc[:,z.columns.values[1:]].apply(lambda x:x/(x.sum()),axis=1)
z1=z1.assign(Outlet_Identifier=z.loc[:,'Outlet_Identifier'])
z1.set_index('Outlet_Identifier',inplace=True)
z1
for i in z1.columns.values:
    for j in z1.index:
        sales.loc[(sales.loc[:,'Outlet_Identifier']==j)&(sales.loc[:,'New_itemtype']==i),'prod_prob_inside']=z1.loc[j,i]
z2=z.loc[:,z.columns.values[1:]].apply(lambda x:x/(x.sum()),axis=0)
z2=z2.assign(Outlet_Identifier=z.loc[:,'Outlet_Identifier'])
z2.set_index('Outlet_Identifier',inplace=True)
z2
for k in z2.columns.values:
    for l in z2.index:
        sales.loc[(sales.loc[:,'Outlet_Identifier']==l)&(sales.loc[:,'New_itemtype']==k),'prod_prob_outside']=z2.loc[l,k]
#Lets also check the brand of the Outlet_shape because people will go to those shops which are present from long time
#How to check for the brand of outlet?
#Simple>Recent outlet establishment year is 2009-Outlet Establishment year gives the number of years of its existence
sales.loc[:,'No of Years for Outlet']=2009-sales.loc[:,'Outlet_Establishment_Year']
sb.distplot(quantile_transform(sales.loc[:,'Item_MRP'].values.reshape(-1,1),output_distribution='normal'),color='green')
sales.loc[:,'Item_MRP']=quantile_transform(sales.loc[:,'Item_MRP'].values.reshape(-1,1),output_distribution='normal')
print(np.unique(sales.loc[:,'Item_Visibility'].values))
#As we can see here there is also item_visibility which is 'zero',how can a item be invisible at all which doesnt make sense
#So lets replace that zero with 'mean/median' value of 'Item_Visibility'
sales.loc[sales.loc[:,'Item_Visibility']==0,'Item_Visibility']=sales.loc[:,'Item_Visibility'].median()
#Before preprocessing the Item_visibility correlation with sales was -0.008 now after this correlation increased to -0.18
visibility_pivot=sales.pivot_table(index='Item_Identifier',values='Item_Visibility')
visibility_pivot.reset_index(inplace=True)
visibility_pivot.set_index('Item_Identifier',inplace=True)
visibility_pivot.head()
for i in visibility_pivot.columns:
    for j in visibility_pivot.index:
        sales.loc[sales.loc[:,'Item_Identifier']==j,'Item_visib_avg']=(sales.loc[sales.loc[:,'Item_Identifier']==j,'Item_Visibility'])/(visibility_pivot.loc[j,'Item_Visibility'])
#Lets check how our newly created features correlated with our Target Variable
plt.figure(figsize=(20,8))
sb.heatmap(sales.corr(),annot=True)
#As we can see some of 'Within' features are correlated well but most of the 'Outside' features are very well correlated with Item_Outlet_Sales
linear=LinearRegression()
mr=sales.loc[~sales.loc[:,'Item_Weight'].isnull(),'Item_MRP'].values.reshape(-1,1)
wgt=sales.loc[~sales.loc[:,'Item_Weight'].isnull(),'Item_Weight'].values.reshape(-1,1)
wgt_tst=sales.loc[sales.loc[:,'Item_Weight'].isnull(),'Item_MRP'].values.reshape(-1,1)
linear.fit(mr,wgt)
sales.loc[sales.loc[:,'Item_Weight'].isnull(),'Item_Weight']=linear.predict(wgt_tst)
l=LogisticRegression()
lg=sales.loc[~sales.loc[:,'Outlet_Size'].isnull(),'Item_MRP'].values.reshape(-1,1)
lgy=sales.loc[~sales.loc[:,'Outlet_Size'].isnull(),'Outlet_Size'].values
lgp=sales.loc[sales.loc[:,'Outlet_Size'].isnull(),'Item_MRP'].values.reshape(-1,1)
l.fit(lg,lgy)
sales.loc[sales.loc[:,'Outlet_Size'].isnull(),'Outlet_Size']=l.predict(lgp)
sales_tr=sales.loc[~sales.loc[:,'Item_Outlet_Sales'].isnull(),:]
sales_tst=sales.loc[sales.loc[:,'Item_Outlet_Sales'].isnull(),:].drop(columns='Item_Outlet_Sales')
train=sales.loc[sales.loc[:,'Source']=='train']

test=sales.loc[sales.loc[:,'Source']=='test']
sales_tr.sort_index(inplace=True)
sales_tst.sort_index(inplace=True)
print('After Feature Engineering Number of columns in train dataset =',len(sales_tr.columns.values))
print('After Feature Engineering number of columns in test dataset =',len(sales_tst.columns.values))
sales_tst.columns
Y=sales_tr.loc[:,'Item_Outlet_Sales'].values
sales_tr.drop(columns=['Source','Item_Outlet_Sales','Item_Identifier','Item_Type','Item_Brand','Outlet_Identifier'],inplace=True)
X=sales_tr
sales_tst.drop(columns=['Source','Item_Identifier','Item_Type','Item_Brand','Outlet_Identifier'],inplace=True)
X2=sales_tr.loc[:,['Item_Weight','Item_Fat_Content','New_itemtype','Item_MRP','Outlet_Establishment_Year',
                  'Outlet_Size','Outlet_Location_Type','Outlet_Type']]
X2.head()
X2_tst=sales_tst.loc[:,['Item_Weight','Item_Fat_Content','New_itemtype','Item_MRP','Outlet_Establishment_Year',
                  'Outlet_Size','Outlet_Location_Type','Outlet_Type']]
cat_x2=['Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Establishment_Year']
int_x2=['Item_Weight','Item_MRP']
onehot_x2=['New_itemtype']
model1=[]
cv1=[]
algo1=[]
algo1.append(('knn',KNeighborsRegressor()))
algo1.append(('linear',LinearRegression()))
algo1.append(('random',RandomForestRegressor()))
algo1.append(('lasso',Lasso()))
algo1.append(('gradient',GradientBoostingRegressor()))
algo1.append(('bayesian',BayesianRidge()))
algo1.append(('gamma',GammaRegressor()))
algo1.append(('decision',DecisionTreeRegressor()))
for i,j in algo1:
    model1.append(i)
    category_pipeline=Pipeline([('ordinal',OrdinalEncoder())])
    onehot_pipeline=Pipeline([('onehot',OneHotEncoder())])
    integer_pipeline=Pipeline([('numeric',StandardScaler())])
    transformer1=ColumnTransformer(transformers=[('cat',category_pipeline,cat_x2),('int',integer_pipeline,int_x2),('onehot',onehot_pipeline,onehot_x2)
                                               ])
    classifier1=Pipeline([('trans',transformer1),(i,j)])
    val=cross_val_score(classifier1,X2,Y,scoring='neg_root_mean_squared_error')
    cv1.append(val)
plt.figure(figsize=(10,5))
sb.boxplot(x=model1,y=cv1)
boosting=Pipeline([('trans',transformer1),('gradient',RandomForestRegressor(max_depth=7,n_estimators=500,max_features=0.4))])
boost_val=cross_val_score(boosting,X2,Y,scoring='neg_root_mean_squared_error')
boost_val
X_btr,X_bcv,y_btr,y_bcv=train_test_split(X2,Y,test_size=0.4)
clfb=boosting.fit(X_btr,y_btr)
print(mean_squared_error(clfb.predict(X_btr),y_btr,squared=False))
print(mean_squared_error(clfb.predict(X_bcv),y_bcv,squared=False))
pred1=clfb.predict(X2_tst)
dummies1=pd.get_dummies(X2,columns=['New_itemtype'])
pd.Series(clfb.steps[1][1].feature_importances_,index=dummies1.columns).plot(kind='barh')
sb.distplot(pred1)
result1=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result1=result1.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result1=result1.assign(Item_Outlet_Sales=pred1)
result1.to_csv('result111.csv')
sklearn.metrics.SCORERS.keys()
cat_cols=list(sales_tr.select_dtypes(include='object').columns.values)
cat_cols.append('Outlet_Establishment_Year')
cat_cols.remove('New_itemtype')
onehot=['New_itemtype']
print(cat_cols)
int_cols=list(sales_tr.select_dtypes(include=['int','float']).columns.values)
print(int_cols)
cat_cols.append('No of Years for Outlet')
print(len(int_cols)+len(cat_cols))
model=[]
cv=[]
algo=[]
algo.append(('knn',KNeighborsRegressor()))
algo.append(('linear',LinearRegression()))
algo.append(('random',RandomForestRegressor()))
algo.append(('lasso',Lasso()))
algo.append(('gradient',GradientBoostingRegressor()))
algo.append(('Bayesian',BayesianRidge()))
algo.append(('gamma',GammaRegressor()))
algo.append(('decision',DecisionTreeRegressor()))
for i,j in algo:
    model.append(i)
    category_pipeline=Pipeline([('ordinal',OrdinalEncoder())])
    onehot_pipeline=Pipeline([('onehot',OneHotEncoder())])
    integer_pipeline=Pipeline([('numeric',StandardScaler())])
    transformer=ColumnTransformer(transformers=[('cat',category_pipeline,cat_cols),('int',integer_pipeline,int_cols),('onehot',onehot_pipeline,onehot),
                                               ])
    classifier=Pipeline([('trans',transformer),(i,j)])
    val=cross_val_score(classifier,X,Y,scoring='neg_root_mean_squared_error')
    cv.append(val)
plt.figure(figsize=(10,5))
sb.boxplot(x=model,y=cv)
params_label={'n_estimators':randint(2,400),'max_features':truncnorm(a=0, b=1, loc=0.4, scale=0.1),'min_samples_split': uniform(0.01, 0.199)}
rand_label=RandomizedSearchCV(RandomForestRegressor(),params_label,cv=5,return_train_score=True,scoring='neg_root_mean_squared_error')
rand_labelclass=Pipeline([('trans',transformer),('rand_label',rand_label)])
rand_labelclass.fit(X,Y)
label_data=pd.DataFrame(rand_labelclass.steps[1][1].cv_results_)
label_data.head(2)
rand_labelclass.steps[1][1].best_params_
boosting_classifier=Pipeline([('trans',transformer),('gradient',RandomForestRegressor(max_features=0.433,min_samples_split=0.0320,n_estimators=250))])
boost_val1=cross_val_score(boosting_classifier,X,Y,scoring='neg_root_mean_squared_error')
boost_val1
X_tr,X_cv,y_tr,y_cv=train_test_split(X,Y,test_size=0.4)
clf=boosting_classifier.fit(X_tr,y_tr)
print(mean_squared_error(clf.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf.predict(X_cv),y_cv,squared=False))
pred=clf.predict(sales_tst)
dummies2=pd.get_dummies(X,columns=['New_itemtype'])
pd.Series(clf.steps[1][1].feature_importances_,index=dummies2.columns).plot(kind='barh',figsize=(15,6))
sb.distplot(pred)
result=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result=result.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result=result.assign(Item_Outlet_Sales=pred)
result.to_csv('result00111.csv')
gradient_label={'n_estimators':randint(2,400),'max_features':truncnorm(a=0, b=1, loc=0.4, scale=0.1),'min_samples_split': uniform(0.01, 0.199)}
grad_label=RandomizedSearchCV(GradientBoostingRegressor(),gradient_label,cv=5,return_train_score=True,scoring='neg_root_mean_squared_error')
grad_labelclass=Pipeline([('trans',transformer),('rand_label',grad_label)])
grad_labelclass.fit(X,Y)
grad_labelclass.steps[1][1].best_params_
gradientreg_data=pd.DataFrame(grad_labelclass.steps[1][1].cv_results_)
gradientreg_data.head(2)
tuned_grad=GradientBoostingRegressor(max_features=0.45,min_samples_split=0.192,n_estimators=90)
tgrad_pipeline=Pipeline([('trans',transformer),('grad',tuned_grad)])
clf_grad=tgrad_pipeline.fit(X_tr,y_tr)
print(mean_squared_error(clf_grad.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf_grad.predict(X_cv),y_cv,squared=False))
pred_grad=clf.predict(sales_tst)
sb.distplot(pred_grad)
result=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result=result.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result=result.assign(Item_Outlet_Sales=pred_grad)
result.to_csv('result00112.csv')
para_extrareg={'n_estimators':randint(2,400),'max_features':truncnorm(a=0, b=1, loc=0.4, scale=0.1),'min_samples_split': uniform(0.01, 0.199)}
extra_rand=RandomizedSearchCV(ExtraTreesRegressor(),para_extrareg,cv=5,return_train_score=True,scoring='neg_root_mean_squared_error')
extrareg_pipeline=Pipeline([('trans',transformer),('extra',extra_rand)])
extrareg_pipeline.fit(X,Y)
extrareg_pipeline.steps[1][1].best_params_
extra_data=pd.DataFrame(extrareg_pipeline.steps[1][1].cv_results_)
extra_data.head(2)
tuned_extra=ExtraTreesRegressor(max_features=0.46,min_samples_split=0.0203,n_estimators=140)
extra_pipeline=Pipeline([('trans',transformer),('grad',tuned_extra)])
clf_extra=extra_pipeline.fit(X_tr,y_tr)
print(mean_squared_error(clf_extra.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf_extra.predict(X_cv),y_cv,squared=False))
pred_extra=clf_extra.predict(sales_tst)
sb.distplot(pred_extra)
result=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result=result.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result=result.assign(Item_Outlet_Sales=pred_extra)
result.to_csv('result00110.csv')
from sklearn.neural_network import MLPRegressor
mlpreg=MLPRegressor()
param_mlp={'hidden_layer_sizes':randint(10,500),'activation':['tanh','relu'],'learning_rate':['constant','adaptive']}
random_mlp=RandomizedSearchCV(mlpreg,param_mlp,cv=5,scoring='neg_root_mean_squared_error',return_train_score=True)
mlp_pipeline=Pipeline([('trans',transformer2),('mlp',random_mlp)])
mlp_pipeline.fit(X,Y)
mlp_pipeline.steps[1][1].best_params_
mlp_data=pd.DataFrame(mlp_pipeline.steps[1][1].cv_results_)
mlp_data.head(2)
mlp_class1=Pipeline([('trans',transformer),('mlp',MLPRegressor(hidden_layer_sizes=(300,120,60),learning_rate='adaptive',activation='relu'))])
clf51=mlp_class1.fit(X_tr,y_tr)
print(mean_squared_error(clf51.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf51.predict(X_cv),y_cv,squared=False))
pred71=clf51.predict(sales_tst)
sb.distplot(pred71)
from sklearn.ensemble import VotingRegressor
rand_forest1=RandomForestRegressor(max_features=0.433,min_samples_split=0.0320,n_estimators=250)
grad_boost1=GradientBoostingRegressor(max_features=0.45,min_samples_split=0.192,n_estimators=90)
extra_regres=ExtraTreesRegressor(max_features=0.46,min_samples_split=0.0203,n_estimators=140)
mlp_regres=MLPRegressor(hidden_layer_sizes=(300,120,60),learning_rate='adaptive',activation='relu')
voting=VotingRegressor(estimators=[('rand',rand_forest),('grad',grad_boost),('extra',extra_regres),('mlp',mlp_regres)])
vot_labelclass=Pipeline([('trans',transformer2),('voting',voting)])
clf_label=vot_labelclass.fit(X_tr,y_tr)
print(mean_squared_error(clf_label.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf_label.predict(X_cv),y_cv,squared=False))
pred_vot=clf_label.predict(sales_tst)
sb.distplot(pred_vot)
result=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result=result.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result=result.assign(Item_Outlet_Sales=pred_vot)
result.to_csv('result00129.csv')
onehot1=['Item_Fat_Content','Outlet_Location_Type','Outlet_Type','New_itemtype']
ordinal_cat=['Outlet_Size','Outlet_Establishment_Year','No of Years for Outlet']
print(onehot1)
print(ordinal_cat)
print(int_cols)
print(len(onehot1)+len(ordinal_cat)+len(int_cols))
model2=[]
cv2=[]
algo2=[]
algo2.append(('knn',KNeighborsRegressor()))
algo2.append(('linear',LinearRegression()))
algo2.append(('random',RandomForestRegressor()))
algo2.append(('lasso',Lasso()))
algo2.append(('gradient',GradientBoostingRegressor()))
algo2.append(('Bayesian',BayesianRidge()))
algo2.append(('gamma',GammaRegressor()))
algo2.append(('decision',DecisionTreeRegressor()))
for i,j in algo2:
    model2.append(i)
    ordinal_pipeline=Pipeline([('ordinal',OrdinalEncoder())])
    onehot_pipeline=Pipeline([('onehot',OneHotEncoder())])
    int_pipeline=Pipeline([('scaler',StandardScaler())])
    transformer2=ColumnTransformer([('onehot',onehot_pipeline,onehot1),('ordi',ordinal_pipeline,ordinal_cat),('int',int_pipeline,int_cols)])
    clas=Pipeline([('trans',transformer2),(i,j)])
    valid=cross_val_score(clas,X,Y,cv=8,scoring='neg_root_mean_squared_error')
    cv2.append(valid)
plt.figure(figsize=(10,5))
sb.boxplot(x=model2,y=cv2)
params={'n_estimators':randint(2,400),'max_features':truncnorm(a=0, b=1, loc=0.4, scale=0.1),'min_samples_split': uniform(0.01, 0.199)}
rand=RandomizedSearchCV(RandomForestRegressor(),params,cv=5,return_train_score=True,scoring='neg_root_mean_squared_error')
onehot_classifier=Pipeline([('trans',transformer2),('random',rand)])
onehot_classifier.fit(X,Y)
hyper_random=pd.DataFrame(onehot_classifier.steps[1][1].cv_results_)
hyper_random.head(3)
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.plot(hyper_random.loc[:,'param_max_features'],hyper_random.loc[:,'mean_train_score'],label='train')
plt.plot(hyper_random.loc[:,'param_max_features'],hyper_random.loc[:,'mean_test_score'],label='cv')
plt.xlabel('max_features')
plt.subplot(132)
plt.plot(hyper_random.loc[:,'param_n_estimators'],hyper_random.loc[:,'mean_train_score'],label='train')
plt.plot(hyper_random.loc[:,'param_n_estimators'],hyper_random.loc[:,'mean_test_score'],label='cv')
plt.xlabel('n_estimators')
plt.subplot(133)
plt.plot(hyper_random.loc[:,'param_min_samples_split'],hyper_random.loc[:,'mean_train_score'],label='train')
plt.plot(hyper_random.loc[:,'param_min_samples_split'],hyper_random.loc[:,'mean_test_score'],label='cv')
plt.xlabel('min_samples_split')
plt.legend()
onehot_classifier.steps[1][1].best_params_
random_class=Pipeline([('trans',transformer2),('random',RandomForestRegressor(max_features=0.433,min_samples_split=0.0300,n_estimators=366))])
clf2=random_class.fit(X_tr,y_tr)
print(mean_squared_error(clf2.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf2.predict(X_cv),y_cv,squared=False))
pred3=clf2.predict(sales_tst)
X11=X.copy()
dummies=pd.get_dummies(X11,columns=onehot1)
pd.Series(clf2.steps[1][1].feature_importances_,index=dummies.columns).plot(kind='barh',figsize=(7,8))
sb.distplot(pred3)
result=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result=result.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result=result.assign(Item_Outlet_Sales=pred3)
result.to_csv('result00v18.csv')
params_gradient={'n_estimators':randint(2,400),'max_features':truncnorm(a=0, b=1, loc=0.35, scale=0.1),'min_samples_split': uniform(0.01, 0.199)}
gradient=RandomizedSearchCV(GradientBoostingRegressor(),params,cv=5,return_train_score=True,scoring='neg_root_mean_squared_error')
onehot_gradclassifier=Pipeline([('trans',transformer2),('random',gradient)])
onehot_gradclassifier.fit(X,Y)
grad_data=pd.DataFrame(onehot_gradclassifier.steps[1][1].cv_results_)
grad_data.head(3)
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.plot(grad_data.loc[:,'param_max_features'],grad_data.loc[:,'mean_train_score'],label='train')
plt.plot(grad_data.loc[:,'param_max_features'],grad_data.loc[:,'mean_test_score'],label='cv')
plt.xlabel('max_features')
plt.subplot(132)
plt.plot(grad_data.loc[:,'param_n_estimators'],grad_data.loc[:,'mean_train_score'],label='train')
plt.plot(grad_data.loc[:,'param_n_estimators'],grad_data.loc[:,'mean_test_score'],label='cv')
plt.xlabel('n_estimators')
plt.subplot(133)
plt.plot(grad_data.loc[:,'param_min_samples_split'],grad_data.loc[:,'mean_train_score'],label='train')
plt.plot(grad_data.loc[:,'param_min_samples_split'],grad_data.loc[:,'mean_test_score'],label='cv')
plt.xlabel('min_samples_split')
plt.legend()
onehot_gradclassifier.steps[1][1].best_params_
grad_class=Pipeline([('trans',transformer2),('gradient',GradientBoostingRegressor(max_features=0.451,min_samples_split=0.177,n_estimators=137))])
clf3=grad_class.fit(X_tr,y_tr)
print(mean_squared_error(clf3.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf3.predict(X_cv),y_cv,squared=False))
pred4=clf3.predict(sales_tst)
pd.Series(clf3.steps[1][1].feature_importances_,index=dummies.columns).plot(kind='barh',figsize=(20,7))
sb.distplot(pred4)
result4=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result4=result4.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result4=result4.assign(Item_Outlet_Sales=pred4)
result4.to_csv('result00v14.csv')
from sklearn.neural_network import MLPRegressor
mlpreg=MLPRegressor()
param_mlp={'hidden_layer_sizes':(randint(10,300)),'activation':['tanh','relu'],'learning_rate':['constant','adaptive']}
random_mlp=RandomizedSearchCV(mlpreg,param_mlp,cv=5,scoring='neg_root_mean_squared_error',return_train_score=True)
mlp_pipeline=Pipeline([('trans',transformer2),('mlp',random_mlp)])
mlp_pipeline.fit(X,Y)
mlp_data=pd.DataFrame(mlp_pipeline.steps[1][1].cv_results_)
mlp_data.head(3)
mlp_pipeline.steps[1][1].best_params_
mlp_class=Pipeline([('trans',transformer2),('mlp',MLPRegressor(hidden_layer_sizes=(500,250,100),learning_rate='constant',activation='relu'))])
clf5=mlp_class.fit(X_tr,y_tr)
print(mean_squared_error(clf5.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf5.predict(X_cv),y_cv,squared=False))
pred7=clf5.predict(sales_tst)
sb.distplot(pred7)
from sklearn.ensemble import VotingRegressor
rand_forest=RandomForestRegressor(max_features=0.48,min_samples_split=0.0300,n_estimators=500)
grad_boost=GradientBoostingRegressor(max_features=0.451,min_samples_split=0.177,n_estimators=137)
mlp_regres=MLPRegressor(hidden_layer_sizes=(174,140,100),learning_rate='adaptive',activation='relu')
voting=VotingRegressor(estimators=[('rand',rand_forest),('grad',grad_boost),('mlp',mlp_regres)])
vot_class=Pipeline([('trans',transformer2),('voting',voting)])
clf4=vot_class.fit(X_tr,y_tr)
print(mean_squared_error(clf4.predict(X_tr),y_tr,squared=False))
print(mean_squared_error(clf4.predict(X_cv),y_cv,squared=False))
pred5=clf4.predict(sales_tst)
sb.distplot(pred5)
result5=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result5=result5.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result5=result5.assign(Item_Outlet_Sales=pred5)
result5.to_csv('result00v30.csv')

X2=X.copy()
X2_test=sales_tst.copy()
X2=pd.get_dummies(X2,columns=['Item_Fat_Content','Outlet_Identifier','Outlet_Location_Type','Outlet_Type','New_itemtype'])
X2_test=pd.get_dummies(X2_test,columns=['Item_Fat_Content','Outlet_Identifier','Outlet_Location_Type','Outlet_Type','New_itemtype'])
ordin=OrdinalEncoder()
ordin.fit(X2['Outlet_Establishment_Year'].values.reshape(-1,1))
X2.loc[:,'Outlet_Establishment_Year']=ordin.transform(X2['Outlet_Establishment_Year'].values.reshape(-1,1))
X2_test.loc[:,'Outlet_Establishment_Year']=ordin.transform(X2_test['Outlet_Establishment_Year'].values.reshape(-1,1))
stand=StandardScaler()
stand.fit(X2['Item_MRP'].values.reshape(-1,1))
X2.loc[:,'Item_MRP']=stand.transform(X2['Item_MRP'].values.reshape(-1,1))
X2_test.loc[:,'Item_MRP']=stand.transform(X2_test['Item_MRP'].values.reshape(-1,1))
stand.fit(X2['Item_Visibility'].values.reshape(-1,1))
X2.loc[:,'Item_Visibility']=stand.transform(X2['Item_Visibility'].values.reshape(-1,1))
X2_test.loc[:,'Item_Visibility']=stand.transform(X2_test['Item_Visibility'].values.reshape(-1,1))
stand.fit(X2['Item_Weight'].values.reshape(-1,1))
X2.loc[:,'Item_Weight']=stand.transform(X2['Item_Weight'].values.reshape(-1,1))
X2_test.loc[:,'Item_Weight']=stand.transform(X2_test['Item_Weight'].values.reshape(-1,1))
ordin.fit(X2['Outlet_Size'].values.reshape(-1,1))
X2.loc[:,'Outlet_Size']=ordin.transform(X2['Outlet_Size'].values.reshape(-1,1))
X2_test.loc[:,'Outlet_Size']=ordin.transform(X2_test['Outlet_Size'].values.reshape(-1,1))
model = Sequential()
model.add(Dense(128, input_shape=(31,), activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(16,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mean_squared_error',optimizer='adam',metrics=tf.keras.metrics.RootMeanSquaredError())
from tensorflow.keras.callbacks import EarlyStopping
hist=model.fit(X2,Y,epochs=100,batch_size=64,validation_split=0.15,callbacks=[EarlyStopping(patience=5,restore_best_weights=True)])
plt.plot(hist.history['root_mean_squared_error'],label='Train')
plt.plot(hist.history['val_root_mean_squared_error'],label='Val')
plt.legend()
pred6=model.predict(X2_test)
sb.distplot(pred2)
result5=pd.DataFrame(test.loc[:,'Item_Identifier'],columns=['Item_Identifier'])
result5=result5.assign(Outlet_Identifier=test.loc[:,'Outlet_Identifier'])
result5=result5.assign(Item_Outlet_Sales=pred5)
result5.to_csv('result00v20.csv')
result.to_csv('result88.csv')