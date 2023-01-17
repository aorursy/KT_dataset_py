import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import numpy as np

import matplotlib

matplotlib.use('PS')

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import HTML

from scipy import stats

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter("ignore", UserWarning)

                      

%matplotlib inline
dados=pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

print('\n Data info \n',dados.info())

print('\n Data Shape\n ',dados.shape)

print('\n Data TypeS\n ',dados.dtypes)

print('\n Number of missing data\n',dados.isna().sum())

HTML(pd.DataFrame(dados.columns).to_html())
dados.head()
plt.style.use('seaborn-whitegrid')

dados.hist(bins=30,figsize=(20,20))

plt.show()
discrete=['city','rooms','bathroom','parking spaces','animal','furniture','floor']

fig,axis=plt.subplots(4,2,figsize=(20,30))

k=0

for discrete in discrete:

    plt.style.use('seaborn-whitegrid')

    data=dados[discrete].value_counts().to_dict()

    axis=axis.flatten()

    sns.countplot(x=dados[discrete],hue=dados[discrete],ax=axis[k])

    k=k+1

plt.show()
def categorize(columns):

    numerical,nominal=[],[]

    for i in columns:

        if dados[i].dtype==object:

            nominal.append(i)

        else:

            numerical.append(i)

    print('Nominal features {}|:'.format(nominal))

    print('Numerical Features {}:'.format(numerical))

    return nominal,numerical
nominal,numerical=categorize(dados.columns)

dados_num=dados.copy()

dados_num=dados_num.drop(nominal,axis=1)

dados_num.describe()
for i in dados_num:

    r=dados_num[i].describe()

    HTML(pd.DataFrame(r.T).to_html())

    totais=dados_num[i].size

    plt.style.use('seaborn-whitegrid')

    print('\nSkew:',dados_num[i].name,' \n',dados_num[i].skew())

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,4))

    g1=sns.distplot(dados_num[i],ax=ax1,fit=stats.norm,bins=50)

    g2=sns.boxplot(dados_num[i],ax=ax2)

    plt.show()
sns.pairplot(dados_num)

plt.style.use('seaborn-whitegrid')
correla=dados_num.corr()

plt.figure(figsize =(8,8))

sns.heatmap(correla,annot =True)

plt.show()

print(correla['rent amount (R$)'].sort_values(ascending=False))
for i in nominal:

    k=0

    fig,axis=plt.subplots(2,2,figsize=(20,25))

    for j in nominal:

        plt.style.use('seaborn-whitegrid')

        axis=axis.flatten()

        imag=sns.countplot(x=dados[i],hue=dados[j],data=dados,ax=axis[k])

        k=k+1

    plt.show()
nominal2=nominal.copy()

nominal2.remove('floor')

print(nominal2)
for i in nominal2:

    k=0

    fig,axis=plt.subplots(5,2,figsize=(20,35))

    fig2,axis2=plt.subplots(5,2,figsize=(20,35))

    for j in numerical:

        axis=axis.flatten()

        axis2=axis2.flatten()

        sns.countplot(x=dados[j],hue=dados[i],data=dados,ax=axis[k])

        sns.violinplot(x=dados[i],y=dados[j],hue=dados[i],data=dados,ax=axis2[k])

        plt.style.use('seaborn-whitegrid')

        k=k+1

    plt.show()
dados_enc=dados.copy()

lb_make=LabelEncoder()

for i in nominal:

    dados_enc[i]=lb_make.fit_transform(dados_enc[i])

dados_enc.head()
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(dados_enc.drop(['rent amount (R$)'],axis=1),dados_enc['rent amount (R$)'],

                                               train_size=0.7,random_state=42)

name_list=[]

model_list=[]

MAE_list=[]
from sklearn.linear_model import LinearRegression

model_lr=LinearRegression(n_jobs=-1)

model_lr.fit(x_train,y_train)

MAE=(mean_absolute_error(y_test,model_lr.predict(x_test)))

name='Linear Reg'

name_list.append(name);model_list.append(model_lr);MAE_list.append(MAE)
from sklearn.ensemble import RandomForestRegressor 

model_rf =RandomForestRegressor(random_state=42)

model_rf.fit(x_train,y_train)

MAE=mean_absolute_error(y_test,model_rf.predict(x_test))

name='Random Forest Reg'

name_list.append(name);model_list.append(model_rf);MAE_list.append(MAE)
from xgboost import XGBRegressor 

model_xgbr =XGBRegressor(objective="reg:squarederror")

model_xgbr.fit(x_train,y_train)

MAE=(mean_absolute_error(y_test,model_xgbr.predict(x_test)))

name='XGB Reg'

name_list.append(name);model_list.append(model_xgbr);MAE_list.append(MAE)
from sklearn.neighbors import KNeighborsRegressor

model_knn =KNeighborsRegressor()

model_knn.fit(x_train,y_train)

MAE=(mean_absolute_error(y_test,model_knn.predict(x_test)))

name='KNN Reg'

name_list.append(name);model_list.append(model_knn);MAE_list.append(MAE)
from sklearn.linear_model import Ridge 

model_rid=Ridge(random_state=42)

model_rid.fit(x_train,y_train)

MAE=(mean_absolute_error(y_test,model_rid.predict(x_test)))

name='Ridge Reg'

name_list.append(name);model_list.append(model_rid);MAE_list.append(MAE)
from sklearn.linear_model import Lasso 

model_las =Lasso(random_state=42,max_iter=30000)

model_las.fit(x_train,y_train)

MAE=(mean_absolute_error(y_test,model_las.predict(x_test)))

name='Lasso Reg'

name_list.append(name);model_list.append(model_las);MAE_list.append(MAE)
fig,axis=plt.subplots(3,2,figsize=(20,25))

for i in range(len(name_list)):

    print('Name= {}  MAE = {} \n'.format(name_list[i],MAE_list[i]))

    axis=axis.flatten()

    ax1=sns.distplot(y_test,hist=False,kde =True,color ="r",label ="Actual Value",ax=axis[i])

    sns.distplot(model_list[i].predict(x_test),color ="b",hist = False,kde =True, label = "Preicted Value",ax=axis[i]).set_title('Name= {}  MAE = {} \n'.format(name_list[i],MAE_list[i]))

    plt.style.use('seaborn-whitegrid')

plt.show()
from skopt import dummy_minimize,gp_minimize

from skopt.plots import plot_convergence
name_list_mini=[]

MAE_list_mini=[]

name_result_mini=[]
def train_model_lr(param):

    normalize=param[0]

    

    mdl=LinearRegression(normalize=normalize,n_jobs=-1)

    mdl.fit(x_train,y_train)

    return mean_absolute_error(y_test,mdl.predict(x_test))



space=[(True,False)]

result_lr=gp_minimize(train_model_lr,space,random_state=42,n_calls=30,n_random_starts=10)



print('Mean absolute error: ',result_lr.fun,'Best parameters:normalize=%s'%(result_lr.x[0]))



name_list_mini.append('Linear Reg');name_result_mini.append(result_lr);MAE_list_mini.append(result_lr.fun)
def train_model_rf(param):

    n_estimators,criterion,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_features,max_leaf_nodes,oob_score,warm_start=(param[0],

                                        param[1],param[2],param[3],param[4],param[5],param[6],param[7],param[8],param[9])



    mdl=RandomForestRegressor(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,oob_score=oob_score,warm_start=warm_start,

                              min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,max_leaf_nodes=max_leaf_nodes,random_state=42)

    mdl.fit(x_train,y_train)

    return mean_absolute_error(y_test,mdl.predict(x_test))



space=[(10,1000),('mse','mae'),(1,10),(2,10),(1,10),(0,0.5),('auto','sqrt','log2'),(2,10),(bool,False),(bool,False)]

result_rf=gp_minimize(train_model_rf,space,random_state=42,n_calls=30,n_random_starts=10)



print('Mean absolute error: ',result_rf.fun,

      'Best parameters:n_estimators= %6f,criterion=%s,max_depth=%6f,min_samples_split=%6f,min_samples_leaf=%6f,min_weight_fraction_leaf=%6f,max_features=%s,max_leaf_nodes=%6f,oob_score=%s,warm_start=%s'%

                                                (result_rf.x[0],result_rf.x[1],result_rf.x[2],result_rf.x[3],result_rf.x[4],result_rf.x[5],result_rf.x[6],result_rf.x[7],result_rf.x[8],result_rf.x[9]))



name_list_mini.append('Random Forest Reg');name_result_mini.append(result_rf);MAE_list_mini.append(result_rf.fun)
def train_model_xgbr(param):

    booster,reg_lambda,alpha,update,feature_selector,rate_drop,normalize_type=param[0],param[1],param[2],param[3],param[4],param[5],param[6]



    mdl_xgbr=XGBRegressor(objective="reg:squarederror",booster=booster,reg_lambda=reg_lambda,alpha=alpha,update=update,feature_selector=feature_selector,rate_drop=rate_drop,normalize_type=normalize_type)

    mdl_xgbr.fit(x_train,y_train)

    return mean_absolute_error(y_test,mdl_xgbr.predict(x_test))



space=[('gblinear','gbtree','dart'),(0,5),(0,5),('shotgun','coord_descent'),('cyclic','shuffle'),(0,1),('tree','forest')]

result_xgbr=gp_minimize(train_model_xgbr,space,random_state=42,n_calls=30,n_random_starts=10)



print('Mean absolute error: ',result_xgbr.fun,

      'Best parameters:booster=%s,reg_lambda=%6f,alpha=%6f,update=%s,feature_selector=%s,rate_drop=%6f,normalize_type=%s'%

                          (result_xgbr.x[0],result_xgbr.x[1],result_xgbr.x[2],result_xgbr.x[3],result_xgbr.x[4],result_xgbr.x[5],result_xgbr.x[6]))



name_list_mini.append('XGBR Reg');name_result_mini.append(result_xgbr);MAE_list_mini.append(result_xgbr.fun)
def train_model_knn(param):

    n_neighbors,weights,algorithm,leaf_size,p=param[0],param[1],param[2],param[3],param[4] 



    mdl=KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm,leaf_size=leaf_size,p=p)

    mdl.fit(x_train,y_train)

    return mean_absolute_error(y_test,mdl.predict(x_test))



space=[(1,10),('uniform','distance'),('auto','ball_tree','kd_tree','brute'),(15,50),(1,2)]

result_knn=gp_minimize(train_model_knn,space,random_state=42,n_calls=30,n_random_starts=10)



print('Mean absolute error: ',result_knn.fun,'.Best parameters:n_estimators=%6f, weights=%s, algorithm=%s, leaf_size=%6f, p=%6f '%

                                                                          (result_knn.x[0],result_knn.x[1],result_knn.x[2],result_knn.x[3],result_knn.x[4]))



name_list_mini.append('KNN Reg');name_result_mini.append(result_knn);MAE_list_mini.append(result_knn.fun)
def train_model_ridge_reg(param):

    alpha,normalize,tol,solver,max_iter=param[0],param[1],param[2],param[3],param[4]



    mdl=Ridge(alpha=alpha,normalize=normalize,tol=tol,solver=solver,max_iter=max_iter,random_state=42)

    mdl.fit(x_train,y_train)

    return mean_absolute_error(y_test,mdl.predict(x_test))



space=[(0.1,10),(bool,False),(1e-9,1),('auto','svd','cholesky','lsqr','sparse_cg'), (500,5000)]

result_rr=gp_minimize(train_model_ridge_reg,space,random_state=42,n_calls=30,n_random_starts=10)



print('Mean absolute error: ',result_rr.fun,'.Best parameters: alpha=%6f, normalize=%s, tol=%6f, solver =%s,max_iter=%6f'%

                                                                          (result_rr.x[0],result_rr.x[1],result_rr.x[2],result_rr.x[3],result_rr.x[4]))



name_list_mini.append('Ridge Reg');name_result_mini.append(result_rr);MAE_list_mini.append(result_rr.fun)
def train_model_lasso(param):

    alpha,normalize,tol,warm_start,positive,selection,precompute=param[0],param[1],param[2],param[3],param[4],param[5],param[6]

    

    mdl=Lasso(alpha=alpha,normalize=normalize,max_iter=30000,tol=tol,warm_start=warm_start,positive=positive,selection=selection,random_state=42,precompute=precompute)

    mdl.fit(x_train,y_train)

    return mean_absolute_error(y_test,mdl.predict(x_test))



space=[(0.1,10),(bool,False),(1e-9,1),(bool,False),(bool,False,True),('cyclic','random'),(False,True)]

result_lasso=gp_minimize(train_model_lasso,space,random_state=42,n_calls=30,n_random_starts=10)



print('Mean absolute error: ',result_lasso.fun,'.Best parameters:alpha= %6f, normalize=%s, tol=%9f, warm_start=%s, positive=%6s, selection=%s, precompute=%s'%

                                                                          (result_lasso.x[0],result_lasso.x[1],result_lasso.x[2],result_lasso.x[3],result_lasso.x[4],result_lasso.x[5],result_lasso.x[6]))



name_list_mini.append('Lasso Reg');name_result_mini.append(result_lasso);MAE_list_mini.append(result_lasso.fun)
fig,axis=plt.subplots(3,2,figsize=(20,20))

for i in range(len(name_list)):

    print('{}  MAE = {} \n'.format(name_list_mini[i],MAE_list_mini[i]))

    axis=axis.flatten()

    plot_convergence(name_result_mini[i],ax=axis[i]).set_title('Convergence plot {}   MAE = {} \n'.format(name_list_mini[i],MAE_list_mini[i]))

    plt.style.use('seaborn-whitegrid')

plt.show()
from sklearn.feature_selection import SelectKBest,f_regression

from sklearn.ensemble import RandomForestRegressor



k_vs_scoremae=[]

for k in range(1,len(x_train.columns)+1,1):

    selector=SelectKBest(score_func=f_regression,k=k)

    xtrain2=selector.fit_transform(x_train,y_train)

    xval2=selector.transform(x_test)

    mdl=Ridge(alpha=6.213067, normalize=False, tol=0.007066, solver ='auto',max_iter=2861.000000,random_state=42)

    mdl.fit(xtrain2,y_train)

    p=mdl.predict(xval2)

    mask=selector.get_support()

    columns_selected=pd.Index.tolist((x_test.columns[mask]))

    scoremae=mean_absolute_error(y_test,p)

    print('k= {}     -  MAE = {}   -Selected columns = {} \n'.format(k,scoremae,columns_selected))

    k_vs_scoremae.append(scoremae)
fig,axis=plt.subplots(1,1,figsize=(10,8))

sns.lineplot(range(1,len(x_train.columns)+1,1),k_vs_scoremae,ax=axis)

plt.style.use('seaborn-whitegrid')

axis.set_title('MAE x variables')

plt.show()
print(pd.concat([pd.Series(selector.scores_,index=x_train.columns,name='F Regression'),

                  pd.Series(selector.pvalues_,index=x_train.columns,name= 'P values')],axis=1).sort_values(by='F Regression',ascending=False))
from sklearn.feature_selection import SelectFromModel

k_vs_score2=[]



for l in range(1,len(x_train.columns)+1,1):

    selector_model=Ridge(alpha=6.213067, normalize=False, tol=0.007066, solver ='auto',max_iter=2861.000000,random_state=42)

    selector=SelectFromModel(selector_model,max_features=l,threshold=-np.inf)

    x_train2=selector.fit_transform(x_train,y_train)

    x_test2=selector.transform(x_test)

    

    mdl=Ridge(alpha=6.213067, normalize=False, tol=0.007066, solver ='auto',max_iter=2861.000000,random_state=42)

    mdl.fit(x_train2,y_train)

    p=mdl.predict(x_test2)

    mask2=selector.get_support()

    columns_selected2=pd.Index.tolist((x_test.columns[mask2]))

    scoremae2=mean_absolute_error(y_test,p)

    k_vs_score2.append(scoremae2)

    print('l= {}     -MAE = {}   -Selected columns = {} \n'.format(l,scoremae2,columns_selected2))

    k_vs_scoremae.append(scoremae)
fig,axis=plt.subplots(1,1,figsize=(10,8))

sns.lineplot(range(1,len(x_train.columns)+1,1),k_vs_score2,ax=axis)

plt.style.use('seaborn-whitegrid')

axis.set_title('MAE x variables')

plt.show()
pd.Series(k_vs_score2[3:],index=range(3,x_train.shape[1],)).plot(figsize=(10,7))
score_past=500



k_vs_score3=[]

for j in range(1,len(x_train.columns)+1,1):

    for seed in range(1000):

        selected=np.random.choice(x_train.columns,j,replace=False)



        x_train2=x_train[selected]

        x_test2=x_test[selected]



        mdl=Ridge(alpha=6.213067, normalize=False, tol=0.007066, solver ='auto',max_iter=2861.000000,random_state=42)

        mdl.fit(x_train2,y_train)

        p=mdl.predict(x_test2)

        score=mean_absolute_error(y_test,p)

        k_vs_score3.append(score)

        if score<score_past:

            print('Number of features = {} seed = {} - MAE = {} - Features ={} \n'.format(j,seed,score,selected))

            score_past=score
x_train3,x_test3,y_train3,y_test3=train_test_split(dados_enc[['total (R$)','hoa (R$)','fire insurance (R$)','property tax (R$)','area',

 'furniture'] ],dados_enc['rent amount (R$)'],train_size=0.7,random_state=42)
model_lr3=Ridge(alpha=6.213067, normalize=False, tol=0.007066, solver ='auto',max_iter=2861.000000,random_state=42)

model_lr3.fit(x_train3,y_train3)

MAE=(mean_absolute_error(y_test3,model_lr3.predict(x_test3)))

print(MAE)