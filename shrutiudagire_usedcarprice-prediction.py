import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("../input/usedcar-price-prediction/Used car-Train.csv")
df_test=pd.read_csv("../input/usedcar-price-prediction/Used car-Test.csv")
df.shape,df_test.shape
df.tail()
df.isnull().sum()
df_test.isnull().sum()
df.drop('New_Price',axis=1,inplace=True)
df
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df_test.drop(['Unnamed: 0'],axis=1,inplace=True)
df_test.shape,df.shape
df['Fuel_Type'].value_counts()
df['Transmission'].value_counts()
df['Owner_Type'].value_counts()
df['Seats'].value_counts()
df[df['Seats']==0]
df[df['Mileage']=='0.0 kmpl']

df[(df['Power'].isnull())]
df.loc[(df['Power'].isnull()),'Power']='0 bhp'
df[(df['Power']=='0 bhp')]
df[(df['Power']=='0 bhp')]
df[(df['Mileage'].isnull())]
df[(df['Engine'].isnull())]
df.loc[(df['Engine'].isnull()),'Engine']='0 CC'
df[(df['Engine']=='0 CC')]
df.shape
#  SPlitting the data so as to convert it to float
list1=['Mileage','Engine','Power']
for j in list1:
    list2=[]
    for i in df[j]:
        i=i.split(" ")[0]
        if(i=='null' or i=='0.0'):
            i='0'
        list2.append(i)
    df[j]=list2 

df[(df['Mileage'].isnull())]
df[(df['Engine']=='0') | (df['Mileage']=='0')]
df.info()
df_test.info()
# Lets convert Mileage,Engine and Power  to float64
for i in list1:      
    df[i]=df[i].astype(float)

# Combining test and train to df_combine an dthen performing null value imputation
df_combine=pd.concat([df,df_test],axis=0,sort=False)
df_combine.head()
df_combine=df_combine.reset_index()
df_combine.drop('index',axis=1,inplace=True)
df_combine
df_combine.info()
# Based on Engine we are imputing Mileage and Power wwith median value
values=dict(list(df_combine.groupby('Engine')))
for i in values:
    df2=pd.DataFrame()
    df2=values[i]
    df_combine.loc[(df_combine['Engine']==i) & (df_combine['Mileage']==0),'Mileage']=df2['Mileage'].median()


for i in values:
    df2=pd.DataFrame()
    df2=values[i]
    df_combine.loc[(df_combine['Engine']==i) & (df_combine['Power']==0.0),'Power']=df2['Power'].median()
# We have missing values for  power and Mileage yet and we have to impute for Engine as well
df_combine[(df_combine['Power']==0.0)]
df_combine[(df_combine['Mileage']==0.0)]
df_combine[(df_combine['Engine']==0)]
# Manually imputing few values

df_combine.loc[(df_combine['Name']=='Mahindra Jeep MM 540 DP')  & (df_combine['Power']==0.0),'Power']=62
df_combine.loc[(df_combine['Name']=='Skoda Superb 3.6 V6 FSI')  & (df_combine['Mileage']==0.0),'Mileage']=7
df_combine.loc[(df_combine['Name']=='Porsche Cayman 2009-2012 S tiptronic')  & (df_combine['Power']==0.0),'Power']=325
df_combine.loc[(df_combine['Name']=='Maruti 1000 AC')  & (df_combine['Power']==0.0),'Power']=37
df_combine.loc[(df_combine['Name']=='Audi A4 3.2 FSI Tiptronic Quattro')  & (df_combine['Power']==0.0),'Power']=255
df_combine.loc[(df_combine['Name']=='Mahindra Jeep MM 540 DP')  & (df_combine['Power']==0.0),'Power']=62
df_combine.loc[(df_combine['Name']=='Mahindra Jeep MM 540 DP')  & (df_combine['Mileage']==0.0),'Mileage']=18
df_combine.loc[(df_combine['Name']=='Porsche Cayman 2009-2012 S')  & (df_combine['Power']==0.0),'Power']=315
df_combine.loc[(df_combine['Name']=='Fiat Siena 1.2 ELX')  & (df_combine['Power']==0.0),'Power']=57
df_combine.loc[(df_combine['Name']=='Nissan Teana 230jM')  & (df_combine['Power']==0.0),'Power']=179.5
df_combine.loc[(df_combine['Name']=='Fiat Petra 1.2 EL')  & (df_combine['Power']==0.0),'Power']=57
df_combine.loc[(df_combine['Name']=='Skoda Superb 3.6 V6 FSI')  & (df_combine['Mileage']==0.0),'Mileage']=7
df_combine.loc[(df_combine['Name']=='Maruti Swift 1.3 VXi')  & (df_combine['Engine']==0.0),'Engine']=1248
df_combine.loc[(df_combine['Name']=='Maruti Swift 1.3 VXi')  & (df_combine['Power']==0.0),'Power']=74


# df_combine.loc[(df_combine['Name']=='Audi A4 3.2 FSI Tiptronic Quattro')  & (df_combine['Seats']==0.0),'Seats']="5.0"
df_combine[(df_combine['Engine']==0) | (df_combine['Mileage']==0) | (df_combine['Power']==0)]
# df_combine.loc[df_combine['Mileage']==0,'Mileage']=df_combine['Mileage'].median()
# df_combine.loc[df_combine['Power']==0,'Power']=df_combine['Power'].median()
df_combine.info()
# Imputing for Seats
df_combine.loc[(df_combine['Seats']==0),'Seats']=df_combine['Seats'].mode()
df_combine[df_combine['Seats']==0]
df=df_combine.loc[0:6018,:].copy(deep=True)
df_test=df_combine.loc[6019:].copy(deep=True)
sns.distplot(df['Price'])
from scipy.stats import boxcox
df['Price'],lambda_value=boxcox(df['Price'])
sns.distplot(df['Price'])
df['Price'].skew()
sns.boxplot(df['Price'])
df['Price'].describe()
Q1=1.137597
Q3=1.929845
IQR=Q3-Q1
v1=Q1-1.5*IQR
v2=Q3+1.5*IQR
df[df['Price']<=v1]
df[df['Price']>=v2]
cat=["Transmission","Fuel_Type",'Owner_Type','Seats']
for i in cat:
    sns.boxplot(df[i],df['Price'])
    plt.show()
# We have to predict the price of the car based on following features Year,Kilometers_Driven,Fuel_Type,Transmission,Owner_Type,Mileage,Engine,Power,Seats rather than name and Location
# Hence Droping name and Location column.
df_combine.drop(['Name','Location'],axis=1,inplace=True)
df.drop(['Name','Location'],axis=1,inplace=True)
df_test.drop(['Name','Location'],axis=1,inplace=True)

df.head()
df_test.head()
df['Seats']=df['Seats'].astype(object)
df_test['Seats']=df_test['Seats'].astype(object)
df_combine['Seats']=df_combine['Seats'].astype(object)

#df_original 'y' variable has no transformation done
df_original=df_combine.loc[0:6018,:]
df_test_original=df_combine.loc[6019:]
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.xticks(np.arange(1998,2020))
sns.scatterplot(df['Year'],df['Price'],hue=df['Owner_Type'])
plt.figure(figsize=(20,10))
plt.xticks(np.arange(1998,2020))
plt.yticks(np.arange(0,180,10))
sns.boxplot(df['Year'],df['Price'],hue=df['Owner_Type'])
plt.figure(figsize=(10,5))
sns.boxplot(df['Owner_Type'],df['Price'],hue=df['Fuel_Type'])
plt.figure(figsize=(10,5))
sns.boxplot(df['Price'],df['Transmission'])
plt.figure(figsize=(10,5))
plt.xticks(rotation=90)
sns.scatterplot(df['Engine'],df['Price'],hue=df['Transmission'])
km1=df_original[df_original['Transmission']=='Manual']
plt.figure(figsize=(10,5))
plt.xticks(rotation=90)
plt.yticks(np.arange(0,170,5))
plt.xticks(np.arange(0,600,10))
sns.scatterplot(km1['Power'],km1['Price'])
km2=df_original[df_original['Transmission']=='Automatic']
# km2[km2['Kilometers_Driven']>6000000]
km2.drop(index=60,inplace=True)
plt.figure(figsize=(20,10))
plt.xticks(rotation=90)
plt.yticks(np.arange(0,170,5))
plt.xticks(np.arange(0,700,10))

sns.scatterplot(km2['Power'],km2['Price'])
df_combine=pd.concat([df,df_test],axis=0,sort=False)
df_dummies=pd.get_dummies(df_combine[["Transmission","Fuel_Type",'Owner_Type','Seats']],drop_first=True)
df_final=pd.concat([df_dummies,df_combine],axis=1)
df_final.drop(["Transmission","Fuel_Type",'Owner_Type','Seats'],axis=1,inplace=True)
df_final.shape

df=df_final.loc[0:6018,:].copy(deep=True)
df_test=df_final.loc[6019:].copy(deep=True)
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
df.corr()['Price']
from sklearn.preprocessing import StandardScaler
X=df.drop('Price',axis=1)
y=df['Price']
std=StandardScaler()
df1=std.fit_transform(X)
df_multi_scaled=pd.DataFrame(df1,columns=X.columns)
df_multi_scaled.head()
df_test.drop('Price',axis=1,inplace=True)
df2=std.transform(df_test)
df_multi_scaled_test=pd.DataFrame(df2,columns=df_test.columns)
df_multi_scaled_test.head()
df.shape,df_multi_scaled.shape,df_test.shape,df_multi_scaled_test.shape
import warnings 
warnings.filterwarnings('ignore')
import statsmodels.api as sm

X_constant = sm.add_constant(df_multi_scaled)
lin_reg = sm.OLS(y,X_constant).fit()
lin_reg.summary()
cols=list(df_multi_scaled.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = df_multi_scaled[cols]
    X_constant = sm.add_constant(X_1)
    model = sm.OLS(y,X_constant).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
df_multi_scaled=df_multi_scaled[selected_features_BE]
df_multi_scaled_test=df_multi_scaled_test[selected_features_BE]
df_multi_scaled.shape,df_multi_scaled_test.shape
df_combine=pd.concat([df_multi_scaled,df_multi_scaled_test],axis=0)
df_combine
from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(df_combine.values, j) for j in range(1, df_combine.shape[1])]
def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):
        a = np.argmax(vif)
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
    return(output)
Xnew=calculate_vif(df_combine)
Xnew.head()
from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(Xnew.values, j) for j in range(1, Xnew.shape[1])]
Xnew=Xnew.reset_index()
Xnew.drop('index',axis=1,inplace=True)
Xnew
df=Xnew.loc[0:6018,:].copy(deep=True)
df_test=Xnew.loc[6019:,:].copy(deep=True)
df.shape,df_test.shape
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.linear_model import Ridge,ElasticNet,Lasso
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor,VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
ridge=Ridge(max_iter=100,random_state=0)
lasso=Lasso(max_iter=100,random_state=0)

variance=[]
means=[]
for n in np.arange(1,500,1):
    ridge=Ridge(alpha=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=model_selection.cross_val_score(ridge,df,y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means.append(np.mean(rmse))
    variance.append(np.std(rmse,ddof=1))
x_axis=np.arange(1,500,1)
plt.plot(x_axis,variance) 

np.argmin(variance),variance[np.argmin(variance)],means[np.argmin(variance)]
np.argmin(means),variance[np.argmin(means)],means[np.argmin(means)]
from sklearn.linear_model import ElasticNetCV, ElasticNet
cv_model = ElasticNetCV(l1_ratio=np.arange(1,2), fit_intercept=True, max_iter=100,n_jobs=-1, random_state=0)
cv_model.fit(df_multi_scaled,y)
print('Optimal alpha: %.8f'%cv_model.alpha_)
print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)
print('Number of iterations %d'%cv_model.n_iter_)
# 3)tunning k value
knn=KNeighborsRegressor()
knn_params={'n_neighbors':np.arange(1,100),'weights':['uniform','distance']}
randomsearch=RandomizedSearchCV(knn,knn_params,cv=3,scoring='neg_mean_squared_error',random_state=0)
randomsearch.fit(df_multi_scaled,y)
randomsearch.best_params_
means_knn=[]
variance_knn=[]
for n in np.arange(1,50):
    KNN=KNeighborsRegressor(weights='distance',n_neighbors=n)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    value=model_selection.cross_val_score(KNN,df_multi_scaled,y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(value))
    variance_knn.append(np.std(rmse,ddof=1))
    means_knn.append(np.mean(rmse))
    
x_axis=np.arange(len(means_knn))
plt.plot(x_axis,means_knn)
np.argmin(means_knn),means_knn[np.argmin(means_knn)],variance_knn[np.argmin(means_knn)]
# Lets perform Random Forest

variance_rf=[]
means_rf=[]
for n in np.arange(1,150):
    RF=RandomForestRegressor(criterion='mse',n_estimators=n,random_state=0)
    kfold = model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    scores=cross_val_score(RF,df_multi_scaled,y,cv=kfold,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_rf.append(np.mean(rmse))
    variance_rf.append(np.std(rmse,ddof=1))
    

x_axis=np.arange(len(variance_rf))
plt.plot(x_axis,variance_rf) 
print(np.argmin(variance_rf),variance_rf[np.argmin(variance_rf)],means_rf[np.argmin(variance_rf)])
print(np.argmin(means_rf),variance_rf[np.argmin(means_rf)],means_rf[np.argmin(means_rf)])
# Bagging of LR
LR=LinearRegression()
means_Bag_LR=[]
variance_Bag_LR=[]
for n in np.arange(1,40):
    Bag=BaggingRegressor(base_estimator=LR,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,df_multi_scaled ,y,cv=3,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_Bag_LR.append(np.mean(rmse))
    variance_Bag_LR.append(np.std(rmse,ddof=1))
x_axis=np.arange(len(variance_Bag_LR))
plt.plot(x_axis,variance_Bag_LR)
np.argmin(variance_Bag_LR),variance_Bag_LR[np.argmin(variance_Bag_LR)],means_Bag_LR[np.argmin(variance_Bag_LR)]
np.argmin(means_Bag_LR),variance_Bag_LR[np.argmin(means_Bag_LR)],means_Bag_LR[np.argmin(means_Bag_LR)]
knn_grid=KNeighborsRegressor(n_neighbors=3,weights='distance')
knn_cust=KNeighborsRegressor(n_neighbors=5,weights='distance')
means_Bag_KNN=[]
variance_Bag_KNN=[]
for n in np.arange(1,50):
    Bag=BaggingRegressor(base_estimator=knn_grid,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,df_multi_scaled ,y,cv=3,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_Bag_KNN.append(np.mean(rmse))
    variance_Bag_KNN.append(np.std(rmse,ddof=1))
x_axis=np.arange(len(variance_Bag_KNN))
plt.plot(x_axis,variance_Bag_KNN)
knn_cust=KNeighborsRegressor(n_neighbors=5,weights='distance')

means_Bag_KNNcust=[]
variance_Bag_KNNcust=[]
for n in np.arange(1,50):
    Bag=BaggingRegressor(base_estimator=knn_cust,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,df_multi_scaled ,y,cv=3,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_Bag_KNNcust.append(np.mean(rmse))
    variance_Bag_KNNcust.append(np.std(rmse,ddof=1))
x_axis=np.arange(len(variance_Bag_KNNcust))
plt.plot(x_axis,variance_Bag_KNNcust)
np.argmin(means_Bag_KNNcust),variance_Bag_KNNcust[np.argmin(means_Bag_KNNcust)],means_Bag_KNNcust[np.argmin(means_Bag_KNNcust)]
DT=DecisionTreeRegressor()
means_Bag_DT=[]
variance_Bag_DT=[]
for n in np.arange(1,50):
    Bag=BaggingRegressor(base_estimator=DT,n_estimators=n,random_state=0)
    scores=cross_val_score(Bag,df_multi_scaled ,y,cv=3,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    means_Bag_DT.append(np.mean(rmse))
    variance_Bag_DT.append(np.std(rmse,ddof=1))
x_axis=np.arange(len(variance_Bag_DT))
plt.plot(x_axis,variance_Bag_DT)
np.argmin(variance_Bag_DT),variance_Bag_DT[np.argmin(variance_Bag_DT)],means_Bag_DT[np.argmin(variance_Bag_DT)]
np.argmin(means_Bag_DT),variance_Bag_DT[np.argmin(means_Bag_DT)],means_Bag_DT[np.argmin(means_Bag_DT)]
rmse_ada_DT=[]
variance_ada_DT=[]
for n in np.arange(1,50):
    AB=AdaBoostRegressor(n_estimators=n,random_state=0)
    scores=cross_val_score(AB,df_multi_scaled,y,cv=3,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    rmse_ada_DT.append(np.mean(rmse))
    variance_ada_DT.append((np.std(rmse,ddof=1)))
x_axis=np.arange(len(rmse_ada_DT))
plt.plot(x_axis,rmse_ada_DT)
np.argmin(rmse_ada_DT),rmse_ada_DT[np.argmin(rmse_ada_DT)],variance_ada_DT[np.argmin(rmse_ada_DT)]
LR=LinearRegression()
rmse_ada_LR=[]
variance_ada_LR=[]
for n in np.arange(1,60):
    AB=AdaBoostRegressor(base_estimator=LR,n_estimators=n,random_state=0)
    scores=cross_val_score(AB,df_multi_scaled,y,cv=3,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    rmse_ada_LR.append(np.mean(rmse))
    variance_ada_LR.append((np.std(rmse,ddof=1)))
x_axis=np.arange(len(rmse_ada_LR))
plt.plot(x_axis,rmse_ada_LR)
LR=LinearRegression()
rmse_ada_LR=[]
variance_ada_LR=[]
for n in np.arange(1,60):
    AB=AdaBoostRegressor(base_estimator=LR,n_estimators=n,random_state=0)
    scores=cross_val_score(AB,df_multi_scaled,y,cv=3,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    rmse_ada_LR.append(np.mean(rmse))
    variance_ada_LR.append((np.std(rmse,ddof=1)))
x_axis=np.arange(len(rmse_ada_LR))
plt.plot(x_axis,rmse_ada_LR)
randomforest=RandomForestRegressor(n_estimators=114,random_state=0)
rmse_ada_RF=[]
variance_ada_RF=[]
for n in np.arange(1,60):
    AB=AdaBoostRegressor(base_estimator=randomforest,n_estimators=n,random_state=0)
    scores=cross_val_score(AB,df_multi_scaled,y,cv=3,scoring='neg_mean_squared_error')
    rmse=np.sqrt(np.abs(scores))
    rmse_ada_RF.append(np.mean(rmse))
    variance_ada_RF.append((np.std(rmse,ddof=1)))
x_axis=np.arange(len(rmse_ada_RF))
plt.plot(x_axis,rmse_ada_RF)
np.argmin(rmse_ada_RF),rmse_ada_RF[np.argmin(rmse_ada_RF)],variance_ada_RF[np.argmin(rmse_ada_RF)]
np.argmin(variance_ada_RF),rmse_ada_RF[np.argmin(variance_ada_RF)],variance_ada_RF[np.argmin(variance_ada_RF)]
LR=LinearRegression()
ridge=Ridge(alpha=368,random_state=0)
elasticnet = ElasticNet(l1_ratio= 1, alpha =0.00220166,  fit_intercept=True)
randomforest=RandomForestRegressor(n_estimators=114,random_state=0)
Bagged_DT=BaggingRegressor(base_estimator=DT,n_estimators=4,random_state=0)
knn_grid=KNeighborsRegressor(n_neighbors=3,weights='distance')
knn_cust=KNeighborsRegressor(n_neighbors=5,weights='distance')
randomfored_ada=AdaBoostRegressor(base_estimator=randomforest,n_estimators=32,random_state=0)
stacked = VotingRegressor(estimators = [('ADAboost randomFOrest',randomfored_ada),('KNN_grid', knn_grid), ('KNN_customized', knn_cust)])

models = []
models.append(('LinearRegression', LR))
models.append(('Ridge', ridge))
# models.append(('Lasso',lasso))
models.append(('ElasticNet',elasticnet))
models.append(('Random Forest',randomforest))
models.append(('Bagged DT',Bagged_DT))
models.append(('KNN_grid',knn_grid))
models.append(('KNN_customized',knn_cust))
models.append(('ADAboost randomFOrest',randomfored_ada))
models.append(('Stacked',stacked))
means=[]
rmse=[]
names=[]
variance=[]
df_result=pd.DataFrame()
for name,model in models:
    kfold=model_selection.KFold(shuffle=True,n_splits=3,random_state=0)
    cv_result=model_selection.cross_val_score(model,df_multi_scaled,y,cv=kfold,scoring='neg_mean_squared_error')
    value=np.sqrt(np.abs(cv_result))
    means.append(value)
    names.append(name)
    rmse.append(np.mean(value))
    variance.append(np.std((value),ddof=1))
df_result['Names']=names
df_result['RMSE']=rmse
df_result['Variance']=variance
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(means)
ax.set_xticklabels(names)
plt.show()

df_result
df_result.sort_values('RMSE',ascending=True)
df_result.sort_values(by='Variance',ascending=True)
stacked.fit(df_multi_scaled,y)
predicted_values=stacked.predict(df_multi_scaled_test)
from scipy.special import inv_boxcox
predicted=inv_boxcox(predicted_values,lambda_value)
predicted
