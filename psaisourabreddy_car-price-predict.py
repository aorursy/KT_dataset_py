import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('display.max_columns',None)

from sklearn.model_selection import train_test_split,cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as r
from scipy import stats

#loading the data set 
df=pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
df.head()
#before exploring the further, lets check if there are any null values in the data.

df.isnull().sum()[df.columns[df.isnull().sum()>0]]
#visualizing nulls
sns.heatmap(df.isnull(),cbar=False,yticklabels=False)
plt.show()
#no nulls in the dataset
#removing car id and symboling, Car name
df.drop(['car_ID','symboling'],1,inplace=True)
#We will proceed with analysing numerical and categorical columns saperately

df_numeric=df.select_dtypes(include=['int64','float64'])
df_catg=df.select_dtypes(include=object)
df_numeric.head()
#1:Deriving Avg mpg based on city and highway mpg 
def Avg_mpg(x):
    city=x[0]
    highway=x[1]
    return ( (city+highway)/2)
Avgmpg=df_numeric[['citympg','highwaympg']].apply(Avg_mpg,axis=1)
df_numeric.insert(len(df_numeric.columns)-1,'Avgmpg',Avgmpg)
#dropping the features
df_numeric.drop(['citympg','highwaympg'],1,inplace=True)
df_numeric.describe().T
#viewing the pair plot `
sns.pairplot(df_numeric,diag_kind='kde')
plt.show()
#heatmap
plt.figure(figsize=(20,10))
sns.heatmap(df_numeric.corr(),annot=True)
plt.show()
#removing the above said columns .
cols_to_remove=['wheelbase','carheight','stroke','compressionratio','peakrpm']
df_numeric.drop(cols_to_remove,1,inplace=True)
df_catg=pd.concat((df_catg,df.iloc[:,-1]),1)
df_catg.head()
#Extracting Car Maker
Car_Maker=df_catg['CarName'].apply(lambda x : x.split()[0])
df_catg.insert(1,'Car_Maker',Car_Maker)
df_catg.drop('CarName',1,inplace=True)
df_catg.Car_Maker.unique()
#We have the same Makers names repeated, lets combine them

df_catg['Car_Maker']=df_catg['Car_Maker'].replace('maxda','mazda')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('Nissan','nissan')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('Nissan','nissan')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('porcshce','porsche')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('toyouta','toyota')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('vokswagen','volkswagen')
df_catg['Car_Maker']=df_catg['Car_Maker'].replace('vw','volkswagen')

df_catg['Car_Maker']=df_catg['Car_Maker'].apply(lambda x : x.capitalize())
plt.figure(figsize=(10,5))
df_catg['Car_Maker'].value_counts().plot(kind='bar')
plt.xlabel('Car Brands')
plt.ylabel('Number of Units in Market')
plt.show()


plt.figure(figsize=(20,8))
sns.barplot(df_catg['Car_Maker'],df_catg['price'],ci=None,estimator=np.min)
plt.xlabel('Car Brands')
plt.ylabel('Avg Price')
plt.show()

def Car_Class(x):
    price=x
    if price>0 and price<10000:
        return 'Low Budget'
    elif price>=10000 and price<20000:
        return 'Medium Budget'
    else:
        return 'High Bidget'
    
Car_class=df_catg['price'].apply(Car_Class)
df_catg.insert(len(df_catg.columns)-1,'Car_Class',Car_class)
df_catg.drop('Car_Maker',1,inplace=True)
d={'Low Budget':0,'Medium Budget':1,'High Bidget':2}

df_catg['Car_Class']=df_catg['Car_Class'].map(d)

plt.figure(figsize=(20,20))
plt.subplot(521)
df_catg['fueltype'].value_counts().plot('bar')
plt.xlabel('Type of Fuel')
plt.ylabel('Number of Units')

plt.subplot(522)
sns.boxplot(df_catg['fueltype'],df_catg['price'])

#####################################################

plt.figure(figsize=(20,20))
plt.subplot(525)
df_catg['aspiration'].value_counts().plot('bar')
plt.xlabel('Type of Aspiration')
plt.ylabel('Number of Units')

plt.subplot(526)
sns.boxplot(df_catg['aspiration'],df_catg['price'])

######################################################

plt.figure(figsize=(20,20))
plt.subplot(529)
df_catg['doornumber'].value_counts().plot('bar')
plt.xlabel('Number of Doors')
plt.ylabel('Number of Units')

plt.subplot(5,2,10)
sns.boxplot(df_catg['doornumber'],df_catg['price'])
plt.show()

plt.figure(figsize=(20,20))
plt.subplot(321)
sns.boxplot(df_catg['fueltype'],df_numeric['horsepower'])

plt.subplot(322)
sns.boxplot(df_catg['fueltype'],df['compressionratio'])

##########################################################

plt.figure(figsize=(20,20))
plt.subplot(325)
sns.boxplot(x=df_catg['aspiration'],y=df_numeric['horsepower'])

plt.subplot(326)
sns.boxplot(df_catg['aspiration'],df_numeric['Avgmpg'])
plt.show()
plt.figure(figsize=(20,20))
plt.subplot(521)
df_catg['carbody'].value_counts().plot('bar')
plt.xlabel('Type of Car')
plt.ylabel('Number of Units')

plt.subplot(522)
sns.boxplot(df_catg['carbody'],df_catg['price'])
##################################################

plt.figure(figsize=(20,20))
plt.subplot(525)
df_catg['drivewheel'].value_counts().plot('bar')
plt.xlabel('Type of Drive')
plt.ylabel('Number of Units')

plt.subplot(526)
sns.boxplot(df_catg['drivewheel'],df_catg['price'],hue=df_catg['enginelocation'])
###################################################

plt.figure(figsize=(20,20))
plt.subplot(529)
df_catg['enginelocation'].value_counts().plot('bar')
plt.xlabel('Location of Engine')
plt.ylabel('Number of Units')

plt.subplot(5,2,10)
sns.boxplot(hue=df_catg['drivewheel'],y=df_numeric['Avgmpg'],x=df_catg['enginelocation'])
plt.show()
plt.figure(figsize=(20,5))
plt.subplot(131)
sns.boxplot(hue=df_catg['drivewheel'],y=df_numeric['enginesize'],x=df_catg['enginelocation'])

plt.subplot(133)
sns.boxplot(df_catg['aspiration'],df_numeric['enginesize'])
plt.show()
##################################################################

plt.figure(figsize=(14,7))
sns.boxplot(df_catg['cylindernumber'],df_numeric['enginesize'],order=['two','three','four','five','six','eight','twelve'])
plt.show()

plt.figure(figsize=(20,20))
plt.subplot(321)
sns.boxplot(df_catg['carbody'],df['horsepower'])

plt.subplot(322)
sns.boxplot(df_catg['carbody'],df_numeric['Avgmpg'])
#########################################################

plt.figure(figsize=(20,20))
plt.subplot(325)
sns.boxplot(df_catg['cylindernumber'],df_numeric['Avgmpg'])

plt.subplot(326)
sns.boxplot(df_catg['drivewheel'],df_numeric['Avgmpg'])

plt.show()
plt.figure(figsize=(20,25))
plt.subplot(521)
df_catg['enginetype'].value_counts().plot('bar')
plt.xlabel('Type of Engine')
plt.ylabel('Number of Units')

plt.subplot(522)
sns.boxplot(df_catg['enginetype'],df_catg['price'],hue=df_catg['enginelocation'])
##################################################################################

plt.figure(figsize=(20,25))
plt.subplot(525)
df_catg['fuelsystem'].value_counts().plot('bar')
plt.xlabel('Type of fuel injection')
plt.ylabel('Number of Units')

plt.subplot(526)
sns.boxplot(df_catg['fuelsystem'],df_numeric['price'])
plt.show()

df_numeric.head()
#removig as already said above
df_numeric.drop('boreratio',1,inplace=True)
df_numeric.head()
#removing price column from catg data frame as already present in numeric one
df_catg=df_catg[df_catg.columns[:-1]]
#removig as already said above
df_catg=df_catg.drop(['cylindernumber','doornumber'],1)
cat_cols=df_catg.columns[:-1]
#renaming the data points in categorical feature with count <20 to other 
for col in cat_cols:
    a=df_catg[col].value_counts()
    for  j,i in enumerate(a):
        if i<20:
            name=a.index[j]
            df_catg=df_catg.replace(name,'other')
for i in df_catg.columns:
    print(df_catg[i].value_counts())
#removing the other data points whose counts is <20

for col in cat_cols:
    count = df_catg[col].value_counts()
    k = count.index[count>20][:-1]
    
    for cat in k:
        name = col + ' ' + cat
        df_catg[name] = (df_catg[col] == cat).astype(int)
    del df_catg[col]
df=pd.concat((df_numeric,df_catg),1)
df.head()
df=df.transform(lambda x : np.log1p(x))
#target feature is approx normal 
sns.distplot(df['price'])
X=df.drop('price',1)
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=45)
print(X.shape)
#as there are 14 columns , lets select around 5 featres 

lr=LinearRegression()
rfe=RFE(lr,5)
rfe=rfe.fit(X_train,y_train)

X_train_new=X_train[X_train.columns[rfe.support_]]
X_test_new=X_test[X_train_new.columns]
X_train_new.head()
model_lr=lr.fit(X_train_new,y_train)

print(f'The R square value for the training data is {round(model_lr.score(X_train_new,y_train)*100,2)}')
print(f'The R square value for the testing data is {round(model_lr.score(X_test_new,y_test)*100,2)}')

#as the data points are only 200, lets perforn 3 fold CV
X1=X[X_train_new.columns]

print(f'The mean R square value we get by this model is {round(cross_val_score(lr,X1,y,cv=3).mean()*100,2)}')
pd.DataFrame(zip(X_train_new.columns,model_lr.coef_),columns=['Col_name','Coeff_value'])
X_const=sm.add_constant(X_train_new)
model_ols=sm.OLS(y_train,X_const).fit()
model_ols.summary()
#cheking for multicollinearity

v=[VIF(X_const.values,i) for i in range(X_const.shape[1])]

pd.DataFrame(zip(X_const,v),columns=['col_name','vif_value'])
sns.distplot(model_ols.resid)
plt.show()
rf=RandomForestRegressor()
params={'max_depth':r(1,25), 'min_samples_split':r(2,20),
       'min_samples_leaf':r(2,15),'max_samples':r(50,75),'max_features':r(5,7),'n_estimators':r(1,50)}

rsearch=RandomizedSearchCV(rf,param_distributions=params,n_jobs=-1,cv=3,return_train_score=True,random_state=45)
rsearch.fit(X_train_new,y_train)
#the best estimator
print(rsearch.best_estimator_)
rsearch.best_params_
rfr=RandomForestRegressor(**rsearch.best_params_,random_state=45)
model_rfr=rfr.fit(X_train_new,y_train)
print(f'The R square value for the training data is {round(model_rfr.score(X_train_new,y_train)*100,2)}')
print(f'The R square value for the testing data is {round(model_rfr.score(X_test_new,y_test)*100,2)}')


X2=X[X_train_new.columns]
print(f'The mean R square value we get by this model is {round(cross_val_score(rfr,X2,y,cv=3).mean()*100,2)}')
#checking the normality of residuals
resid=y_train-model_rfr.predict(X_train_new)

print(f'P value for normality check is {stats.jarque_bera(resid)[1]}')
#Residuals are nomrmally disributed
gbr=GradientBoostingRegressor()
params_gb={'n_estimators':r(10,50),'min_samples_split':r(2,50),'min_samples_leaf':r(2,50),'max_depth':r(3,50)}
rsearch_gb=RandomizedSearchCV(gbr,param_distributions=params_gb,random_state=45,n_jobs=-1)
rsearch_gb.fit(X_train_new,y_train)
rsearch_gb.best_params_
g=GradientBoostingRegressor(**rsearch_gb.best_params_,random_state=45)
model_gb=g.fit(X_train_new,y_train)
print(f'The R square value for the training data is {round(model_gb.score(X_train_new,y_train)*100,2)}')
print(f'The R square value for the testing data is {round(model_gb.score(X_test_new,y_test)*100,2)}')
print(f'The mean R square value we get by this model is {round(cross_val_score(g,X2,y,cv=3).mean()*100,2)}')
resid=y_train-model_gb.predict(X_train_new)

print(f'P value for normality check is {stats.jarque_bera(resid)[1]}')
#Residuals are nomrmally disributed
from sklearn.pipeline import Pipeline
pipeline_lr=Pipeline([('lr_regression',LinearRegression())])
pipeline_rfr=Pipeline([('random_regression',RandomForestRegressor(**rsearch.best_params_,random_state=45))])
pipeline_gbr=Pipeline([('gradient_regression',GradientBoostingRegressor(**rsearch_gb.best_params_,random_state=45))])
pipelines = [pipeline_lr, pipeline_rfr, pipeline_gbr]
pipe_dict = {0: 'Linear Regression', 1: 'Random Forest Regression ', 2: 'Gradient Boost Regression'}
for model in pipelines:
    model.fit(X_train_new,y_train)
for i,model in enumerate(pipelines):
    print("{} Mean Accuracy: {}".format(pipe_dict[i],round((cross_val_score(model,X2,y,cv=5).mean()) *100),2))