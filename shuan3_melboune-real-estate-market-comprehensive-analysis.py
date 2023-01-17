%matplotlib inline

import pandas as pd

import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection  import train_test_split

import numpy as np

from scipy.stats import norm # for scientific Computing

from scipy import stats, integrate

import matplotlib.pyplot as plt
melbourne_data  = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")
melbourne_data.head(10)
melbourne_data.shape
melbourne_data.info()
## verifying columns with object data type

print(melbourne_data.select_dtypes(["object"]).columns)
##changing all object data types to category - This step is necessary to be able to plot categorical data for our analysis

objdtype_cols = melbourne_data.select_dtypes(["object"]).columns

melbourne_data[objdtype_cols] = melbourne_data[objdtype_cols].astype('category')
melbourne_data.info()
## looking at data information above, we can notice that "Data" is also converted to category. 

## In this step we will cast date to datetime

melbourne_data['Date'] =  pd.to_datetime( melbourne_data['Date'])
## the following command suggests that all our data types are now in required format

melbourne_data.info()
## describe command will give us all statstical information about our numeric variables

melbourne_data.describe().T
melbourne_data["Postcode"] = melbourne_data["Postcode"].astype('category')
melbourne_data.describe().T
## in this step we will first confirm our above statemnt by obesrving "Rooms" and "Bedroom2"



melbourne_data['b 2 r'] = melbourne_data["Bedroom2"] - melbourne_data["Rooms"]

melbourne_data[['b 2 r', 'Bedroom2', 'Rooms']].head()
## We can see that the difference is very minimal here that is will be wise to remove one of the 2 columns

melbourne_data = melbourne_data.drop(['b 2 r', 'Bedroom2'], 1)
## visualizing missing values

fig, ax = plt.subplots(figsize=(15,7))

sns.heatmap(melbourne_data.isnull(), yticklabels=False,cmap='viridis')
# Percentage of missing values

melbourne_data.isnull().sum()/len(melbourne_data)*100
melbourne_data = melbourne_data.drop(["Landsize", "BuildingArea", "YearBuilt"], axis=1)
## Also since our target variable is price, it makes sense to drop rows for Price columns wher price values are missing

melbourne_data.dropna(subset=["Price"], inplace=True)
#from sklearn.preprocessing import Imputer

#X=melbourne_data[['Bathroom','Car']]

#imp=Imputer(missing_values='NaN',strategy='median',axis=0)

#imp.fit(X)

#X=pd.DataFrame(data=imp.transform(X),columns=X.columns)

#melbourne_data[['Bathroom','Car']]=X
melbourne_data['Car']=melbourne_data['Car'].fillna(melbourne_data['Car'].mode()[0])

melbourne_data['Bathroom']=melbourne_data['Bathroom'].fillna(melbourne_data['Bathroom'].mode()[0])
melbourne_data.shape
# Percentage of missing values

melbourne_data.isnull().sum()/len(melbourne_data)*100
melbourne_data.describe().T
## to findout outliers lets divide data into different price ranges to identify number of occurences of data in different price ranges

melbourne_data['PriceRange'] = np.where(melbourne_data['Price'] <= 100000, '0-100,000',  

                                       np.where ((melbourne_data['Price'] > 100000) & (melbourne_data['Price'] <= 1000000), '100,001 - 1M',

                                                np.where((melbourne_data['Price'] > 1000000) & (melbourne_data['Price'] <= 3000000), '1M - 3M',

                                                        np.where((melbourne_data['Price']>3000000) & (melbourne_data['Price']<=5000000), '3M - 5M',

                                                                np.where((melbourne_data['Price']>5000000) & (melbourne_data['Price']<=6000000), '5M - 6M',

                                                                        np.where((melbourne_data['Price']>6000000) & (melbourne_data['Price']<=7000000), '6M - 7M',

                                                                                np.where((melbourne_data['Price']>7000000) & (melbourne_data['Price']<=8000000), '7M-8M', 

                                                                                         np.where((melbourne_data['Price']>8000000) & (melbourne_data['Price']<=9000000), '8M-9M', 

                                                                                                 np.where((melbourne_data['Price']>9000000) & (melbourne_data['Price']<=10000000), '9M-10M', 

                                                                                                         np.where((melbourne_data['Price']>10000000) & (melbourne_data['Price']<=11000000), '10M-11M', 

                                                                                                                 np.where((melbourne_data['Price']>11000000) & (melbourne_data['Price']<=12000000), '11M-12M', '')

                                                                                                                 ))))))))))
melbourne_data.groupby(['PriceRange']).agg({'PriceRange': ['count']})

melbourne_data.info()
melbourne_data.describe().T
melbourne_data.drop(melbourne_data[(melbourne_data['PriceRange'] == '0-100,000') |

                                   (melbourne_data['PriceRange'] == '7M-8M') |

                                   (melbourne_data['PriceRange'] == '8M-9M') |

                                   (melbourne_data['PriceRange'] == '11M-12M')].index, inplace=True)
melbourne_data.describe().T
melbourne_data.groupby(['Rooms'])['Rooms'].count()
melbourne_data.drop(melbourne_data[(melbourne_data['Rooms'] == 12) | 

                                   (melbourne_data['Rooms'] == 16)].index, inplace=True)
melbourne_data.describe().T
##sns.distplot(melbourne_data, kde=False, bins=20).set(xlabel='Price');

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

##melbourne_data.select_dtypes(include = numerics)

melbourne_data.select_dtypes(include = numerics).hist(bins=15, figsize=(15, 6), layout=(2, 4))
melbourne_data['Distance'] = round(melbourne_data['Distance'])
melbourne_data.shape
## extract year from date

melbourne_data['Year']=melbourne_data['Date'].apply(lambda x:x.year)

melbourne_data.head(5)
#data subset by type

#house price

melbourne_data_h=melbourne_data[melbourne_data['Type']=='h']

#condo price

melbourne_data_u=melbourne_data[melbourne_data['Type']=='u']

#townhouse price

melbourne_data_t=melbourne_data[melbourne_data['Type']=='t']

#house,condo and town house price groupby year and mean

melbourne_data_h_y=melbourne_data_h.groupby('Year').mean()

melbourne_data_u_y=melbourne_data_u.groupby('Year').mean()

melbourne_data_t_y=melbourne_data_t.groupby('Year').mean()

melbourne_data_h_y.head()

#sns.lmplot(x="Year", y="Price", hue="Type", data=melbourne_data,  x_estimator=np.mean);

melbourne_data_h_y['Price'].plot(kind='line', color='r',label='House')

melbourne_data_u_y['Price'].plot(kind='line', color='g',label='Condo')

melbourne_data_t_y['Price'].plot(kind='line', color='b',label='Townhouse')

year_xticks=[2016,2017,2018]

plt.ylabel('Price')

plt.xticks( year_xticks)

plt.title('Melboune price trend Vs Year per type')

plt.legend()
melbourne_data.shape
melbourne_data.columns


melbourne_data_South_M=melbourne_data[melbourne_data['Regionname']=='Southern Metropolitan']

melbourne_data_South_M_average=melbourne_data_South_M.groupby(['Year'])['Price'].mean()

# Series.to_frame()

# create X and y



X = melbourne_data_South_M[[ 'Year']]

y = melbourne_data_South_M[['Price']]



# instantiate and fit

lm2 = LinearRegression()

lm2.fit(X, y)



# print the coefficients

print (lm2.intercept_)

print (lm2.coef_)


### STATSMODELS ###



# you have to create a DataFrame since the Statsmodels formula interface expects it

X_new = pd.DataFrame({'Year': [2019,2020,2021]})



# predict for a new observation

lm2.predict(X_new)
melbourne_data_SM=melbourne_data[melbourne_data['Regionname']=='Southern Metropolitan']

melbourne_data_SM_u=melbourne_data_SM[melbourne_data_SM['Type']=='u']

melbourne_data_SM_u.shape


### STATSMODELS ###



# create a fitted model

lm1 = smf.ols(formula='Price ~ Year', data=melbourne_data_SM_u).fit()



# print the coefficients

lm1.params


# you have to create a DataFrame since the Statsmodels formula interface expects it

X_new = pd.DataFrame({'Year': [2016,2017,2018,2019,2020,2021]})



# predict for a new observation

lm1.predict(X_new)
lm1.rsquared


melbourne_data_E=melbourne_data[melbourne_data['Regionname']=='Eastern Metropolitan']

melbourne_data_E_u=melbourne_data_E[melbourne_data_E['Type']=='u']



lme = smf.ols(formula='Price ~ Year', data=melbourne_data_E_u).fit()



# print the coefficients

lme.params
melbourne_data_E_u.shape
X_new = pd.DataFrame({'Year': [2016,2017,2018,2019,2020,2021]})



# predict for a new observation

lme.predict(X_new)
#get month information from date 

#df['year_month']=df.datetime_column.apply(lambda x: str(x)[:7])

#per = df.Date.dt.to_period("M")

# How many calls, sms, and data entries are in each month?

#data.groupby(['month', 'item'])

#df['birthdate'].groupby([df.birthdate.dt.year, df.birthdate.dt.month]).agg('count')

melbourne_data['Month']=pd.DatetimeIndex(melbourne_data['Date']).month



#lois[_y_m]=lois['Price'].groupby(['Month']).mean()

#Prepare data for pie chart to check sales based on month in order to see which month sell most.

melbourne_data_2016=melbourne_data[melbourne_data['Year']==2016]

melbourne_data_2017=melbourne_data[melbourne_data['Year']==2017]

melbourne_data_2018=melbourne_data[melbourne_data['Year']==2018]

melbourne_data_2016_count=melbourne_data_2016.groupby(['Month']).count()

melbourne_data_2017_count=melbourne_data_2017.groupby(['Month']).count()

melbourne_data_2018_count=melbourne_data_2018.groupby(['Month']).count()

Comparison={2016:melbourne_data_2016.shape,2017:melbourne_data_2017.shape,2018:melbourne_data_2018.shape}

Comparison
label_2016=['January','March','April','May','June','July','August','September','October','November','December']

plt.pie(melbourne_data_2016_count['Price'],labels=label_2016,autopct='%.1f %%')

plt.title('Year 2016')

plt.show()
label_2017=['January','February','March','April','May','June','July','August','September','October','November','December']

plt.pie(melbourne_data_2017_count['Price'],labels=label_2017,autopct='%.1f %%')

plt.title('Year 2017')
label_2018=['January','February','March','June','October']

plt.pie(melbourne_data_2018_count['Price'],labels=label_2018,autopct='%.1f %%')

plt.title('Year 2018')


# Abbreviate Regionname categories for presentation

melbourne_data['Regionabb'] = melbourne_data['Regionname'].map({'Northern Metropolitan':'N Metro',

                                            'Western Metropolitan':'W Metro', 

                                            'Southern Metropolitan':'S Metro', 

                                            'Eastern Metropolitan':'E Metro', 

                                            'South-Eastern Metropolitan':'SE Metro', 

                                            'Northern Victoria':'N Vic',

                                            'Eastern Victoria':'E Vic',

                                            'Western Victoria':'W Vic'})


sns.lmplot(x="Year", y="Price",hue="Type", data=melbourne_data,col='Regionabb', x_estimator=np.mean,col_wrap=2)

plt.ylim(200000, 2000000)

plt.xlim(2015,2020)
#South region price change vs year per type

sns.lmplot(x="Year", y="Price",hue="Type", data=melbourne_data[melbourne_data['Regionabb']=='S Metro'], x_estimator=np.mean);
# East region price change vs year one type

melbourne_data_S=melbourne_data[melbourne_data['Regionabb']=='S Metro']

sns.lmplot(x="Year", y="Price", data=melbourne_data_S[melbourne_data_S['Type']=='u'], x_estimator=np.mean);
Pct_change=melbourne_data.groupby(['Year','Regionabb','Type'],as_index=False)['Price'].mean()

Pct_change = Pct_change.sort_values(['Regionabb', 'Type','Year']).set_index(np.arange(len(Pct_change.index)))



Pct_change.info()
melbourne_data_count_region_y=melbourne_data.groupby(['Year','Regionabb','Type'],as_index=False)['Price'].count()

melbourne_data_count_region_y = melbourne_data_count_region_y.sort_values(['Regionabb', 'Type','Year']).set_index(np.arange(len(melbourne_data_count_region_y.index)))

melbourne_data_count_region_y.rename(columns={'Price':'Count'}, inplace=True)

# define fucntion to get year growth rate again price per region and type

def PCTM(gg):

    df=pd.DataFrame(gg['Price'].pct_change())

    df['Year']=gg['Year']

    df['region']=gg['Regionabb']

    df['Type']=gg['Type']

    df=df[df['Year']!=2016]

    return df
#df2[df2['id'].isin(['SP.POP.TOTL','NY.GNP.PCAP.CD'])]

melboune_growthrate_y_t=PCTM(Pct_change)

melboune_growthrate_y_t1=melboune_growthrate_y_t[melboune_growthrate_y_t['region'].isin(['N Metro','S Metro','E Metro','SE Metro','W Metro','S Metro'])]

melboune_growthrate_y_t1.rename(columns={'Price':'Price Growth Rate'}, inplace=True)

melboune_growthrate_y_t1[melboune_growthrate_y_t1['Price Growth Rate']>0.05]
Sales_count=melbourne_data.groupby(['Regionabb'])['Price'].count()

Sales_count.head(10)
Sales_count=melbourne_data.groupby(['Regionabb','Type'])['Price'].count()

Sales_count.nlargest(20)


            # define fucntion to get year growth rate again count per region and type

def PCTMC(gg):

    df=pd.DataFrame(gg['Count'].pct_change())

    df['Year']=gg['Year']

    df['region']=gg['Regionabb']

    df['Type']=gg['Type']

    df=df[df['Year']!=2016]

    return df
#df2[df2['id'].isin(['SP.POP.TOTL','NY.GNP.PCAP.CD'])]

melboune_growthrate_y_c=PCTMC(melbourne_data_count_region_y)

melboune_growthrate_y_c1=melboune_growthrate_y_c[melboune_growthrate_y_c['region'].isin(['N Metro','S Metro','E Metro','SE Metro','W Metro','S Metro'])]



melboune_growthrate_y_c1.rename(columns={'Count':'Count Growth Rate'}, inplace=True)

melboune_growthrate_y_c1[melboune_growthrate_y_c1['Count Growth Rate']>0.2]
melboune_count1=melbourne_data_count_region_y[melbourne_data_count_region_y['Regionabb'].isin(['S Metro','E Metro','SE Metro','W Metro','S Metro','N Metro'])]

melboune_count1[melboune_count1['Count']>1000]
sns.boxplot(x = 'Method', y = 'Price', data = melbourne_data)

plt.show()

#Sold method did not affect price
sns.lmplot(x="Year", y="Price", hue="Rooms", data=melbourne_data,  x_estimator=np.mean);
sns.lmplot(x="Distance", y="Price", data=melbourne_data, x_estimator=np.mean);
sns.lmplot(x="Car", y="Price", data=melbourne_data, x_estimator=np.mean);




Ideal_House=melbourne_data.groupby(['Regionabb','Type','Rooms','Bathroom'])['Price'].count()





Ideal_House.loc[['S Metro'],'h'].nlargest(10)
Ideal_House.nlargest(10)
Ideal_House.loc[['E Metro'],'u'].nlargest(10)
corrmat=melbourne_data.corr()
fig,ax=plt.subplots(figsize=(12,10))

sns.heatmap(corrmat,annot=True,annot_kws={'size': 12})
#define function to refine those correlation more than 0.3 with abs value

def getCorrelatedFeature(corrdata,threshold):

    feature=[]

    value=[]

    

    for i, index in enumerate(corrdata.index):

        if abs(corrdata[index])>threshold:

            feature.append(index)

            value.append(corrdata[index])

    df=pd.DataFrame(data=value,index=feature,columns=['Corr Value'])

    return df
threshold=0.4

corr_value=getCorrelatedFeature(corrmat['Price'],threshold)

corr_value
melbourne_data.isnull().sum()
melbourne_data['Type_Code'] = melbourne_data['Type'].map({'h':3,

                                            't':2, 

                                            'u':1, 

                                            'dev site':0, 

                                            'o res':0, 

                                            'br':0})

# Group Regionname categories 

melbourne_data1 = pd.get_dummies(melbourne_data['Regionabb'],drop_first=False)

melbourne_data=pd.concat([melbourne_data,melbourne_data1],axis=1)

melbourne_data.columns.values


#fig,ax=plt.subplots(figsize=(12,10))

#df=melbourne_data[['Price','Rooms','Distance', 'Bathroom',  'Year', 'Type_Code','RegionCode']]

#sns.heatmap(df,annot=True)

#dff=melbourne_data[['Price','Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code',]].groupby('RegionCode')

#dff.head()
melbourne_data_NN=melbourne_data[['Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code','N Metro','W Metro','S Metro','E Metro','SE Metro','N Vic','E Vic','W Vic','Price']].dropna()

melbourne_data_NN[['Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code','N Metro','W Metro','S Metro','E Metro','SE Metro','N Vic','E Vic','W Vic','Price']].isnull().sum()
melbourne_data_NN.shape
#Finding coefficient



X=melbourne_data_NN[['Rooms','Distance', 'Bathroom', 'Car', 'Year', 'Propertycount','Type_Code','N Metro','W Metro','S Metro','E Metro','SE Metro','N Vic','E Vic','W Vic']]

y=melbourne_data_NN['Price']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state=5)
# Fit

# Import model

from sklearn.linear_model import LinearRegression



# Create linear regression object

regressor = LinearRegression()



# Fit model to training data

regressor.fit(X_train,y_train)


# Predict

# Predicting test set results

y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)


from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_test,y_pred))

print('MSE:',metrics.mean_squared_error(y_test,y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('R^2 =',metrics.explained_variance_score(y_test,y_pred))
plt.scatter(y_test, y_pred)
# Histogram of the distribution of residuals

sns.distplot((y_test - y_pred))
cdf = pd.DataFrame(data = regressor.coef_, index = X.columns, columns = ['Coefficients'])

cdf
X.head()
from sklearn.ensemble import RandomForestClassifier

#model=RandomForestClassifier(n_estimators=20)

#model.fit(X_train,y_train)
clf=RandomForestClassifier(n_jobs=2,random_state=0)

clf.fit(X,y)
clf.predict(X)
clf.score(X_test,y_test)
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LinearRegression



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

folds=StratifiedKFold(n_splits=3)


def get_score(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    return model.score(X_test, y_test)
print(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))
print(get_score(LinearRegression(), X_train, X_test, y_train, y_test))
correlated_data=melbourne_data_NN[corr_value.index]

correlated_data.head()
corr_value.index
sns.pairplot(correlated_data)

plt.tight_layout()
sns.heatmap(correlated_data.corr(),annot=True,annot_kws={'size':12})
X1=correlated_data.drop(labels=['Price'],axis=1)

y1=correlated_data['Price']

X1.head()
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=0)
X1_train.shape,X1_test.shape
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error,mean_squared_error
model = LinearRegression()

model.fit(X1_train,y1_train)
y1_predict=model.predict(X1_test)
y1_predict,y1_test
df=pd.DataFrame(data=[y1_predict,y1_test])

df.T.head(5)

from sklearn.metrics import r2_score
score=r2_score(y1_test,y1_predict)

mae=mean_absolute_error(y1_test,y1_predict)

mse=mean_squared_error(y1_test,y1_predict)

print("r2_score", score)

print("mae", mae)

print("mse", mse)
#store feature performance

total_features=[]

total_features_name=[]

selected_correlation_value=[]

r3_score=[]

mae_value=[]

mse_value=[]

def performance_metrics(features, th, y_true,y_pred):

    score=r2_score(y_true,y_pred)

    mae=mean_absolute_error(y_true,y_pred)

    mse=mean_squared_error(y_true,y_pred)

    

    total_features.append(len(features)-1)

    total_features_name.append(str(features))

    selected_correlation_value.append(th)

    r3_score.append(score)

    mae_value.append(mae)

    mse_value.append(mse)

    

    metrics_dataframe=pd.DataFrame(data=[total_features_name, total_features,selected_correlation_value,r3_score,mae_value,mse_value],index=['Features name','Total features','corre value','r2 score','mae','mse'])

    return metrics_dataframe.T
def get_y_predict(corrdata):

    X=corrdata.drop(labels=['Price'],axis=1)

    y=corrdata['Price']

    

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    model=LinearRegression()

    model.fit(X_train,y_train)

    y_predict=model.predict(X_test)

    return y_predict

    
th5=0.4

corr_value=getCorrelatedFeature(corrmat['Price'],th5)

correlated_data=melbourne_data_NN[corr_value.index]

y_predict=get_y_predict(correlated_data)

performance_metrics(correlated_data.columns,th5,y_test,y_predict)
#Ploting learning curves

from sklearn.model_selection import learning_curve, ShuffleSplit
def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=None,train_sizes=np.linspace(0.1,1.0,10)):

    plt.figure()

    plt.title(title)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    

    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)

    

    train_scores_mean=np.mean(train_scores,axis=1)

    train_scores_std=np.std(train_scores,axis=1)

    test_scores_mean=np.mean(test_scores,axis=1)

    test_scores_std=np.std(test_scores,axis=1)

    

    plt.grid()

    

    plt.fill_between(train_sizes,train_scores_mean - train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")

    plt.fill_between(train_sizes,test_scores_mean - test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="g")

    plt.plot(train_sizes,train_scores_mean,'o-',color="r",label="Training score")

    plt.plot(train_sizes,test_scores_mean,'o-',color="g",label="Cross-validation score")

    

    plt.legend(loc="best")

    return plt



X=correlated_data.drop(labels=['Price'],axis=1)

y=correlated_data['Price']



title="learning curves (linear regression)" + str(X.columns.values)

cv=ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)



estimator=LinearRegression()

plot_learning_curve(estimator,title,X1,y1,ylim=(0.7,1.01),cv=cv,n_jobs=-1)



plt.show()