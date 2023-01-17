"""

Lets import all the libraries needed for our code

"""



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score   

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

import xgboost as xgb

from math import sqrt

from scipy.stats import norm, skew 

from sklearn.preprocessing import PolynomialFeatures 

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import ElasticNet

from sklearn import linear_model

from scipy import stats

import os

print(os.listdir("../input"))
"""

Reading file from the specific folder

"""

train=pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

print("Shape of dataset",train.shape,"\n")

print("Basic information of our data",train.info(),"\n")

print("Basic view of our rows",train.head())

test=pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",

"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",

"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)

test.head()
df = pd.concat([train, test])

df = pd.merge_ordered(df, macro, on='timestamp', how='left')



print(df.shape)
df.head()
print(pd.set_option("display.max_rows",999))



print("Basic description of our data ",df.describe())
"""

Lets check how many null values are present in our dataset

"""

null_columns=df.columns[df.isnull().any()]



df[null_columns].isnull().sum()
"""

Lets fetch Year column from our Transaction timestamp 

"""

df["trans_year"]=df.timestamp.dt.year



month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df['month'] = df.timestamp.dt.month

df['dow'] = df.timestamp.dt.dayofweek



# Remove timestamp column (may overfit the model in train)

df.drop(['timestamp'], axis=1, inplace=True)

#df[df.build_year>300].build_year.min()



df=df[df.build_year!=20052009.0]

df.build_year=np.where(df.build_year<1691, np.nan,df.build_year)
"""

Lets replace NULL values in Buildyear as mean of the build_year in that subarea

"""

df.build_year = df.groupby(["sub_area"])['build_year'].apply(lambda x: x.fillna(x.mean()))

df["build_year"]=df["build_year"].astype("int64")

print("Lets see count of null value left in build_year now ",df.build_year.isnull().sum())
#Even after replacement with mean we have build_year as zero which is quite irrelevant 

#hence without any groupping lets replace these values with mean





df.build_year=np.where(df.build_year<1691, np.nan,df.build_year)

df.build_year.fillna(df.build_year.mean(),inplace=True)
df.build_year.isnull().sum()
sns.scatterplot(df.index,df.build_year)
"""

Lets replace NULL values in max_floor as mean of the max_floor in that subarea

"""

df.max_floor = df.groupby(["sub_area"])['max_floor'].apply(lambda x: x.fillna(x.mean()))

df["max_floor"]=df["max_floor"].astype("int64")

print("Lets see count of null value left in max_floor now ",df.max_floor.isnull().sum())



"""

Lets create bins as per building size

"""

bins=[-1,0,1,2,5,10,50,120]

labels=["Single storey","One floor","Only 2 floor","2-5 floor","6-10 floors","11-50 floors","50+ floors"]



df["build_size"]=pd.cut(df["max_floor"],bins,labels=labels)

"""

Lets convert the build_size as object type

"""

df["build_size"]=df["build_size"].astype("object")

print("Build size column overview \n",df["build_size"].head())



train["build_size"]=pd.cut(train["max_floor"],bins,labels=labels)

train["build_size"]=train["build_size"].astype("object")
"""

Lets create a column that tells the age of the building

"""

df["Building_Age"]=0

df["Building_Age"] = df["build_year"].apply(lambda x: 2019-x)

print ("Building age overview \n",df["Building_Age"].head())

df["Building_Age"]=df["Building_Age"].astype("int64")



"""

Lets create a column that tells the type as per the age of the building

"""

bins=[0,2,5,10,20,50,100,500]

labels=["2017-18","2017-2014","2014-2009","2009-1999","1999-1969","1969-1919","1919& before"]

df["building_agetype"]=pd.cut(df["Building_Age"],bins,labels=labels)

df["building_agetype"]=df["building_agetype"].astype("object")



print ("Building age overview \n",df["building_agetype"].head())

"""

Lets try to replace max_floor on the basis of subarea groupped mean's

"""



df.floor = df.groupby(["sub_area"])['floor'].apply(lambda x: x.fillna(x.mean()))

print ("Lets see count of null value left in max_floor now ",df.floor.isnull().sum())
"""

Lets try to replace max_floor on the basis of subarea and build year groupped mean's

"""

df.max_floor = df.groupby(["sub_area","build_year"])['max_floor'].apply(lambda x: x.fillna(x.mean()))

#print (df.max_floor.isnull().sum())

df.max_floor = df.groupby(["build_year"])['max_floor'].apply(lambda x: x.fillna(x.mean()))

#print (df.max_floor.isnull().sum())

df.max_floor = df.groupby(["sub_area"])['max_floor'].apply(lambda x: x.fillna(x.mean()))





"""

Since all values should be integer 

"""

df["max_floor"]=df["max_floor"].astype("int64")

print ("Lets see count of null value left in max_floor now ",df.max_floor.isnull().sum())
"""

Lets try to replace preschool_quota,school_quota as its mean

"""

df.preschool_quota = df["preschool_quota"].fillna(df["preschool_quota"].mean())

df.school_quota = df["school_quota"].fillna(df["school_quota"].mean())



"""

Since all values should be integer 

"""

#df["preschool_quota"]=df["preschool_quota"].astype("int64")

print ("Lets see count of null value left in preschool_quota now ",df.preschool_quota.isnull().sum())



print ("Lets see count of null value left in school_quota now ",df.school_quota.isnull().sum())
"""

Lets try to replace state on the basis of subarea and build year groupped mean's

"""

df.state = df.groupby(["sub_area","build_year"])['state'].apply(lambda x: x.fillna(x.mean()))



df.state = df.groupby(["build_year"])['state'].apply(lambda x: x.fillna(x.mean()))

df.state = df.groupby(["sub_area"])['state'].apply(lambda x: x.fillna(x.mean()))



"""

Since all values should be integer 

"""

df["state"]=df["state"].astype("int64")

print ("Lets see count of null value left in max_floor now ",df.state.isnull().sum())
"""

We could see that as per num_room we had approx 7k+ null rows so lets remove them

"""

df.num_room.fillna(df.num_room.median(),inplace=True)



df["num_room"]=df["num_room"].astype("int64")

print ("Lets see count of null value left in num_room now ",df.num_room.isnull().sum())
"""

Lets try to replace material on the basis of subarea median

"""



df.material = df.groupby(["sub_area"])['material'].apply(lambda x: x.fillna(x.median()))

"""

Since all values are numeric itself  

"""

df["material"]=df["material"].astype("int64")



print ("Lets see count of null value left in material now ",df.material.isnull().sum())
"""

We have assumed that below columns are of not much relevance for us and hence lets delete these columns from our dataset





to_be_del_col=["kitch_sq","cafe_sum_500_min_price_avg","cafe_sum_500_max_price_avg","cafe_avg_price_500","id",

              "cafe_sum_1000_min_price_avg","cafe_sum_1000_max_price_avg","cafe_avg_price_1000","cafe_sum_1500_min_price_avg"

               ,"cafe_sum_1500_max_price_avg","cafe_avg_price_1500","timestamp","life_sq","hospital_beds_raion"

              ]



df_old=df.copy(deep=True)

#to_be_del=['cafe_sum_1000_max_price_avg', 'cafe_avg_price_1000',]

df.drop(to_be_del_col,inplace=True,axis=1)

"""
"""

Lets round off the float values upto 2 decimal places 

"""

for col in df.columns:

    if (df[col].dtype == 'float64'): 

        df[col] = round(df[col],2)

        
"""

Its not a relevant case ot have max_floor less than floor

Hence lets have only data where max_floor is greater than or equal to floor

"""        

df["floor"]=df['floor'].astype("int64")

df["max_floor"]=df['max_floor'].astype("int64")

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

sns.set_palette(flatui)

sns.palplot(sns.color_palette())
fig, ax = plt.subplots(figsize=(15,10))

b = sns.boxplot(x="build_size", y="price_doc", data=train)

plt.xlabel("Building Size")

plt.ylabel("Price ")

plt.title("Price variation as per Building Type")

plt.show(b)
sns.countplot("trans_year",hue="product_type",data=df)

plt.xlabel("Year of transaction")

plt.ylabel("Total Number of Transactions")

plt.title("Transaction count as per year and Product Type")

sns.scatterplot("full_sq","price_doc",data=train)

plt.xlabel("Full square")



plt.ylabel("Price")

plt.title("Price v/s Full square")

"""

Hence we can see that full_sq >300 is an outlier so lets remove this value and check our plot

"""



#The outliers should not be there for price_doc as well and hence removing it

train_new=train[train["price_doc"]<train[["price_doc"]].quantile(.95).values[0]]



sns.scatterplot("full_sq","price_doc",data=train_new)

plt.xlabel("Full square")

plt.ylabel("Price")

plt.title("Price v/s Full square")
df_invest=df[df["product_type"]=="Investment"].groupby("sub_area")[["product_type"]].count()

df_owner=df[df["product_type"]=="OwnerOccupier"].groupby("sub_area")[["product_type"]].count()



fig, ax = plt.subplots(figsize=(15,30))

p1 = plt.barh(df_invest.index.values,df_invest.product_type)

p2 = plt.barh(df_owner.index.values,df_owner.product_type)

plt.ylabel('Sub Area')

plt.title('Number of investment and owner occupied properties as per sub area')

plt.legend((p1[0], p2[0]), ('Investment', 'OwnerOccupier'))



plt.show()
#df_2011=df[df["trans_year"]==2011].groupby("sub_area")[["id"]].count()

#df_2012=df[df["trans_year"]==2012].groupby("sub_area")[["id"]].count()

df_2013=df[df["trans_year"]==2013].groupby("sub_area")[["trans_year"]].count()

df_2014=df[df["trans_year"]==2014].groupby("sub_area")[["trans_year"]].count()

df_2015=df[df["trans_year"]==2015].groupby("sub_area")[["trans_year"]].count()



fig, ax = plt.subplots(figsize=(15,30))

#p2011 = plt.barh(df_2011.index.values,df_2011.id)

#p2012 = plt.barh(df_2012.index.values,df_2012.id)

p2013 = plt.barh(df_2013.index.values,df_2013.trans_year)

p2014 = plt.barh(df_2014.index.values,df_2014.trans_year)

p2015 = plt.barh(df_2015.index.values,df_2015.trans_year)

plt.ylabel('Subarea')

plt.title('Price  by subarea and Transaction year')

plt.legend((p2013[0], p2014[0],p2015[0]), ('2013', '2014','2015'))



#plt.legend((p2011[0], p2012[0],p2013[0], p2014[0],p2015[0]), ('2011', '2012','2013', '2014','2015'))



plt.show()
fig, ax = plt.subplots(figsize=(25,10))

a=sns.countplot("building_agetype",data=df,orient="v")

plt.ylabel("Number of houses ")

plt.xlabel("Building age ")

plt.title("Count of houses per Building age group ")



"""

#We removed some columns so lets see now how many columns have

#null values and if there are any lets replace it with mode

null_columns=df.columns[df.isnull().any()]

# basically just for the sake of keeping all columns and applying vif we replaced all null values with mode



for column in null_columns:

    df[column].fillna(df[column].mode()[0], inplace=True)

"""

    

    
"""

Since XGBoost works only on numeric data lets convert our object type data to numeric form and then apply our model



"""

#Since there are so many columns and it takes a long time we have removed the below code ,

#one can run and see for reference



"""

correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True)

plt.show()

"""
rs = np.random.RandomState(0)

#df = pd.DataFrame(rs.rand(10, 10))

corr = df[['green_part_5000', 'prom_part_5000', 'office_count_5000', 'office_sqm_5000','trc_count_5000', 'trc_sqm_5000', 'cafe_count_5000',

 'cafe_sum_5000_min_price_avg', 'cafe_sum_5000_max_price_avg','cafe_avg_price_5000', 'cafe_count_5000_na_price',

 'cafe_count_5000_price_500', 'cafe_count_5000_price_1000','cafe_count_5000_price_1500', 'cafe_count_5000_price_2500', 'cafe_count_5000_price_4000', 'cafe_count_5000_price_high',

 'big_church_count_5000', 'church_count_5000', 'mosque_count_5000',

 'leisure_count_5000', 'sport_count_5000', 'market_count_5000']].corr()

corr.style.background_gradient(cmap='coolwarm')



"""

df.drop(["trc_count_1500","cafe_count_1500","cafe_count_1500_na_price","cafe_count_1500_price_500","cafe_count_1500_price_1000","cafe_count_1500_price_1500","cafe_count_1500_price_2500","cafe_count_1500_price_4000","cafe_count_1500_price_high","big_church_count_1500","church_count_1500","mosque_count_1500","leisure_count_1500","sport_count_1500","market_count_1500"],axis=1,inplace=True)

df.drop(["culture_objects_top_25_raion","culture_objects_top_25","thermal_power_plant_raion","incineration_raion","oil_chemistry_raion","radiation_raion","railroad_terminal_raion","big_market_raion","nuclear_reactor_raion","detention_facility_raion","trc_count_1000","cafe_count_1000","cafe_count_1000_na_price","cafe_count_1000_price_500","cafe_count_1000_price_1000","cafe_count_1000_price_1500","cafe_count_1000_price_2500","cafe_count_1000_price_4000","cafe_count_1000_price_high","big_church_count_1000","church_count_1000","mosque_count_1000","leisure_count_1000","sport_count_1000","market_count_1000",],axis=1,inplace=True)



df.drop(["office_count_3000","trc_count_3000","cafe_count_3000","cafe_sum_3000_min_price_avg","cafe_sum_3000_max_price_avg","cafe_avg_price_3000","cafe_avg_price_3000","cafe_count_3000_na_price","cafe_count_3000_price_500","cafe_count_3000_price_1000","cafe_count_3000_price_1500","cafe_count_3000_price_2500","cafe_count_3000_price_4000","cafe_count_3000_price_high","big_church_count_3000","church_count_3000","mosque_count_3000","leisure_count_3000","sport_count_3000","market_count_3000"],axis=1,inplace=True)

df.drop(["office_count_2000","trc_count_2000","cafe_count_2000","cafe_sum_2000_min_price_avg","cafe_sum_2000_max_price_avg","cafe_avg_price_2000","cafe_count_2000_na_price","cafe_count_2000_price_500","cafe_count_2000_price_1000","cafe_count_2000_price_1500","cafe_count_2000_price_2500","cafe_count_2000_price_4000","cafe_count_2000_price_high","big_church_count_2000","church_count_2000","mosque_count_2000","leisure_count_2000","sport_count_2000","market_count_2000"],axis=1,inplace=True)

df.drop(["male_f","female_f","young_all","young_male","young_female","work_all","work_male","work_female","ekder_all",

             "ekder_male","ekder_female","0_6_all","0_6_male","0_6_female","7_14_all","7_14_male","7_14_female","0_17_all",

              "0_17_male","0_17_female","16_29_all","16_29_male","16_29_female","0_13_all","0_13_male","0_13_female"],

             axis=1,inplace=True)



df.drop(["office_count_500","office_sqm_500","trc_count_500","trc_sqm_500","cafe_count_500","cafe_count_500_na_price","cafe_count_500_price_500","cafe_count_500_price_1000","cafe_count_500_price_1500","cafe_count_500_price_2500","cafe_count_500_price_4000","cafe_count_500_price_high","big_church_count_500","mosque_count_500","leisure_count_500","sport_count_500","market_count_500"],axis=1,inplace=True)

df.drop(["cafe_count_5000_na_price","cafe_count_5000_price_500","cafe_count_5000_price_1000","cafe_count_5000_price_1500","cafe_count_5000_price_2500","cafe_count_5000_price_4000","cafe_count_5000_price_high","big_church_count_5000","church_count_5000","leisure_count_5000","office_count_5000","office_sqm_5000"],axis=1,inplace=True)



"""
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True)

plt.show()
a=list(df.select_dtypes(include=['object']).dtypes.index.values)



for i in a :

    # use pd.concat to join the new columns with your original dataframe

    df_xg = pd.concat([df,pd.get_dummies(df[i], prefix='sub_area',drop_first=True)],axis=1,)



    # now drop the original 'country' column (you don't need it anymore)

    #df_xg.drop([i],axis=1, inplace=True)
df_xg.columns.values
df_xg["log_price"]=np.log1p(df_xg["price_doc"])

                            

new_train=df_xg[df_xg.id<30474]

new_train.shape

new_test=df_xg[df_xg.id>30473]

new_test.shape


x=new_train.drop(["price_doc","log_price"],axis=1).select_dtypes(exclude=['object'])

y=new_train["log_price"]



x -= x.mean(axis=0)

x /= x.std(axis=0)





x_test=new_test.drop(["price_doc","log_price"],axis=1).select_dtypes(exclude=['object'])

y_test=new_test["log_price"]



x_test -= x_test.mean(axis=0)

x_test /= x_test.std(axis=0)



x_train,x_val,y_train,y_val=train_test_split(x, y, test_size=0.33, random_state=42)

"""

Its better to convert data into Dmatrix before applying xgb 

"""

dtrain = xgb.DMatrix(data=x_train,label= y_train)

dval = xgb.DMatrix(data=x_val, label=y_val)

dtest = xgb.DMatrix(data=x_test, label=y_test)



"""

Intialising XGB model and then fitting train dataa

"""



xgb_params = {

    'eta': 0.05,

    'max_depth': 6,

    'subsample': 1.0,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



# Uncomment to tune XGB `num_boost_rounds`

partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],

                       early_stopping_rounds=20, verbose_eval=20)



num_boost_round = partial_model.best_iteration
num_boost_round = partial_model.best_iteration

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)
ylog_pred = model.predict(dtest)

y_pred = np.exp(ylog_pred) - 1



df_sub = pd.DataFrame({'id': new_test.id, 'price_doc': y_pred})



df_sub.to_csv('sub.csv', index=False)
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}



cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results.head()
"""

Plotting the feature importance graph

"""



fig, ax = plt.subplots(1, 1, figsize=(8, 16))

xgb.plot_importance(partial_model, max_num_features=50, height=0.5, ax=ax)
feature_important = partial_model.get_score(importance_type='weight')

keys = list(feature_important.keys())

values = list(feature_important.values())



"""

Lets create a dataframe with all the details about feature importance as per our model 

"""

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

#Fetching top 50 features for our model building 

(data.sort_values(by=['score'], ascending=False)[:50]).index.values



"""

We selected top 50 fetaures and then tried to fit linear model

"""





z=df[['full_sq', 'floor', 'max_floor', 'build_year', 'trans_year',

       'num_room', 'state', 'railroad_km', 'metro_min_avto',

       'cemetery_km', 'public_healthcare_km', 'radiation_km',

       'kindergarten_km', 'green_zone_km', 'industrial_km', 'park_km',

       'workplaces_km', 'swim_pool_km', 'mosque_km', 'nuclear_reactor_km',

       'school_km', 'big_church_km', 'material',

       'public_transport_station_min_walk', 'green_part_500',

       'additional_education_km', 'metro_km_avto', 'university_km',

       'ttk_km', 'cafe_count_5000', 'catering_km', 'metro_min_walk',

       'water_km', 'area_m', 'theater_km', 'ice_rink_km', 'Building_Age',

       'green_part_1500', 'shopping_centers_km', 'big_road2_km',

       'fitness_km', 'stadium_km', 'church_synagogue_km',

       'zd_vokzaly_avto_km', 'big_market_km', 'bus_terminal_avto_km',

       'hospice_morgue_km', 'green_part_1000', 'ts_km',

       'railroad_station_avto_km']].select_dtypes(exclude=['object'])

y=df["price_doc"]



from sklearn.preprocessing import StandardScaler

standradize=True

if standradize:

    print('Standradizing the data..')

    #inf values can result from squaring

    scaler = StandardScaler()

    

    z= scaler.fit_transform(z)



    print('Data Standradized!')

    

x_train,x_test,y_train,y_test=train_test_split(z, y, test_size=0.33, random_state=42)

#wo_null1.columns.values
"""

Fitting linear regression on our dataset

"""



model=LinearRegression()



model.fit(x_train,y_train)
pred=model.predict(x_test)



print("mse",mean_squared_error(pred,y_test))

print("mse",mean_absolute_error(pred,y_test))

"""

Lets try and analyse our target variable and understand if its skewed or normally distributed 

"""



sns.distplot(train['price_doc'] , fit=norm);



"""

Finding mu and sigma after fitting it to a normalized form 

"""

(mu, sigma) = norm.fit(train['price_doc'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Price distribution')

"""

Lets also draw the qqplot of the same , basically qqplot are 

Q–Q (quantile-quantile) plot is a probability plot, which is a graphical method for comparing two probability 

distributions by plotting their quantiles against each other.

"""

fig = plt.figure()

res = stats.probplot(train['price_doc'], plot=plt)

plt.show()
"""

We could see that the price_doc value is right skewed

and as (linear) models work well on normally distributed data , 

we need to transform this variable and make it more normally distributed. 

Lets try log transformation and understand if it improved model perfomance

"""



sns.distplot(new_train["log_price"], fit=norm);



"""

Lets try and analyse our  transformed price and understand if its skewed or normally distributed 

"""

(mu, sigma) = norm.fit(new_train["log_price"])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Log Price distribution')
"""

Lets also draw the qqplot of the transformed price

"""

fig = plt.figure()

res = stats.probplot(new_train["log_price"], plot=plt)

plt.show()
"""

Lets apply regression now on Logtransformmed data

"""


z=new_train[['full_sq', 'floor', 'max_floor', 'build_year', 'trans_year',

       'num_room', 'state', 'railroad_km', 'metro_min_avto',

       'cemetery_km', 'public_healthcare_km', 'radiation_km',

       'kindergarten_km', 'green_zone_km', 'industrial_km', 'park_km',

       'workplaces_km', 'swim_pool_km', 'mosque_km', 'nuclear_reactor_km',

       'school_km', 'big_church_km', 'material',

       'public_transport_station_min_walk', 'green_part_500',

       'additional_education_km', 'metro_km_avto', 'university_km',

       'ttk_km', 'cafe_count_5000', 'catering_km', 'metro_min_walk',

       'water_km', 'area_m', 'theater_km', 'ice_rink_km', 'Building_Age',

       'green_part_1500', 'shopping_centers_km', 'big_road2_km',

       'fitness_km', 'stadium_km', 'church_synagogue_km',

       'zd_vokzaly_avto_km', 'big_market_km', 'bus_terminal_avto_km',

       'hospice_morgue_km', 'green_part_1000', 'ts_km',

       'railroad_station_avto_km']].select_dtypes(exclude=['object'])

y=new_train["log_price"]



from sklearn.preprocessing import StandardScaler

standradize=True

if standradize:

    print('Standradizing the data..')

    #inf values can result from squaring

    scaler = StandardScaler()

    

    z= scaler.fit_transform(z)



    print('Data Standradized!')

    

x_train,x_test,y_train,y_test=train_test_split(z, y, test_size=0.33, random_state=42)

    

#x_train,x_test,y_train,y_test=train_test_split(z, y, test_size=0.33, random_state=42)

#wo_null1.columns.values
"""

Fitting linear regression on our dataset

"""



model=LinearRegression()



model.fit(x_train,y_train)
pred=model.predict(x_test)



print("rmse",sqrt(mean_squared_error(pred,y_test)))

print("mse",mean_absolute_error(pred,y_test))

print("Rsquare values",r2_score(pred,y_test))
pred_price=np.expm1(pred)

y_test_price=np.expm1(y_test)



print("mse",mean_squared_error(pred_price,y_test_price))

print("mse",mean_absolute_error(pred_price,y_test_price))

print("Rsquare values",r2_score(pred_price,y_test_price))
"""

Lets try implementing Polynomial regression 

"""

  

poly = PolynomialFeatures(degree = 2) 

X_poly = poly.fit_transform(x_train) 



poly.fit(X_poly, y_train) 

lin2 = LinearRegression() 

lin2.fit(X_poly, y_train) 
pred=lin2.predict(poly.fit_transform(x_test)) 

print("Mean absolute error",mean_absolute_error(pred,y_test))

print("Rsquare values",r2_score(pred,y_test))
"""

Lets try implementing elasticnet i.e combination of lasso and ridge regresion 

and see if there is any improvement in our model

"""



regr = ElasticNet(random_state=0)

regr.fit(x_train, y_train)

ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,

      max_iter=1000, normalize=False, positive=False, precompute=False,

      random_state=0, selection='cyclic', tol=0.0001, warm_start=False)

#print(regr.coef_) 

print(regr.intercept_) 

pred=regr.predict(x_test)

print("Root Mean square error",sqrt(mean_squared_error(pred,y_test)))

print("Rsquare values",r2_score(pred,y_test))
#Again lets implement xgboost on selected features along with Log transformed price value
"""

We will implement k fold cross validation with Linear model as its a better way to train

Below is the sample copied code for reference 

"""



x_train, x_test, y_train, y_test = train_test_split(z,y,test_size=0.2)

clf = linear_model.Lasso()

clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

print("accuracy",accuracy)

pred=clf.predict(x_test)

print("Mean absolute error",mean_absolute_error(pred,y_test))





scores = cross_val_score(clf, x_train, y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores)
"""

Its better to convert data into Dmatrix before applying xgb 

"""

dtrain = xgb.DMatrix(data=x_train,label= y_train)

dval = xgb.DMatrix(data=x_test, label=y_test)



"""

Intialising XGB model and then fitting train dataa

"""



xgb_params = {

    'eta': 0.1,

    'max_depth': 6,

    'subsample': 1.0,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



# Uncomment to tune XGB `num_boost_rounds`

partial_model = xgb.train(xgb_params, dtrain, num_boost_round=1000, evals=[(dval, 'val')],

                       early_stopping_rounds=20, verbose_eval=20)



num_boost_round = partial_model.best_iteration


pred=partial_model.predict(dval)

print("Mean absolute error",mean_absolute_error(pred,y_test))

print("rmse",sqrt(mean_squared_error(pred,y_test)))

print("Rsquare values",r2_score(pred,y_test))


test1=new_test[['full_sq', 'floor', 'Building_Age', 'max_floor', 'trans_year',

       'kindergarten_km', 'state', 'metro_min_avto', 'num_room',

       'public_transport_station_km', 'park_km', 'green_zone_km',

       'railroad_km', 'school_km', 'industrial_km', 'water_km',

       'catering_km', 'metro_km_avto', 'additional_education_km',

       'cemetery_km', 'build_year', 'big_road1_km', 'fitness_km',

       'public_healthcare_km', 'material', 'mosque_km', 'metro_min_walk',

       'big_market_km', 'hospice_morgue_km', 'radiation_km',

       'big_road2_km', 'water_treatment_km', 'green_part_1000',

       'thermal_power_plant_km', 'shopping_centers_km', 'area_m',

       'swim_pool_km', 'stadium_km', 'market_shop_km',

       'railroad_station_walk_km', 'ts_km', 'theater_km', 'preschool_km',

       'office_km', 'green_part_500', 'power_transmission_line_km',

       'nuclear_reactor_km', 'prom_part_5000', 'zd_vokzaly_avto_km',

       'bus_terminal_avto_km']].select_dtypes(exclude=['object'])



from sklearn.preprocessing import StandardScaler

standradize=True

if standradize:

    print('Standradizing the data..')

    #inf values can result from squaring

    scaler = StandardScaler()

    

    test1= scaler.fit_transform(test1)



    print('Data Standradized!')

dval1 = xgb.DMatrix(data=test1, label=y_test)

print(test1.shape)
print(new_test.shape)

test.shape
"""

pred1=partial_model.predict(dval1)

#pred2=pd.DataFrame(pred1)

df_sub = pd.DataFrame({'id': test["id"], 'price_doc': pred1})



df_sub.to_csv('sub.csv', index=False)

"""
df_sub.shape