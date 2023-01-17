import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform
from sklearn import preprocessing
def daytime (row):
    if (row['hour'] <= 6) or (row['hour'] > 22):
        return ("night")
    elif (row['hour'] > 6) and (row['hour'] <= 12):
        return ("morning")
    elif (row['hour'] > 12) and (row['hour'] <= 17):
        return ("afternoon")
    elif (row['hour'] > 17) and (row['hour'] <= 22):
        return ("evening")

    
def add_time_features(df):
    df['year'] = df['pickup_datetime'].apply(lambda x: x.year)
    df['month'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['day'] = df['pickup_datetime'].apply(lambda x: x.day)
    df['hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['weekday'] = df['pickup_datetime'].apply(lambda x: x.weekday())
    df['pickup_datetime'] =  df['pickup_datetime'].apply(lambda x: str(x))
    df['daytime'] = df.apply (lambda x: daytime(x), axis=1)
    df = df.drop('pickup_datetime', axis=1)
    df=df.drop('hour',axis=1)
    df=df.drop('day',axis=1)
    return df
df = pd.read_csv("../input/cabfare/Data.csv")
df.info()
df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z',errors='coerce')
df.isnull().sum()
df= add_time_features(df)
df["year"] = df["year"].astype(object)
df["month"] = df["month"].astype(object)
df["weekday"] = df["weekday"].astype(object)
from geopy.distance import geodesic
from geopy.distance import great_circle
df['great_circle']=df.apply(lambda x: great_circle((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)
df['geodesic']=df.apply(lambda x: geodesic((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)
df.info()
def time_analysis(df):
    return pd.DataFrame({"FareAverage":np.mean(df.fare_amount),"Count":np.size(df.fare_amount),"FareSum":sum(df.fare_amount)},index=["Time"] )
df_yearly=df.groupby('year').apply(time_analysis).reset_index()
sns.catplot(x="year", y="FareAverage", kind="bar", data=df_yearly,color="c",palette="dark",height=3, aspect=1.5)
sns.catplot(x="year", y="Count", kind="bar", data=df_yearly,color="g",palette="dark",height=3, aspect=1.5)
sns.catplot(x="year", y="FareSum", kind="bar", data=df_yearly,color="m",palette="dark",height=3, aspect=1.5)
df_monthly=df.groupby('month').apply(time_analysis).reset_index()
sns.catplot(x="month", y="FareAverage", kind="bar", data=df_monthly,color="c",palette="dark",height=3, aspect=1.5)
sns.catplot(x="month", y="Count", kind="bar", data=df_monthly,color="g",palette="dark",height=3, aspect=1.5)
sns.catplot(x="month", y="FareSum", kind="bar", data=df_monthly,color="m",palette="dark",height=3, aspect=1.5)
df_weekly=df.groupby('weekday').apply(time_analysis).reset_index()
sns.catplot(x="weekday", y="FareAverage", kind="bar", data=df_weekly,color="c",palette="dark",height=3, aspect=1.5)
sns.catplot(x="weekday", y="Count", kind="bar", data=df_weekly,color="g",palette="dark",height=3, aspect=1.5)
sns.catplot(x="weekday", y="FareSum", kind="bar", data=df_weekly,color="m",palette="dark",height=3, aspect=1.5)
df_daily=df.groupby('daytime').apply(time_analysis).reset_index()
sns.catplot(x="daytime", y="FareAverage", kind="bar", data=df_daily,color="c",palette="dark",height=3, aspect=1)
sns.catplot(x="daytime", y="Count", kind="bar", data=df_daily,color="g",palette="dark",height=3, aspect=1)
sns.catplot(x="daytime", y="FareSum", kind="bar", data=df_daily,color="m",palette="dark",height=3, aspect=1)
df1=df[['passenger_count',"pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude", 'great_circle',"geodesic","fare_amount"]]
sns.pairplot(df1)
df["passenger_count"] = df["passenger_count"].astype(object)
ncol=["great_circle","geodesic","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","fare_amount"]
plt.figure(figsize=(10,10))
_ = sns.heatmap(df[ncol].corr(), square=True, cmap='RdYlGn',linewidths=1,linecolor='w',annot=True)
plt.title('Correlation matrix ')
plt.show()
import scipy.stats as stats
_ = sns.jointplot(x='fare_amount',y='geodesic',data=df,kind = 'reg')
_.annotate(stats.pearsonr)
plt.show()
df=df.drop(["great_circle","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"],axis=1)
df.info()
# Import label encoder 
colnames = list(df.columns)
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for col in colnames:
    if df[col].dtype==object:
        df[col]= label_encoder.fit_transform(df[col])
cat_var=["passenger_count","year","month","weekday","daytime"] 
catdf=df[cat_var]
from sklearn.feature_selection import chi2
n= 10
for i in range(0,4):
    X=catdf.iloc[:,i+1:n]
    y=catdf.iloc[:,i]
    chi_scores = chi2(X,y)
    p_values = pd.Series(chi_scores[1],index = X.columns)
    print("for",i)
    print(p_values)
    for j in range (0, len(p_values)):
        if (p_values[j]<0.01):
            print(p_values[j])
df=df.drop(["year","month","weekday"],axis=1)
df.info()
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('fare_amount ~ C(passenger_count)+C(daytime)',data=df).fit()
aov_table = sm.stats.anova_lm(model)
aov_table
probanova=list(aov_table["PR(>F)"])
for i in range(0,3):
    if probanova[i]>0.05:
        print(i)
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
df1=df.drop(["fare_amount"],axis=1)
calc_vif(df1)
df["passenger_count"] = df["passenger_count"].astype(object)
df["daytime"] = df["daytime"].astype(object)
df = pd.get_dummies(df, drop_first=True)
df.info()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
x = df.drop('fare_amount',axis=1).values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)
y = df['fare_amount'].values
model = sm.OLS(y,X).fit()
model.summary()