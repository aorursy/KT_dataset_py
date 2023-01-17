# import modules 

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

import numpy as np

import pandas as pd 

import sklearn as sk

from math import sqrt

import warnings

warnings.filterwarnings('ignore')

 
# load ambrogia file

df = pd.read_csv("../input/ambrosia.csv",encoding='utf-8') 
# We'll start with site1 and year 2014

ambel = df[(df.site =='site1') & (df.year == 2014) ] 
# shape of dataset, rows and columns

print("rows, columns: "+str(ambel.shape))

print(len(str(ambel.shape))*'-')

print(ambel.dtypes.value_counts())

print(len(str(ambel.shape))*'-')

ambel.head(2)
ambel.info()
#convert to date

ambel['harvest_date'] = pd.to_datetime(ambel['harvest_date'])

#create month column

ambel['month'] = ambel['harvest_date'].dt.month

ambel.head()
# check for null values 

ambel.isnull().sum() 
# Descriptive statistics of dataset

ambel.describe(include='all').transpose()
ax = sns.swarmplot(x="type_crop", y="fallen_seeds", data=ambel)

ax.set_title("Fallen seeds by type of crop")

plt.xlabel("Type of crop")

plt.ylabel("Fallen seeds")

plt.show() 
ax = sns.swarmplot(x="month", y="fallen_seeds" , hue="type_crop", data=ambel)

ax.set_title("Fallen seeds by type of crop")

plt.xlabel("Month")

plt.ylabel("Fallen seeds")

plt.legend(title='Crop')

plt.show() 
#subset of main ambel file, filtered for corn

corn = ambel[(ambel.type_crop == 'corn') ]  

#renaming harvest_date column to date 

corn.rename(columns={'harvest_date':'date'}, inplace=True)

corn.head(2)
# Unique harverst dates

corn.date.value_counts()
# group by date and sum fallen seeds

def date_seeds(df,col,col2):

    df = df.groupby([col])[[col2]].sum().reset_index()

    return df 



# creates columns for cumulative seed total, and for cumulative % based on cumulative sum of seeds by grouped date

def seeds_cumul(df,col,col2,col3):

    df[(col)] = df[(col2)].cumsum()

    df[(col3)] = df[(col)] / df[col2].sum() * 100 

    return df 



# Piping the 2 functions, sort of mimicking  %>%  in R

corn_fallen = (corn.pipe(date_seeds, col='date',col2 ='fallen_seeds')

        .pipe(seeds_cumul, col='cumul_fallen_seeds', col2='fallen_seeds', col3='perc_fallen_seeds'))



corn_fallen 
plt.figure(figsize=(10, 6))

ax = sns.lineplot(x = "date", y = "cumul_fallen_seeds",  data = corn_fallen)

ax.set_title("Cumulative Fallen seeds by date")

plt.xlabel("Date")

plt.ylabel("Cumulative Fallen seeds")

plt.show()
soil_temp = pd.read_csv("../input/corn2014_soil_temp.csv",encoding='utf-8') 

soil_temp.columns = soil_temp.columns.str.strip().str.lower()

soil_temp['date'] = pd.to_datetime(soil_temp['date'])

soil_temp['month'] = soil_temp['date'].dt.month 



print("rows, columns: "+str(soil_temp.shape))

print(len(str(soil_temp.shape))*'-')

print(soil_temp.dtypes.value_counts())

print(len(str(soil_temp.shape))*'-')

soil_temp.head(2)
ax = sns.lineplot(x = "month", y = "temp", data = soil_temp)

ax.set_title("Soil temperature by month for Corn")

plt.xlabel("Month")

plt.ylabel("Soil temperature")

plt.show()
# min and max temperature by day in soil_temp for corn in 2014

maxtemp = soil_temp.groupby('date')['temp'].max().reset_index() 

mintemp = soil_temp.groupby('date')['temp'].min().reset_index()   

 
merged_temp = pd.merge(maxtemp, mintemp, on="date", suffixes=("_max","_min"), copy=True)

merged_temp['GDD_day'] = (((merged_temp.temp_max + merged_temp.temp_min) /2)-5)

merged_temp['cumul_GDD'] =  merged_temp.GDD_day.cumsum()

merged_temp['month'] = merged_temp['date'].dt.month

 

print("rows, columns: "+str(merged_temp.shape))

print(len(str(merged_temp.shape))*'-')

print(merged_temp.dtypes.value_counts())

print(len(str(merged_temp.shape))*'-')

merged_temp.head(2)
ax = sns.lineplot(x = "month", y = "cumul_GDD", data = merged_temp)

ax.set_title("Cumulative GDD by month Corn")

plt.xlabel("Month")

plt.ylabel("Cumulative GDD")

plt.show()

 
# drop columns that are of no more use in merge_temp

merged_temp.drop(['temp_max', 'temp_min','GDD_day'], axis=1, inplace=True)



corn_fallen_merge = pd.merge(corn_fallen, merged_temp, on="date") 

corn_fallen_merge.head(2)
## regression plot

ax = sns.lmplot(x = "cumul_GDD", y = "perc_fallen_seeds", data = corn_fallen_merge)

plt.xlabel("Cumulative GDD")

plt.ylabel("Fallen Seeds(%)")

plt.show()
from sklearn.linear_model import LinearRegression



# construct our linear regression model

model = LinearRegression(fit_intercept=True)

x = corn_fallen_merge.cumul_GDD

y = corn_fallen_merge.perc_fallen_seeds



# fit our model to the data

model.fit(x[:, np.newaxis], y)



#

xfit = np.linspace(1300, 2500, 12)

yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)

plt.plot(xfit, yfit);
print("Slope:    ", model.coef_[0])

print("Intercept:", model.intercept_)
y_predict = model.predict(x.values.reshape(-1,1))

print("RMSE:", sqrt(((y-y_predict)**2).values.mean()))
ax = sns.residplot(x = "cumul_GDD", y= "perc_fallen_seeds", data = corn_fallen_merge , lowess = True)

ax.set(ylabel='Observed - Prediction')

plt.xlabel("Cumulative GDD")

plt.ylabel("Observed - Prediction")

plt.show()

plt.show()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline, make_pipeline



model_1 = LinearRegression(fit_intercept=True)

x = corn_fallen_merge.cumul_GDD

y = corn_fallen_merge.perc_fallen_seeds



model_1 = make_pipeline(PolynomialFeatures(degree = 5),LinearRegression())

model_1.fit(x[:, np.newaxis], y)

plt.figure(figsize = (8,5))

plt.scatter(x,y, alpha = .6, label = 'GDD')

yfit = model_1.predict(xfit[:, np.newaxis])

plt.plot(x,yfit, color = 'red', label = 'Polynomial')

plt.ylim(-2, 110)

plt.title('Polynomial degree 5')

plt.xlabel('GDD'), plt.ylabel('Fallen Seeds %')

plt.legend(), plt.show()
from sklearn.metrics import mean_squared_error

y_pred = model_1.predict(x.values.reshape(-1,1))

print("RMSE:", sqrt(mean_squared_error(y, y_pred)))