import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Importing data
df = pd.read_csv ("../input/Melbourne_housing_FULL.csv")
#df.isnull().sum()
#Changing date column to a pandas datetime format and creating a new column quarter
df['Date'] = pd.to_datetime (df['Date'])
df['Quarter'] = df.Date.dt.quarter
#df_work = df.dropna().sort_values("Date")
df_work = df
from datetime import date

all_Data = []
##Find out number of days since start
days_since_start = [(x - df_work["Date"].min()).days for x in df_work["Date"]]

#Find correlations to target = price
corr_matrix = df_work.corr().abs()
#print (corr_matrix['Price'].sort_values(ascending = False).head(20))

#Visualizing the correlation matrix
#select upper traingle of correlation matrix
upper = corr_matrix.where (np.triu(np.ones(corr_matrix.shape), k = 1).astype (np.bool))
#print (corr_matrix.where (np.triu(np.ones(corr_matrix.shape), k = 1).astype (np.bool)))
sns.heatmap(upper)
plt.show()
region_south_metro = pd.DataFrame (df_work [(df_work['Type'] == 'h' ) & (df_work ['Regionname'] == "Southern Metropolitan")]) 
region_south_metro['Date']= pd.to_datetime(region_south_metro['Date'] )
region_south_metro = region_south_metro.set_index ('Date' ).resample ('M').mean().dropna()
#---------------------------
region_north_metro = pd.DataFrame (df_work [(df_work['Type'] == 'h' ) & (df_work ['Regionname'] == "Northern Metropolitan")]) 
region_north_metro['Date']= pd.to_datetime(region_north_metro['Date'] )
region_north_metro = region_north_metro.set_index ('Date' ).resample ('M').mean().dropna()
#--------------------
region_west_metro = pd.DataFrame (df_work[(df_work['Type'] == 'h' ) & (df_work ['Regionname'] == "Western Metropolitan")]) 
region_west_metro['Date']= pd.to_datetime(region_west_metro['Date'] )
region_west_metro = region_west_metro.set_index ('Date' ).resample ('M').mean().dropna()
#-------------------------
region_east_metro = pd.DataFrame (df_work [(df_work['Type'] == 'h' ) & (df_work ['Regionname'] == "Eastern Metropolitan")]) 
region_east_metro['Date']= pd.to_datetime(region_east_metro['Date'] )
region_east_metro = region_east_metro.set_index ('Date' ).resample ('M').mean().dropna()
#------------------------
region_south_east_metro = pd.DataFrame (df_work [(df_work['Type'] == 'h' ) & (df_work ['Regionname'] == "South-Eastern Metropolitan")]) 
region_south_east_metro['Date']= pd.to_datetime(region_south_east_metro['Date'] )
region_south_east_metro = region_south_east_metro.set_index ('Date' ).resample ('M').mean().dropna()
#-------------------------
region_east_vict = pd.DataFrame (df_work [(df_work['Type'] == 'h' ) & (df_work ['Regionname'] == "Eastern Victoria")]) 
region_east_vict['Date']= pd.to_datetime(region_east_vict['Date'] )
region_east_vict = region_east_vict.set_index ('Date' ).resample ('M').mean().dropna()
#------------------------------------
region_north_vict = pd.DataFrame (df_work [(df_work['Type'] == 'h' ) & (df_work ['Regionname'] == "Northern Victoria")]) 
region_north_vict['Date']= pd.to_datetime(region_north_vict['Date'] )
region_north_vict = region_north_vict.set_index ('Date' ).resample ('M').mean().dropna()
#-----------------------------------------
region_west_vict = pd.DataFrame (df_work[(df_work['Type'] == 'h' ) & (df_work ['Regionname'] == "Western Victoria")]) 
region_west_vict['Date']= pd.to_datetime(region_west_vict['Date'] )
region_west_vict = region_west_vict.set_index ('Date' ).resample ('M').mean().dropna()
region_south_metro ['Price']
fig,ax = plt.subplots(figsize = (15,8))
ax.plot (region_south_metro ['Price'] , label = "Southern Metro")
ax.plot (region_north_metro ['Price'], label = "Northern Metro")
ax.plot (region_west_metro ['Price'], label = "Western Metro")
ax.plot (region_east_metro ['Price'], label = "Eastern Metro")
ax.plot (region_south_east_metro ['Price'], label = "Sout-east Metro")
ax.plot (region_east_vict ['Price'], label = "East Vict")
ax.plot (region_north_vict ['Price'], label = "North Vict")
ax.plot (region_west_vict ['Price'], label = "West Vict")
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
suburb_northcote = pd.DataFrame (df_work [(df_work['Suburb'] == 'Northcote' ) ]) 
sns.countplot (x = 'Type' , data = suburb_northcote)
plt.title ("There are more 'h' ")
plt.show
fig, ax = plt.subplots (figsize = (10,10))
sns.catplot ( ax= ax, x = 'Type' , y = 'Price' , kind = "swarm" , data = suburb_northcote )
#ax.set_title ("The prices of the 'h' are higher than u &t , also the majority of the houses price range is in between $1000000 to $1500000 ")
plt.show()
suburb_northcote_type = pd.DataFrame(suburb_northcote.loc[(suburb_northcote["Type"] == "h") & (suburb_northcote["Rooms"] == 2)])
suburb_northcote_type['Date']= pd.to_datetime(suburb_northcote_type['Date'])
suburb_northcote_type = suburb_northcote_type.set_index ('Date'). resample ('M').mean().dropna()

plt.figure(figsize = (15,8))
plt.plot (suburb_northcote_type['Price'])
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.legend(loc='best')
plt.show()
sns.kdeplot(suburb_northcote[
           (suburb_northcote["Type"]=="u")
         & (suburb_northcote["Rooms"] == 2)]["Price"])
df_work.groupby("Distance") ["Price"].mean().plot()
plt.show()
df[(df["Rooms"]>2) & (df["Type"] == "h")& (df["Landsize"] <5000)][["Landsize","Distance"]].dropna().groupby("Distance").mean().plot()
plt.show()
df_work['Bathroom'] = df_work['Bathroom'].fillna(-1)
bins = [-10,-1,0,1,2,3,4,np.inf]

group_names = ['unknown', 'zero','single', 'double', 'triple', 'quadruple', 'five' ]
categories = pd.cut (df_work.Bathroom , bins, labels = group_names)
df.Bathroom = categories

#Next step is grouping the required features to fill the bathroom unknown values 
df_work['Car'] = df_work['Car'].fillna(-1)
bins = [-10,-1, 0,1,2,3,4, np.inf]

group_names = ['unknown', 'zero','single', 'double', 'triple', 'quadruple', 'five' ]
categories = pd.cut (df_work.Car , bins, labels = group_names)
df_work.Car = categories
print (df_work.Car.value_counts())

#Next step is grouping the required features to fill the Distance unknown values
bins = [0,2.5,5,7.5,10.5,13,15.5,17.5,20]
df_work['Distance_bins'] = np.digitize (df_work.Distance, bins)
df_work.Distance_bins.value_counts()
#Imputing unknown values of Bathroom column
bathroom_unit_u = df_work.loc [(df_work['Type'] == 'u') & (df_work['Bathroom'] == 'unknown')].Price.mean()

#From this representation it can be seen that ,amy of Unit houses have bathrooms either single or double depending on price,
#so for this purpose mean of price is calculated and for price above mean it is double bathroom and for price belo mean single.
                                                                                             
#df_bath.loc [(df_bath['Type'] == 'u') & (df_bath['Bathroom'] == 'unknown'), 'Bathroom'] = df_bath['Bathroom'].replace ('unknown', 'single') 
df_work.loc [(df_work['Type'] == 'u') & (df_work['Price'] < bathroom_unit_u), 'Bathroom'] = df_work['Bathroom'].replace ('unknown', 'single') 
df_work.loc [(df_work['Type'] == 'u') & (df_work['Price'] > bathroom_unit_u), 'Bathroom'] = df_work['Bathroom'].replace ('unknown', 'double')

#For TYPE "h" replacing nan values
bathroom_unit_h = df_work.loc [(df_work['Type'] == 'h') & (df_work['Bathroom'] == 'unknown')].Price.mean()
#bathroom_unit_h
#df_bath.loc [(df_bath['Type'] == 'h') & (df_bath['Price'] < bathroom_unit_h) & (df_bath['Bathroom'] == 'unknown') & (df_bath['Distance_bins'] < 6 ), 'Bathroom']
df_work.loc [(df_work['Type'] == 'h') & (df_work['Price'] < bathroom_unit_h) & (df_work['Distance_bins'] <= 6 ) & (df_work['Bathroom'] == 'unknown'), 'Bathroom']  = df_work['Bathroom'].replace ('unknown', 'single')
df_work.loc [(df_work['Type'] == 'h') & (df_work['Price'] > bathroom_unit_h) & (df_work['Distance_bins'] >= 6 ) & (df_work['Bathroom'] == 'unknown'), 'Bathroom']  = df_work['Bathroom'].replace ('unknown', 'double')
df_work.loc [(df_work['Type'] == 'h') & (df_work['Price'] < bathroom_unit_h) & (df_work['Distance_bins'] >= 6 ) & (df_work['Bathroom'] == 'unknown'), 'Bathroom']  = df_work['Bathroom'].replace ('unknown', 'double')
df_work.loc [(df_work['Type'] == 'h') & (df_work['Price'] > bathroom_unit_h) & (df_work['Distance_bins'] <= 6 ) & (df_work['Bathroom'] == 'unknown'), 'Bathroom']  = df_work['Bathroom'].replace ('unknown', 'double')
#For TYPE "h" replacing nan values
bathroom_unit_h = df_work.loc [(df_work['Type'] == 't') & (df_work['Bathroom'] == 'unknown')].Price.mean()
#bathroom_unit_h
#df_bath.loc [(df_bath['Type'] == 'h') & (df_bath['Price'] < bathroom_unit_h) & (df_bath['Bathroom'] == 'unknown') & (df_bath['Distance_bins'] < 6 ), 'Bathroom']
df_work.loc [(df_work['Type'] == 't') & (df_work['Price'] < bathroom_unit_h) & (df_work['Distance_bins'] <= 6 ) & (df_work['Bathroom'] == 'unknown'), 'Bathroom']  = df_work['Bathroom'].replace ('unknown', 'single')
df_work.loc [(df_work['Type'] == 't') & (df_work['Price'] > bathroom_unit_h) & (df_work['Distance_bins'] >= 6 ) & (df_work['Bathroom'] == 'unknown'), 'Bathroom']  = df_work['Bathroom'].replace ('unknown', 'double')
df_work.loc [(df_work['Type'] == 't') & (df_work['Price'] < bathroom_unit_h) & (df_work['Distance_bins'] >= 6 ) & (df_work['Bathroom'] == 'unknown'), 'Bathroom']  = df_work['Bathroom'].replace ('unknown', 'double')
df_work.loc [(df_work['Type'] == 't') & (df_work['Price'] > bathroom_unit_h) & (df_work['Distance_bins'] <= 6 ) & (df_work['Bathroom'] == 'unknown'), 'Bathroom']  = df_work['Bathroom'].replace ('unknown', 'double')
#Lets see NaN values for regionname
df_work ['Regionname']= df_work['Regionname'].replace (np.nan, 0)
suburb_region = df_work.loc [df_work ['Regionname'] == 0, 'Suburb'].reset_index(False)
suburb_region
for i in range (len (suburb_region['Suburb'])):
    print (i)
    regionname1 = df_work.loc [df_work['Suburb'] == suburb_region['Suburb'][i], 'Regionname' ].value_counts().idxmax()
    print (suburb_region['Suburb'][i])
    print (regionname1)
    
    #df_bath['Regionname'] = df_bath['Regionname'].replace (0,regionname1)
    df_work.loc [(df_work['Suburb'] == suburb_region['Suburb'][i]) & (df_work ['Regionname'] == 0), "Regionname"] = df_work['Regionname'].replace (0,regionname1)
    
df_work [df_work['Regionname'] == 0]
#Still there is one unfilled region name. For now I fiill it with most frequent regionname
regionname_remain = df_work[(df_work ['SellerG'] == "Brad") & (df_work ['Price'] < 700000) & (df_work ['Price'] < 600000) & (df_work ['Type'] == "h")]['Regionname'].value_counts().idxmax()
df_work.loc [df_work['Regionname'] == 0, "Regionname"] = df['Regionname'].replace (0,regionname_remain) 
df_work [df_work['Regionname'] == 0]
df_work.Car.value_counts()
#Now imputing the values of car column
car_southern_metro = df_work.loc [(df_work ['Regionname'] == "Southern Metropolitan")].Price.mean()
df_work.loc [(df_work['Regionname'] == 'Southern Metropolitan') & (df_work['Price'] < car_southern_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'single')
df_work.loc [(df_work['Regionname'] == 'Southern Metropolitan') & (df_work['Price'] > car_southern_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'Southern Metropolitan') & (df_work['Price'] > car_southern_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'Southern Metropolitan') & (df_work['Price'] < car_southern_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')

#----------------------------
car_northern_metro = df_work.loc [(df_work ['Regionname'] == "Northern Metropolitan")].Price.mean()
df_work.loc [(df_work['Regionname'] == 'Northern Metropolitan') & (df_work['Price'] < car_northern_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'single')
df_work.loc [(df_work['Regionname'] == 'Northern Metropolitan') & (df_work['Price'] > car_northern_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'Northern Metropolitan') & (df_work['Price'] > car_northern_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'Northern Metropolitan') & (df_work['Price'] < car_northern_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
#----------------------------------------
car_western_metro = df_work.loc [(df_work ['Regionname'] == "Western Metropolitan")].Price.mean()
df_work.loc [(df_work['Regionname'] == 'Western Metropolitan') & (df_work['Price'] < car_western_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'single')
df_work.loc [(df_work['Regionname'] == 'Western Metropolitan') & (df_work['Price'] > car_western_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'Western Metropolitan') & (df_work['Price'] > car_western_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'Western Metropolitan') & (df_work['Price'] < car_western_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
#-------------------------------------------------------
car_east_metro = df_work.loc [(df_work ['Regionname'] == "Eastern Metropolitan")].Price.mean()
df_work.loc [(df_work['Regionname'] == 'Eastern Metropolitan') & (df_work['Price'] < car_east_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'single')
df_work.loc [(df_work['Regionname'] == 'Eastern Metropolitan') & (df_work['Price'] > car_east_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'Eastern Metropolitan') & (df_work['Price'] > car_east_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'Eastern Metropolitan') & (df_work['Price'] < car_east_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
#-------------------------------------------------------
car_souteast_metro = df_work.loc [(df_work ['Regionname'] == "South-Eastern Metropolitan")].Price.mean()
df_work.loc [(df_work['Regionname'] == 'South-Eastern Metropolitan') & (df_work['Price'] < car_souteast_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'single')
df_work.loc [(df_work['Regionname'] == 'South-Eastern Metropolitan') & (df_work['Price'] > car_souteast_metro) & (df_work['Distance_bins'] <= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'South-Eastern Metropolitan') & (df_work['Price'] > car_souteast_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work.loc [(df_work['Regionname'] == 'South-Eastern Metropolitan') & (df_work['Price'] < car_souteast_metro) & (df_work['Distance_bins'] >= 6 ) & (df_work['Car'] == 'unknown'), 'Car']  = df_work['Car'].replace ('unknown', 'double')
df_work['Car'] = df_work['Car'].replace ('unknown' , 'double')
df_work = df_work[np.isfinite(df_work['Price'])]   # This is to drop only values in Price column that are null or na
suburb_dummies = pd.get_dummies(df_work[["Type", "Method", "Bathroom", "Car"]], drop_first = True)
all_Data = df_work.drop(["Address","Price","Date", "Bedroom2","Distance",  "Lattitude","Longtitude","Landsize", "Propertycount", "BuildingArea", "YearBuilt", "SellerG","Suburb","Type",'Bathroom', 'Car' ,"Method","CouncilArea","Regionname"],axis=1).join(suburb_dummies)
all_Data.dtypes
print (all_Data.keys())
print (all_Data.shape)
X = all_Data.dropna (axis = 1)
X.dtypes
y = df_work["Price"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
X_train.dtypes
lm.fit (X_train, y_train)
print (lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
ranked_suburbs = coeff_df.sort_values("Coefficient", ascending = False)
ranked_suburbs

X_test.fillna (method = 'ffill' )
X_test.isnull().sum()
predictions = lm.predict (X_test)
from sklearn import metrics
print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
#Fit parameters by Ridge regression with gridsearchcv
alphas = np.array ([1,0.1,0.01,0.001,0.0001,0])
fit_interceptOptions = ([True])
solverOptions = ([ 'svd', 'sparse_cg', 'cholesky'])
model = Ridge (normalize=False) 
grid = GridSearchCV(estimator = model,  param_grid=dict(alpha=alphas, fit_intercept=fit_interceptOptions, solver=solverOptions)                    )
grid.fit (X_train, y_train)
print (grid)
#summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.alpha)
print(grid.best_estimator_.fit_intercept)
print(grid.best_estimator_.solver)

