import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Plots

import folium

from folium import features

from folium.plugins import HeatMap

from folium.plugins import MarkerCluster

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# For Prediction

from fastai.tabular import *

from fastai.tabular import add_datepart
Tweets = pd.read_csv("../input/uk-house-price/Twitter_case_study_2_dataset.csv")

Tweets.drop('index', axis=1, inplace=True)

Tweets.head(3)
Tweets.country.value_counts(ascending=True).tail().plot.barh();
Tweets.source.value_counts(ascending=True).tail().plot.barh();
df_map = folium.Map(location=[54.523293, -1.539852], zoom_start=3)

data = [[x[0], x[1], 1] for x in np.array(Tweets[['latitude', 'longitude']])]

HeatMap(data, radius = 20).add_to(df_map)

df_map
'''map_wb = folium.Map(location=[54.523293, -1.539852],zoom_start=1)

mc = MarkerCluster()

for ind,row in Tweets.iterrows():

    mc.add_child(folium.CircleMarker(location=[row['latitude'],row['longitude']],

                        radius=1,color='#3185cc'))

map_wb.add_child(mc)

'''
Tweets= Tweets[Tweets['latitude']>48.8566]

df_map = folium.Map(location=[54.523293, -1.539852], zoom_start=6)

data = [[x[0], x[1], 1] for x in np.array(Tweets[['latitude', 'longitude']])]

HeatMap(data, radius = 20).add_to(df_map)

df_map
Tweets['date'] = pd.to_datetime(Tweets['date'])

Tweets.head()
Cities = pd.read_csv("../input/uk-house-price/worldcities.csv")

Cities= Cities[Cities['country'] == 'United Kingdom']

Cities.head(3)
Cities.drop('city', axis=1, inplace=True)

Cities.drop('country', axis=1, inplace=True)

Cities.drop('iso2', axis=1, inplace=True)

Cities.drop('iso3', axis=1, inplace=True)

Cities.drop('admin_name', axis=1, inplace=True)

#Cities.drop('capital', axis=1, inplace=True)

Cities.drop('id', axis=1, inplace=True)

Cities['city_ascii']=Cities['city_ascii'].str.upper()

Cities.head(3)




def Count_Tweets(lat_city,long_city,ck):

    Tweets_tmp = Tweets[(Tweets['latitude']<=lat_city+ck) & (Tweets['latitude']>=lat_city-ck) & (Tweets['longitude']>=long_city-ck) & (Tweets['longitude']<=long_city+ck) ]

   #print(Tweets_tmp.shape[0])

    return Tweets_tmp.shape[0]



# Number of Tweets within 1 Degree Lat, Long

ck=1

Cities.loc[:,'Tweets_Count_1']=0

Cities['Tweets_Count_1'] = Cities.apply(lambda row: Count_Tweets(row['lat'], row['lng'], ck), axis=1)



# Number of Tweets within 0.5 Degree Lat, Long

ck=0.5

Cities.loc[:,'Tweets_Count_05']=0

Cities['Tweets_Count_05'] = Cities.apply(lambda row: Count_Tweets(row['lat'], row['lng'], ck), axis=1)



# Number of Tweets within 0.25 Degree Lat, Long

ck=0.25

Cities.loc[:,'Tweets_Count_025']=0

Cities['Tweets_Count_025'] = Cities.apply(lambda row: Count_Tweets(row['lat'], row['lng'], ck), axis=1)



# Number of Tweets within 0.1 Degree Lat, Long

ck=0.01

Cities.loc[:,'Tweets_Count_001']=0

Cities['Tweets_Count_001'] = Cities.apply(lambda row: Count_Tweets(row['lat'], row['lng'], ck), axis=1)





Cities.head(3)
df_map = folium.Map(location=[54.523293, -1.539852], zoom_start=5)

data = [[x[0], x[1], x[2]] for x in np.array(Cities[['lat', 'lng','Tweets_Count_05']])]

HeatMap(data, radius = 20).add_to(df_map)

df_map
Paid_Price = pd.read_csv("../input/uk-house-price/price_paid_records.csv")

Paid_Price=Paid_Price.sample(frac = 0.05) 

Paid_Price.head(3)
Paid_Price['Date of Transfer'] = pd.to_datetime(Paid_Price['Date of Transfer'])

add_datepart(Paid_Price, 'Date of Transfer')

Paid_Price.dtypes
len(Paid_Price.drop_duplicates())
sns.boxplot(y = Paid_Price['Price'])

plt.title('Price')
print(len(Paid_Price))

Paid_Price = Paid_Price.loc[(Paid_Price['Price'] < (500000)) & (Paid_Price['Price'] > (10000))]

print(len(Paid_Price))

sns.boxplot(y = Paid_Price['Price'])

plt.title('Price')
#Let us take a quick exploratory look at the distribution of house prices. We see that the majority of house prices across all years is less than £500,000.

f, ax = plt.subplots(figsize=(8, 7))

#Paid_Price=Paid_Price[Paid_Price['Price']<500000]

Paid_Price['Price'].hist()

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="SalePrice")

ax.set(title="SalePrice distribution")
data = pd.concat([Paid_Price['Price'], Paid_Price['County']], axis=1)

f, ax = plt.subplots(figsize=(600, 12))

fig = sns.boxplot(x=Paid_Price['County'], y="Price", data=data)
Paid_Price_series=Paid_Price.groupby(['Date of TransferYear'])['Price'].mean().reset_index()

Paid_Price_series.set_index('Date of TransferYear', inplace=True)

#Paid_Price_series=Paid_Price_series[Paid_Price_series['Price']<500000]

ax=Paid_Price_series.plot(figsize= (8,6),title = 'Mean Price Variation over Years')#, xlabel='Years',ylabel = 'Mean Price Variation over Years')



# Set the x-axis label

ax.set_xlabel("Year")



# Set the y-axis label

ax.set_ylabel("Mean House Price over Years")
Paid_Price_series=Paid_Price.groupby(['Date of TransferMonth'])['Price'].mean().reset_index()

Paid_Price_series.set_index('Date of TransferMonth', inplace=True)

#Paid_Price_series=Paid_Price_series[Paid_Price_series['Price']<500000]

ax=Paid_Price_series.plot(figsize= (8,6),title = 'Mean Price Variation over Month')#, xlabel='Years',ylabel = 'Mean Price Variation over Years')



# Set the x-axis label

ax.set_xlabel("Months")



# Set the y-axis label

ax.set_ylabel("Mean House Price over Months")
#Date of TransferDayofweek             int64

Paid_Price_series=Paid_Price.groupby(['Date of TransferDayofweek'])['Price'].mean().reset_index()

Paid_Price_series.set_index('Date of TransferDayofweek', inplace=True)

#Paid_Price_series=Paid_Price_series[Paid_Price_series['Price']<500000]

ax=Paid_Price_series.plot(figsize= (8,6),title = 'Mean Price Variation over Day of Week')#, xlabel='Years',ylabel = 'Mean Price Variation over Years')



# Set the x-axis label

ax.set_xlabel("Day of Week")



# Set the y-axis label

ax.set_ylabel("Mean House Price over Day of Week")
#Date of TransferIs_month_end

#Date of TransferIs_month_start

#Date of TransferIs_quarter_end

#Date of TransferIs_quarter_start





Paid_Price_series=Paid_Price.groupby(['Date of TransferIs_month_end'])['Price'].mean().reset_index()

Paid_Price_series.set_index('Date of TransferIs_month_end', inplace=True)

#Paid_Price_series=Paid_Price_series[Paid_Price_series['Price']<500000]

ax=Paid_Price_series.plot(figsize= (8,6),title = 'Mean Price Variation Month End')#, xlabel='Years',ylabel = 'Mean Price Variation over Years')



# Set the x-axis label

ax.set_xlabel("Month End")



# Set the y-axis label

ax.set_ylabel("Mean House Price over Month End")
#Date of TransferIs_month_end

#Date of TransferIs_month_start

#Date of TransferIs_quarter_end

#Date of TransferIs_quarter_start





Paid_Price_series=Paid_Price.groupby(['Date of TransferIs_quarter_end'])['Price'].mean().reset_index()

Paid_Price_series.set_index('Date of TransferIs_quarter_end', inplace=True)

#Paid_Price_series=Paid_Price_series[Paid_Price_series['Price']<500000]

ax=Paid_Price_series.plot(figsize= (8,6),title = 'Mean Price Variation over Quater End')#, xlabel='Years',ylabel = 'Mean Price Variation over Years')



# Set the x-axis label

ax.set_xlabel("Quater End")



# Set the y-axis label

ax.set_ylabel("Mean House Price over Quater End")
print(Paid_Price.shape)

Paid_Price_Tweet= pd.merge (Paid_Price, Cities , left_on= 'Town/City' ,  right_on = 'city_ascii', how='left')

print(Paid_Price_Tweet.shape)

Paid_Price_Tweet.head(3)
Paid_Price_Tweet.drop('Transaction unique identifier', axis=1, inplace=True)

Paid_Price_Tweet.drop('city_ascii', axis=1, inplace=True)

Paid_Price_Tweet.drop('Date of TransferElapsed', axis=1, inplace=True)
#print(Paid_Price.shape)

#Paid_Price= pd.merge (Paid_Price, Cities , left_on= 'Town/City' ,  right_on = 'city_ascii', how='left')

#print(Paid_Price.shape)


# Find Numerical and Categorical variables

Paid_Price_Tweet['Price'] = Paid_Price_Tweet['Price'].astype(float)

Paid_Price_Tweet.dtypes
#Defining the keyword arguments for fastai's TabularList



train_data = Paid_Price_Tweet

dep_var = 'Price'



cat_names = ['Property Type', 'Old/New', 'Duration', 'Town/City',

            'District', 'County', 'PPDCategory Type',

            'Record Status - monthly file only',

            'Date of TransferIs_month_end', 'Date of TransferIs_month_start',

            'Date of TransferIs_quarter_end', 'Date of TransferIs_quarter_start',

            'Date of TransferIs_year_end', 'Date of TransferIs_year_start']





cont_names = ['Date of TransferYear',	'Date of TransferMonth',	

              'Date of TransferWeek',	'Date of TransferDay',	

              'Date of TransferDayofweek',	'Date of TransferDayofyear',

              'lat',	'lng','population',	'Tweets_Count_1',	'Tweets_Count_05',

              'Tweets_Count_025',	'Tweets_Count_001'

              ]

path =''





train_data = train_data[cat_names + cont_names + [dep_var]]



#List of Processes/transforms to be applied to the dataset

procs = [FillMissing, Categorify, Normalize]



#Start index for creating a validation set from train_data

start_indx = len(train_data) - int(len(train_data) * 0.2)



#End index for creating a validation set from train_data

end_indx = len(train_data)





#TabularList for Validation

test = (TabularList.from_df(train_data.iloc[start_indx:end_indx].copy(), path=path, cat_names=cat_names, cont_names=cont_names))



#test = val





#TabularList for training

data = (TabularList.from_df(train_data, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_rand_pct(valid_pct = 0.1)

                           #.split_by_idx(list(range(start_indx,end_indx)))

                           .label_from_df(cols=dep_var, label_cls=FloatList)

                           .add_test(test)

                           .databunch())

max_y = np.max(train_data['Price'])*1.2

y_range = torch.tensor([0, max_y], device=defaults.device)

y_range
learn = tabular_learner(data, layers=[600,300], ps=[0.001,0.01], emb_drop=0.04, y_range=y_range, metrics=[rmse,r2_score])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(1, 5e-2, wd=0.2)
learn.save("Learn_Tabular")

learn.load("Learn_Tabular")

print()
## Not Executed due to high execution time - uncommemnt if required

'''



dep_var = 'Price'



cat_names = ['Property Type', 'Old/New', 'Duration', 'Town/City',

            'District', 'County', 'PPDCategory Type',

            'Record Status - monthly file only',

            'Date of TransferIs_month_end', 'Date of TransferIs_month_start',

            'Date of TransferIs_quarter_end', 'Date of TransferIs_quarter_start',

            'Date of TransferIs_year_end', 'Date of TransferIs_year_start']





cont_names = ['Date of TransferYear',	'Date of TransferMonth',	

              'Date of TransferWeek',	'Date of TransferDay',	

              'Date of TransferDayofweek',	'Date of TransferDayofyear',

              'lat',	'lng','population',	'Tweets_Count_1',	'Tweets_Count_05',

              'Tweets_Count_025',	'Tweets_Count_001'

              ]

              

def preprocess(Paid_Price_Short):

    





    Paid_Price_Short['Date of Transfer'] = pd.to_datetime(Paid_Price_Short['Date of Transfer'])

    add_datepart(Paid_Price_Short, 'Date of Transfer')

    Paid_Price_Short = Paid_Price_Short.loc[(Paid_Price_Short['Price'] < (500000)) & (Paid_Price_Short['Price'] > (10000))]





    Paid_Price_Short_Tweet= pd.merge (Paid_Price_Short, Cities , left_on= 'Town/City' ,  right_on = 'city_ascii', how='left')

    Paid_Price_Short_Tweet.drop('Transaction unique identifier', axis=1, inplace=True)

    Paid_Price_Short_Tweet.drop('city_ascii', axis=1, inplace=True)

    Paid_Price_Short_Tweet.drop('Date of TransferElapsed', axis=1, inplace=True)

    Paid_Price_Short_Tweet['Price'] = Paid_Price_Short_Tweet['Price'].astype(float)

    

    train_data = Paid_Price_Short_Tweet

    train_data = train_data[cat_names + cont_names + [dep_var]]

    data = (TabularList.from_df(train_data, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_rand_pct(valid_pct = 0.1)

                           #.split_by_idx(list(range(start_indx,end_indx)))

                           .label_from_df(cols=dep_var, label_cls=FloatList)

                           #.add_test(test)

                           .databunch())

    

    

   

    #learn = tabular_learner(data, layers=[600,300], ps=[0.001,0.01], emb_drop=0.04, y_range=y_range, metrics=[rmse,r2_score])

    #if COUNT !=0:

    learn.data = data

    learn.load("Learn_Tabular",strict=False,remove_module=True)

    



    #learn.fit_one_cycle(1, 5e-2, wd=0.2)

    learn.save("Learn_Tabular")

    #print(COUNT)

    #COUNT=COUNT+1





reader  = pd.read_csv("/content/drive/My Drive/10FA/price_paid_records.csv", chunksize=65536) # chunksize depends with you RAM

[preprocess(r) for r in reader]

'''
perc_na = (Paid_Price_Tweet.isnull().sum()/len(Paid_Price_Tweet))*100

ratio_na = perc_na.sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Percentage' :ratio_na})

missing_data


# Apply per-column median of that columns and fill with that value for Numerical Variables



numeric_cols=['Date of TransferYear',	'Date of TransferMonth',	

              'Date of TransferWeek',	'Date of TransferDay',	

              'Date of TransferDayofweek',	'Date of TransferDayofyear',

              'lat',	'lng','population',	'Tweets_Count_1',	'Tweets_Count_05',

              'Tweets_Count_025',	'Tweets_Count_001'

              ]



Paid_Price_Tweet[numeric_cols] = Paid_Price_Tweet[numeric_cols].apply(lambda x: x.fillna(x.median()),axis=0)

# Apply per-column Highest Frequency value to address Missing variables

# Convert Non numerical columns to string



for c_cols in  ['Property Type', 'Old/New', 'Duration', 'Town/City',

            'District', 'County', 'PPDCategory Type',

            'Record Status - monthly file only',

            'Date of TransferIs_month_end', 'Date of TransferIs_month_start',

            'Date of TransferIs_quarter_end', 'Date of TransferIs_quarter_start',

            'Date of TransferIs_year_end', 'Date of TransferIs_year_start']:



    

    #train[c_cols] = train[c_cols].cat.add_categories('Unknown')

    Paid_Price_Tweet=Paid_Price_Tweet.fillna(Paid_Price_Tweet[c_cols].value_counts().index[0])

    

    Paid_Price_Tweet[c_cols]=Paid_Price_Tweet[c_cols].astype(str)


Paid_Price_Tweet= Paid_Price_Tweet.drop([

'population',

], axis=1)
for c_cols in  ['Property Type', 'Old/New', 'Duration', 'Town/City',

            'District', 'County', 'PPDCategory Type',

            'Record Status - monthly file only',

            'Date of TransferIs_month_end', 'Date of TransferIs_month_start',

            'Date of TransferIs_quarter_end', 'Date of TransferIs_quarter_start',

            'Date of TransferIs_year_end', 'Date of TransferIs_year_start']:

    print(c_cols, Paid_Price_Tweet[c_cols].nunique() )


Paid_Price_Tweet= Paid_Price_Tweet.drop([

'Town/City',

 'District',

 'County'



], axis=1)
Paid_Price_Tweet['Property Type']=Paid_Price_Tweet['Property Type'].astype('category')

Paid_Price_Tweet['Old/New']=Paid_Price_Tweet['Old/New'].astype('category')

Paid_Price_Tweet['Duration']=Paid_Price_Tweet['Duration'].astype('category')

#Paid_Price_Tweet['Town/City']=Paid_Price_Tweet['Town/City'].astype('category')

#Paid_Price_Tweet['District ']=Paid_Price_Tweet['District '].astype('category')

#Paid_Price_Tweet['County']=Paid_Price_Tweet['County'].astype('category')

Paid_Price_Tweet['PPDCategory Type']=Paid_Price_Tweet['PPDCategory Type'].astype('category')

Paid_Price_Tweet['Record Status - monthly file only']=Paid_Price_Tweet['Record Status - monthly file only'].astype('category')

Paid_Price_Tweet['Date of TransferIs_month_end']=Paid_Price_Tweet['Date of TransferIs_month_end'].astype('category')

Paid_Price_Tweet['Date of TransferIs_month_start']=Paid_Price_Tweet['Date of TransferIs_month_start'].astype('category')

Paid_Price_Tweet['Date of TransferIs_quarter_end']=Paid_Price_Tweet['Date of TransferIs_quarter_end'].astype('category')

Paid_Price_Tweet['Date of TransferIs_quarter_start']=Paid_Price_Tweet['Date of TransferIs_quarter_start'].astype('category')

Paid_Price_Tweet['Date of TransferIs_year_end']=Paid_Price_Tweet['Date of TransferIs_year_end'].astype('category')

Paid_Price_Tweet['Date of TransferIs_year_start']=Paid_Price_Tweet['Date of TransferIs_year_start'].astype('category')

print(Paid_Price_Tweet.shape)

df_train=pd.get_dummies(Paid_Price_Tweet)

print(Paid_Price_Tweet.shape)
print(df_train.shape)



target1=df_train['Price']



df_train= df_train.drop(['Price'], axis=1)

df_train.shape
from sklearn.model_selection import train_test_split



# in the Random Forest method, involves training each decision tree on a different data sample

from sklearn.ensemble import RandomForestRegressor



from sklearn.metrics import r2_score, mean_squared_error # import metrics from sklearn
X_train, X_test, y_train, y_test =train_test_split(df_train, target1)
df_train.head(2)
#!pip  install scikit-learn==0.19.1


from sklearn.metrics import make_scorer, mean_squared_error

#  k-fold CV, the training set is split into k smaller sets 

#from sklearn.cross_validation import cross_val_score



scorer = make_scorer(mean_squared_error, False)



clf = RandomForestRegressor(n_estimators=20, n_jobs=-1)

# For the lack of time I have only used 20 Trees

clf


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


score_r2 = r2_score(y_test, y_pred)

score_mse = mean_squared_error(y_test, y_pred)





d = {

     'RF_Regressor': [score_r2 , score_mse] 

  

    }

d_i = ['R2', 'Mean Squared Error']

df_results = pd.DataFrame(data=d, index = d_i)

df_results



importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



#for f in range(X_train.shape[1]):

    #print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))





# Plot the impurity-based feature importances of the forest

plt.figure(figsize=(20,10))

plt.title("Feature importances")



plt.bar(range(0,9), importances[indices][0:9],

        color="r", yerr=std[indices][0:9], align="center")



#plt.bar(range(X_train.shape[1]), importances[indices],

#        color="r", yerr=std[indices], align="center")

plt.xticks(range(0,9), indices[0:9])

#plt.xlim([-1, X_train.shape[1]])

plt.show()
