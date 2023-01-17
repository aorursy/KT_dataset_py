import geopandas as gpd

import matplotlib.pyplot as plt

import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns # visualization

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

import eli5

from eli5.sklearn import PermutationImportance

from eli5 import show_weights

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

import re

from collections import Counter

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets

from IPython.display import Image  

from sklearn import tree

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.
#creates the bike dataframe from the csv file

bike_file_path = '../input/austin-bike/austin_bikeshare_trips.csv'

bike_data = pd.read_csv(bike_file_path)

#creates the station dataframe from the csv file

station_file_path = '../input/austin-bike/austin_bikeshare_stations.csv'

station_data = pd.read_csv(station_file_path)

#creates copies of the dataframes

cbd = bike_data.copy()

csd = station_data.copy()
cbd.head()
csd.head()
#prints out the station dataframe

csd
#prints out bike dataframe

cbd
cbd.dtypes
print("Columns with Number of Missing Entries(bike_data):")

print(cbd.isnull().sum())

print("Columns with Number of Missing Entries(station_data):")

print(csd.isnull().sum())
x = 0

Date=[]

while x<len(cbd):

    Date.append(cbd.start_time[x][8:10])

    x = x+1
cbd['date'] = Date

cbd["date"] = cbd["date"].astype('int8')

cbd
#dropping start/end IDs

#Since there are no missing start/end names, will encode those to fill previously missing IDs

del cbd['end_station_id']

del cbd['start_station_id']



cbd = cbd[np.isfinite(cbd['month'])]

cbd
for category in cbd:

    cbd[category]=cbd[category].astype('category')

cbd.dtypes
cbd["start_code"] = cbd["start_station_name"].cat.codes

cbd["end_code"] = cbd["end_station_name"].cat.codes

cbd["sub_code"] = cbd["subscriber_type"].cat.codes

print(cbd.isnull().sum())
#drops .4%

cbd = cbd.dropna(axis = 0)

cbd
cbd.checkout_time

#for time in cbd.checkout_time:

time_list = [time[:2] for time in cbd.checkout_time]

time_list
new_times = [time.replace(":","") for time in time_list]

new_times
counted_times = Counter(new_times)

counted_times.most_common()
cbd["start_hour"] = new_times

cbd
cbd['month_name']=cbd['month'].astype('str')







cbd['month_name'] = cbd['month_name'].map({'1.0':'January','2.0':'February','3.0':'March','4.0':'April','5.0':'May','6.0':'June','7.0':'July','8.0':'August','9.0':'September','10.0':'October','11.0':'November','12.0':'December'})

cbd.month_name
cbd = cbd[['bikeid','trip_id','start_station_name','end_station_name','start_time','checkout_time','start_hour','date','month','month_name','year','subscriber_type','duration_minutes','start_code','end_code','sub_code']]

cbd.head()
sns.set(font_scale = 1.5)

g=sns.catplot(x="start_hour",

              kind="count",

              palette="twilight",

              data = cbd,

              order = ["1","2","3","4","5","6","7","8","9","10",

                       "11","12","13","14","15","16","17","18",

                       "19","20","21","22","23","24"])



g.fig.set_size_inches(30,10)

g.fig.suptitle("Start Times", fontsize=40)

plt.ylabel("Count", fontsize = 30)

plt.xlabel("Time", fontsize = 30)



plt.show()
sns.set(font_scale = 1.5)

g=sns.catplot(x="start_station_name",kind="count", palette="Spectral",data = cbd,order=pd.value_counts(cbd['start_station_name']).iloc[:100].index)

g.set_xticklabels(rotation=90)



g.fig.set_size_inches(30,10)

g.fig.suptitle('Top Starting Stations', fontsize=40)

plt.ylabel("Count", fontsize = 30)

plt.xlabel("Stations", fontsize = 30)





plt.show()
sns.set(font_scale = 1.5)

g=sns.catplot(x="end_station_name",kind="count", palette="Spectral",data = cbd,order=pd.value_counts(cbd['end_station_name']).iloc[:100].index)

g.set_xticklabels(rotation=90)



g.fig.set_size_inches(30,10)

g.fig.suptitle('Top Ending Stations', fontsize=40)

plt.ylabel("Count", fontsize = 30)

plt.xlabel("Stations", fontsize = 30)





plt.show()
g = sns.catplot(x="start_station_name",

                hue="month_name",

                kind="count",

                palette="Spectral",

                edgecolor=".2",

                data=cbd,

                height=7,

                aspect =2,

                order=pd.value_counts(cbd['start_station_name']).iloc[:15].index,

                hue_order =["January","February","March","April","May","June","July","August","September","October",

                            "November","December"] )

plt.ylabel('Count',fontsize=30)

plt.xlabel('Stations',fontsize=30)

plt.yticks(fontsize=20)

plt.xticks(fontsize=20)

plt.title('Top 15 Stations by month',fontsize=50)



g.set_xticklabels(rotation=90)
top_cbd = cbd[cbd['start_station_name']=='Riverside @ S. Lamar']

sns.set_style("dark")

plt.figure(figsize=(20, 10))

plt.rc('xtick', labelsize=20)

plt.rc('ytick', labelsize=20)

g = sns.countplot(x="month_name", data=top_cbd,palette="Spectral",order =["January","February","March","April","May","June","July","August","September","October",

                            "November","December"])

plt.ylabel("Count", fontsize=30)

plt.xlabel("Month", fontsize=30)



plt.title("Top Station(Riverside @ S. Lamar) Popularity by Month", fontsize=40)

plt.xticks(rotation =90)

plt.show()
sns.set(font_scale = 1.5)

g=sns.catplot(x="subscriber_type",kind="count", palette="Spectral",data = cbd,order=pd.value_counts(cbd['subscriber_type']).iloc[:6].index)

g.set_xticklabels(rotation=90)



g.fig.set_size_inches(30,10)

g.fig.suptitle('Top Subscriber Types', fontsize=40)

plt.ylabel("Count", fontsize = 30)

plt.xlabel("Subscriber Type", fontsize = 30)





plt.show()
list_of_months = ['January','February','March','April','May','June',

                  'July','August','September','October','November','December']
index = 0

data1=pd.DataFrame()

data2=pd.DataFrame()

data3=pd.DataFrame()

data4=pd.DataFrame()

data5=pd.DataFrame()

data6=pd.DataFrame()

data7=pd.DataFrame()

data8=pd.DataFrame()

data9=pd.DataFrame()

data10=pd.DataFrame()

data11=pd.DataFrame()

data12=pd.DataFrame()

data13=pd.DataFrame()

dfs = [data1,data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13]



while index < len(dfs):

    dfs[index] = cbd[cbd['month'] == index]

    index = index + 1
sns.set_style("dark")

fig, axes = plt.subplots(nrows=4, 

                         ncols=3, 

                         figsize=(30,30),

                         )

index2 = 0

index = 1

row = 0

col = 0

while( index<len(dfs)):

    y = dfs[index]['date'].value_counts()

    

    

    x = y.index

    

    axes[row,col].bar(x,y)

    axes[row,col].set_title('Bike Trips by Day of The Month: ' + list_of_months[index2])

    index = index + 1

    index2 = index2 +1

    if col == 2:

        col = 0

        row = row + 1

        

    elif col != 2:

        col = col + 1
index = 0

data1=pd.DataFrame()

data2=pd.DataFrame()

data3=pd.DataFrame()

data4=pd.DataFrame()

data5=pd.DataFrame()



year = 2013

dfs = [data1,data2, data3, data4, data5]



while index < len(dfs):

    dfs[index] = cbd[cbd['year'] == year]

    index = index + 1

    year = year + 1
sns.set_style("dark")

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(30,30))

year = 2013

index = 0

row = 0

col = 0

while( index<len(dfs)):

    y = dfs[index]['month'].value_counts()

    x = y.index

    axes[row,col].bar(x,y)

    axes[row,col].set_title("{}{}".format("Month Spread in Year: ",year))

    index = index + 1

    year = year + 1

    if col == 1:

        col = 0

        row = row + 1

        

    elif col != 1:

        col = col + 1
def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='500px')
# Create a map

# Create a base map

m_4 = folium.Map(location=[30.2672,-97.7431], tiles='cartodbpositron', zoom_start=13)



def color_producer(STATUS):

    if STATUS == "active":

        return 'forestgreen'

    else:

        return 'darkred'



# Add a bubble map to the base map

for i in range(0,len(station_data)):

    Circle(

        location=[station_data.iloc[i]['latitude'], station_data.iloc[i]['longitude']],

        popup=(station_data.iloc[i]['name'],station_data.iloc[i]['status']),

        radius=20,

        color=color_producer(station_data.iloc[i]['status'])).add_to(m_4)



# Display the map

embed_map(m_4, 'm_4.html')
# Create a map

m_2 = folium.Map(location=[30.2672,-97.7431], tiles='cartodbpositron', zoom_start=13)





# Add points to the map

for i in range(0,len(station_data)):

    Marker(location=[station_data.iloc[i]['latitude'], station_data.iloc[i]['longitude']],

        tooltip=(station_data.iloc[i]['name'],station_data.iloc[i]['status']),

        radius=20).add_to(m_2)



# Display the map

embed_map(m_2, 'm_2.html')
sns.set(font_scale = 1.5)

g=sns.catplot(x='duration_minutes',kind="count", palette="Spectral",data = cbd, order=pd.value_counts(cbd['duration_minutes']).iloc[:100].index)

g.set_xticklabels(rotation=90)



g.fig.set_size_inches(30,10)

g.fig.suptitle('Duration of Trips', fontsize=40)

plt.ylabel("Count", fontsize = 30)

plt.xlabel("Duration in Minutes", fontsize = 30)







plt.show()
sns.set(font_scale = 1.5)

g=sns.catplot(x="bikeid",kind="count", palette="icefire",data = cbd,order=pd.value_counts(cbd['bikeid']).iloc[:100].index)

g.set_xticklabels(rotation=90)



g.fig.set_size_inches(30,10)

g.fig.suptitle('Most used Bikes', fontsize=40)

plt.ylabel("Count", fontsize = 30)

plt.xlabel("Bike ID Number", fontsize = 30)





plt.show()
cbd["year"] = cbd["year"].astype('category')

cbd["month"] = cbd["month"].astype('category')

cbd["trip_id"] = cbd["trip_id"].astype('category')



cbd["date"] = cbd["date"].astype('int8')



cbd["year_code"] = cbd["year"].cat.codes

cbd["month_code"] = cbd["month"].cat.codes

cbd["trip_id_code"] = cbd["trip_id"].cat.codes
y = cbd.date

features = ['year_code','month_code','trip_id_code']

x = cbd[features]

train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)

basic_model = DecisionTreeRegressor()

basic_model.fit(train_x, train_y)

val_predictions = basic_model.predict(val_x)

print("Printing MAE for Basic Decision Tree Regressor:\n", mean_absolute_error(val_y, val_predictions))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    leaf_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    leaf_model.fit(train_x, train_y)

    preds_val = leaf_model.predict(val_x)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)

print("Decision Tree with Leaves\n")

for max_leaf_nodes in [5, 50, 500, 5000,50000]:

    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)

    

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))
forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_x, train_y)

forest_preds = forest_model.predict(val_x)

print("Printing MAE for Random Forest Model:\n",mean_absolute_error(val_y, forest_preds))
perm = PermutationImportance(basic_model, random_state=1).fit(val_x, val_y)

eli5.show_weights(perm, feature_names = val_x.columns.tolist())
y = cbd.date



#choosing features

trip_features = ['year_code','month_code','trip_id_code']

X = cbd[trip_features]



#testing

#X.describe()

X.head()
model = DecisionTreeRegressor(random_state=1)

model.fit(X,y)
print("Making date predictions for the following 10 trips:")

print(X.head(10))

print("The date predictions are")

print(model.predict(X.head(10)))



print('\nOriginal dates')

print(cbd['date'].head(10))
predicted_trip_month = model.predict(X)

print("Printing the mean absolute error", mean_absolute_error(y, predicted_trip_month))
train_X, val_X, train_y, val_y = train_test_split(X,y, random_state = 0)



model = DecisionTreeRegressor()



model.fit(train_X,train_y)



#getting predicted points

val_predictions = model.predict(val_X)

print("Using the DecisionTreeRegressor.. Now\nPrinting the mean absolute value ",mean_absolute_error(val_y, val_predictions))
print("Making date predictions for the following 10 Trips:")

print(X.head(10))

print("The date predictions are")

print(model.predict(X.head(10)))



print('\nOriginal Dates')

print(cbd['date'].head(10))
my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),

                              ('model',

                               RandomForestRegressor(n_estimators=50,random_state=0))])
points_CV = -1 * cross_val_score(my_pipeline, X, y, cv=5, 

                               scoring = 'neg_mean_absolute_error')

print("Using Cross Validation..\nNow Printing Mean Absolute Error points:\n",

       points_CV)
print("Using Cross Validation..\nNow Printing Average Mean Absolute Error points across all experiments: \n", points_CV.mean())
pipe_data = cbd



pipe_data.dropna(axis=0, inplace=True)

y = pipe_data.date





 





X_train_full, X_valid_full, y_train, y_valid = train_test_split(pipe_data, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)

# Select categorical columns

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10000 and 

                    X_train_full[cname].dtype == "object"]



 



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



 



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()





 



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



 



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



 



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



 



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



 



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



 



# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



 



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print('MAE Using Pipeline:', mean_absolute_error(y_valid, preds))
xgbr_model = XGBRegressor(n_estimators=5000, learning_rate=0.05, n_jobs=10)

xgbr_model.fit(train_X, train_y, 

             early_stopping_rounds=5, 

             eval_set=[(val_X, val_y)], 

             verbose=False)
predictionsXBGR = xgbr_model.predict(val_X)

print("Mean Absolute Error using XGBR: " + str(mean_absolute_error(predictionsXBGR, val_y)))