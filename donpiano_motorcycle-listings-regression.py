from matplotlib import pyplot as plt

import numpy as np 

import os

import pandas as pd

import seaborn as sns



from sklearn import preprocessing

from sklearn.metrics import SCORERS, r2_score

from sklearn.model_selection import GridSearchCV





print(os.listdir("../input"))
df = pd.read_csv('../input/motorcycle_listings.csv')
df.info()
df.head()
df.brand.value_counts()
def print_min_max(dataframe):

    for col in dataframe.columns:

        print(col)

        try:

            print("Min:", dataframe[col].min())

            print("Max:", dataframe[col].max(),"\n")

        except:

            print(col, "ISSUE PRINTING MIN/MAX\n")

            

print_min_max(df)
def plot_column_with_price(column_name):

    fig, ax = plt.subplots()

    ax.scatter(x = df[column_name], y = df['price'])

    plt.xlabel(column_name, fontsize=13)

    plt.ylabel('price', fontsize=13)

    plt.xticks(rotation=90)

    plt.show()



plot_column_with_price("km")    

plot_column_with_price("vehicle_engine")
df = df[df['price'] > 0] # Remove zero price

df = df[df['km'] > 10] # Used bikes only

df = df[df['km'] < 300000] # Remove high unrealistic KM

df = df[df['vehicle_engine'] > 10] # Remove zero engine size



# print_min_max(df)



df.reset_index(inplace=True, drop=True)



plot_column_with_price("km")    

plot_column_with_price("vehicle_engine")



df.hist(figsize=(10,10))
ml = df.copy()

ml = ml[

    ['brand',

#      'model',

     'year',

     'body_type',

     'km',

     'number_images',

     'vehicle_engine',

     'price']]

print (ml.dtypes)

ml.head()
label = ml.pop("price")
plt.subplots(figsize=(12,9))

sns.heatmap(df.corr(), square=True, annot=True)

df.corr()
ml['year_since_made'] = 2020 - ml['year']

ml = ml.drop('year', axis=1)

ml.head(10)
# ml['brand_model'] = ml['brand'] + "_" + ml['model']
body_type_one_hot = pd.get_dummies(data=ml['body_type'], prefix="body_")

brand_one_hot = pd.get_dummies(data=ml['brand'], prefix="brand_")
ml = ml.join(body_type_one_hot)

ml = ml.drop("body_type", axis=1)

ml = ml.join(brand_one_hot)

ml = ml.drop("brand", axis=1)
ml.head()
ml.head(10)
ml = ml.astype(dtype='float')
columns_to_convert = ['km', 'number_images', 'vehicle_engine', 'year_since_made']



scaler = preprocessing.MinMaxScaler()



ml2 = ml.copy()



for column in columns_to_convert:

    ml2[column] = scaler.fit_transform(ml2[[column]])

    print("Converted", column)
ml2.head(10)
from sklearn.model_selection import train_test_split

feature_train, feature_test, label_train, label_test = train_test_split(ml2, label, test_size=0.3)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model

from sklearn import svm



from sklearn import tree



regressor_algo = [

    GradientBoostingRegressor(),

    GradientBoostingRegressor(max_depth=3,learning_rate=0.5,n_estimators=110)

]



for item in regressor_algo:

    print(item)

    reg = item

    reg.fit(feature_train, label_train)

    score = reg.score(feature_test, label_test)

    print(score)
fi = pd.DataFrame()

fi['features'] = list(ml2.columns.values)

fi['importance'] = reg.feature_importances_



fi = fi.sort_values(by=['importance','features'], ascending=False)

fi.set_index(keys='features', inplace=True)

fi[fi['importance'] > 0.002].plot(

    kind='barh', 

    figsize=(20,20),

    fontsize=22,

    title="Feature Importance"

)
# grid_param = {  

#     'alpha': [0.1,0.3,0.5,0.7,0.9],

#     'learning_rate': [0.1,0.3,0.4,0.5,0.6,0.8,0.9],

#     'max_depth': [2,3,4,5,6],

#     'n_estimators': [80, 90, 100, 110, 120, 130, 200]

# }

# gd_sr = GridSearchCV(estimator=GradientBoostingRegressor(),  

#                      param_grid = grid_param,

#                      scoring='r2',

#                      cv=5,

#                      n_jobs=-1)



# gd_sr.fit(feature_train, label_train)

# print(gd_sr.best_estimator_)

# print(gd_sr.best_score_)
def set_prediction_data(dataframe, km, num_images, engine_size, year, body, brand):

    temp = dataframe.copy()

    temp['km'] = km

    temp['number_images'] = num_images

    temp['vehicle_engine'] = engine_size

    temp['year_since_made'] = [2020 - year[0]]

    temp['body__'+body] = 1.0

    temp['brand__'+brand] = 1.0

    temp.fillna(value=0,inplace=True)

    return temp



def scale_columns(dataframe):

    temp = dataframe.copy()

    columns_to_convert = ['km', 'number_images', 'vehicle_engine', 'year_since_made']

    scaler = preprocessing.MinMaxScaler()

    ml2 = ml.copy()

    for column in columns_to_convert:

        ml2[column] = scaler.fit(ml2[[column]])

        temp[column] = scaler.transform(temp[[column]])

    return temp



def predict_value(column_names, km, num_images, engine_size, year, body, brand):

    inf = pd.DataFrame(columns = column_names, dtype=float) # Creates df from ml columns

    inf = set_prediction_data(inf, km, num_images, engine_size, year, body, brand)

    inf = scale_columns(inf)

    inf.head()

    pred = reg.predict(X=inf)

    return pred

    

prediction = predict_value(

        column_names = ml2.columns,

        km = [8000],

        num_images = [10],

        engine_size = [655],

        year = [2016],

        body = "Naked",

        brand = "Yamaha"

)

print (prediction[0])
temp_it = []

temp_price = []



i = 2000

while i < 2020:

    prediction = predict_value(

        column_names = ml2.columns,

        km = [8000],

        num_images = [10],

        engine_size = [655],

        year = [i],

        body = "Naked",

        brand = "Yamaha"

    )

    temp_it.append(i)

    temp_price.append(prediction[0].round(2))

    i = i + 1

    

temp = pd.DataFrame()

temp['iterator'] = temp_it

temp['price'] = temp_price
%matplotlib inline

import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid') ## https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html



SMALL_SIZE = 8

MEDIUM_SIZE = 15

BIGGER_SIZE = 20



plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes

plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title

plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels

plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels

plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize



plt.figure(figsize = (15,8))



plt.xlabel("Year")

plt.ylabel("Price")





plt.plot(

    temp['iterator'],

    temp['price'],

    drawstyle='default', 

    color="#4285f4",  

    linewidth=3)



plt.legend()

plt.title("Predicted Price By Year")



# plt.xticks(rotation=85)



plt.show()