import numpy as np #Deal with numbers and arrays

import pandas as pd #Create dataframes and sort/clean data faster



import matplotlib.pyplot as plt #Visualization module

import seaborn as sns #Makes matplotlib prettier



#This blocks imports everything needed from sklearn(models, procession packages, and metrics)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.svm import SVR, LinearSVR

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor



from xgboost import XGBRegressor # XGB regression model

import plotly.express as px #Module for dynamic data visualization
df = pd.read_csv('../input/airbnb-rio-de-janeiro/total_data.csv', index_col=False, usecols=['host_is_superhost', 'host_listings_count', 'latitude', 'longitude',

       'property_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',

       'amenities', 'price', 'require_guest_profile_picture',

       'require_guest_phone_verification', 'month', 'security_deposit','cleaning_fee'])
df.head()
print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')
df.isnull().sum()
total_rows = df.shape[0]

print(f'security_deposit missing values are in {round(361064*100/total_rows,1)}% of rows, and cleaning_fee missing values are in {round(269336*100/total_rows,1)}% of rows.')
rows_before_drop = df.shape[0]

df.drop(['security_deposit','cleaning_fee'], axis=1, inplace=True)

print(f"'security_deposit' and 'cleaning fee' dropped :)")
df.dropna(inplace=True)

print(f'{df.shape} - {rows_before_drop-df.shape[0]} rows dropped')
df.isnull().sum()
df.dtypes
# CHANGING DATA TYPES



#host_listings_count

df['host_listings_count'] = df['host_listings_count'].astype(np.float32, copy=False)

df['host_listings_count'] = df['host_listings_count'].astype(np.int16, copy=False)

#accommodates

df['accommodates'] = df['accommodates'].astype(np.int16, copy=False)

#price

df['price'] = df['price'].str.replace('$', '', regex=False)

df['price'] = df['price'].str.replace(',', '', regex=False)

df['price'] = df['price'].astype(np.float32, copy=False)

df['price'] = df['price'].astype(np.int32, copy=False)
def get_max_fence(column):

    qt = df[column].quantile([0.25,0.75])

    upper = qt.values[1]

    iqr = upper-qt.values[0]

    max_fence = upper + 1.5*(iqr)

    return max_fence
def box_plot(column):

    fig, (ax1, ax2) = plt.subplots(1,2)

    fig.set_size_inches(16,6)

    _ = sns.boxplot(x=df[column], ax = ax1)

    ax1.set_title(f'{column} boxplot')

    ax2.set_title(f'Zooming in the {column} boxplot')

    ax2.set_xlim((-0.1,1.1*get_max_fence(column)))

    _ = sns.boxplot(x=df[column], ax = ax2)
column = 'host_listings_count'

plt.figure(figsize=(16,8))

ax = sns.barplot(x=df[column].value_counts().index, y=df[column].value_counts(), color='g')

ax.set_xticklabels(map(int,df['beds'].index))

ax.set_xlim((-0.5,get_max_fence(column)+2))

ax.set_xlabel('Amount of listings by host')

_ = ax.set_title('Distribution of host_listings_count')
column = 'host_listings_count'

box_plot(column)
print(get_max_fence('host_listings_count'))
rows_before = df.shape[0]

df = df[df['host_listings_count'] <= get_max_fence('host_listings_count')]

print(f'{rows_before-df.shape[0]} rows were deleted.')
df.loc[df['host_listings_count'] == 0.0, 'host_listings_count'] = 1.0
plt.figure(figsize=(12,8))

ax = sns.distplot(df['price'],norm_hist=True)

_ = ax.set_title('Price distribution')
box_plot('price')
get_max_fence('price')
rows_before = df.shape[0]

df = df[df['price'] <= get_max_fence('price')]

print(f'{rows_before-df.shape[0]} rows were removed')
plt.figure(figsize=(12,8))

ax = sns.distplot(df['price'],norm_hist=True)

_ = ax.set_title('Price distribution - Outliers removed')
plt.figure(figsize=(12,8))

ax = sns.countplot('property_type', data=df)

ax.tick_params(axis='x', rotation=90)

_ = ax.set_title('Distribution of property types')
categories_to_append = ('Aparthotel', 'Earth house', 'Chalet', 'Cottage', 'Tiny house',

                        'Boutique hotel', 'Hotel', 'Casa particular (Cuba)', 'Bungalow',

                        'Nature lodge', 'Cabin', 'Castle', 'Treehouse', 'Island', 'Boat', 'Tent',

                        'Resort', 'Hut', 'Campsite', 'Barn', 'Dorm', 'Camper/RV', 'Farm stay', 'Yurt',

                        'Tipi', 'Pension (South Korea)', 'Dome house', 'Igloo', 'Casa particular',

                        'Houseboat', 'Lighthouse', 'Plane', 'Train', 'Parking Space')



for cat in categories_to_append:

    df.loc[df['property_type'] == cat, 'property_type'] = 'Other'
plt.figure(figsize=(12,8))

ax = sns.countplot('property_type', data=df)

ax.tick_params(axis='x', rotation=20)

_ = ax.set_title('Distribution of property types')
aa = df.groupby(by='property_type').mean().sort_values(by='price',ascending=False).iloc[0:6]

fig, (ax1,ax2) = plt.subplots(1,2)

fig.set_size_inches(20,8)

violin_data=df.loc[df['property_type'].isin(aa.index)]

_ =  sns.barplot(x=aa.index, y='price', data=aa,ax=ax1)

_ = ax1.set_title('Average price of property_type')

_ = ax2.set_title('Price distribution of property_type')

_ = sns.violinplot(x = 'property_type', y =  'price',data=violin_data,ax=ax2)
plt.figure(figsize=(12,8))

ax = sns.countplot('beds', data=df)

ax.set_xticklabels(map(int,df['beds'].index))

ax.set_xlabel('Amount of beds')

_ = ax.set_title('Distribution of beds')
box_plot('beds')
get_max_fence('beds')

rows_before = df.shape[0]

df = df[df['beds'] <= get_max_fence('beds')]

print(f'{rows_before-df.shape[0]} rows were removed')
plt.figure(figsize=(12,8))

ax = sns.countplot('beds', data=df)

ax.set_xticklabels(map(int,df['beds'].index))

_ = ax.set_title('Distribution of beds')
df['amenities'].unique()[0]
df['n_amenities'] = df['amenities'].str.split(',').apply(len)+1

df['n_amenities'] = df['n_amenities'].astype('int')

df.loc[df['amenities'] == '{}', 'n_amenities'] = df['n_amenities'].mode()

df = df.drop('amenities', axis=1)
box_plot('n_amenities')
print(get_max_fence('n_amenities'))

rows_before = df.shape[0]

df = df[df['n_amenities'] <= get_max_fence('n_amenities')]

print(f'{rows_before-df.shape[0]} rows were removed')
plt.figure(figsize=(12,8))

ax = sns.countplot('n_amenities', data=df)

ax.tick_params(axis='x')

ax.set_xticklabels(map(int,df['n_amenities'].index))

# ax.locator_params(integer=True)

_ = ax.set_title('Distribution of n_amenities')
df.head()

corr = df.corr()

plt.figure(figsize=(16,12))

_ = sns.heatmap(df.corr(), annot=True, cmap='Greens')
#COORDINATES TO PLOT THE MAP

box = (df.longitude.min(), df.longitude.max(), df.latitude.min(), df.latitude.max())

box1 = (-43.7370,-43.1041,-23.0729,-22.7497)
fig, ax= plt.subplots()

fig.set_size_inches(16,12)

a = plt.imread('https://i.ibb.co/52dDkxT/map-8.png')

ax.imshow(a,zorder=0, extent=(box[0],-43.1041,-23.082,box[3]), aspect='equal')

ax.set_title('Listings location and info about the city')

ax.set_xlim(box[0],box[1])

ax.set_ylim(box[2],box[3])#-23.0729

ax.annotate('offset values',

            xy=(-43.18, -22.82),

            xytext=(-43.2,-22.872), backgroundcolor='yellow',bbox=dict(facecolor='#ffa500', alpha=0.5, edgecolor='red', joinstyle='round'),

            arrowprops=dict(headwidth=8, width=1, color='#ffa500', connectionstyle="arc3, rad=0.3"),

            fontsize=12)

ax.annotate('Parque Olímpico',

            xy=(-43.39, -22.98),

            xytext=(-43.48,-23.055), backgroundcolor='yellow',bbox=dict(facecolor='#ffa500', alpha=0.5, edgecolor='red', joinstyle='round'),

            arrowprops=dict(headwidth=8, width=1, color='#ffa500', connectionstyle="arc3, rad=-0.2"),

            fontsize=12)

ax.annotate('City center',

            xy=(-43.163, -22.91),

            xytext=(-43.165,-23.02), backgroundcolor='yellow',bbox=dict(facecolor='#ffa500', alpha=0.5, edgecolor='red', joinstyle='round'),

            arrowprops=dict(headwidth=8, width=1, color='#ffa500', connectionstyle="arc3, rad=0.5"),

            fontsize=12)

ax.annotate('Touristic spots',

            xy=(-43.19, -22.995),

            xytext=(-43.32,-23.055), backgroundcolor='yellow',bbox=dict(facecolor='#ffa500', alpha=0.5, edgecolor='red', joinstyle='round'),

            arrowprops=dict(headwidth=8, width=1, color='#ffa500', connectionstyle="arc3, rad=0.3"),

            fontsize=12)

_ = sns.scatterplot(x='longitude', y='latitude', data=df, ax=ax, zorder=1,color='black',alpha=0.6, s=0.1,edgecolor=None)
df_density_mapbox = df.sample(n=15000)

map_center = {'lat':df_density_mapbox.latitude.mean(), 'lon':df_density_mapbox.longitude.mean()}

fig = px.density_mapbox(df_density_mapbox, lat='latitude', lon='longitude',z='price',title=10*'  '+'Daily Price density', radius=2.5,

                        center=map_center, zoom=10,

                        mapbox_style='stamen-terrain')

fig.show()
df_label_encoder = df.copy()

for column in ('host_is_superhost', 'require_guest_profile_picture', 'require_guest_phone_verification'):

    df_label_encoder.loc[df_label_encoder[column] == 'f', column] = 0

    df_label_encoder.loc[df_label_encoder[column] == 't', column] = 1

    df_label_encoder[column] = df_label_encoder[column].astype(int)

encoder = LabelEncoder()





df_label_encoder['property_type'] = encoder.fit_transform(df_label_encoder['property_type']) 



print('Columns encoded')
def evaluate(model_name, y_test, predictions):

    RMSE = np.sqrt(mean_squared_error(y_test, predictions))

    MAE = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    return f'model: {model_name}\nMean Absolute Error: {MAE}\nRoot Mean Square Error: {RMSE}\nR² Score: {round(r2*100, 2)}% \n--------------------------------------------'   
y = df_label_encoder['price']

columns_dropped =  ['price']

models = {'Random Forest':RandomForestRegressor(),

          'Lasso':Lasso(),

          'ElasticNet': ElasticNet(),

          'XGBRegressor': XGBRegressor(),

          'Linear Regression': LinearRegression(),

          'Linear SVR': LinearSVR(),

          'sgdregressor' : SGDRegressor(),

          'decision tree': DecisionTreeRegressor(),

          'Extra Tree Regressor': ExtraTreesRegressor()

}



X = df_label_encoder.drop(columns_dropped, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

name = 'Extra Trees Regressor'

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(evaluate(name, y_test, predictions))
fig, ax = plt.subplots()

fig.set_size_inches(12,8)

bp_df = pd.DataFrame({'features': model.feature_importances_}, index=X_test.columns)

bp_df = bp_df.sort_values(by='features', ascending=False)

ax.tick_params(axis='x', rotation=18)

_ = ax.set_title('Features importance')

_ = sns.barplot(x=bp_df.index, y='features', data=bp_df, ax = ax)
cols_lists = ['require_guest_phone_verification', 'require_guest_profile_picture',

              'property_type', 'bedrooms',

              'host_listings_count', 'host_is_superhost']

for column in cols_lists: 

    removed = X[column]

    X.drop(column, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

    selected_model = ExtraTreesRegressor(random_state=1)

    selected_model.fit(X_train, y_train)

    predictions = selected_model.predict(X_test)

    print(f'col removed: {column}\n{evaluate(name, y_test, predictions)}')

    X = pd.concat([X, removed], axis=1)
X.drop('host_listings_count', axis=1, inplace=True)
removed = X['host_is_superhost']

X.drop('host_is_superhost', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

selected_model = ExtraTreesRegressor(random_state=1)

selected_model.fit(X_train, y_train)

predictions = selected_model.predict(X_test)

print(f'cols removed: host_is_superhost\n{evaluate(name, y_test, predictions)}')
column = 'require_guest_phone_verification'

removed = X[column]

X.drop(column, axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

selected_model = ExtraTreesRegressor(random_state=1)

selected_model.fit(X_train, y_train)

predictions = selected_model.predict(X_test)

print(f'cols removed: {column}\n{evaluate(name, y_test, predictions)}')
column = 'require_guest_profile_picture'

removed = X[column]

X.drop(column, axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

selected_model = ExtraTreesRegressor(random_state=1)

selected_model.fit(X_train, y_train)

predictions = selected_model.predict(X_test)

print(f'cols removed: {column}\n{evaluate(name, y_test, predictions)}')
i=1

for feature in X.columns:

    print(f'Feature number {i}: {feature}')

    i+=1
fig, ax = plt.subplots()

fig.set_size_inches(12,8)

bp_df = pd.DataFrame({'features': selected_model.feature_importances_}, index=X_test.columns)

bp_df = bp_df.sort_values(by='features', ascending=False)

ax.tick_params(axis='x', rotation=0)

_ = ax.set_title('Features importance')

_ = sns.barplot(x=bp_df.index, y='features', data=bp_df, ax = ax)