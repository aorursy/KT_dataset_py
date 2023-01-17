# Required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Importing dataset
kchp=pd.read_csv('../input/kc_house_data.csv')
# Viewing the first 5 rows of the dataset
kchp.head()
# Checking if there are any missing values
kchp.isnull().sum()
# Information about the dataset
kchp.info()
# Using the date column to obtain the year that the house was sold
kchp['date']=kchp['date'].astype('str')

for i in range(len(kchp.index)):
    kchp.loc[i,'date']=kchp.loc[i,'date'][0:4]
    
kchp['date']=kchp['date'].astype('int64')
# Renaming the column date as year_sold
kchp.rename(columns={'date':'year_sold'},inplace=True)

# If the sqft_living, sqft_living15 and sqft_lot, sqft_lot15 columns are not the same then it implies  
# that the house has been renovated. A column renovated is created with 1 - renovated, 0 - not renovated
# kchp.loc[:,['sqft_living','sqft_lot','sqft_living15','sqft_lot15','yr_renovated']].head(10)
kchp['renovated']=np.where((kchp['sqft_living']!=kchp['sqft_living15'])|(kchp['sqft_lot']!=kchp['sqft_lot15']),1,0)
# kchp.loc[:,['sqft_living','sqft_lot','sqft_living15','sqft_lot15','yr_renovated','renovated']].head(20)

# The yr_renovated column has mostly 0 values and we have obtained the renovation information, so it is dropped
# Columns id, sqft_living and sqft_lot won't be used, so they are dropped as well
kchp.drop(['id','sqft_living','sqft_lot','yr_renovated'],axis=1,inplace=True)

# The age of the buidlding at the time it is sold is added as a new column
kchp['age_of_building']=kchp['year_sold']-kchp['yr_built']

# Column yr_built is now  dropped since column age_of_building is created
kchp.drop('yr_built',axis=1,inplace=True)

# Columns bathrooms and floors have float values wihich is not possible, so they are floored
kchp['bathrooms']=np.floor(kchp['bathrooms'])
kchp['floors']=np.floor(kchp['floors'])

# Columns are changed to appropriate data types
kchp['waterfront']=kchp['waterfront'].astype('category')
kchp['condition']=kchp['condition'].astype('category')
kchp['grade']=kchp['grade'].astype('category')
kchp['bathrooms']=kchp['bathrooms'].astype('int64')
kchp['floors']=kchp['floors'].astype('int64')
kchp['renovated']=kchp['renovated'].astype('category')
kchp['zipcode']=kchp['zipcode'].astype('category')
# Viewing the first 5 rows and information about the dataset after the changes made
print(kchp.head())
print(kchp.info())
sns.set(rc={'figure.figsize':(15,15)})
fig=plt.figure()

ax1=fig.add_subplot(4,3,1)
sns.boxplot(y=kchp['price'],ax=ax1,width=0.3,color='turquoise')
ax1.set_title('Price of the houses')

ax2=fig.add_subplot(4,3,2)
sns.boxplot(y=kchp['bedrooms'],ax=ax2,width=0.3,color='royalblue')
ax2.set_title('Number of bedrooms')

ax3=fig.add_subplot(4,3,3)
sns.boxplot(y=kchp['bathrooms'],ax=ax3,width=0.3,color='cyan')
ax3.set_title('Number of bathrooms')

ax4=fig.add_subplot(4,3,4)
sns.boxplot(y=kchp['floors'],ax=ax4,width=0.3,color='gold')
ax4.set_title('Number of floors')

ax5=fig.add_subplot(4,3,5)
sns.boxplot(y=kchp['view'],ax=ax5,width=0.3,color='plum')
ax5.set_title('Number of times viewed')

ax6=fig.add_subplot(4,3,6)
sns.boxplot(y=kchp['sqft_above'],ax=ax6,width=0.3,color='red')
ax6.set_title('Square footage of house apart from basement')

ax7=fig.add_subplot(4,3,7)
sns.boxplot(y=kchp['sqft_basement'],ax=ax7,width=0.3,color='indigo')
ax7.set_title('Square footage of basement')

ax8=fig.add_subplot(4,3,8)
sns.boxplot(y=kchp['sqft_living15'],ax=ax8,width=0.3,color='salmon')
ax8.set_title('Living room area')

ax9=fig.add_subplot(4,3,9)
sns.boxplot(y=kchp['sqft_lot15'],ax=ax9,width=0.3,color='silver')
ax9.set_title('Lot size area')

ax10=fig.add_subplot(4,3,10)
sns.boxplot(y=kchp['age_of_building'],ax=ax10,width=0.3,color='mediumaquamarine')
ax10.set_title('Age of buiding')

plt.show()
print(kchp.loc[:,['bedrooms','price','sqft_above']].head())
print(kchp.loc[kchp['bedrooms']>10,['bedrooms','price','sqft_above']])
# Number of bedrooms is changed from 33 to 3
kchp.loc[kchp['bedrooms']==33,'bedrooms']=3
# kchp.loc[15870,'bedrooms']
kchp['sqft_lot15'].describe()
sns.set(rc={'figure.figsize':(10,25)})
sns.boxplot(y=kchp['sqft_lot15'],width=0.3,color='silver')
plt.show()
pd.set_option('display.max_columns',None)
print(kchp.loc[kchp['sqft_lot15']>100000,['sqft_lot15','price','bedrooms','bathrooms','sqft_above',
                                          'sqft_basement']].head())
print(kchp.loc[kchp['sqft_lot15']>400000,['sqft_lot15','price','bedrooms','bathrooms','sqft_above',
                                          'sqft_basement']])
kchp['sqft_above'].describe()
print(kchp.groupby(['bedrooms'])['sqft_above','price'].mean())
kchp.loc[(kchp['bedrooms']==0)&(kchp['bathrooms']==0),['bedrooms','bathrooms','floors','price']]
len(kchp.loc[(kchp['bedrooms']==0)|(kchp['bathrooms']==0),['bedrooms','bathrooms','floors','price']].index)
# Removing rows with zero bedrooms or bathrooms
indices=kchp.loc[(kchp['bedrooms']==0)|(kchp['bathrooms']==0),['bedrooms','bathrooms','floors','price']].index
kchp.drop(labels=indices,axis=0,inplace=True)
fig1=plt.figure()

sns.set(rc={'figure.figsize':(15,33)})

ax01=fig1.add_subplot(5,2,1)
sns.regplot(x='bedrooms', y='price', data=kchp,fit_reg=False,ax=ax01,color='turquoise')
ax01.set_title('Association between Number of bedrooms and Price')

ax11=fig1.add_subplot(5,2,2)
sns.regplot(x='bathrooms', y='price', data=kchp,fit_reg=False,ax=ax11,color='royalblue')
ax11.set_title('Association between Number of bathrooms and Price')

ax21=fig1.add_subplot(5,2,3)
sns.regplot(x='floors', y='price', data=kchp,fit_reg=False,ax=ax21,color='rebeccapurple')
ax21.set_xticks([1,2,3])
ax21.set_title('Association between Number of floors and Price')

ax31=fig1.add_subplot(5,2,4)
sns.regplot(x='view', y='price', data=kchp,fit_reg=False,ax=ax31,color='orangered')
ax31.set_xticks([0,1,2,3,4])
ax31.set_title('Association between Number of times viewed and Price')

ax41=fig1.add_subplot(5,2,5)
sns.regplot(x='sqft_above', y='price', data=kchp,fit_reg=False,ax=ax41,color='plum')
ax41.set_title('Association between Square footage of house apart from basement and Price')

ax51=fig1.add_subplot(5,2,6)
sns.regplot(x='sqft_basement', y='price', data=kchp,fit_reg=False,ax=ax51,color='darkorange')
ax51.set_title('Association between Square footage of basement and Price')

ax61=fig1.add_subplot(5,2,7)
sns.regplot(x='sqft_living15', y='price', data=kchp,fit_reg=False,ax=ax61,color='indigo')
ax61.set_title('Association between Living room area and Price')

ax71=fig1.add_subplot(5,2,8)
sns.regplot(x='sqft_lot15', y='price', data=kchp,fit_reg=False,ax=ax71,color='salmon')
ax71.set_title('Association between Lot size area and Price')

ax81=fig1.add_subplot(5,2,9)
sns.regplot(x='age_of_building', y='price', data=kchp,fit_reg=False,ax=ax81,color='mediumaquamarine')
ax81.set_title('Association between Age of building and Price')

plt.show()

fig=plt.figure()

sns.set(rc={'figure.figsize':(15,12)})

ax0=fig.add_subplot(2,2,1)
sns.boxplot(x="waterfront", y="price", data=kchp,dodge=False,palette="gist_rainbow",width=0.4,ax=ax0)
ax0.set_xticklabels(['No','Yes'])
ax0.set_title('Waterfront vs. Price')

ax1=fig.add_subplot(2,2,2)
sns.boxplot(x="condition", y="price", data=kchp,dodge=False,palette="gist_rainbow",width=0.4,ax=ax1)
ax1.set_title('Condition vs. Price')

ax2=fig.add_subplot(2,2,3)
sns.boxplot(x="grade", y="price", data=kchp,dodge=False,palette="bright",width=0.4,ax=ax2)
ax2.set_title('Grade vs. Price')

ax3=fig.add_subplot(2,2,4)
sns.boxplot(x="renovated", y="price", data=kchp,dodge=False,palette="gist_rainbow",width=0.4,ax=ax3)
ax3.set_xticklabels(['No','Yes'])
ax3.set_title('Renovated vs. Price')
plt.show()
# Obtaining the mean price for all zipcodes
zipinfo=kchp.groupby('zipcode')['price'].mean()
zipinfo=pd.DataFrame(zipinfo)
zipinfo.reset_index(inplace=True)

# Obtaining one of the latitude and longitude values corresponding to the zipcodes
latitude=[]
longitude=[]
for zipc in zipinfo['zipcode']:
    lt=kchp.loc[kchp['zipcode']==zipc,'lat']
    latitude.append(lt.iloc[0])
    lg=kchp.loc[kchp['zipcode']==zipc,'long']
    longitude.append(lg.iloc[0])

zipinfo['lat']=latitude
zipinfo['long']=longitude
# Plotting the mean price for each zipcode using the zipinfo dataframe in the map
import folium
kchp['price']=kchp['price'].astype('str')
location = kchp['lat'].mean(), kchp['long'].mean()

locationlist = zipinfo[['lat','long']].values.tolist()
zips=zipinfo['zipcode'].values.tolist()
labels = kchp['price'].values.tolist()

# Empty map
map1 = folium.Map(location=location, zoom_start=14)

# Accesing the latitude
for point in range(1,len(zipinfo.index)): 
    popup = folium.Popup('Price : {} , Zipcode : {}'.format(labels[point],zips[point]), parse_html=True)
    folium.Marker(locationlist[point], popup=popup).add_to(map1)

map1

# Click the marker to view price and zipcode information
# Changing the data type of price back to float64
kchp['price']=kchp['price'].astype('float64')
# Dividing the dataset into development and validation dataset
y=kchp.loc[:,'price']
x=kchp.loc[:,['year_sold', 'bedrooms', 'bathrooms', 'floors', 'waterfront',
       'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'zipcode',
       'sqft_living15', 'sqft_lot15', 'renovated', 'age_of_building']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

from sklearn.ensemble import RandomForestRegressor

n_estimators_options=[50,100,150,200,250,300]
for n_est in n_estimator_options :
    rf = RandomForestRegressor(n_estimators = n_est,criterion='mse',oob_score = True, n_jobs = -1,
                               random_state =50, max_features = "sqrt", min_samples_leaf = 50)
    rf.fit(x_train, y_train)
#     train dataset
    predictions_tr = rf.predict(x_train)
    abs_errors_tr = abs(predictions_tr - y_train)
    ape_tr = 100 * (abs_errors_tr / predictions_tr)
    accuracy_tr = 100 - np.mean(ape_tr)
#     test dataset
    predictions = rf.predict(x_test)
    abs_errors = abs(predictions - y_test)
    ape = 100 * (abs_errors / predictions)
    accuracy = 100 - np.mean(ape)
    print('n_estimators:{}\n'.format(n_est),'Accuracy for development dataset :',
          round(accuracy_tr, 2), '%.\n','Accuracy for validation dataset :', round(accuracy, 2), '%.\n')

# Random Forest with number of estimators equal to 200
rf = RandomForestRegressor(n_estimators = 150,criterion='mse',oob_score = True, n_jobs = -1,
                               random_state =50, max_features = "sqrt", min_samples_leaf = 50)
rf.fit(x_train, y_train)
print(rf.feature_importances_)
# Feature Importance
feature_importance=pd.DataFrame({'Variable':x_train.columns,'Importance':rf.feature_importances_})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importance)
sns.barplot(x='Importance',y='Variable',data=feature_importance,orient="h",palette='Blues_r')
plt.show()