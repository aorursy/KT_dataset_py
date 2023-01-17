# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats



from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
data  =pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head(2)
sns.pairplot(data)
top_10_host_brooklyn=data[data['neighbourhood_group']=='Brooklyn']['host_name'].value_counts().head(10)

top_10_host_manhattan=data[data['neighbourhood_group']=='Manhattan']['host_name'].value_counts().head(10)

top_10_host_queens=data[data['neighbourhood_group']=='Queens']['host_name'].value_counts().head(10)

top_10_host_bronx=data[data['neighbourhood_group']=='Bronx']['host_name'].value_counts().head(10)

top_10_host_Staten_Island=data[data['neighbourhood_group']=='Staten Island']['host_name'].value_counts().head(10)
plt.figure(figsize=(12,5))

sns.barplot(top_10_host_brooklyn.index,top_10_host_brooklyn.values)

plt.xlabel('Host Names')

plt.ylabel('Frequency')

plt.title('Top 10 Host of Brooklyn')

plt.figure(figsize=(12,5))

sns.barplot(top_10_host_manhattan.index,top_10_host_manhattan.values)

plt.xlabel('Host Names')

plt.ylabel('Frequency')

plt.title('Top 10 Host of Manhattan')
plt.figure(figsize=(12,5))

sns.barplot(top_10_host_queens.index,top_10_host_queens.values)

plt.xlabel('Host Names')

plt.ylabel('Frequency')

plt.title('Top 10 Host of Queens')

plt.figure(figsize=(12,5))

sns.barplot(top_10_host_Staten_Island.index,top_10_host_Staten_Island.values,palette='Blues_d')

plt.xlabel('Host Names')

plt.ylabel('Frequency')

plt.title('Top 10 Host of State Island')

plt.figure(figsize=(12,5))

sns.barplot(top_10_host_bronx.index,top_10_host_bronx.values)

plt.xlabel('Host Names')

plt.ylabel('Frequency')

plt.title('Top 10 Host of Bronx')

data_brooklyn=data[data['neighbourhood_group']=='Brooklyn']

data_manhattan=data[data['neighbourhood_group']=='Manhattan']

data_bronx=data[data['neighbourhood_group']=='Bronx']

data_queens=data[data['neighbourhood_group']=='Queens']

data_state=data[data['neighbourhood_group']=='Staten Island']

data_brooklyn['neighbourhood'].unique()

data_brooklyn.groupby('neighbourhood')[['host_name']].count().sort_values('host_name',ascending=False).head(10).plot.bar(figsize=(12,5))

plt.ylabel('Host_names Frequency')

plt.title("Top 10 Host's Frequency of Brooklyn")
data_manhattan.groupby('neighbourhood')[['host_name']].count().sort_values('host_name',ascending=False).head(10).plot.bar(figsize=(12,5),color='green')

plt.ylabel('Host_names Frequency')

plt.title("Top 10 Host's Frequency of Manhattan")

data_queens.groupby('neighbourhood')[['host_name']].count().sort_values('host_name',ascending=False).head(10).plot.bar(figsize=(12,5),color='Orange')

plt.ylabel('Host_names Frequency')

plt.title("Top 10 Host's Frequency of Queens")

data_state.groupby('neighbourhood')[['host_name']].count().sort_values('host_name',ascending=False).head(10).plot.bar(figsize=(12,5),color='red')

plt.ylabel('Host_names Frequency')

plt.title("Top 10 Host's Frequency of State's Island")



data_bronx.groupby('neighbourhood')[['host_name']].count().sort_values('host_name',ascending=False).head(10).plot.bar(figsize=(12,5))

plt.ylabel('Host_names Frequency')

plt.title("Top 10 Host's Frequency of Bronx")

plt.figure(figsize=(12,5))

data['host_name'].value_counts().head(50).plot.bar()

plt.xlabel('Host names')

plt.ylabel('Frequency')

plt.title("Top 50 Host's Names")

data_brooklyn.groupby('host_name')[['calculated_host_listings_count']].count().sort_values(by='calculated_host_listings_count',ascending=False).head(20).plot.bar(figsize=(12,5))

plt.title('As per the Calculated Host Listings Top 20 Host names of Brooklyn')

from wordcloud import WordCloud

from wordcloud import ImageColorGenerator

text = " ".join(review for review in data['host_name'].dropna())

word = WordCloud(width=1000,height=800,margin=0,max_font_size=150,background_color='white').generate(text)



plt.figure(figsize=[10,10])

plt.imshow(word,interpolation='bilinear')

plt.axis('off')

plt.show()

text1 = " ".join(review for review in data['neighbourhood'].dropna())

word = WordCloud(width=1000,height=800,margin=0,max_font_size=150).generate(text1)



plt.figure(figsize=[10,10])

plt.imshow(word,interpolation='bilinear')

plt.axis('off')

plt.show()

text2 = " ".join(review for review in data['name'].dropna())

word = WordCloud(width=1000,height=800,margin=0,max_font_size=150,background_color='white').generate(text2)



plt.figure(figsize=[10,10])

plt.imshow(word,interpolation='bilinear')

plt.axis('off')

plt.show()

sns.countplot(data['room_type'])

fig,ax = plt.subplots(2,2,figsize=(12,5))

sns.distplot(data['price'],ax=ax[0,0])

sns.distplot(data['minimum_nights'],ax=ax[0,1])

sns.distplot(data['availability_365'],ax=ax[1,0])

sns.distplot(data['reviews_per_month'].dropna(),ax=ax[1,1])

plt.tight_layout()

data = data.drop(columns=['id','name','host_id','host_name','last_review'])

data.head()

# 10052 missing values in the Reviews per month

print(data.isna().sum())



#imputing the missing values

data['reviews_per_month'] = data['reviews_per_month'].fillna(data['reviews_per_month'].mean())

data.isna().sum()
sns.distplot(np.log(data['price']+1))
#IQR method for Outliers Detection

q1 = data['price'].quantile(0.25)

q3 =data['price'].quantile(0.75)

print(q1,q3)

iqr = q3-q1

print('IQR value',iqr)



upper_limit = q3+1.5*iqr

lower_limit = q1-1.5*iqr



print("Upper value",upper_limit)

print("Lower value",lower_limit)

#Zscore method

print('Z score upper value',data['price'].mean()+3*data['price'].std())

print('Z score lower value',data['price'].mean()-3*data['price'].std())

data['price'].describe()
#price is missing 0

print(data[data['price']==0].shape)



#price is missing 999

print(data[data['price']==999].shape)



##price is missing 9999

print(data[data['price']==9999].shape)



##price is missing 99

print(data[data['price']==99].shape)



data.info()

sns.scatterplot(data['reviews_per_month'],data['number_of_reviews'])
# Taking the Average reviews per month by group by Neighbourhood group

data.groupby(['neighbourhood_group']).agg({'reviews_per_month':'mean'})

avg_reviews_neigh_20=data.groupby(['neighbourhood']).agg({'reviews_per_month':'mean'}).sort_values(by='reviews_per_month',ascending=False).head(20)

plt.figure(figsize=(12,5))

sns.barplot(avg_reviews_neigh_20['reviews_per_month'],avg_reviews_neigh_20.index,palette='gist_stern')

plt.title('As per the Avg Review Top 20 Neighbourhood ')

avg_price_neigh_20=data.groupby(['neighbourhood']).agg({'price':'mean'}).sort_values(by='price',ascending=False).head(20)

plt.figure(figsize=(12,5))

sns.barplot(avg_price_neigh_20['price'],avg_price_neigh_20.index,palette='RdBu_r')

plt.title('As per Avg price Top 20 Neighbourhoood')

avg_price_bly_20=data_brooklyn.groupby(['neighbourhood']).agg({'price':'mean'}).sort_values(by='price',ascending=False).head(20)

avg_price_man_20=data_manhattan.groupby(['neighbourhood']).agg({'price':'mean'}).sort_values(by='price',ascending=False).head(20)



fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.barplot(avg_price_bly_20['price'],avg_price_bly_20.index,palette='RdBu',ax=ax[0])

sns.barplot(avg_price_man_20['price'],avg_price_man_20.index,palette='rainbow',ax=ax[1])



ax[0].set_title('As per Avg Price Top 20 Neighbourhood of Brooklyn')

ax[1].set_title('As per Avg Price Top 20 Neighbourhood of Manhattan')

plt.tight_layout()

avg_price_que_20=data_queens.groupby(['neighbourhood']).agg({'price':'mean'}).sort_values(by='price',ascending=False).head(20)

avg_price_sat_20=data_state.groupby(['neighbourhood']).agg({'price':'mean'}).sort_values(by='price',ascending=False).head(20)



fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.barplot(avg_price_que_20['price'],avg_price_que_20.index,palette='gist_rainbow',ax=ax[0])

sns.barplot(avg_price_sat_20['price'],avg_price_sat_20.index,palette='gnuplot',ax=ax[1])



ax[0].set_title('As per Avg Price Top 20 Neighbourhood of Queens')

ax[1].set_title('As per Avg Price Top 20 Neighbourhood of State Island')

plt.tight_layout()

avg_price_bro_20=data_bronx.groupby(['neighbourhood']).agg({'price':'mean'}).sort_values(by='price',ascending=False).head(20)

plt.figure(figsize=(10,5))



sns.barplot(avg_price_bro_20['price'],avg_price_bro_20.index,palette='Spectral')

plt.title('As per the Avg Price Top 20 Neighbourhood of Bronx ')

fig,ax = plt.subplots(1,2,figsize=(10,5))

sns.scatterplot(data['number_of_reviews'],data['price'],ax=ax[0])

sns.scatterplot(data['reviews_per_month'],data['price'],ax=ax[1])

plt.tight_layout()

data.groupby(['room_type']).agg({'price':'mean'})

# Replacing the 0,99,999,9999 values as missing 

data['price']=data['price'].astype('str')

data['price']=data['price'].replace('0',np.nan)

data['price']=data['price'].replace('99',np.nan)

data['price']=data['price'].replace('999',np.nan)

data['price']=data['price'].replace('9999',np.nan)

data['price'].isna().sum()
#checking the percentage of the NAN

#1.5 so we will dropn the nan values

print('Percentage of Missing values',data['price'].isna().sum()/data.shape[0]*100)

data=data.dropna()
data['price']=data['price'].astype(float)

print("Mean Price",data['price'].mean())

print("Median Price",data['price'].median())
sns.boxplot(y=data['price'])
#As per the IQR method we are removing the Outliers

data_price_upper_limit=data[data['price']>334]

data_price_wi_out_upper_values = data[data['price']<=334]

print(data_price_upper_limit.shape)

print(data_price_wi_out_upper_values.shape)
sns.distplot(data_price_wi_out_upper_values['price'])

sns.distplot(np.sqrt(data_price_wi_out_upper_values['price']))

#zscore method for minimum nights

data_price_wi_out_upper_values['minimum_nights'].mean()+3*data_price_wi_out_upper_values['minimum_nights'].std()



#decided to go with the IQR method where the upper limit for minimum nights 11

data_price_wi_out_upper_values=data_price_wi_out_upper_values[data_price_wi_out_upper_values['minimum_nights']<=11]

sns.boxplot(y=data_price_wi_out_upper_values['minimum_nights'])

# IQRmethod for the calculated host listing with max value as 3.5

data_price_wi_out_upper_values=data_price_wi_out_upper_values[data_price_wi_out_upper_values['calculated_host_listings_count']<=3.5]

sns.boxplot(y=data_price_wi_out_upper_values['availability_365'])

data_price_wi_out_upper_values['reviews_per_month'].quantile(0.75)+1.5*1.47

data_price_wi_out_upper_values=data_price_wi_out_upper_values[data_price_wi_out_upper_values['reviews_per_month']<=3.935]

print(data_price_wi_out_upper_values.describe())

data_price_wi_out_upper_values=data_price_wi_out_upper_values[data_price_wi_out_upper_values['number_of_reviews']<=53.5]

#data_pre data ready for the Analysis and model building

data_pre = data_price_wi_out_upper_values

data_pre.isna().sum()


data_pre = data_pre.drop(columns=['neighbourhood'])

data_pre = pd.get_dummies(data=data_pre,columns=['neighbourhood_group','room_type'],drop_first=True)

# Data ready for the model without the outliers

data_pre.head()

# ### Building the Model 



# #### Removing the longitude,latitude,calculated_host_listings_count because of High Multicollinearity

#Building the raw model only by removinng the outliers in all the variables.

X = data_pre.drop(columns=['price','longitude','latitude','calculated_host_listings_count'])

y = data_pre['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_constant = sm.add_constant(X_train)

lin_reg = sm.OLS(y_train,X_constant).fit()

lin_reg.summary()



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

pd.DataFrame({'vif':vif},index=X.columns)



#Building the Sklearn Decision Tree Regressor

model_1 = DecisionTreeRegressor(random_state=1)

model_1.fit(X_train,y_train)

print(model_1.score(X_train,y_train))

print(model_1.score(X_test,y_test))

data_pre.columns

# ### Normalizing the Data

norm = data_pre.drop(['latitude','longitude'],axis=1)

df_nor = preprocessing.normalize(norm)

df_nor= pd.DataFrame(df_nor)

df_nor.columns = ['price', 'minimum_nights', 'number_of_reviews',

       'reviews_per_month', 'calculated_host_listings_count',

       'availability_365', 'neighbourhood_group_Brooklyn',

       'neighbourhood_group_Manhattan', 'neighbourhood_group_Queens',

       'neighbourhood_group_Staten Island', 'room_type_Private room',

       'room_type_Shared room']

df_nor.head()
# Removing the below features because of Multicollinearity

# calculated_host_listings_count,neighbourhood_group_Brooklyn,room_type_Private room

# neighbourhood_group_Staten Island,neighbourhood_group_Queens



X = df_nor.drop(columns=['price','calculated_host_listings_count','neighbourhood_group_Brooklyn','room_type_Private room'

                        ,'neighbourhood_group_Staten Island','neighbourhood_group_Queens'])

y=df_nor['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_constant = sm.add_constant(X_train)

lin_reg = sm.OLS(y_train,X_constant).fit()

lin_reg.summary()

model = LinearRegression()

model.fit(X_train,y_train)

print(model.score(X_train,y_train))

print(model.score(X_test,y_test))
model.coef_
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

pd.DataFrame({'vif':vif},index=X.columns)
X = df_nor.drop(columns=['price','neighbourhood_group_Manhattan','neighbourhood_group_Queens',

                         'neighbourhood_group_Staten Island','neighbourhood_group_Brooklyn','calculated_host_listings_count'])

y=df_nor['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_constant = sm.add_constant(X_train)

lin_reg = sm.OLS(y_train,X_constant).fit()

lin_reg.summary()
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

pd.DataFrame({'vif':vif},index=X.columns)
# ### SVR model

ml = SVR(kernel='linear')

ml.fit(X_train,y_train)

print(ml.score(X_train,y_train))

print(ml.score(X_test,y_test))
# ### Ridge Regression

clf  = RidgeCV(alphas=[10,100,1000,10000],cv=10).fit(X_train,y_train)

print(clf.score(X_train,y_train))

print(clf.score(X_test,y_test))
clf.alpha_,clf.alphas,clf.cv,clf.coef_
# ### Standardize Scaling and Model Building

df_scale=data_pre[['latitude', 'longitude','price','minimum_nights', 'number_of_reviews',

       'reviews_per_month', 'calculated_host_listings_count',

       'availability_365']]

df_scale.head()



scaler= StandardScaler()

df_scale=scaler.fit_transform(df_scale)



df_scale = pd.DataFrame(df_scale)

df_scale.columns=['latitude', 'longitude','price','minimum_nights', 'number_of_reviews',

       'reviews_per_month', 'calculated_host_listings_count',

       'availability_365']

df_scale.head()

print(df_scale.shape)
X = df_scale.drop(columns=['price'],axis=1)

y = df_scale['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

X_constant = sm.add_constant(X_train)

lin_reg = sm.OLS(y_train,X_constant).fit()

lin_reg.summary()
