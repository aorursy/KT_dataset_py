import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df
df.dtypes
#we're going to do an entry level prediction algorithm so we will drop unnecessary columns
df.drop(['id', 'host_id', 'host_name', 'last_review'], axis=1, inplace=True)
df.isna().sum()
#We can see we have multiple columns missing values
#let's remove missing values
df['name'].replace((np.nan, 'l'), inplace= True)
df['name'].isnull().sum()
#replacing all missing values in 'reviews_per_month' column
df['reviews_per_month'].replace((np.nan, 0), inplace=True)
df['reviews_per_month'].isna().sum()
#let's take a look at correlation between our different variables
corr=df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
#Which neighborhoods have the most Airbnb's
fig=px.histogram(df, x='neighbourhood')
fig.show()

#What are the most common room types in Williamsburg?
fig=px.histogram(df, x='room_type', facet_col='neighbourhood_group', facet_col_wrap=1)
fig.update_yaxes(matches=None)
fig.show()
#let's visualize location vs price on an interactive map
fig=px.scatter_mapbox(df, lat= 'latitude', lon= 'longitude', size='price', color='price')
fig.update_layout(mapbox_style="open-street-map")
fig.show()
from wordcloud import WordCloud
plt.subplots(figsize=(25,15))
wcloud= WordCloud().generate(''.join(df['name']))
plt.imshow(wcloud)
plt.show()
#we'll start off by doing some preprocessing
#dropping the name column
df2=df.drop('name', axis=1)
df2
#label encoding vs One Hot encoding: seeing which performs better
#i wanted to see which method of encoding works best for this problem type so I defined both
from sklearn.preprocessing import LabelEncoder

df_drp= df.drop(['name', 'neighbourhood_group', 'neighbourhood', 'room_type'], axis=1) #the dropped columns df for upcoming encoding

#OHE
df_OHE=pd.get_dummies(df, columns=['neighbourhood_group', 'neighbourhood', 'room_type'])
df_finalOHE=pd.concat([df_drp, df_OHE], axis=1)

#LE
df_le=df[['neighbourhood_group', 'neighbourhood', 'room_type']].apply(LabelEncoder().fit_transform)
df_finalLE=pd.concat([df_drp, df_le], axis=1)
df_finalLE


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics

#We'll first try the OHE data
#we first scale the data
scl=StandardScaler()
x1=df_finalOHE.loc[:, df_finalOHE.columns!= 'price']
df_finalOHE = df_finalOHE.loc[:,~df_finalOHE.columns.duplicated()] #for some reason I had a duplicate price column which I got rid of
y1=df_finalOHE['price']
x1

x2=df_finalLE.loc[:, df_finalLE.columns != 'price']
y2=df_finalLE['price']
x2
df_finalOHE.drop('name', axis= 1, inplace= True) #for some reason the 'name' column carried over so I just dropped it again
df_finalOHE
#let's split our data into training and testing datasets for One Hot Encoding
x1_train, x1_test, y1_train, y1_test= train_test_split(x1, y1, test_size= 0.2, random_state= 1)
x1_train
#splitting the data into train/test sets for the label encoding method
x2_train, x2_test, y2_train, y2_test= train_test_split(x2, y2, test_size=0.2, random_state= 1)
x2_train
#now applying polynomial Linear Regression using pipelines
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree=2)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
#training the model using the One Hot Encoding method
pipe.fit(x1_train.drop('name', axis= 1), y1_train)

pipe.fit(x2_train, y2_train)
pipe.fit(x2_train, y2_train)
y1_hat=pipe.predict(x1_test)
y1_hat
y2_hat=pipe.predict(x2_test)
y2_hat
from sklearn.metrics import r2_score
OHE_acc=r2_score(y1_test, y1_hat)
OHE_acc
OHE_acc2=r2_score(y2_test, y2_hat)
OHE_acc2
#my model scores are stil very shitty, so let's see if dropping some of the other columns help with improving model
#accuracy
from sklearn.metrics import mean_squared_error
rmses = []
dgr = np.arange(1, 5)
min_rmse, min_deg = 1e10, 0

for i in dgr:
    Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree=i)), ('model',LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(x2_train, y2_train)
    
    # Compare with test data
    y2_hat = pipe.predict(x2_test)
    LE_MSE = mean_squared_error(y2_test, y2_hat)
    LE_rmse = np.sqrt(LE_MSE)
    rmses.append(LE_rmse)

    # Cross-validation of degree
    if min_rmse > LE_rmse:
        min_rmse = LE_rmse
        min_deg = i

# Plot and present results
print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dgr, rmses)
ax.set_yscale('log')
ax.set_xlabel('Degree')
ax.set_ylabel('RMSE')
    
min_rmse


