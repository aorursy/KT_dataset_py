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
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor 

from sklearn.preprocessing import PolynomialFeatures 

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from mpl_toolkits.mplot3d import Axes3D

import folium

from folium.plugins import HeatMap

%matplotlib inline

import warnings 

warnings.filterwarnings('ignore')



evaluation=pd.DataFrame({'Model':[]

                         ,'Details':[]

                         ,'Root mean squared error(RMSE)':[]

                         ,'R-Squared(training)':[]

                         ,'Adjusted R Squared(training)':[]

                         ,'R-Squared(test)':[]

                         ,'Adjusted R-Squared Test':[]

                         ,'5-Fold Cross Validation':[]

                        }

                       )

df=pd.read_csv("../input/kc-house-dataset/kc_house_data.csv")
df.head()
df.columns
df.shape
df.describe()
df.info()
ax=sns.heatmap(df.corr())
train_data,test_data=train_test_split(df,train_size=0.8,random_state=3)



LR=linear_model.LinearRegression()

X_train=np.array(train_data['sqft_living'],dtype=pd.Series).reshape(-1,1)

Y_train=np.array(train_data['price'],dtype=pd.Series)

LR.fit(X_train,Y_train)



X_test=np.array(test_data['sqft_living'],dtype=pd.Series).reshape(-1,1)

Y_test=np.array(test_data['price'],dtype=pd.Series)



pred=LR.predict(X_test)

rmsesm=float(format(np.sqrt(metrics.mean_squared_error(Y_test,pred)),'.3f'))

rtrsm=float(format(LR.score(X_train,Y_train),'.3f'))

rtesm=float(format(LR.score(X_test,Y_test),'.3f'))

cv=float(format(cross_val_score(LR,df[['sqft_living']],df[['price']],cv=5).mean(),'.3f'))



print("Average price for test data:{:.3f}".format(Y_test.mean()))

print("Intercept:{}".format(LR.intercept_))

print("Coefficient:{}".format(LR.coef_))



r=evaluation.shape[0]

evaluation.loc[r]=['Simple Linear Regression','-',rmsesm,rtrsm,'-',rtesm,'-',cv]

evaluation
sns.set(style="white",font_scale=1)
plt.figure(figsize=(6.5,5))

plt.scatter(X_test,Y_test,color='red',label='data',alpha=0.1)

plt.plot(X_test,LR.predict(X_test),color='black',label='predicted regression line')

plt.xlabel("Living Spce(sqft)",fontsize=15)

plt.ylabel("Price ($)",fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.legend()

plt.show()
DF=df[['id' ,'date','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']]

H=DF.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)

[x.title.set_size(12) for x in H.ravel()];

[x.yaxis.tick_left() for x in H.ravel()];
sns.set(style="whitegrid",font_scale=1)
f,axes=plt.subplots(1,2,figsize=(15,5))

sns.boxplot(x=df['bedrooms'],y=df['price'],ax=axes[0])

sns.boxplot(x=df['floors'],y=df['price'],ax=axes[1])

sns.despine(left=True,bottom=True)

axes[0].set(xlabel='Bedrooms',ylabel='Price')

axes[0].yaxis.tick_left()

axes[1].set(xlabel='Floors',ylabel='Price')

axes[1].yaxis.tick_right()

axes[1].yaxis.set_label_position('right')



f,axe=plt.subplots(1,1,figsize=(12,5))

sns.despine(left=True,bottom=True)

sns.boxplot(x=df['bathrooms'],y=df['price'],ax=axe)

axe.yaxis.tick_left()

axe.set(xlabel='Bathrooms/Bedrooms',ylabel='Price');
fig=plt.figure(figsize=(18,12))

ax=fig.add_subplot(2,2,1,projection='3d')

ax.scatter(df['floors'],df['bedrooms'],df['bathrooms'],c='darkgreen',alpha=0.5)

ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nBathrooms/Bedrooms')

ax.set(ylim=[0,12]);



ax=fig.add_subplot(2,2,2,projection="3d")

ax.scatter(df['floors'],df['bedrooms'],df['sqft_living'],c='darkgreen',alpha=0.5)

ax.set(xlabel='\nFloors',ylabel='\nBedrooms',zlabel='\nsqft Living')

ax.set(ylim=[0,12]);



ax=fig.add_subplot(2,2,3,projection="3d")

ax.scatter(df['sqft_living'],df['sqft_lot'],df['bathrooms'],c='darkgreen',alpha=0.5)

ax.set(xlabel='\sqft living',ylabel='\nsqft lot',zlabel='\bathrooms/Bedrooms')

ax.set(ylim=[0,25000]);



ax=fig.add_subplot(2,2,4,projection="3d")

ax.scatter(df['sqft_living'],df['sqft_lot'],df['bedrooms'],c='darkgreen',alpha=0.5)

ax.set(xlabel='\nsqft living',ylabel='\nsqft lot',zlabel='\Bedrooms')

ax.set(ylim=[0,25000]);





f ,axes= plt.subplots(1,2, figsize=(15,5))

sns.boxplot(x=df['waterfront'],y=df['price'],ax=axes[0])

sns.boxplot(x=df['view'],y=df['price'],ax=axes[1])

sns.despine(left=True,bottom=True)

axes[0].set(xlabel='Waterfront',ylabel='Price')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position('right')

axes[1].set(xlabel='View',ylabel='Price')

axes[1].yaxis.tick_right()



f ,axe= plt.subplots(1,1, figsize=(15,5))

sns.boxplot(x=df['grade'],y=df['price'],ax=axe)

sns.despine(left=True,bottom=True)

axe.set(xlabel='Grade',ylabel='Price')



axe.yaxis.tick_left();





fig=plt.figure(figsize=(10,5))

ax=fig.add_subplot(1,1,1,projection='3d')

ax.scatter(train_data['view'],train_data['grade'],train_data['yr_built'],alpha=0.5)

ax.set(xlabel='\nView',ylabel='\nGrade',zlabel='\nYear Built');
# Creating a Map using folium library to see Map view distribution of houses
maxpr=df.loc[df['price'].idxmax()]



def generateBaseMap(default_location= [47.5112, -122.257],default_zoom_start=9.4):

    base_map=folium.Map(location=default_location,control_scale=True,zoom_start=default_zoom_start)

    return base_map

    

df_copy=df.copy()

# df['zipcode'].head()

# this will give first 4 zipcode. Select any one to put in the heatmap



df_copy['count']=1

basemap=generateBaseMap()



folium.TileLayer('cartodbpositron').add_to(basemap)

s=folium.FeatureGroup(name='icon').add_to(basemap)





folium.Marker([maxpr['lat'],maxpr['long']],popup='Highest Price : $'+str(format(maxpr['price'],'0.0f')),

             icon=folium.Icon(color='red')).add_to(s)



HeatMap(data=df_copy[['lat','long','count']].groupby(['lat','long']).sum().reset_index().values.tolist(),

       radius=8,max_zoom=13,name='Heat Map').add_to(basemap)

folium.LayerControl(collapsed=False).add_to(basemap)



basemap
features=['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']

mask=np.zeros_like(df[features].corr(),dtype=np.bool)

mask[np.triu_indices_from(mask)]=True



f,ax=plt.subplots(figsize=(15,12))

plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap='BuGn',

           linecolor='w',annot=True,annot_kws={'size':8},mask=mask,cbar_kws={'shrink':0.9});
df_dm=df.copy()

df_dm.head()

# Above information has a 'date' column from which we will extract only the year to know the year of sale of house



df_dm['sales_yr']=df_dm['date'].astype(str).str[:4]



# Now to extract feature for better outlook of data, we will find out the age of house which will be a difference between 'sales_yr' & 'yr_built'

df_dm['age']=df_dm['sales_yr'].astype(int)-df_dm['yr_built']



# To better our preprcessing steps, we will take inot account the age of house renovated; which is an exact difference between 'yr_sales' and 'yr_renovated'

df_dm['age_rnv']=0

df_dm['age_rnv']=df_dm['sales_yr'][df_dm['yr_renovated']!=0].astype(int)-df_dm['yr_renovated'][df_dm['yr_renovated']!=0]

df_dm['age_rnv'][df_dm['age_rnv'].isnull()]=0
# Paritioning age into bins



bins=[-2,0,5,10,25,50,75,100000]

labels=['<1','1-5','6-10','11-25','26-50','51-75','>75']

df_dm['age_binned']=pd.cut(df_dm['age'],bins=bins,labels=labels)





#Partitioning age_rnv into bins



bins=[-2,0,5,10,25,50,75,100000]

labels=['<1','1-5','6-10','11-25','26-50','51-75','>75']

df_dm['age_rnv_binned']=pd.cut(df_dm['age_rnv'],bins=bins,labels=labels)

f,axes=plt.subplots(1,2,figsize=(12,5))

p1=sns.countplot(df_dm['age_binned'],ax=axes[0])

for i in p1.patches:

        height=i.get_height()

        p1.text(i.get_x()+i.get_width()/2,height+200,height,ha="center")

        

p2=sns.countplot(df_dm['age_rnv_binned'],ax=axes[1])



for i in p2.patches:

    height=i.get_height()

    p2.text(i.get_x()+i.get_width()/2,height+200,height,ha="center")

    

axes[0].set(xlabel='Age')

axes[0].yaxis.tick_left()

axes[1].yaxis.set_label_position('right')

axes[1].yaxis.tick_right()

axes[1].set(xlabel='Renovation Age')



# Transform the factor values to be able to use in the model



df_dm=pd.get_dummies(df_dm,columns=['age_binned','age_rnv_binned'])
def adjustedR2(r2,n,k):

    return r2- (k-1)/(n-k)*(1-r2)



train_data_dm,test_data_dm=train_test_split(df_dm,train_size=0.8,random_state=3)



features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']



complex_model_1=linear_model.LinearRegression()

complex_model_1.fit(train_data_dm[features],train_data_dm['price'])



print('Intercept:{}'.format(complex_model_1.intercept_))

print('Coefficients:{}'.format(complex_model_1.coef_))





pred=complex_model_1.predict(test_data_dm[features])

rmsecm=float(format(np.sqrt(metrics.mean_squared_error(test_data_dm['price'],pred)),'0.3f'))

rtrcm=float(format(complex_model_1.score(train_data_dm[features],train_data_dm['price']),'0.3f'))

artrcm=float(format(adjustedR2(complex_model_1.score(train_data_dm[features],train_data_dm['price']),train_data_dm.shape[0],len(features)),'0.3f'))

rtecm=float(format(complex_model_1.score(test_data_dm[features],test_data_dm['price']),'0.3f'))

artecm=float(format(adjustedR2(complex_model_1.score(test_data_dm[features],test_data_dm['price']),test_data_dm.shape[0],len(features)),'0.3f'))



cv=float(format(cross_val_score(complex_model_1,df_dm[features],df_dm['price'],cv=5).mean(),'0.3f'))



r=evaluation.shape[0]

evaluation.loc[r]=['Multiple Regression-1','selected features', rmsecm,rtrcm,artrcm,rtecm,artecm,cv]



evaluation.sort_values(by='5-Fold Cross Validation',ascending=False)


