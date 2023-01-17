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
crop=pd.read_csv('/kaggle/input/crop-production-statistics-from-1997-in-india/apy.csv')
def basic_description (dataframe):

    print('shape:\n',dataframe.shape)

    print('head:\n',dataframe.head())

    print('info:\n',dataframe.info())

    print('null value:\n',dataframe.isnull().sum()*100/len(dataframe))

    print('describe:\n',dataframe.describe())

basic_description(crop)
crop.dropna(subset=['Production'],axis=0,inplace=True)
basic_description(crop)
pd.set_option('display.max_rows', 200000)

pd.set_option('display.max_columns', 10)

pd.set_option('display.width', 1000)
import seaborn as sns

import matplotlib.pyplot as plt
crop.columns
# for i in ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop']: 

#     sns.boxplot(data=crop,y='Production',x=crop[i])

#     plt.show()

    
sns.heatmap(crop.corr(),annot=True)

plt.show()
crop.Crop.value_counts()
# cat_crop={'Cereal':['Rice','Maize','Wheat','Barley','Varagu','Other Cereals & Millets','Ragi','Small millets','Bajra','Jowar'],

#           'Pulses':['Moong','Urad','Arhar/Tur','Peas & beans','Masoor','Other Kharif pulses','other misc. pulses','Ricebean (nagadal)','Rajmash Kholar','Lentil','Samai','Blackgram','Korra','Cowpea(Lobia)','Other  Rabi pulses','Other Kharif pulses','Peas & beans (Pulses)'],

#          'Fruits':['Peach','Apple','Litchi','Pear','Plums','Ber','Sapota','Lemon','Pome Granet','Other Citrus Fruit','Water Melon','Jack Fruit','Grapes','Pineapple','Orange','Pome Fruit','Citrus Fruit','Other Fresh Fruits','Mango','Papaya','Coconut','Banana'],

#          ('Bean','Lab-Lab','Moth','Guar seed','Tapioca','Soyabean','Horse-gram','Gram'):'Beans',

#          ('Turnip','Peas','Beet Root','Carrot','Yam','Ribed Guard','Ash Gourd ','Pump Kin','Redish','Snak Guard','Bottle Gourd','Bitter Gourd','Cucumber','Drum Stick','Cauliflower','Beans & Mutter(Vegetable)','Cabbage','Bhindi','Tomato','Brinjal','Khesari','Sweet potato','Potato','Onion'):'Vegetable',

#           ('Perilla','Colocosia','Ginger','Cardamom','Black pepper','Dry ginger','Garlic','Coriander','Turmeric','Dry chillies'):'Species',

#          ('Jobster','Cond-spcs other'):'Others',

#          ('other fibres','Kapas','Jute & mesta','Jute','Mesta','Cotton(lint)'):'fibres',

#          ('Arcanut (Processed)','Atcanut (Raw)','Cashewnut Processed','Cashewnut Raw','Cashewnut','Arecanut','Groundnut'):'Nuts',

#          'Rubber':'Natural Polymer',

#          'Coffee':'Coffee',

#          'Tea':'Tea',

#          'Total foodgrain':'Total foodgrain',

#          'Pulses total':'Pulses total',

#          'Oilseeds total':'Oilseeds total',

#          'Paddy':'Paddy',

#          ('other oilseeds','Safflower','Niger seed','Castor seed','Linseed','Sunflower','Rapeseed &Mustard','Sesamum'):'Oilseeds',

#          'Sannhamp':'Fertile Plant',

#          'Tobacco':'Commercial',

#          'Sugarcane':'Sugarcane'}
cc=crop['Crop']

def cat_crop(cc):

    for i in ['Rice','Maize','Wheat','Barley','Varagu','Other Cereals & Millets','Ragi','Small millets','Bajra','Jowar']:

        if cc==i:

            return 'Cereal'

    for i in ['Moong','Urad','Arhar/Tur','Peas & beans','Masoor',

              'Other Kharif pulses','other misc. pulses','Ricebean (nagadal)',

              'Rajmash Kholar','Lentil','Samai','Blackgram','Korra','Cowpea(Lobia)',

              'Other  Rabi pulses','Other Kharif pulses','Peas & beans (Pulses)']:

        if cc==i:

            return 'Pulses'

    for i in ['Peach','Apple','Litchi','Pear','Plums','Ber','Sapota','Lemon','Pome Granet',

               'Other Citrus Fruit','Water Melon','Jack Fruit','Grapes','Pineapple','Orange',

               'Pome Fruit','Citrus Fruit','Other Fresh Fruits','Mango','Papaya','Coconut','Banana']:

        if cc==i:

            return 'Fruits'

    for i in ['Bean','Lab-Lab','Moth','Guar seed','Tapioca','Soyabean','Horse-gram','Gram']:

        if cc==i:

            return 'Beans'

    for i in ['Turnip','Peas','Beet Root','Carrot','Yam','Ribed Guard','Ash Gourd ','Pump Kin','Redish','Snak Guard','Bottle Gourd',

              'Bitter Gourd','Cucumber','Drum Stick','Cauliflower','Beans & Mutter(Vegetable)','Cabbage',

              'Bhindi','Tomato','Brinjal','Khesari','Sweet potato','Potato','Onion']:

        if cc==i:

            return 'Vegetables'

    for i in ['Perilla','Colocosia','Ginger','Cardamom','Black pepper','Dry ginger','Garlic','Coriander','Turmeric','Dry chillies']:

        if cc==i:

            return 'Species'

    for i in ['Jobster','Cond-spcs other']:

        if cc==i:

            return 'Other'

    for i in ['other fibres','Kapas','Jute & mesta','Jute','Mesta','Cotton(lint)']:

        if cc==i:

            return 'fibres'

    for i in ['Arcanut (Processed)','Atcanut (Raw)','Cashewnut Processed','Cashewnut Raw','Cashewnut','Arecanut','Groundnut']:

        if cc==i:

            return 'Nuts'

    for i in ['Rubber']:

        if cc==i:

            return 'Natural Polymer'

    for i in ['Coffee']:

        if cc== i:

            return 'Coffee'

    for i in ['Tea']:

        if cc==i:

            return 'Tea'

    for i in ['Total foodgrain']:

        if cc==i:

            return 'Total foodgrain'

    for i in ['Pulses total']:

        if cc==i:

            return 'Pulses total'

    for i in ['Oilseeds total']:

        if cc==i:

            return 'Oilseeds total'

    for i in ['Paddy']:

        if cc==i:

            return 'Paddy'

    for i in ['other oilseeds','Safflower','Niger seed','Castor seed','Linseed','Sunflower','Rapeseed &Mustard','Sesamum']:

        if cc==i:

            return 'Oilseeds'

    for i in ['Sannhamp']:

        if cc==i:

            return 'Fertile Plant'

    for i in ['Tobacco']:

        if cc==i:

            return 'Commercial'

    for i in ['Sugarcane']:

        if cc==i:

            return 'Sugarcane'
crop['cat_crop']=crop['Crop'].apply(cat_crop)
crop
crop.cat_crop.value_counts()
plt.figure(figsize=(20,8))

sns.scatterplot(data=crop,x='State_Name',y='cat_crop',hue='Season')

plt.xticks(rotation=90)

plt.show()
vc=crop['cat_crop'].value_counts()

explodelist=[0]*len(vc)

explodelist[11]=0.8

vc.plot(kind="pie",autopct="%1.1f",radius=3,rotatelabels=True,pctdistance=0.6,labeldistance=1.1,explode=explodelist)

plt.show()
%matplotlib inline

plt.figure(figsize=(18,5))

sns.barplot(data=crop,x='Crop_Year',y='Production',hue='cat_crop',errwidth=0.0,palette='magma') 

plt.xticks(rotation=90)

plt.show()

    
from datetime import datetime
trend=crop[['Crop_Year','Production']]
trend['Crop_Year']=pd.to_datetime(trend['Crop_Year'],format='%Y')

trend['year']=trend['Crop_Year'].dt.year
trend=trend.set_index(['year'])

trend.drop('Crop_Year',axis=1,inplace=True)

plt.figure(figsize=(20,8))

trend.plot(grid=True,figsize=(10,5))

plt.show()
crop.columns
import warnings

warnings.filterwarnings('ignore')

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split
crop_ml=crop[['State_Name','Crop_Year','Season','cat_crop','Production']]
crop_ml_dummy=pd.get_dummies(data=crop_ml,columns=['State_Name','Season','cat_crop'],drop_first=True)
X=crop_ml_dummy.drop('Production',axis=1)

y=crop_ml_dummy.Production
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
model=LinearRegression()

rfe=RFE(model,50)

rfe.fit(X_train,y_train)

pd.DataFrame(list(zip(X.columns, rfe.support_, rfe.ranking_)), 

            columns=['cols', 'select', 'rank'])
y_pred = rfe.predict(X_test)

y_train_pred=rfe.predict(X_train)

print('R2 test:  ', r2_score(y_test, y_pred))

print('R2 train:', r2_score(y_train,y_train_pred))

print('rmse test:',np.sqrt(mean_squared_error(y_train, y_train_pred)))

print('rmse test:  ', np.sqrt(mean_squared_error(y_test, y_pred)))
model=DecisionTreeRegressor()

rfe1=RFE(model,50)

rfe1.fit(X_train,y_train)

pd.DataFrame(list(zip(X.columns, rfe1.support_, rfe1.ranking_)), 

            columns=['cols', 'select', 'rank'])
y_pred_dtr=rfe1.predict(X_test)

y_train_pred_dtr=rfe1.predict(X_train)

print('R2 test:  ', r2_score(y_test, y_pred_dtr))

print('R2 train:  ', r2_score(y_train, y_train_pred_dtr))

print('difference of r2 score:',r2_score(y_train, y_train_pred_dtr)-r2_score(y_test, y_pred_dtr))

print('rmse test:  ', np.sqrt(mean_squared_error(y_test, y_pred_dtr)))

print('rmse train:  ', np.sqrt(mean_squared_error(y_train, y_train_pred_dtr)))
model=RandomForestRegressor()

rfe2=RFE(model,50)

rfe2.fit(X,y)

pd.DataFrame(list(zip(X.columns, rfe2.support_, rfe2.ranking_)), 

            columns=['cols', 'select', 'rank'])
y_pred_rf=rfe2.predict(X)

print('R2:  ', r2_score(y, y_pred_rf))

print('rmse:  ', np.sqrt(mean_squared_error(y, y_pred_rf)))