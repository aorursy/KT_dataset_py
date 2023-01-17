import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/car-data/CarPrice_Assignment.csv');

data.head()
#Checking dataset's info

print(data.shape);

data.info()
data['CarName'].unique()
#Too many unique values. Need to split it into single car names

data['CarName']=data['CarName'].apply(lambda x: x.split(' ')[0]);

print(data['CarName'].head());

data['CarName'].unique()
# Similar car names have different spellings;



def Name_replace(name1,name2):

    data['CarName'].replace(name1,name2,inplace=True);



Name_replace('maxda','mazda');

Name_replace('Nissan','nissan');

Name_replace('porcshce','porsche');

Name_replace('toyouta','toyota');

Name_replace('vokswagen','volkswagen');

Name_replace('vw','volkswagen');



data['CarName'].unique()
data.columns
data.skew()
def scatter_plot(x,figure):

    plt.subplot(5,2,figure);

    plt.scatter(data[x],data['price']);

    plt.title(x+' vs Price');

    plt.ylabel('price');

    plt.xlabel(x);

    plt.xticks(rotation=90);



def count_plot(y,palette):

    fig=plt.subplots(figsize=(20,6));

    sns.countplot(y,data=data,palette=palette);

    plt.xticks(rotation=90);

    

def avg_var_plot(z):

    data1=pd.DataFrame(data.groupby([z])['price'].mean().sort_values(ascending=False)).plot.bar(figsize=(20,8));

    plt.title(z+' vs AvgPrice');

    plt.xticks(rotation=90);

    plt.ylabel('price')
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline



fig=plt.subplots(figsize=(20,6));

plt.bar(data['CarName'],data['price']);

plt.xlabel('Car Name');

plt.ylabel('Price');

plt.title('Price vs CarName')

plt.xticks(rotation=90);
count_plot('CarName','hls');

avg_var_plot('CarName');

#Toyota is the most preferred choice of car and is less expensive.
count_plot('symboling','viridis');

avg_var_plot('symboling');
count_plot('fueltype','gnuplot');

avg_var_plot('fueltype');



#Gas is the most preferred fuel type and is cheaper than diesel.
count_plot('aspiration','tab10');

avg_var_plot('aspiration');



#Std is most preferred and is cheaper.
count_plot('doornumber','husl');

avg_var_plot('doornumber');



#there is not much difference between the door numbers in terms of price and count
count_plot('carbody','husl');

avg_var_plot('carbody');



#Sedan is the most preferred choice and is cheaper
count_plot('drivewheel','Set2');

avg_var_plot('drivewheel');



#fwd is preferred and is cheap
count_plot('enginelocation','Paired');

avg_var_plot('enginelocation');



#Cars with engine at the front are more in number and relatively cheap
count_plot('enginetype','rocket');

avg_var_plot('enginetype');



#Ohc is the preferred engine type and is the cheapest
count_plot('cylindernumber','mako');

avg_var_plot('cylindernumber');



#cars with 4 cylinder number are in abundance and are relatively cheaper
count_plot('fuelsystem','magma');

avg_var_plot('fuelsystem');
fig=plt.subplots(figsize=(20,6))

sns.distplot(data.price)
fig=plt.subplots(figsize=(20,6));

sns.barplot(x='CarName',y='price',hue='doornumber',data=data)

plt.title('Cars with number of doors each');
data['fueltype'].unique()
fig=plt.subplots(figsize=(20,6));

sns.barplot(x='CarName',y='price',hue='fueltype',data=data);
plt.figure(figsize=(20,8));

sns.boxplot(x=data['symboling'],y=data['price'],palette='gnuplot')

plt.title('Price vs Symboling');
plt.figure(figsize=(30,15));

scatter_plot('carlength',1);

scatter_plot('carwidth',2);

scatter_plot('carheight',3);

scatter_plot('curbweight',4);

plt.tight_layout();



#Carlength, carwidth, curbweigt show linear positive trend with price whereas carheight show no trend with price;
plt.figure(figsize=(30,15));

scatter_plot('wheelbase',1);

scatter_plot('enginesize',2);

scatter_plot('boreratio',3);

scatter_plot('stroke',4);

plt.tight_layout()

#wheelbase, enginesize, boreratio show positive trend with price;
plt.figure(figsize=(30,15));

scatter_plot('compressionratio',1);

scatter_plot('horsepower',2);

scatter_plot('peakrpm',3);

scatter_plot('citympg',4);

scatter_plot('highwaympg',5);

plt.tight_layout();

#highwaympg, citympg, peakrpm, compression ratio show negative or no trend with price
cols_to_drop=['highwaympg','citympg','peakrpm','compressionratio','stroke','doornumber','car_ID','symboling','enginelocation','CarName'];

data.drop(cols_to_drop,axis=1,inplace=True);

data.head()
print(data.shape);

data.info()
sns.pairplot(data)
y=data['price'];

X=data.drop('price',axis=1,inplace=True);

data.head()
data.describe()
#Splitting the data

from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test=train_test_split(data,y,train_size=0.7, test_size=0.3,random_state=0);



#Separating Categorical and Numerical Columns

cat_cols=[cname for cname in X_train.columns if X_train[cname].nunique()<10 and X_train[cname].dtype=='object'];

num_cols=[cname for cname in X_train.columns if X_train[cname].dtype in ['int64','float64']];



my_cols=cat_cols + num_cols;

X_train=X_train[my_cols].copy();

X_test=X_test[my_cols].copy();
X_train['enginetype'].unique()
X_test['enginetype'].unique()
print(X_train['cylindernumber'].unique());

print(X_test['cylindernumber'].unique());

print(X_train['fuelsystem'].unique());

print(X_test['fuelsystem'].unique());
print(X_train.shape);

print(X_test.shape);

print(y_train.shape);

print(y_test.shape);
good_label_cols = [col for col in cat_cols if 

                   set(X_train[col]) == set(X_test[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(cat_cols)-set(good_label_cols))
good_label_cols
bad_label_cols
#Label Encoding Categorical Data;

from sklearn.preprocessing import LabelEncoder



label_X_train=X_train.drop(bad_label_cols,axis=1);

label_X_test=X_test.drop(bad_label_cols,axis=1);



encoded=LabelEncoder();

for col in good_label_cols:

    label_X_train[col]=encoded.fit_transform(X_train[col]);

    label_X_test[col]=encoded.transform(X_test[col]);

label_X_train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score



model=RandomForestRegressor(n_estimators=100,random_state=0);

model.fit(label_X_train,y_train);

prediction=model.predict(label_X_test);

print(r2_score(prediction,y_test));