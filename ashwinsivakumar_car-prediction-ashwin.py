# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading the dataset
cars = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
cars.columns
cars
car=cars.select_dtypes(exclude='object')
car
f=cars.select_dtypes(exclude='float')
d=cars.select_dtypes(exclude='object')
cars.describe()
#Extracting Car Company from the CarName as per direction in Problem 
CompanyName = cars['CarName'].apply(lambda x : x.split(' ')[0])
cars.insert(2,"CompanyName",CompanyName)
carname = cars['CarName'].apply(lambda y : y.split(' ')[-1])
cars.insert(4,"carname",carname)
del cars['CarName']
cars.head()
#Changing spelling mistakes in carname
def replace_name(a,b):
    cars.CompanyName.replace(a,b,inplace=True)

replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')
cars.CompanyName.unique()

del cars['citympg']
del cars['highwaympg']
del cars['car_ID']
cars
plt.figure(figsize=(8,8))
sns.heatmap(cars.corr())
plt.show()

plt.title('copmpany distribution in dataset')
cars['CompanyName'].value_counts().plot(kind='bar')
plt.title('company distribution')


def ordinality_check(a):
    sns.countplot(x=a)
plt.figure(figsize=(25,25))
plt.subplot(3,3,1)
ordinality_check(f.fueltype)
plt.subplot(3,3,2)
ordinality_check(f.aspiration)
plt.subplot(3,3,3)
ordinality_check(f.drivewheel)
plt.subplot(3,3,4)
ordinality_check(f.doornumber)
plt.subplot(3,3,5)
ordinality_check(f.enginelocation)
plt.subplot(3,3,6)
ordinality_check(f.enginetype)
plt.subplot(3,3,7)
ordinality_check(f.carbody)
cars
#labelencode
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
cars['fueltype']=le.fit_transform(f.fueltype)
cars['aspiration']=le.fit_transform(f.aspiration)
cars['doornumber']=le.fit_transform(f.doornumber)
cars['enginelocation']=le.fit_transform(f.enginelocation)
cars


def ordinality_check(a):
    sns.barplot(x=a,y=cars.price)
plt.figure(figsize=(25,25))
plt.subplot(3,3,1)
ordinality_check(f.fueltype)
plt.subplot(3,3,2)
ordinality_check(f.aspiration)
plt.subplot(3,3,3)
ordinality_check(f.drivewheel)
plt.subplot(3,3,4)
ordinality_check(f.doornumber)
plt.subplot(3,3,5)
ordinality_check(f.enginelocation)
plt.subplot(3,3,6)
ordinality_check(f.enginetype)
plt.subplot(3,3,7)
ordinality_check(f.carbody)





d.columns

def ordinality_check(a):
    sns.boxplot( x=a)    
plt.figure(figsize=(25,25))
plt.subplot(4,3,1)
ordinality_check(d.price)
plt.subplot(4,3,2)
ordinality_check(d.boreratio)
plt.subplot(4,3,3)
ordinality_check(d.compressionratio)
plt.subplot(4,3,4)
ordinality_check(d.horsepower)
plt.subplot(4,3,5)
ordinality_check(d.stroke)
plt.subplot(4,3,6)
ordinality_check(d.peakrpm)
plt.subplot(4,3,7)
ordinality_check(d.carheight)
plt.subplot(4,3,8)
ordinality_check(d.carlength)
plt.subplot(4,3,9)
ordinality_check(d.carwidth)
plt.subplot(4,3,10)
ordinality_check(d.wheelbase)



   

cars['price'].skew()
from scipy import stats
price_log=stats.boxcox(cars['price'])[0]
pd.Series(price_log).skew()
ordinality_check(cars.price)

