# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
plt.rc('figure', figsize = (15,8))
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
data.head()
# Yes our data has 10692 lines and 13 columns.
data.shape
# checking types of data, is imporant to check the data.head() and then look at this infos
# we see that floor is an object
data.info()
# counting how many null values has in data
# we see that floor is an object
data.isna().sum()
# first checking unique values from floor
# we found that we have '-' values and an outlier '301'
data['floor'].unique()
# so lets see how many is the distribuition for '-' and '301'
# we see that '-' has a lot of values,  
data['floor'].value_counts()
# we will transform '-' and '301' to median, but first we have to replace to 0 both of them and them transform to integers
data['floor_num'] = data['floor'].replace(['-','301'], 0)
data['floor_num'].value_counts()
# transforming an object to integer, for this we first convert to str and them to int
data['floor_num'] = data['floor_num'].astype(str).astype(int)
# now we have a good floor type now, as integer.
data.info()
data['floor_num'].value_counts()
# we have to replace the value '0' for the median
median_floor = data['floor_num'].median()
data['floor_num'] = data['floor_num'].replace(0, median_floor)
data['floor_num'].value_counts()
# checking summary statistics
data.describe()
# lets explore the categorical data
# floor doesn't count
categorical = [var for var in data.columns if data[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

# checking unique variables from all cathegorical variables
data['animal'].unique(), data['furniture'].unique(), data['city'].unique()
# so we have 3 categorical data, we can explore these data here
# view the frequency 
for var in categorical:
    print(data[var].value_counts()/np.float(len(data)))
# lets plot the data and see how it goe
ax = data['city'].value_counts().plot(kind='bar',
                                    figsize=(10,8))
ax.set_xlabel('Cities', fontsize = 20)
ax.set_ylabel('Count', fontsize = 20)
ax.set_title('Count of cities', fontsize = 25)
ax = data['animal'].value_counts().plot(kind = 'bar',
                                       figsize = (10,8),
                                       color = 'brown')
ax.set_title('Accept or not animals in the $House$', fontsize = 22)
ax.set_xlabel('Y or N', fontsize = 15)
ax.set_ylabel('Quantitiy of houses' , fontsize = 15)

ax = data['furniture'].value_counts().plot(kind = 'bar',
                                       figsize = (10,8),
                                       color = 'green')
ax.set_title('Houses with furnitire or not', fontsize = 22)
ax.set_xlabel('Y or N', fontsize = 15)
ax.set_ylabel('Quantitiy of houses' , fontsize = 15)


# now we are going to explore numerical data.
numerical = [var for var in data.columns if data[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)
data.describe()
         
# really a 1120000 rent ? who pays that ?
(data['total (R$)'].describe()).round(2)
# boxplot
# we can look that there are a lot of outliers in this data
# probably the worst boxplot EVER!
sns.boxplot(x = 'city', y ='total (R$)', data = data)
# let's remove the outliers just to make sure that we have GOOD data for the analysis
city_group = data.groupby('city')['total (R$)']
# checking if its a series... because we will use that later
type(city_group)
# so lets remove the outliers by this method
Q1 = city_group.quantile(.25)
Q3 = city_group.quantile(.75)
IIQ = Q3 - Q1  #interquartile range
lower_limit = Q1 - 1.5* IIQ
upper_limit = Q3 + 1.5* IIQ
# checking if its ok
print('Q1 Result is ',  Q1)
print('Q3 Result is ',  Q3)
## seems to be working these values..
# now lets remove the outliers by group city
# creating a new dataframe with only the values that they are between lower limit and upper limit
data_new = pd.DataFrame()

for city in city_group.groups.keys():
        is_city = data['city'] == city
        accept_limit = (data['total (R$)'] >= lower_limit[city]) & (data['total (R$)'] <= upper_limit[city])
        select = is_city & accept_limit
        data_select = data[select]
        data_new = pd.concat([data_new, data_select])
data_new.head()
sns.boxplot(x = 'city', y ='total (R$)', data = data)
sns.boxplot(x = 'city', y ='total (R$)', data = data_new)
# now we are going to explore numerical data.
# correlation plot
fig = plt.figure(figsize=(10,10))
corr = data_new.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True);
# i will make a scatter plot for each variable that has a good correlation with total (R$)
area = plt.figure()
g1 = area.add_subplot(2,2,1)
g2 = area.add_subplot(2,2,2)
g3 = area.add_subplot(2,2,3)
g4 = area.add_subplot(2,2,4)

## Lets check scatter plot total (R$) with fire insurance, property tax(R$), rent amount (R$),bathroom
g1.scatter(data_new['fire insurance (R$)'], data_new['total (R$)'])
g1.set_title('Fire insurance x total')


g2.scatter(data_new['property tax (R$)'], data_new['total (R$)'])
g2.set_title('property x total')
g3.scatter(data_new['rent amount (R$)'], data_new['total (R$)'])
g3.set_title('rent R$ x total')
g4.scatter(data_new['bathroom'], data_new['total (R$)'])
g4.set_title('bathrooms x total')
area
