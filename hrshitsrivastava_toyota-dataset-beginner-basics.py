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
# Set up the Toyota csv file as a DataFrame

cars_df = pd.read_csv('/kaggle/input/toyotacorollacsv/ToyotaCorolla.csv', index_col = 0)



# Let's see a preview of the data

cars_df.head()
cars_df.columns
cars = cars_df.copy(deep = True)   #true is default and false is for shallow copy

cars.head()
cars.memory_usage().head()
cars.loc[:,['Fuel_Type','Price']].head(3)  #Access a group of rows and columns by label(s).
cars.dtypes.value_counts()
#cars_cpy.select_dtypes(include = None, exclude = None).head(2) #default
cars.select_dtypes(exclude = [object]).head(3)  
cars.info()  #entries are 1436, row labels are 1 to 1442
print(np.unique(cars['Doors']))

print(np.unique(cars['cc']))

print(np.unique(cars['Automatic']))
cars.info()
cars['Met_Color'] = cars['Met_Color'].astype('object')

cars['Automatic'] = cars['Automatic'].astype('object')

cars.info()
cars.isnull().sum()
columns_to_drop = ['Model','Mfg_Month', 'Mfg_Year','Cylinders',

       'Gears', 'Quarterly_Tax','Mfr_Guarantee', 'BOVAG_Guarantee',

       'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2', 'Airco',

       'Automatic_airco', 'Boardcomputer', 'CD_Player', 'Central_Lock',

       'Powered_Windows', 'Power_Steering', 'Radio', 'Mistlamps',

       'Sport_Model', 'Backseat_Divider', 'Metallic_Rim', 'Radio_cassette','Tow_Bar']

cars.drop(columns_to_drop, axis = 1, inplace= True)

cars.sample(5)

     


cars.describe().T
cars.rename(columns = {'Age_08_04':'Age_Months', 'cc':'CC'}, inplace = True)

cars.sample(2)
cars.insert(10, "Price_Class", " ")
for i in range(0, len(cars['Price']), 1):

    if cars['Price'].iloc[i] <= 8450:

        cars["Price_Class"].iloc[i] = 'Cheap'

    elif cars['Price'].iloc[i] >= 11950:

         cars["Price_Class"].iloc[i] = 'Expensive'

    else:

        cars['Price_Class'].iloc[i] = 'Average'

        
cars.insert(11, "Age", " ")
cars.describe().T
i = 0

while i < len(cars['Age_Months']):

    if cars['Age_Months'].iloc[i] <= 44:

        cars['Age'].iloc[i] = 'New Model'

        

    elif cars['Age_Months'].iloc[i] >= 70:

        cars['Age'].iloc[i] = 'Very Old'

               

    else:

        cars['Age'].iloc[i] = 'Old'

    i+=1           
cars['Age'].value_counts()
cars['Price_Class'].value_counts()
cars.head()
cars.insert(12, "Age-Year",0)
cars.insert(12, 'KM/Month', 0)
cars.head()
def conversion(val1, val2):

    val_con = val1/12

    ratio   = val2/val1

    return [val_con, ratio]
cars['Age-Year'], cars['KM/Month'] = conversion(cars['Age_Months'], cars['KM'])

cars.sample(5)
cars.info()
pd.crosstab(index = cars['Fuel_Type'], columns = 'count', dropna = True)
pd.crosstab(index = cars['Automatic'], columns = cars['Fuel_Type'], dropna = True)  #two-way table
pd.crosstab(index = cars['Automatic'], columns = cars['Fuel_Type'], normalize = True, dropna = True)  #two-way table, joint-probability
pd.crosstab(index = cars['Automatic'], columns = cars['Fuel_Type'],margins = True, normalize = True, dropna = True) #two-way table, marginal probability
pd.crosstab(index = cars['Automatic'], columns = cars['Fuel_Type'],margins = True, normalize = 'index', dropna = True) #two-way-conditional-probability
pd.crosstab(index = cars['Automatic'], columns = cars['Fuel_Type'],margins = True, normalize = 'columns', dropna = True)
numerical_data = cars.select_dtypes(exclude = [object])

print(numerical_data.shape)
corr_matrix = numerical_data.corr()

corr_matrix
import matplotlib.pyplot as plt
plt.scatter(cars['Age-Year'], cars['Price'], c = 'red')

plt.title('Price vs Age of the Cars')

plt.xlabel('Age in Years')

plt.ylabel('Price(Euros)')

plt.show()
plt.hist(cars['KM'], edgecolor = 'white', bins = 5)

plt.title('Histogram of Kilometer')

plt.xlabel('Kilometer')

plt.ylabel('Frequency')

plt.show()
fuel_count = pd.value_counts(cars['Fuel_Type'].values, sort = True)

plt.xlabel('Frequency')

plt.ylabel('Fuel Type')

plt.title('Bar plot of Fuel Type')

fuel_count.plot.barh()
import seaborn as sns
cars.head()
sns.set(style = 'darkgrid')

sns.regplot(x = cars['Age-Year'], y = cars['Price'], marker = '*')
sns.lmplot(x = 'Age-Year', y = 'Price', data = cars, hue = 'Fuel_Type', fit_reg= False, legend = True, palette ='Set1')
sns.distplot(cars['Age_Months'], kde = False, bins = 5)
sns.countplot(x = 'Fuel_Type', data = cars, hue = 'Automatic')
sns.boxplot(y = cars['Price'], x = cars['Fuel_Type'], hue = cars['Automatic'])
f, (ax_box, ax_hist) = plt.subplots(2, gridspec_kw={'height_ratios':(.20,.80)})
f, (ax_box, ax_hist) = plt.subplots(2, gridspec_kw={'height_ratios':(.20,.80)})

sns.boxplot(cars['Price'], ax = ax_box)

sns.distplot(cars['Price'], ax = ax_hist, kde = False)
sns.pairplot(cars, kind = 'scatter', hue = 'Fuel_Type')