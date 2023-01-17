import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        import sqlite3
import pandas as pd

ggl = pd.read_csv("../input/google-app/google_appstore.csv")

ggl.head()
ggl.shape
ggl.describe()
ggl.dtypes

ggl.hist()
ggl.info()
ggl.boxplot()
ggl.isnull().sum()
ggl.describe(include='all')
ggl[ggl.Rating >5]
ggl.drop([10472],inplace = True)

ggl[10470:10475]
ggl.boxplot()
ggl.hist()
ggl.columns

ggl.index
ggl.isnull().sum()
ggl.shape
ggl.dropna(thresh = 1084, axis = 1, inplace = True)
ggl.shape
ggl['Rating'].isnull().sum()
# 1/First we must transform the type of our data: from the 'DataFrame' type to the 'array' type, because the sklean's function can't handl the dataframe type

# PS: the axis in sklearn is the ivers of numpy and pandas, so axis = 0: looking for the columns, axis = 1: looking for the rows.

data = ggl.iloc[:,:].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values= 'NaN', strategy='mean', axis = 0)

imputer = imputer.fit(data[:,2:3])

data[:,2:3] = imputer.transform(data[:,2:3])

data[:,2:3]

# Another method to fill the messing data by the median/mode/mean

#ggl.fillna(value = (ggl['Rating'].median()))



# 2/ Then we replace the original column of 'Rating' by the new column that we preprocess it

ggl['Rating'] = data[:,2:3]

ggl.isnull().sum()

# for removing a several columns we have just to put it between brackets eg: ggl.drop(['R1','R2','R3'], axis = 1, i)

#ggl.drop('Rating1', axis = 1, inplace = True)

#Selection of The rows wich have 14M of her size but the result is a dataframe

size = ggl[ggl["Size"] == '14M']

# transform the dataframe to an array

size = size.iloc[:,:].values

type(size)

size
ggl.isnull().sum()
ggl['Android Ver'].mode()

ggl['Current Ver'].mode()

ggl['Type'].mode()
ggl['Android Ver'].fillna(str(ggl['Android Ver'].mode().values[0]), inplace = True)

ggl['Type'].fillna(str(ggl['Type'].mode().values[0]), inplace = True)

ggl['Current Ver'].fillna(str(ggl['Current Ver'].mode().values[0]), inplace = True)

ggl.isnull().sum()
ggl['Price'] =  ggl['Price'].apply(lambda x: str(x).replace('$','') if '$' in str(x) else str(x) )

ggl['Price'] =  ggl['Price'].apply(lambda x:float(x))



ggl['Reviews'] =  ggl['Reviews'].apply(lambda x:float(x))

ggl['Installs'] = ggl['Installs'].apply(lambda x: str(x).replace('+','') if '+' in str(x) else str(x))

ggl['Installs'] = ggl['Installs'].apply(lambda x: str(x).replace(',','') if ',' in str(x) else str(x))

ggl['Installs'] = ggl['Installs'].apply(lambda x : float(x))

ggl['Reviews'] = pd.to_numeric(ggl['Reviews'], errors = 'coerce')

ggl['Rating'] = pd.to_numeric(ggl['Rating'], errors = 'coerce')
ggl.head()
ggl.index
ggl['Rating'].describe()
grp = ggl.groupby('Category')

grp.mean() # give the mean of each group of each column

x = grp['Rating'].mean()

y = grp['Installs'].sum()

z = grp['Reviews'].std()

w = grp['Price'].max()

import matplotlib.pyplot as plt

# A simple plot

plt.plot(x)
 # A lot only with spot

plt.plot(x, 'ro')
#change the size of the figure

plt.figure(figsize=(12,6))

# We can modefie the sheap and the color of the plot

# bs: carré, ro: spot, r-- : disconued line, g: line

# r: red, 'b':blue, 'g':green

plt.plot(x, 'g', color = 'r')

# we need to chanche the orientation of the ticks

plt.xticks(rotation = 90)

plt.title('Category wise Rating')

plt.xlabel('Category')

plt.ylabel('Rating')

plt.show()



ggl['Rating'].hist()
ggl.head()
x = ggl.iloc[:,2:4].values

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x = sc_x.fit_transform(x)

#To replace Rating by the new values scaled: x[:, 0:1] = ggl['Rating']

#To replace Reviews by the new values scaled: x[:,1] = ggl['Reviews']