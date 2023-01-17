# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

df.head(15)
print(f"row:{df.shape[0]}, column:{df.shape[1]}")                # Total Rows and columns
df['city'].value_counts()
null_data = df.loc[df['floor']=='-']                      # elemenate not available datas.

df1=df.drop(null_data.index,axis=0)
df1
df1['price_per_area'] = df1['total (R$)']/df1['area']             # creating one extra column for removing the misleading datas.

df1.head(10) 
df1['price_per_area'].describe()
mean = df1['price_per_area'].mean()

std = df1['price_per_area'].std()

low_price_per_area = df1.loc[df1['price_per_area']>mean+std]

high_price_per_area = df1.loc[df1['price_per_area']<mean-std]

print(len(low_price_per_area),len(high_price_per_area))

df1.shape[0]
df2 = df1.drop(low_price_per_area.index,axis=0)

df2 = df2.drop(high_price_per_area.index,axis=0)

df2.shape[0]
df2.head()
df3 = df2.drop((df2.iloc[:,8:12]),axis=1)

df3
plt.scatter(df3['area'],df3['total (R$)'],c='r')               # Detecting the extreem points by visualization.

plt.show()
df3=df3[df3['area']<500]                                       # Removeing the extreem datapoints.

df3.shape
plt.scatter(df3['area'],df3['total (R$)'],c='r')                 # visualization 

plt.show()
df3.price_per_area.describe()
lower_bound=df3.price_per_area.quantile(0.80)                    # Set the upper bound and lower bound or range of our dataset.

upper_bound = df3.price_per_area.quantile(0.20)
df4 = df3[df3.price_per_area<lower_bound]

df4
df4 = df4[df4.price_per_area>upper_bound]

df4
plt.scatter(df4['area'],df4['total (R$)'],c='r')               # Visualization after elemenate the outlier datas 

plt.show()
dummies = pd.get_dummies(df4.city)                  # Creating the dummy variables for city column

dummies
df5 = pd.concat([df4,dummies],axis=1)                   # concat with original dataset

df5
df5 = df5.drop(['city','São Paulo'],axis=1)           # Drop the city column and São Paulo column for avoiding Dummy variable trap

df5
df5.isnull().sum()                     # check for any NAN value
x = df5[['area','rooms','bathroom','parking spaces','floor','Belo Horizonte','Campinas','Porto Alegre', 'Rio de Janeiro']]

x                                                # Independent variable column
y = df5['total (R$)']                    # Dependent variable column

y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train,y_train)                        # Model Training using training dataset
print(len(x_train),len(y_train),len(x_test),len(y_test))
reg.score(x_test,y_test)                          # Accuracy of the Model
def locate_index(place):                           # this function return the index value of city.

  for i in range(len(x.columns)):

    if x.columns[i] == place:

      return(i)
u_area = input('Enter the area:')

u_sqft = int(input("Enter the no of sqft: "))

u_rooms = int(input("Enter the no of rooms: "))

u_bathroom = int(input("Enter the no of bathrooms: "))

u_floor = int(input("Enter the no of floor: "))

u_pspace = int(input("Enter the no of parking space: "))

def price_predict(u_area,u_sqft,u_rooms,u_bathroom,u_floor,u_psapce):

  pred_list = np.zeros(len(x.columns))

  pred_list[0] = u_sqft

  pred_list[1] = u_rooms

  pred_list[2] = u_bathroom

  pred_list[3] = u_pspace

  pred_list[4] = u_floor

  if u_area != 'São Paulo':

    pred_list[locate_index(u_area)] == 1.0

  print(f"The Estimated price is: {reg.predict([pred_list])}")

price_predict(u_area,u_sqft,u_rooms,u_bathroom,u_floor,u_pspace)