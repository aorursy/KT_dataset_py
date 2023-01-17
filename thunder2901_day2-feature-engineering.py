import pandas as pd

length = [12,10,11,14,15,16]

breadth = [9,4,2,6,10,12]

type_room = ['Big','Small','Small','Medium','Big','Big']  

df = pd.DataFrame({'length':length,'breadth':breadth,'type_room':type_room})

df
#creating a new column

df['area'] = df['length'] * df['breadth']

df
city = ['CityA','CityB','CityA','CityC','CityA','CityB']

roll = [12,14,15,16,13,19]

df1 = pd.DataFrame({"city_of_origin":city, "roll":roll})

df1
df1=pd.get_dummies(df1)

df1
city = ['CityA','CityB','CityA','CityC','CityA','CityB']

roll = [12,14,15,16,13,19]

df2 = pd.DataFrame({"city_of_origin":city, "roll":roll})

df2
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

df2['city_of_origin'] = lb.fit_transform(df2['city_of_origin'])

df2.head()