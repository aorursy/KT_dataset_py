import pandas as pd
import numpy as np

data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
data.head()
data.columns
data.describe()
data.describe(include='object')
cat = (data.dtypes == 'object')
object_cols = list(cat[cat].index)
print("Categorical variables are:")
print(object_cols)
cat_data = data[object_cols]

cat_data.head()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

cat_data['Regionname'] = label_encoder.fit_transform(cat_data['Regionname'])
cat_data.head()
cat_data = pd.get_dummies(cat_data,columns=['Method'],prefix = ['cat'])
cat_data.head()
cat_data['Type'].unique()
type = {'h':0,'u':1,'t':2}
cat_data = cat_data.replace({'Type':type})

cat_data.head()
drop = ['Suburb','Address','SellerG','Date','CouncilArea']

cat_data = cat_data.drop(drop, axis=1)
cat_data.head()
num_data = data.select_dtypes(exclude=['object'])
num_data.head()
new_data = pd.concat([num_data,cat_data],axis=1)
new_data.head()
print(new_data.isnull().sum())
new_data = new_data.drop(['BuildingArea','YearBuilt','Car'],axis=1)
new_data.head()
print(new_data.isnull().sum())
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

data_values = new_data.values

data_scaled = scaler.fit_transform(data_values)

normalized_data = pd.DataFrame(data_scaled)

normalized_data.head()
normalized_data.describe()
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

data_values = new_data.values

std_data = std_scaler.fit_transform(data_values)

standardized_data = pd.DataFrame(std_data)

standardized_data.head()
standardized_data.describe()