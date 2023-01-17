import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_column',None) 
data = pd.read_csv("/kaggle/input/vehicle-dataset-from-cardekho/car data.csv")
data
data.shape
# categorical values

print(data.Seller_Type.unique())
print(data['Transmission'].unique())
print(data['Owner'].unique())
print(data.Fuel_Type.unique())
# missing values
data.isnull().sum()
data.describe()
data.columns
df = data[['Year' , 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
df
df["Current_Year"]=2020
df["Number_of_years"] = df["Current_Year"] - df["Year"]
df = df.drop(['Year' , 'Current_Year'] , axis=1)
df
# encoding

df = pd.get_dummies(df,drop_first=True)
# catégorical 
print(df.Seller_Type_Individual.unique())
print(df['Transmission_Manual'].unique())
print(df['Owner'].unique())
print(df.Fuel_Type_Diesel.unique())
#distribution 
sns.pairplot(df)
threshold = 0.2
corrmat = df.corr() # correlation between variables
target = 'Selling_Price'
print(corrmat)
top_corr_feats = corrmat.nlargest(len(df.columns) , target)[target]
top_corr_feats = top_corr_feats.abs()
top_corr_feats = list(top_corr_feats[top_corr_feats.values>threshold].index)
top_corr_feats
# heat map correlation 
plt.figure(figsize=(20,20))
g = sns.heatmap(df[top_corr_feats].corr() , annot=True , cmap='RdYlGn')
# Train test split
# target and features 
X = df[['Present_Price' , 'Fuel_Type_Diesel' , 'Transmission_Manual' , 'Fuel_Type_Petrol' , 'Seller_Type_Individual' , 'Number_of_years']]
y = df.iloc[:,0] # target
from sklearn.preprocessing import RobustScaler # normalization supporting outliers

rob = RobustScaler()
df_rb = rob.fit_transform(df)
df_rb = pd.DataFrame(df_rb)

X_rb = df_rb.drop([0 , 2 , 3] , axis=1)
y_rb = df_rb.iloc[:,0]
df_rb
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X_rb,y_rb , test_size=0.2 , random_state=0)
X_train
X_test
from sklearn.linear_model import *
model = LinearRegression()
model.fit(X_train, y_train)
print('Train score :' , model.score(X_train, y_train)) # R2 score
print('Test score : ' , model.score(X_test , y_test))
prediction = model.predict(X_test)
print(prediction)
from sklearn.metrics import *
print('MAE', mean_absolute_error(y_test,prediction))
print('MSE' , mean_squared_error(y_test , prediction))
print('RMSE' , np.sqrt(mean_squared_error(y_test , prediction)))
print('R2 score' , r2_score(y_test , prediction))
model.coef_
model.intercept_
sns.distplot(y_test-prediction)

