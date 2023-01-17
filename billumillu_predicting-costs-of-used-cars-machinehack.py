# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



df_full = pd.read_excel("../input/Data_Train.xlsx")

df_test = pd.read_excel("../input/Data_Test.xlsx")

df_full.head(10)
df_full.shape
df_full.info()

df_full.isnull().sum()
df_full['Name'] = df_full.Name.str.split().str.get(0)

df_test['Name'] = df_test.Name.str.split().str.get(0)

df_full.head()
df_full['Name'].value_counts().sum()
# Get names of columns with missing values

cols_with_missing = [col for col in df_full.columns

                     if df_full[col].isnull().any()]

print("Columns with missing values:")

print(cols_with_missing)
# Let's deal with them one by one.



df_full['Seats'].fillna(df_full['Seats'].mean(),inplace=True)

df_test['Seats'].fillna(df_test['Seats'].mean(),inplace=True)
data = pd.concat([df_full,df_test], sort=False)
import matplotlib.pyplot as plt



plt.figure(figsize=(20,5))

data['Mileage'].value_counts().head(100).plot.bar()

plt.show()
df_full['Mileage'] = df_full['Mileage'].fillna('17.0 kmpl')

df_test['Mileage'] = df_test['Mileage'].fillna('17.0 kmpl')



#I noticed the 14th entry (and others) have 0.0 kmpl. Let's replace that too.



df_full['Mileage'] = df_full['Mileage'].replace("0.0 kmpl", "17.0 kmpl")

df_test['Mileage'] = df_test['Mileage'].replace("0.0 kmpl", "17.0 kmpl")
plt.figure(figsize=(20,5))

data['Engine'].value_counts().head(100).plot.bar()

plt.show()
df_full['Engine'] = df_full['Engine'].fillna('1197 CC')

df_test['Engine'] = df_test['Engine'].fillna('1197 CC')
plt.figure(figsize=(20,5))

data['Power'].value_counts().head(100).plot.bar()

plt.show()
df_full['Power'] = df_full['Power'].fillna('74 bhp')

df_test['Power'] = df_test['Power'].fillna('74 bhp')



#I noticed the 76th entry (and others) have null bhp. Let's replace that too. 

#This was creating problems during LabelEncoding.



df_full['Power'] = df_full['Power'].replace("null bhp", "74 bhp")

df_test['Power'] = df_test['Power'].replace("null bhp", "74 bhp")
plt.figure(figsize=(20,5))

data['New_Price'].value_counts().head(100).plot.bar()

plt.show()
# # I'll select 4.78 cuz the others are way too high.



# df_full['New_Price'] = df_full['New_Price'].fillna('4.78 Lakh')

# df_test['New_Price'] = df_test['New_Price'].fillna('4.78 Lakh')



# # Run the method get_number() defined below first.

# # Converting it to float.



# df_full['New_Price'] = df_full['New_Price'].apply(get_number).astype('float')

# df_test['New_Price'] = df_test['New_Price'].apply(get_number).astype('float')
# Method to extract 'float' from 'object' 



import re



def get_number(name):

    title_search = re.search('([\d+\.+\d]+\W)', name)

    

    if title_search:

        return title_search.group(1)

    return ""
data['New_Price'] = data['New_Price'].fillna('0.0 Lakh') # dealt with missing values.



data['New_Price'] = data['New_Price'].apply(get_number).astype('float') #converted to float



total = data['New_Price'].groupby(data['Name'])

print(total.mean().round(2))
df_full['New_Price'] = df_full['New_Price'].fillna('0.0 Lakh') # dealt with missing values.

df_full['New_Price'] = df_full['New_Price'].apply(get_number).astype('float') #converted to float
df_full.loc[df_full['Name']=="Audi", 'New_Price'] = df_full.loc[df_full['Name']=="Audi", 'New_Price'].replace(0.0,5.02)

df_full.loc[df_full['Name']=="BMW", 'New_Price'] = df_full.loc[df_full['Name']=="BMW", 'New_Price'].replace(0.0,11.14)

df_full.loc[df_full['Name']=="Bentley", 'New_Price'] = df_full.loc[df_full['Name']=="Bentley", 'New_Price'].replace(0.0,1.88)

df_full.loc[df_full['Name']=="Datsun", 'New_Price'] = df_full.loc[df_full['Name']=="Datsun", 'New_Price'].replace(0.0,3.14)

df_full.loc[df_full['Name']=="Fiat", 'New_Price'] = df_full.loc[df_full['Name']=="Fiat", 'New_Price'].replace(0.0,0.95)



df_full.loc[df_full['Name']=="Ford", 'New_Price'] = df_full.loc[df_full['Name']=="Ford", 'New_Price'].replace(0.0,1.16)

df_full.loc[df_full['Name']=="Honda", 'New_Price'] = df_full.loc[df_full['Name']=="Honda", 'New_Price'].replace(0.0,1.30)

df_full.loc[df_full['Name']=="Hyundai", 'New_Price'] = df_full.loc[df_full['Name']=="Hyundai", 'New_Price'].replace(0.0,1.03)

df_full.loc[df_full['Name']=="Isuzu", 'New_Price'] = df_full.loc[df_full['Name']=="Isuzu", 'New_Price'].replace(0.0,16.84)

df_full.loc[df_full['Name']=="ISUZU", 'New_Price'] = df_full.loc[df_full['Name']=="ISUZU", 'New_Price'].replace(0.0,16.84)



df_full.loc[df_full['Name']=="Jaguar", 'New_Price'] = df_full.loc[df_full['Name']=="Jaguar", 'New_Price'].replace(0.0,8.52)

df_full.loc[df_full['Name']=="Jeep", 'New_Price'] = df_full.loc[df_full['Name']=="Jeep", 'New_Price'].replace(0.0,22.75)

df_full.loc[df_full['Name']=="Land", 'New_Price'] = df_full.loc[df_full['Name']=="Land", 'New_Price'].replace(0.0,4.39)

df_full.loc[df_full['Name']=="Mahindra", 'New_Price'] = df_full.loc[df_full['Name']=="Mahindra", 'New_Price'].replace(0.0,1.20)

df_full.loc[df_full['Name']=="Maruti", 'New_Price'] = df_full.loc[df_full['Name']=="Maruti", 'New_Price'].replace(0.0,1.29)



df_full.loc[df_full['Name']=="Mercedes-Benz", 'New_Price'] = df_full.loc[df_full['Name']=="Mercedes-Benz", 'New_Price'].replace(0.0,7.97)

df_full.loc[df_full['Name']=="Mini", 'New_Price'] = df_full.loc[df_full['Name']=="Mini", 'New_Price'].replace(0.0,25.06)

df_full.loc[df_full['Name']=="Mitsubishi", 'New_Price'] = df_full.loc[df_full['Name']=="Mitsubishi", 'New_Price'].replace(0.0,12.03)

df_full.loc[df_full['Name']=="Nissan", 'New_Price'] = df_full.loc[df_full['Name']=="Nissan", 'New_Price'].replace(0.0,1.89)

df_full.loc[df_full['Name']=="Porsche", 'New_Price'] = df_full.loc[df_full['Name']=="Porsche", 'New_Price'].replace(0.0,0.07)



df_full.loc[df_full['Name']=="Renault", 'New_Price'] = df_full.loc[df_full['Name']=="Renault", 'New_Price'].replace(0.0,1.49)

df_full.loc[df_full['Name']=="Skoda", 'New_Price'] = df_full.loc[df_full['Name']=="Skoda", 'New_Price'].replace(0.0,3.63)

df_full.loc[df_full['Name']=="Tata", 'New_Price'] = df_full.loc[df_full['Name']=="Tata", 'New_Price'].replace(0.0,2.00)

df_full.loc[df_full['Name']=="Toyota", 'New_Price'] = df_full.loc[df_full['Name']=="Toyota", 'New_Price'].replace(0.0,4.38)

df_full.loc[df_full['Name']=="Volksvagen", 'New_Price'] = df_full.loc[df_full['Name']=="Volksvagen", 'New_Price'].replace(0.0,1.53)

df_full.loc[df_full['Name']=="Volvo", 'New_Price'] = df_full.loc[df_full['Name']=="Volvo", 'New_Price'].replace(0.0,4.62)
df_test['New_Price'] = df_test['New_Price'].fillna('0.0 Lakh') # dealt with missing values.

df_test['New_Price'] = df_test['New_Price'].apply(get_number).astype('float') #converted to float
# Modify df_test too...



df_test.loc[df_full['Name']=="Audi", 'New_Price'] = df_test.loc[df_test['Name']=="Audi", 'New_Price'].replace(0.0,5.02)

df_test.loc[df_full['Name']=="BMW", 'New_Price'] = df_test.loc[df_test['Name']=="BMW", 'New_Price'].replace(0.0,11.14)

df_test.loc[df_full['Name']=="Bentley", 'New_Price'] = df_test.loc[df_test['Name']=="Bentley", 'New_Price'].replace(0.0,1.88)

df_test.loc[df_full['Name']=="Datsun", 'New_Price'] = df_test.loc[df_test['Name']=="Datsun", 'New_Price'].replace(0.0,3.14)

df_test.loc[df_full['Name']=="Fiat", 'New_Price'] = df_test.loc[df_test['Name']=="Fiat", 'New_Price'].replace(0.0,0.95)



df_test.loc[df_full['Name']=="Ford", 'New_Price'] = df_test.loc[df_test['Name']=="Ford", 'New_Price'].replace(0.0,1.16)

df_test.loc[df_full['Name']=="Honda", 'New_Price'] = df_test.loc[df_test['Name']=="Honda", 'New_Price'].replace(0.0,1.30)

df_test.loc[df_full['Name']=="Hyundai", 'New_Price'] = df_test.loc[df_test['Name']=="Hyundai", 'New_Price'].replace(0.0,1.03)

df_test.loc[df_full['Name']=="Isuzu", 'New_Price'] = df_test.loc[df_test['Name']=="Isuzu", 'New_Price'].replace(0.0,16.84)

df_test.loc[df_full['Name']=="ISUZU", 'New_Price'] = df_test.loc[df_test['Name']=="ISUZU", 'New_Price'].replace(0.0,16.84)



df_test.loc[df_full['Name']=="Jaguar", 'New_Price'] = df_test.loc[df_test['Name']=="Jaguar", 'New_Price'].replace(0.0,8.52)

df_test.loc[df_full['Name']=="Jeep", 'New_Price'] = df_test.loc[df_test['Name']=="Jeep", 'New_Price'].replace(0.0,22.75)

df_test.loc[df_full['Name']=="Land", 'New_Price'] = df_test.loc[df_test['Name']=="Land", 'New_Price'].replace(0.0,4.39)

df_test.loc[df_full['Name']=="Mahindra", 'New_Price'] = df_test.loc[df_test['Name']=="Mahindra", 'New_Price'].replace(0.0,1.20)

df_test.loc[df_full['Name']=="Maruti", 'New_Price'] = df_test.loc[df_test['Name']=="Maruti", 'New_Price'].replace(0.0,1.29)



df_test.loc[df_full['Name']=="Mercedes-Benz", 'New_Price'] = df_test.loc[df_test['Name']=="Mercedes-Benz", 'New_Price'].replace(0.0,7.97)

df_test.loc[df_full['Name']=="Mini", 'New_Price'] = df_test.loc[df_test['Name']=="Mini", 'New_Price'].replace(0.0,25.06)

df_test.loc[df_full['Name']=="Mitsubishi", 'New_Price'] = df_test.loc[df_test['Name']=="Mitsubishi", 'New_Price'].replace(0.0,12.03)

df_test.loc[df_full['Name']=="Nissan", 'New_Price'] = df_test.loc[df_test['Name']=="Nissan", 'New_Price'].replace(0.0,1.89)

df_test.loc[df_full['Name']=="Porsche", 'New_Price'] = df_test.loc[df_test['Name']=="Porsche", 'New_Price'].replace(0.0,0.07)



df_test.loc[df_full['Name']=="Renault", 'New_Price'] = df_test.loc[df_test['Name']=="Renault", 'New_Price'].replace(0.0,1.49)

df_test.loc[df_full['Name']=="Skoda", 'New_Price'] = df_test.loc[df_test['Name']=="Skoda", 'New_Price'].replace(0.0,3.63)

df_test.loc[df_full['Name']=="Tata", 'New_Price'] = df_test.loc[df_test['Name']=="Tata", 'New_Price'].replace(0.0,2.00)

df_test.loc[df_full['Name']=="Toyota", 'New_Price'] = df_test.loc[df_test['Name']=="Toyota", 'New_Price'].replace(0.0,4.38)

df_test.loc[df_full['Name']=="Volksvagen", 'New_Price'] = df_test.loc[df_test['Name']=="Volksvagen", 'New_Price'].replace(0.0,1.53)

df_test.loc[df_full['Name']=="Volvo", 'New_Price'] = df_test.loc[df_test['Name']=="Volvo", 'New_Price'].replace(0.0,4.62)
plt.figure(figsize=(20,5))

df_full['New_Price'].value_counts().head(100).plot.bar()

plt.show()



plt.figure(figsize=(20,5))

df_test['New_Price'].value_counts().head(100).plot.bar()

plt.show()
df_full.loc[df_full['Name']=="Ambassador", 'New_Price'] = df_full.loc[df_full['Name']=="Ambassador", 'New_Price'].replace(0.0,1.29)

df_full.loc[df_full['Name']=="Chevrolet", 'New_Price'] = df_full.loc[df_full['Name']=="Chevrolet", 'New_Price'].replace(0.0,1.29)

df_full.loc[df_full['Name']=="Force", 'New_Price'] = df_full.loc[df_full['Name']=="Force", 'New_Price'].replace(0.0,1.29)

df_full.loc[df_full['Name']=="Lamborghini", 'New_Price'] = df_full.loc[df_full['Name']=="Lamborghini", 'New_Price'].replace(0.0,1.29)

df_full.loc[df_full['Name']=="OpelCorsa", 'New_Price'] = df_full.loc[df_full['Name']=="OpelCorsa", 'New_Price'].replace(0.0,1.29)



df_test.loc[df_full['Name']=="Ambassador", 'New_Price'] = df_test.loc[df_test['Name']=="Ambassador", 'New_Price'].replace(0.0,1.29)

df_test.loc[df_full['Name']=="Chevrolet", 'New_Price'] = df_test.loc[df_test['Name']=="Chevrolet", 'New_Price'].replace(0.0,1.29)

df_test.loc[df_full['Name']=="Force", 'New_Price'] = df_test.loc[df_test['Name']=="Force", 'New_Price'].replace(0.0,1.29)

df_test.loc[df_full['Name']=="Lamborghini", 'New_Price'] = df_test.loc[df_test['Name']=="Lamborghini", 'New_Price'].replace(0.0,1.29)

df_test.loc[df_full['Name']=="OpelCorsa", 'New_Price'] = df_test.loc[df_test['Name']=="OpelCorsa", 'New_Price'].replace(0.0,1.29)
# pd.pivot_table(data, index = 'Name', values='New_Price',

#                                    aggfunc=lambda x: x.mode().iat[0])



# # IndexError: index 0 is out of bounds for axis 0 with size 0

df_full.isnull().sum()
df_full.head(10)
df_full.info()
#Using the above defined method get_number()



df_full['Mileage'] = df_full['Mileage'].apply(get_number).astype('float')

df_full['Engine'] = df_full['Engine'].apply(get_number).astype('int')

df_full['Power'] = df_full['Power'].apply(get_number).astype('float')



df_test['Mileage'] = df_test['Mileage'].apply(get_number).astype('float')

df_test['Engine'] = df_test['Engine'].apply(get_number).astype('int')

df_test['Power'] = df_test['Power'].apply(get_number).astype('float')



df_full.info()
help(re) # This baby was realy helpful!
df_test.info()
df_full.head()
from sklearn.model_selection import train_test_split



y = np.log1p(df_full.Price)  # Made a HUGE difference. MAE went down from 1.8 to 0.1!! Thanks to Rishi - @littleraj30 for pointing it out.

X = df_full.drop(['Price'],axis=1)

# df_test = df_test.drop('New_Price',axis=1)



X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.82,test_size=0.18,random_state=0)
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()



# X_train[object_cols] = label_encoder.fit_transform(X_train[object_cols])

# X_valid[object_cols] = label_encoder.transform(X_valid[object_cols])

# df_test[object_cols] = label_encoder.fit_transform(df_test[object_cols])



# ValueError: bad input shape (4815, 5)

# That's why I did it manually.



X_train['Name'] = label_encoder.fit_transform(X_train['Name'])

X_valid['Name'] = label_encoder.transform(X_valid['Name'])

df_test['Name'] = label_encoder.fit_transform(df_test['Name'])



X_train['Location'] = label_encoder.fit_transform(X_train['Location'])

X_valid['Location'] = label_encoder.transform(X_valid['Location'])

df_test['Location'] = label_encoder.fit_transform(df_test['Location'])



X_train['Fuel_Type'] = label_encoder.fit_transform(X_train['Fuel_Type'])

X_valid['Fuel_Type'] = label_encoder.transform(X_valid['Fuel_Type'])

df_test['Fuel_Type'] = label_encoder.fit_transform(df_test['Fuel_Type'])



X_train['Transmission'] = label_encoder.fit_transform(X_train['Transmission'])

X_valid['Transmission'] = label_encoder.transform(X_valid['Transmission'])

df_test['Transmission'] = label_encoder.fit_transform(df_test['Transmission'])



X_train['Owner_Type'] = label_encoder.fit_transform(X_train['Owner_Type'])

X_valid['Owner_Type'] = label_encoder.transform(X_valid['Owner_Type'])

df_test['Owner_Type'] = label_encoder.fit_transform(df_test['Owner_Type'])

X_train.head()
X_train.info()
# # Let's try scaling too.



# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler().fit(X_train)

# rescaled_X_train = scaler.transform(X_train)



# scaler = StandardScaler().fit(X_valid)

# rescaled_X_valid = scaler.transform(X_valid)



# scaler = StandardScaler().fit(df_test)

# rescaled_df_test = scaler.transform(df_test)



# from xgboost import XGBRegressor

# from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error



# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# my_model.fit(rescaled_X_train, y_train, 

#              early_stopping_rounds=5, 

#              eval_set=[(rescaled_X_valid, y_valid)], 

#              verbose=False)



# predictions = my_model.predict(rescaled_X_valid)

# print("MAE: " + str(mean_absolute_error(predictions, y_valid)))

# print("MSE: " + str(mean_squared_error(predictions, y_valid)))

# print("MSLE: " + str(mean_squared_log_error(predictions, y_valid)))



# # MAE: 2.115451765105513

# # MSE: 17.56415019000094

# # MSLE: 0.058881434868999126
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_squared_log_error



my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)



predictions = my_model.predict(X_valid)

print("MAE: " + str(mean_absolute_error(predictions, y_valid)))

print("MSE: " + str(mean_squared_error(predictions, y_valid)))

print("MSLE: " + str(mean_squared_log_error(predictions, y_valid)))
preds_test = my_model.predict(df_test)

preds_test = np.exp(preds_test)-1 #converting target to original state

print(preds_test)



# The Price is in the format xx.xx So let's round off and submit.



preds_test = preds_test.round(2)

print(preds_test)
output = pd.DataFrame({'Price': preds_test})

output.to_excel('submission.xlsx', index=False)