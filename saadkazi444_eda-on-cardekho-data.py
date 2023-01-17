import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/car data.csv", )

df.head()
car_depreciation = df['Present_Price'] - df['Selling_Price']

df['depreciation'] = car_depreciation

car_depreciation
depr = df[['Car_Name', 'depreciation']]

depr.head()
grouped = depr.groupby('Car_Name').mean()

grouped.sort_values('depreciation', ascending=False)



df['Car_Name'].value_counts().head(30)
temp = df.loc[df['Car_Name'] == 'fortuner']

temp.head(11)# since fortuner has 11 samples
plt.bar(x = temp['Year'], height=temp['depreciation'])

plt.xlabel('Year')

plt.ylabel('Depreciation in lacs')
temp_1 = df.loc[df['Car_Name'] == 'city']

temp_1.head()
plt.bar(x = temp_1['Year'], height=temp_1['depreciation'])

plt.xlabel('Year')

plt.ylabel('Depreciation in lacs')
temp_2 = df.loc[df['Car_Name'] == 'corolla altis']

temp_2.head()
plt.bar(x = temp_2['Year'], height=temp_2['depreciation'])

plt.xlabel('Year')

plt.ylabel('Depreciation in lacs')
temp_3 = df.loc[df['Car_Name'] == 'verna']

temp_3.head()
plt.bar(x = temp_3['Year'], height=temp_3['depreciation'])

plt.xlabel('Year')

plt.ylabel('Depreciation in lacs')
kms = df[['Car_Name', 'Kms_Driven']]

kms.head()
by_kms_driven = kms.groupby('Car_Name').mean().tail(30)

by_kms_driven.sort_values('Kms_Driven', ascending=False)
temp_4 = df.loc[df['Car_Name'] == 'city']

temp_4.head()
plt.bar(x = temp_4['depreciation'], height=temp_4['Kms_Driven'])

plt.xlabel('Depreciation in lacs')

plt.ylabel('Kms Driven')
temp_5 = df.loc[df['Car_Name'] == 'fortuner']

temp_5.head()
plt.bar(x = temp_5['depreciation'], height=temp_5['Kms_Driven'])

plt.xlabel('Depreciation in lacs')

plt.ylabel('Kms Driven')
temp_6 = df.loc[df['Car_Name'] == 'corolla altis']

temp_6.head()
plt.bar(x = temp_6['depreciation'], height=temp_6['Kms_Driven'])

plt.xlabel('Depreciation in lacs')

plt.ylabel('Kms Driven')
temp_7 = df.loc[df['Car_Name'] == 'verna']

temp_7.head()
plt.bar(x = temp_7['depreciation'], height=temp_7['Kms_Driven'])

plt.xlabel('Depreciation in lacs')

plt.ylabel('Kms Driven')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Fuel_Type'])
le.classes_
df['Fuel_Type'] = le.transform(df['Fuel_Type'])
df.head()
le.fit(df['Seller_Type'])
df['Seller_Type'] = le.transform(df['Seller_Type'])
le.classes_
df.head()
le.fit(df['Transmission'])
df['Transmission'] = le.transform(df['Transmission'])
df.head()
by_fuel_type = df[['Car_Name', 'Fuel_Type', 'depreciation']]

by_fuel_type.head()
fuel_t = by_fuel_type.groupby('Fuel_Type').mean()



fuel_t.head()
by_seller_t = df[['Car_Name', 'Seller_Type', 'depreciation']]

by_seller_t.head()
seller_t = by_seller_t.groupby('Seller_Type').mean()

# by_kms_driven.sort_values('Kms_Driven', ascending=False)

seller_t.head()
new_df = df[['Car_Name', 'Year', 'Kms_Driven', 'depreciation']]

new_df.head()
sns.pairplot(new_df)
final_val = df[['Car_Name', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'depreciation','Year', 'Transmission']]

final_val.head()
sns.pairplot(final_val)