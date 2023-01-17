import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/Train.csv")
df.head(5)
df.info()
df['Item_Identifier'].nunique()

#There are 1559 different items in the dataset
df['Item_Fat_Content'].unique()

#Basically there are only 2 types, Low Fat and Regular
def replace_value(string):

    if string == "Low Fat" or string == "low fat" or string == "LF":

        return 0

    else:

        return 1



df['Item_Fat_Content'] = df['Item_Fat_Content'].apply(replace_value)
df['Item_Fat_Content'].nunique()
#Replace Missing weight of an Item with mean of that particular item overall

df['Item_Weight'].isnull().sum()
df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(lambda x:x.fillna(x.mean()))
df[df['Item_Weight'].isnull()]

#These four items are present only once in dataset, better to drop these four values
#Even though the number of regular content is very low, Item outlet sale mean is almost equal for both

#Total sale value is high for Low Fat items compared to Regular items
df.groupby('Item_Fat_Content')['Item_Outlet_Sales'].sum()
df.groupby('Item_Fat_Content')['Item_MRP'].mean()
df['Item_Type'].value_counts()

#Looks like Fruits and Snacks are more
df.groupby('Item_Type')['Item_MRP'].mean().sort_values(ascending=False)

#Household items have high MRP

#Wheras Baking Goods have low MRP
df['Outlet_Identifier'].nunique()

#There are 10 different outlets compared
outlet = df.groupby('Outlet_Identifier')
outlet['Item_Outlet_Sales'].mean().sort_values(ascending=False)

#OUT027 has high sales value
df['Outlet_Establishment_Year'].nunique()
df.groupby(['Outlet_Establishment_Year','Outlet_Identifier'])['Item_Outlet_Sales'].count

#OUT027 is the oldest and have sold large number of items

#Also OUT019 is oldest but has very low number of items
df['Outlet_Location_Type'].value_counts()

#More stores are in Tier Three location
df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean()

#Tier 2 has more mean sale overall
df.groupby(['Outlet_Identifier','Outlet_Location_Type'])['Item_Outlet_Sales'].mean().sort_values(ascending=False)
df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean()
df.groupby(['Outlet_Identifier','Outlet_Type'])['Item_Outlet_Sales'].mean().sort_values(ascending=False)

#Supermarket Type3 has more mean sale which is only OUT027
df['Outlet_Size'].isnull().sum()
df.groupby('Outlet_Identifier')['Outlet_Size'].count()

#Size of Three outlets are not given
df.groupby('Outlet_Type')['Outlet_Size'].count()
df.groupby('Outlet_Type').count()
df.groupby(['Outlet_Type','Outlet_Location_Type'])['Outlet_Size'].count()
df['Item_Weight'].isnull().sum()
df['Outlet_Size'].loc[df['Outlet_Type'] == 'Grocery Store'] = df['Outlet_Size'].loc[df['Outlet_Type'] == 'Grocery Store'].fillna("Small")

df['Outlet_Size'].loc[df['Outlet_Type'] == 'Supermarket Type1'] = df['Outlet_Size'].loc[df['Outlet_Type'] == 'Supermarket Type1'].fillna("Small")
df['Outlet_Size'].unique()
df.dropna(inplace=True)
df.groupby('Item_Type')['Item_Visibility'].mean().sort_values(ascending=False)

#Breakfast,Seafood and Dairy has more visibility
#Vizualization
sns.countplot(df['Item_Identifier'])
sns.countplot(df['Item_Fat_Content'])

#Most of the items are of Low Fat
sns.countplot(df['Item_Type'],hue=df['Item_Fat_Content'])

plt.xticks(rotation=45,ha='right')

#Household item is full of low fat
sns.scatterplot(df['Outlet_Identifier'],df['Item_Outlet_Sales'])

plt.xticks(rotation=45,ha="right")

#As confirmed ealier, OUT027 has high sales
sns.boxplot(df['Outlet_Size'],df['Item_Outlet_Sales'])

#Mean of Medium sized shops is more
sns.boxplot(df['Outlet_Location_Type'],df['Item_Outlet_Sales'])

#Tier 2 has slightly higher sales
sns.boxplot(df['Outlet_Type'],df['Item_Outlet_Sales'])
sns.barplot(df['Item_Type'],df['Item_MRP'],hue=df['Item_Fat_Content'])

plt.xticks(rotation=45,ha="right")

#Seafood price is more if it is regular

#Starchy food price is more if it is low fat
#Plotting correlation

corr = df.corr()

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with sns heatmap

sns.heatmap(corr, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df.head()
cat_columns=['Item_Identifier','Item_Type','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']

for each in cat_columns:

    df[cat_columns] = df[cat_columns].astype('category')

df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
df.head()
df.drop('Outlet_Establishment_Year',axis=1,inplace=True)
#Shuffling the dataframe

df = df.sample(frac=1).reset_index(drop=True)
df.head()
#Seperating the predictor variable

y = df['Item_Outlet_Sales']

X = df.drop('Item_Outlet_Sales',axis=1)
#ormalize two columns between zero and one

X.drop('Item_Identifier',axis=1,inplace=True)

cols_to_norm = ['Item_Weight','Item_MRP']

X[cols_to_norm] = X[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

X.head()
#Adding one hot encoding for Item Type column

Item_label = pd.get_dummies(df['Item_Type'],prefix='Type')
type(Item_label)
X = pd.concat([X,Item_label],axis=1)

X.shape
X.drop('Item_Type',axis=1,inplace=True)

X.head()
#Normalize between zero and one

y = (y-y.min())/(y.max()-y.min())
#Train Test split

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
from sklearn.svm import SVR

#Fitting the data in SVM model

model = SVR()

model.fit(X_train,y_train)
#Model prediction

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error

#Using Mean Squared error as metric

print(mean_squared_error(y_test,y_pred))