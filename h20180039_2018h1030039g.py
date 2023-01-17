import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")
df.head()
df.isnull().sum()
df.dtypes
df.describe()
df.fillna(df.mean(),inplace=True)
df.isnull().sum()
# df['feature7'].value_counts()
df[df.rating==6].count()
sns.boxplot(x='rating',y='feature6',data=df)
# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
df['type']=df['type'].astype('category').cat.codes
sns.countplot(x = df['rating'])
corr_values=corr['rating'].sort_values(ascending=False)

corr_values=abs(corr_values).sort_values(ascending=False)

print("Correlation of mentioned features wrt outcome in ascending order")

print(abs(corr_values).sort_values(ascending=False))
sns.distplot(df['feature1'],kde = False)
features=['feature6','feature8','feature11','type','feature4','feature2']

#           ,'feature7','feature3','feature5','feature8','feature10','feature1','feature9','feature11']
X_data=df.drop(['id','rating'],axis=1)
Y_data=df['rating']
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_data) 

from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X_scaled,Y_data,test_size=0.35,random_state=42) 

# X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.35,random_state=42) 
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=250)

model=rfc.fit(X_train, y_train)
from math import sqrt

from sklearn.metrics import mean_squared_error

rfc_pred = model.predict(X_val)

amae_lr = sqrt(mean_squared_error(rfc_pred,y_val))



print("Mean Squared Error of Linear Regression: {}".format(amae_lr))
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.35,random_state=42) 
from sklearn.ensemble import RandomForestClassifier

rfc1 = RandomForestClassifier(n_estimators=200)

model1=rfc1.fit(X_train, y_train)
rfc_pred1 = model1.predict(X_val)

amae_lr = sqrt(mean_squared_error(rfc_pred1,y_val))



print("Mean Squared Error of Linear Regression: {}".format(amae_lr))
test=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
# data1=test[features]

# data1['type']=data1['type'].astype('category').cat.codes

data1=test.drop(['id'],axis=1)
data1.fillna(data1.mean(),inplace=True)
data1['type']=data1['type'].astype('category').cat.codes
data1.isnull().sum()
# data1 = scaler.transform(data1)
rfc_pred = model1.predict(data1)

# amae_lr = sqrt(mean_squared_error(rfc_pred,y_val))



# print("Mean Squared Error of Linear Regression: {}".format(amae_lr))
rfc_pred
compare = pd.DataFrame({'id': test['id'], 'rating' : rfc_pred})
compare.to_csv('submission1.csv',index=False)