import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
% matplotlib inline
df=pd.read_csv('../input/BlackFriday.csv')
df.head()
new_df=df.copy()
new_df=new_df.drop(['User_ID','Product_ID'],axis=1)
df.describe()
df.info()
df.fillna(0,inplace=True)
df.head()
df.info()
df.Age.unique()
df=df.replace(['F','M'],[0,1])
df=df.replace(['A','B','C'],[0,1,2])
df=df.replace(['0-17','18-25', '26-35','36-45', '46-50', '51-55', '55+'],[0,1,2,3,4,5,6])
df=df.replace(['4+'],[4])
df.head()
df['Product_Category_1']=df['Product_Category_1'].astype(int)
df['Product_Category_2']=df['Product_Category_2'].astype(int)
df['Product_Category_3']=df['Product_Category_3'].astype(int)
df=df.drop(['Product_ID'],axis=1)
df.head()
df['sex']=df['Gender'].apply(lambda y: 1 if y=='M' else 0)
df.head()
plt.figure(figsize=(10,6))
sns.heatmap(new_df.corr(),annot=True,linewidths=0.3)
plt.figure(figsize=(10,6))
sns.barplot(x='Occupation',y='Purchase',data=df,hue='Marital_Status',ci=0)
plt.figure(figsize=(10,6))
sns.barplot(x='City_Category',y='Purchase',data=df,ci=0)


#df=df.replace(['A','B','C'],[0,1,2])

plt.figure(figsize=(10,6))
sns.barplot(x='City_Category',y='Purchase',data=df,hue='Marital_Status',ci=0,palette='spring')
plt.figure(figsize=(10,6))
sns.barplot(x='Product_Category_1',y='Purchase',data=df,ci=0)
plt.figure(figsize=(10,6))
sns.barplot(x='Product_Category_2',y='Purchase',data=df,ci=0)
plt.figure(figsize=(10,6))
sns.barplot(x='Product_Category_3',y='Purchase',data=df,ci=0)
x=df[['Occupation','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3',
      'Purchase']]
y=df['sex']
x.shape,y.shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=7)
x_train.shape,x_test.shape
y_train.shape,y_test.shape
model=RandomForestClassifier(n_estimators=150)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
from sklearn.metrics import r2_score
r2_score(y_test,prediction)