#importing important libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#loading the dataset

data=pd.read_csv("../input/clicks-conversion-tracking/KAG_conversion_data.csv")
data.head()
data.shape
#checking if any missing value is there

data.isnull().sum()
#Checking the info of the data

data.info()
data.describe()
#Taking a copy of the dataset to do further operations

df=data.copy()
df.head()
#Checking the unique elements of 'age' column

df['age'].unique()

#replace the range with average values

df['age']=df['age'].replace(['30-34','35-39','40-44','45-49'],[32,37,42,47])

df[['age']] = df[['age']].apply(pd.to_numeric) 
df['age']
#replace 'Male' with '0' and 'Female'with '1'

df['gender']=df['gender'].replace(['M','F'],[0,1])
df['gender']
#checking types of columns

df.dtypes
#taking important columns for processing

ds=df[['age','gender','interest','Impressions','Clicks','Spent','Total_Conversion','Approved_Conversion']]
ds.head()
#creating some new calculated columns of our use

#How much amount is spent per click, i.e. SPC=Spent/Clicks

ds["SPC"]=df["Spent"]/df["Clicks"]

#How many impression turned into clicks, i.e. CPI%=(Clicks/Impressions)*100

ds["CPI"]=(df["Clicks"]/df["Impressions"])*100
#Checking the complete preprocessed dataset

ds.head()
#Let's check the correlation between features using heatmap

f,ax = plt.subplots(figsize=(15, 10))

sns.heatmap(ds.corr(method='pearson'), annot=True, fmt= '.1f',ax=ax, cmap="BrBG")
cba=ds.groupby("age")["Clicks"].count() #Clicks per age group

Iba=ds.groupby("age")["Impressions"].count() #Impressions per age group

conv_age=ds.groupby("age")["Total_Conversion"].count() #Conversions per age group

CPI_age=ds.groupby("age")["CPI"].count() #CPI per age group

plt.subplot(221)

ax = cba.plot(kind='bar', figsize=(10,6), color="blue", fontsize=10)

ax.set_title("Clicks by age", fontsize=16)

ax.set_xlabel("Age", fontsize=12);

ax.set_ylabel("Clicks", fontsize=12);

plt.subplot(222)

ix = Iba.plot(kind='bar', figsize=(10,6), color="gray", fontsize=10)

ix.set_title("Impressions by age", fontsize=16)

ix.set_xlabel("Age", fontsize=12);

ix.set_ylabel("Impressions", fontsize=12);

plt.subplot(223)

bx = conv_age.plot(kind='bar', figsize=(10,6), color="green", fontsize=10)

bx.set_title("Conversion by age", fontsize=16)

bx.set_xlabel("Age", fontsize=12);

bx.set_ylabel("Conversion", fontsize=12);

plt.subplot(224)

cx = CPI_age.plot(kind='bar', figsize=(10,6), color="maroon", fontsize=10)

cx.set_title("CPI by age", fontsize=16)

cx.set_xlabel("Age", fontsize=12);

cx.set_ylabel("CPI", fontsize=12);

plt.tight_layout()

plt.show()
#Now let's try to find some relation between age, gender and other important parameters

for column in ds[['Clicks','Impressions','Spent','Total_Conversion','CPI']]:

    with sns.axes_style(style='ticks'):

             g = sns.catplot("age", column, "gender", data=ds, height=8, kind="box")

             g.set_axis_labels("Age", column);
#Checking the unique Campaign IDs

df['xyz_campaign_id'].unique()
#Let's Check the amount spent on each Campaign

sns.boxplot(x='xyz_campaign_id', y='Spent',data=df, color='gray', width=1)
#Storing the 1178 campaign stats in a different dataframe

df_1178=df.loc[df['xyz_campaign_id'] == 1178]
df_1178.shape
df_1178.head()
df_1178=df_1178.drop(['xyz_campaign_id'],axis=1)
#Check the correlation heatmap

f,ax = plt.subplots(figsize=(15, 10))

sns.heatmap(df_1178.corr(method='pearson'), annot=True, fmt= '.1f',ax=ax)
#Let us get the insights based on age

sns.boxplot(x='age', y='Clicks',data=df_1178, color='gray', width=1)
cbag=df_1178.groupby("age")["Clicks"].count()

ax = cbag.plot(kind='bar', figsize=(10,6), color="blue", fontsize=10)

ax.set_title("Clicks by age", fontsize=16)

ax.set_xlabel("Age", fontsize=12);

ax.set_ylabel("Clicks", fontsize=12);

plt.show()
#Let's check the conversion for different age groups

conv_ages=df_1178.groupby("age")["Total_Conversion"].count() #Conversions per age group

bx = conv_ages.plot(kind='bar', figsize=(10,6), color="lime", fontsize=10)

bx.set_title("Conversion by age", fontsize=16)

bx.set_xlabel("Age", fontsize=12);

bx.set_ylabel("Conversion", fontsize=12);
for column in df_1178[['Clicks','Impressions','Spent','Total_Conversion']]:

    with sns.axes_style(style='ticks'):

             g = sns.catplot("age", column, "gender", data=df_1178, height=8, kind="box")

             g.set_axis_labels("Age", column);

        
ds1=df[['xyz_campaign_id','age','gender','interest','Impressions','Clicks','Spent','Total_Conversion','Approved_Conversion']]
#Using Elbow method!

from sklearn.cluster import KMeans

wcss=[]

K_rng=10



for i in range(1,K_rng):

    K=KMeans(i)

    K.fit(ds1)

    w=K.inertia_

    wcss.append(w)

    

Clusters=range(1,K_rng)

plt.figure(figsize=(12,8))

plt.plot(Clusters,wcss)

plt.xlabel('Clusters')

plt.ylabel('WCSS Values') #Within Cluster Sum of Squares

plt.title('Elbow Method Visualisation')
#Fitting the model

K2= KMeans(2)

K2.fit(ds1)
#Prediction using the model

ds1_pred=ds1.copy()

ds1_pred['Predicted']=K2.fit_predict(ds1)
#Visualise the clusters (Clicks and Conversion) after prediction

plt.figure(figsize=(8,5))

plt.scatter(ds1_pred['Clicks'], ds1_pred['Total_Conversion'], c=ds1_pred['Predicted'], cmap = 'rainbow')

plt.xlabel('Clicks')

plt.ylabel('Conversion')

plt.title('Clicks VS Conversion(K=2)')
#Visualise the clusters (Impressions and Clicks) after prediction

plt.figure(figsize=(8,5))

plt.scatter(ds1_pred['Impressions'], ds1_pred['Clicks'], c=ds1_pred['Predicted'], cmap = 'jet')

plt.xlabel('Impressions')

plt.ylabel('Clicks')

plt.title('Impressions VS Clicks(K=2)')
#Fitting the model

K3= KMeans(3)

K3.fit(ds1)
#Prediction using the model

ds1_pred2=ds1.copy()

ds1_pred2['Predicted']=K3.fit_predict(ds1)
#Visualise the clusters (Clicks and Conversion) after prediction

plt.figure(figsize=(8,5))

plt.scatter(ds1_pred2['Clicks'], ds1_pred2['Total_Conversion'], c=ds1_pred2['Predicted'], cmap = 'rainbow')

plt.xlabel('Clicks')

plt.ylabel('Conversion')

plt.title('Clicks VS Conversion(K=3)')
#Visualise the clusters (Impressions and Clicks) after prediction

plt.figure(figsize=(8,5))

plt.scatter(ds1_pred2['Impressions'], ds1_pred2['Clicks'], c=ds1_pred2['Predicted'], cmap = 'jet')

plt.xlabel('Impressions')

plt.ylabel('Clicks')

plt.title('Impressions VS Clicks(K=3)')
#Fitting the model

K4= KMeans(4)

K4.fit(ds1)
#Prediction using the model

ds1_pred3=ds1.copy()

ds1_pred3['Predicted']=K4.fit_predict(ds1)
#Visualise the clusters (Clicks and Conversion) after prediction

plt.figure(figsize=(8,5))

plt.scatter(ds1_pred3['Clicks'], ds1_pred3['Total_Conversion'], c=ds1_pred3['Predicted'], cmap = 'rainbow')

plt.xlabel('Clicks')

plt.ylabel('Conversion')

plt.title('Clicks VS Conversion(K=4)')
#Visualise the clusters (Impressions and Clicks) after prediction

plt.figure(figsize=(8,5))

plt.scatter(ds1_pred3['Impressions'], ds1_pred3['Clicks'], c=ds1_pred3['Predicted'], cmap = 'jet')

plt.xlabel('Impressions')

plt.ylabel('Clicks')

plt.title('Impressions VS Clicks(K=4)')