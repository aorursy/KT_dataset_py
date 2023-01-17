import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

df.head()
df.info()
df.fillna(value=df.mean(),inplace=True)  #Since there is no categorical data of int64 dtype, we can do this freely. Had there been categorical data of type int64 we would have to do df.fillna(value=df[numerical_features].mean(),inplace=True)

# Only NaNs in float and int dtype columns are replaced with mean. [See output of df.mean() to understand this]

df.head()

categorical_features=['gender','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod']

numerical_features=['MonthlyCharges','TotalCharges','tenure']
from sklearn.preprocessing import LabelEncoder 

from sklearn.preprocessing import StandardScaler 

  

le = LabelEncoder() 

df['gender']= le.fit_transform(df['gender'])

df['Married']= le.fit_transform(df['Married'])

df['Children']= le.fit_transform(df['Children'])

df['TVConnection']= le.fit_transform(df['TVConnection'])

df['Channel1']= le.fit_transform(df['Channel1'])

df['Channel2']= le.fit_transform(df['Channel2'])

df['Channel3']= le.fit_transform(df['Channel3'])

df['Channel4']= le.fit_transform(df['Channel4'])

df['Channel5']= le.fit_transform(df['Channel5'])

df['Channel6']= le.fit_transform(df['Channel6'])

df['Internet']= le.fit_transform(df['Internet'])

df['HighSpeed']= le.fit_transform(df['HighSpeed'])

df['AddedServices']= le.fit_transform(df['AddedServices'])

df['Subscription']= le.fit_transform(df['Subscription'])

df['PaymentMethod']= le.fit_transform(df['PaymentMethod'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

scaler = StandardScaler()





df.head()
df.fillna(value=df.mean(),inplace=True)

df.isnull().any().any()
corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
X=df[['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','tenure','PaymentMethod','MonthlyCharges','TotalCharges']].copy()

y=df["Satisfied"].copy()

from sklearn.model_selection import train_test_split

X[numerical_features] = scaler.fit_transform(X[numerical_features])



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.33,random_state=50) #changed from 50
#for public



from sklearn.cluster import KMeans



from sklearn.metrics import roc_auc_score





n_clusters=4

cluster_out={}

mi=0

bi=-1

for pp in range(2**n_clusters):

    clf=KMeans(n_clusters=n_clusters,random_state=42).fit(X)

    y1=clf.predict(X)

    cluster_out=[0]*n_clusters

    for i in range(n_clusters):

        if pp&(1<<i)!=0:

            cluster_out[i]=1

    for i in range(len(y1)):

        y1[i]=cluster_out[y1[i]]

    acc=roc_auc_score(y,y1)

    if acc>mi:

        print(acc,pp)

        mi=acc

        bi=pp



        
df1=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

df1.info()
from sklearn.preprocessing import LabelEncoder 

  

le = LabelEncoder() 

df1['gender']= le.fit_transform(df1['gender'])

df1['Married']= le.fit_transform(df1['Married'])

df1['Children']= le.fit_transform(df1['Children'])

df1['TVConnection']= le.fit_transform(df1['TVConnection'])

df1['Channel1']= le.fit_transform(df1['Channel1'])

df1['Channel2']= le.fit_transform(df1['Channel2'])

df1['Channel3']= le.fit_transform(df1['Channel3'])

df1['Channel4']= le.fit_transform(df1['Channel4'])

df1['Channel5']= le.fit_transform(df1['Channel5'])

df1['Channel6']= le.fit_transform(df1['Channel6'])

df1['Internet']= le.fit_transform(df1['Internet'])

df1['HighSpeed']= le.fit_transform(df1['HighSpeed'])

df1['AddedServices']= le.fit_transform(df1['AddedServices'])

df1['Subscription']= le.fit_transform(df1['Subscription'])

df1['PaymentMethod']= le.fit_transform(df1['PaymentMethod'])

df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'], errors='coerce')





df1.head()
df1.fillna(value=df1.mean(),inplace=True)

df1.isnull().any().any()
X_test=df1[['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','tenure','PaymentMethod','MonthlyCharges','TotalCharges']].copy()



X_test[numerical_features] = scaler.fit_transform(X_test[numerical_features])

n_clusters=4

cluster_out={}

pp = 5

clf=KMeans(n_clusters=n_clusters,random_state=42).fit(X)

y_pred=clf.predict(X_test)

cluster_out=[0]*n_clusters

for i in range(n_clusters):

    if pp&(1<<i)!=0:

        cluster_out[i]=1

for i in range(len(y_pred)):

    y_pred[i]=cluster_out[y_pred[i]]
df4=pd.DataFrame()

df4['custId']=df1['custId']

df4['Satisfied']=y_pred

df4.head()

for i in range(len(y_pred)):

    print(y_pred[i])



df4.to_csv('sol1.csv',index=False)