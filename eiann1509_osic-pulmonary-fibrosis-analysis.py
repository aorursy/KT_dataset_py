import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Input,Dense,Dropout,Lambda
#lets work with csv files



train_df=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

test_df=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

sub_df=pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
train_df.head()
len(train_df['Patient'].unique())
train_df.groupby(['SmokingStatus']).count()['Sex']
train_df.info()
train_df.count()
train_df.columns
patient_train_df=set(train_df['Patient'].unique())

patient_test_df= set(test_df['Patient'].unique())
patient_train_df.intersection(patient_test_df)
unique_train_df=train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates()

unique_train_df.head()
train_df['Patient'].value_counts().max()
train_df['Patient'].value_counts().min()
train_df[train_df['Patient']=='ID00422637202311677017371']
import matplotlib.pyplot as plt



#1 visualization



#smoking status



unique_train_df['SmokingStatus'].value_counts().plot(kind='bar', 

                                              color='blue',

                                              title='Distribution of the SmokingStatus column')



plt.ylabel('counts')

plt.xticks(rotation = 0)
train_df['Weeks'].value_counts().head()
train_df['Weeks'].value_counts().head()
#week distribution

plt.figure(figsize=(50,40))

train_df['Weeks'][:int(len(train_df['Weeks'].unique())/2)].value_counts().plot(kind='barh', color='blue',

                                              title='Distribution of the weeks column')

plt.tick_params(axis="x", labelsize=20)

plt.tick_params(axis="y", labelsize=20)

train_df.head()
train_df['FVC'].max()
#fvc vs percent 

import plotly.express as px

plt.figure(figsize=(30,20))

px.scatter(train_df,x='Percent',y='FVC',color='Age')
#FVC vs age



px.bar(train_df,x='SmokingStatus',y='FVC',color='Age')
#percent



train_df['Percent'].value_counts()
plt.scatter(train_df['Percent'],train_df['FVC'],color='red')
data = train_df.append([test_df, sub_df])
len(train_df),len(test_df),len(sub_df)
data.info()
data.isnull().sum()
train_df = pd.concat( (train_df,test_df) )
from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()

train_df['SmokingStatus']=label_encoder.fit_transform(train_df['SmokingStatus'])

train_df['Sex']=label_encoder.fit_transform(train_df['Sex'])
train_df['Percent']       = (train_df['Percent'] - train_df['Percent'].mean()) / train_df['Percent'].std()

train_df['Age']           = (train_df['Age'] - train_df['Age'].mean()) / train_df['Age'].std()

train_df['Sex']           = (train_df['Sex'] - train_df['Sex'].mean()) / train_df['Sex'].std()

train_df['SmokingStatus'] = (train_df['SmokingStatus'] - train_df['SmokingStatus'].mean()) / train_df['SmokingStatus'].std()

train_df.head(10)
#keras model



# model architecture https://www.kaggle.com/chrisden/6-82-quantile-reg-lr-schedulers-checkpoints



from tensorflow.keras.models import Model



i = Input(shape=(5,))

x = Dense(100, activation="relu", name="d1")(i)

x = Dense(100, activation="relu", name="d2")(x)

p1 = Dense(3, activation="linear", name="p1")(x)

p2 = Dense(3, activation="relu", name="p2")(x)

preds=preds =Lambda(lambda x: x[0] + tf.cumsum(x[1], axis = 1), 

                     name = "preds")([p1, p2])





model=Model(i,[p1,p2])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False),loss='mse')
model.summary()
r=model.fit(train_df[['Weeks','Percent','Sex','SmokingStatus','Age']],train_df['FVC'],epochs=100,verbose=1)
#quantile regression model

#linear model



from statsmodels.formula.api import quantreg



model_1=quantreg('FVC ~ Weeks+Sex+Age+SmokingStatus+Percent', train_df).fit(q=0.15)

model_2=quantreg('FVC ~ Weeks+Sex+Age+SmokingStatus+Percent', train_df).fit(q=0.50)

model_3=quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train_df).fit( q=0.85 )
train_df['FVC1'],train_df['FVC2'],train_df['FVC3']=model_1.predict(train_df),model_2.predict(train_df),model_3.predict(train_df)
train_df.head()
len(sub_df),len(train_df)
df=pd.DataFrame()
df['Confidence1'] = train_df.iloc[:730,9] - train_df.iloc[:730,8]

df['Patient_Week']=train_df['Patient'][:730]+'_'+train_df['Weeks'].astype('str')[:730]

df['Confidence']=sub_df['Confidence']

df['FVC']=train_df['FVC'][:730]

df['FVC1']=train_df['FVC1'][:730]

# get rid of unused data and show some non-empty data

submission = df[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

submission.loc[~submission.FVC1.isnull()].head(10)
