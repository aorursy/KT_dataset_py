import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

%matplotlib inline
df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

tf = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
df.head()
df.info()
for col_name in df.columns:

    print(col_name,'\t\t\t',df[col_name].unique())
#removing the ' ' value from TotalCharges attribute and converting it to number

df.TotalCharges = df.TotalCharges.str.replace(' ','')

df.TotalCharges = pd.to_numeric(df.TotalCharges).fillna(0.0)
categorical_columns = ['gender','Married','Children','TVConnection','Channel1','Channel2',

                       'Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed',

                       'AddedServices','Subscription','PaymentMethod']

df.TotalCharges = df.TotalCharges.astype('float64').fillna(0.0) 

df = pd.get_dummies(df,columns=categorical_columns)
scaler = StandardScaler()

scaler.fit(df)

scaler.transform(df)
corr = df.corr()

print(corr['Satisfied'][corr.Satisfied>0.2] )

# corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# important_columns = ['tenure','TotalCharges','TVConnection_No','Channel1_No tv connection',

#                     'Channel2_No tv connection','Channel3_No tv connection',

#                     'Channel4_No tv connection','Channel5_No tv connection',

#                     'Channel6_No tv connection','Subscription_Biannually']



important_columns = ['tenure','Subscription_Biannually']

y = df['Satisfied']

new_df = df[important_columns]

print(new_df.columns)
sns.countplot(y='Satisfied' , data=df)
new_df.columns
X = np.array(new_df)

# y = np.array(new_df['Satisfied'])
kmeans = KMeans(n_clusters=2, random_state=42,max_iter=1000,n_jobs=-1,n_init=3)

y_pred = kmeans.fit_predict(new_df)

correct = 0

for i in range(len(X)):

    if y_pred[i] == y[i]:

        correct += 1



print("accuracy ",correct/len(X))

print('roc ',roc_auc_score(y, y_pred))
#removing the ' ' value from TotalCharges attribute and converting it to number

tf.TotalCharges = tf.TotalCharges.str.replace(' ','')

tf.TotalCharges = pd.to_numeric(tf.TotalCharges).fillna(0.0)
# converting categorical data to numerical data

categorical_columns = ['gender','Married','Children','TVConnection','Channel1','Channel2',

                       'Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed',

                       'AddedServices','Subscription','PaymentMethod']



tf.TotalCharges = tf.TotalCharges.astype('float64').fillna(0.0) 

tf = pd.get_dummies(tf,columns=categorical_columns)
scaler.fit_transform(tf)
# important_columns = ['tenure','TotalCharges','TVConnection_No','Channel1_No tv connection',

#                     'Channel2_No tv connection','Channel3_No tv connection',

#                     'Channel4_No tv connection','Channel5_No tv connection',

#                     'Channel6_No tv connection','Subscription_Biannually',]



important_columns = ['tenure','Subscription_Biannually']

tf_id = tf['custId']

new_tf = tf[important_columns]
X = np.array(new_tf)

result = []

for i in range(len(X)):

    predict_me = np.array(X[i])

    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)

    result.append(prediction[0])
pred = pd.DataFrame({

    'custId' : tf_id,

    'Satisfied' : result

})
pred
# pred.to_csv('kmean_2nd_0.66062.csv',index=False)