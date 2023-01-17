# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# machine learning

from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error,make_scorer

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer

from sklearn.model_selection import GridSearchCV
df= pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

test_df= pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")
df.info()
test_df.info()
#correct data format

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')

test_df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
missing_count = df.isnull().sum()

missing_count[missing_count > 0]
f = df["TotalCharges"].mean()

df["TotalCharges"].fillna(value=f, inplace=True)
missing_count = test_df.isnull().sum()

missing_count[missing_count > 0]
f = test_df["TotalCharges"].mean()

test_df["TotalCharges"].fillna(value=f, inplace=True)
#Convert ordinal features to numerical



from sklearn import preprocessing

le = preprocessing.LabelEncoder()

features = ["gender","Married","Children","TVConnection","Channel1","Channel2","Channel3","Channel4",

             "Channel5","Channel6","Internet","HighSpeed","AddedServices","Subscription","PaymentMethod"]

le=preprocessing.LabelEncoder()



for x in features:

     df[x]=le.fit_transform(df.__getattr__(x))
df.head(5)
for x in features:

     test_df[x]=le.fit_transform(test_df.__getattr__(x))
plt.figure(figsize=(15,15))

sns.heatmap(data=df.corr(),annot=True)
X = np.array(df.drop(['custId','Satisfied','TVConnection','tenure'], 1).astype(float))

y = df['Satisfied']

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
x_train = df.drop(['custId','Satisfied','TVConnection','tenure'], axis=1)

y_train = df['Satisfied']

x_test = test_df.drop(['custId','TVConnection','tenure'], axis=1)
scaler = MinMaxScaler()

# scaler = RobustScaler()

#scaler = Normalizer()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test) 
kmeans = KMeans(n_clusters=2, max_iter=700, algorithm = 'auto')

kmeans.fit(X)

label = kmeans.predict(X_scaled)
correct = 0

for i in range(len(X_scaled)):

    predict_me = np.array(X_scaled[i].astype(float))

    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = kmeans.predict(predict_me)

    if prediction == 1-y[i]:

        correct += 1



print(correct/len(X_scaled))
kmeans = KMeans(n_clusters=2, max_iter=700, algorithm = 'auto')

kmeans.fit(x_test)

label = kmeans.predict(x_test)
submission = pd.DataFrame({'custId':test_df['custId'],'Satisfied':label})

# submission.head(40)
filename = 'result.csv'

submission.to_csv(filename,index=False)