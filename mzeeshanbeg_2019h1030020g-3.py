import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





%matplotlib inline



pd.set_option('display.max_columns', 100)
filename = "/kaggle/input/eval-lab-3-f464/train.csv"
customer = pd.read_csv(filename)
customer.head(n=5)
customer.isnull().any().any()
customer.info()

# df.describe(include='object')

# df.isnull().any().any()
sns.boxplot(x="PaymentMethod", y="Satisfied", data=customer)
sns.regplot(x="MonthlyCharges", y="Satisfied", data=customer)
sns.boxplot(x="SeniorCitizen", y="Satisfied", data=customer)
customer.corr()
customer['gender']=customer['gender'].astype('category').cat.codes

customer['Married']=customer['Married'].astype('category').cat.codes

customer['Children']=customer['Children'].astype('category').cat.codes

customer['TVConnection']=customer['TVConnection'].astype('category').cat.codes

customer['Channel1']=customer['Channel1'].astype('category').cat.codes

customer['Channel2']=customer['Channel2'].astype('category').cat.codes

customer['Channel3']=customer['Channel3'].astype('category').cat.codes

customer['Channel4']=customer['Channel4'].astype('category').cat.codes

customer['Channel5']=customer['Channel5'].astype('category').cat.codes

customer['Channel6']=customer['Channel6'].astype('category').cat.codes

customer['Internet']=customer['Internet'].astype('category').cat.codes

customer['HighSpeed']=customer['HighSpeed'].astype('category').cat.codes

customer['AddedServices']=customer['AddedServices'].astype('category').cat.codes

# customer['MonthlyCharges']=customer['MonthlyCharges'].astype('category').cat.codes

customer['Subscription']=customer['Subscription'].astype('category').cat.codes

customer['PaymentMethod']=customer['PaymentMethod'].astype('category').cat.codes
customer.head()
customer.info()
customer['TotalCharges']=customer['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)

customer.info()

# customer['TotalCharges']=customer['TotalCharges'].astype('category').cat.codes

sns.regplot(x="TotalCharges", y="Satisfied", data=customer)
X = customer[["SeniorCitizen","MonthlyCharges","Married","Children","TVConnection", "Channel1","Channel2", "Channel3", "Channel4","Channel5","Channel6" ,"Internet", "HighSpeed","AddedServices", "Subscription","TotalCharges","tenure","PaymentMethod"]].copy()

y = customer[['Satisfied']].copy()
# features = ["Married","Children","TVConnection", "Channel1","Channel2", "Channel3", "Channel4","Channel5","Channel6" ,"Internet", "HighSpeed","AddedServices", "Subscription","PaymentMethod",'MonthlyCharges','SeniorCitizen','tenure','TotalCharges']

features=['Married','Children','SeniorCitizen','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','AddedServices','Subscription','PaymentMethod','tenure']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(customer[features])

X_scaled

y_scaled=scaler.fit_transform(customer[['Satisfied']])



# standardizing the data

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# X_scaled = scaler.fit_transform(X[numerical_features])
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X_scaled,y_scaled,test_size=0.20,random_state=42)
from sklearn.cluster import KMeans

#take only 2 clusters, Satisfied or Not Satisfied

# km = KMeans(

#     n_clusters=2, init='k-means++',

#     n_init=10, max_iter=300, 

#     tol=1e-04, random_state=0

# )

km = KMeans( n_clusters=2)

y_km = km.fit(X_train,y_train)
filename_test = "/kaggle/input/eval-lab-3-f464/train.csv"
custTest = pd.read_csv(filename)

custTest.info()

# df2.head()
custTest['gender']=custTest['gender'].astype('category').cat.codes

custTest['Married']=custTest['Married'].astype('category').cat.codes

custTest['Children']=custTest['Children'].astype('category').cat.codes

custTest['TVConnection']=custTest['TVConnection'].astype('category').cat.codes

custTest['Channel1']=custTest['Channel1'].astype('category').cat.codes

custTest['Channel2']=custTest['Channel2'].astype('category').cat.codes

custTest['Channel3']=custTest['Channel3'].astype('category').cat.codes

custTest['Channel4']=custTest['Channel4'].astype('category').cat.codes

custTest['Channel5']=custTest['Channel5'].astype('category').cat.codes

custTest['Channel6']=custTest['Channel6'].astype('category').cat.codes

custTest['Internet']=custTest['Internet'].astype('category').cat.codes

custTest['HighSpeed']=custTest['HighSpeed'].astype('category').cat.codes

custTest['AddedServices']=custTest['AddedServices'].astype('category').cat.codes

custTest['Subscription']=custTest['Subscription'].astype('category').cat.codes

custTest['PaymentMethod']=custTest['PaymentMethod'].astype('category').cat.codes

# custTest['MonthlyCharges']=custTest['MonthlyCharges'].astype('category').cat.codes

# df.info()



custTest['TotalCharges']=custTest['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)

# custTest['TotalCharges']=custTest['TotalCharges'].astype('category').cat.codes



custTest.info()
X_t = custTest[["SeniorCitizen","Married","Children","TVConnection", "Channel1","Channel2", "Channel3", "Channel4","Channel5","Channel6" ,"Internet", "HighSpeed","AddedServices", "Subscription","MonthlyCharges","TotalCharges","tenure","PaymentMethod"]].copy()
# features = ["Married","Children","TVConnection", "Channel1","Channel2", "Channel3", "Channel4","Channel5","Channel6" ,"Internet", "HighSpeed","AddedServices", "Subscription","PaymentMethod",'MonthlyCharges','SeniorCitizen','tenure','TotalCharges']

features=['Married','Children','SeniorCitizen','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','AddedServices','Subscription','PaymentMethod','tenure']

from sklearn.preprocessing import MinMaxScaler

scaler_t = MinMaxScaler()

X_t_scaled = scaler_t.fit_transform(custTest[features])



X_t_scaled



# from sklearn.preprocessing import StandardScaler

# scaler_t = StandardScaler()

# X_t_scaled = scaler_t.fit_transform(X_t[numerical_features])
y_t = km.predict(X_t_scaled)
y_t
final_pred = custTest[['custId']].copy()

final_pred['Satisfied'] = y_t
final_pred
final_pred.to_csv("eval_3_pred.csv",index=False,encoding='utf-8')