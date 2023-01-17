import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
df_train_raw = pd.read_csv('train.csv')

df_test_raw = pd.read_csv('test.csv')
df_train_raw.head()
df_train_raw.info()
df_train_raw.loc[544, 'TotalCharges'] = 2276.90

df_train_raw.loc[1348, 'TotalCharges'] = 2276.90

df_train_raw.loc[1553, 'TotalCharges'] = 2276.90

df_train_raw.loc[2504, 'TotalCharges'] = 2276.90

df_train_raw.loc[3083, 'TotalCharges'] = 2276.90

df_train_raw.loc[4766, 'TotalCharges'] = 2276.90

df_train_raw = df_train_raw.astype({"TotalCharges":float}) 
df_test_raw.loc[71, 'TotalCharges'] = 2298.24

df_test_raw.loc[580, 'TotalCharges'] = 2298.24

df_test_raw.loc[637, 'TotalCharges'] = 2298.24

df_test_raw.loc[790, 'TotalCharges'] = 2298.24

df_test_raw.loc[1505, 'TotalCharges'] = 2298.24

df_test_raw = df_test_raw.astype({"TotalCharges":float}) 
df_test_raw.info()
df_train_ohe = pd.get_dummies(df_train_raw)

df_test_ohe = pd.get_dummies(df_test_raw)
corr = df_train_ohe.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 20))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(250, 30, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
df_train_raw.info()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import Birch

from sklearn.preprocessing import minmax_scale

from sklearn.metrics import accuracy_score
cols = list(df_train_raw.columns)
dic = {}

for i in cols:

#     print(i)

    if(i=='Satisfied'):

        continue

    knn1 = KNeighborsClassifier(n_neighbors=2)

    data = np.asarray(df_train_raw[i].astype("category").cat.codes)

    data = minmax_scale(data)

    knn1.fit(data.reshape(-1, 1), df_train_raw['Satisfied'])

    try:

        val = roc_auc_score(knn1.predict(data.reshape(-1, 1)),  df_train_raw['Satisfied'])

        print(i, max(val, 1-val))

        dic[i] = max(val, 1-val)

    except:

        print(i, "Not working")
sorted(dic, key=dic.get, reverse=True)
train_df
def preprocess(data):

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"],errors = 'coerce')

#     data.dropna(axis = 0,inplace=True)

    data["HighSpeed"].replace({'No internet' : 0,'No' : 1 ,'Yes' : 2},inplace = True)

    data['gender'].replace({'Male' : 0, 'Female' : 1},inplace=True)

    data[["Married","Children","AddedServices"]] = data[["Married","Children","AddedServices"]].replace({ 'No' : 0 , 'Yes' : 1})

    data[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]]=data[["Channel1","Channel2","Channel3","Channel4","Channel5","Channel6"]].replace({'No tv connection' : 0 , 'No' : 1 , 'Yes' : 2 })

    data["Subscription"].replace({'Monthly':1,'Biannually':6,'Annually':12},inplace=True)

    data = pd.get_dummies(data = data,columns=['TVConnection','PaymentMethod'])

    return data
# using roc_auc_score



cat_cols = ['TVConnection', 'Channel6', 'SeniorCitizen']

num_cols = ['TotalCharges']

train_df = df_train_raw[num_cols+cat_cols]

test_df = df_test_raw[num_cols+cat_cols]

minmaxScalar = MinMaxScaler()



# train_df['Subscription'] = df_train_raw['Subscription'].astype("category").cat.codes

# test_df['Subscription'] = df_test_raw['Subscription'].astype("category").cat.codes



for i in cat_cols:

    train_df[i] = df_train_raw[i].astype("category").cat.codes

    test_df[i] = df_test_raw[i].astype("category").cat.codes



train_df = minmaxScalar.fit_transform(train_df)

test_df = minmaxScalar.transform(test_df)

km1 = KMeans(n_clusters=2)

km1.fit(train_df)

train_score = roc_auc_score(df_train_raw['Satisfied'], km1.predict(train_df))

ans = km1.predict(test_df)

if(train_score<0.5):

    ans = [not x for x in ans]

    print(1-train_score)

else:

    print(train_score)
sub1 = pd.merge(left=df_test_raw[['custId']], right=pd.DataFrame(data=ans, columns=['Satisfied']), left_index=True, right_index=True)
pd.DataFrame.to_csv(sub1, "sub.csv", index=False)
cols_subset = ['tenure', 'MonthlyCharges', 'TotalCharges','Married_Yes', 

       'Children_No', 'TVConnection_Cable','Channel1_No tv connection',

       'Channel2_No tv connection', 'Channel3_No', 'Channel3_No tv connection',

       'Channel4_No', 'Channel4_No tv connection', 'Channel5_No',

       'Channel5_No tv connection', 'Channel6_No',

       'Channel6_No tv connection','AddedServices_Yes', 

       'Subscription_Annually', 'Subscription_Biannually','Subscription_Monthly', 'PaymentMethod_Net Banking']

df_train_ohe_ = df_train_ohe[cols_subset]

df_train_ohe_ = MinMaxScaler().fit_transform(df_train_ohe_)

df_test_ohe_ = df_test_ohe[cols_subset]

df_test_ohe_ = MinMaxScaler().fit_transform(df_test_ohe_)
birch = Birch(n_clusters=2).fit(df_train_ohe_)

ans = birch.predict(df_test_ohe_)

ans = [not x for x in ans]
df_train_raw = pd.read_csv('train.csv')

df_test_raw = pd.read_csv('test.csv')

cols_subset = ['gender', 'SeniorCitizen', 'Married', 'Children',

       'Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5',

       'Channel6', 'HighSpeed', 'AddedServices',

       'Subscription', 'tenure', 'MonthlyCharges', 'TotalCharges', 'TVConnection_Cable', 'TVConnection_DTH',

       'TVConnection_No', 'PaymentMethod_Bank transfer',

       'PaymentMethod_Cash', 'PaymentMethod_Credit card',

       'PaymentMethod_Net Banking']

tmp_data = df_train_raw

df_train_k2 = preprocess(tmp_data)

df_train_k2.drop(['Internet', 'custId', 'Satisfied'], axis=1, inplace=True)

df_train_k2.loc[:,  ['TotalCharges','tenure','MonthlyCharges']] = MinMaxScaler().fit_transform(df_train_k2[['TotalCharges','tenure','MonthlyCharges']])

tmp_data = df_test_raw

df_test_k2 = preprocess(tmp_data)

df_test_k2.drop(['Internet', 'custId'], axis=1, inplace=True)

df_test_k2.loc[:,  ['TotalCharges','tenure','MonthlyCharges']] = MinMaxScaler().fit_transform(df_test_k2[['TotalCharges','tenure','MonthlyCharges']])
df_train_k2.fillna(df_train_k2.mean(), inplace=True)

df_test_k2.fillna(df_test_k2.mean(), inplace=True)
kmeans_2 = KMeans(n_clusters=2).fit(df_train_k2)



y_pred_2 = kmeans_2.predict(df_test_k2)
# train_df = df_train_ohe[['SeniorCitizen',

#  'tenure',

#  'MonthlyCharges',

#  'TotalCharges',

#  'Married_No',

#  'Married_Yes',

#  'Children_No',

#  'Children_Yes',

#  'TVConnection_Cable',

#  'TVConnection_DTH',

#  'TVConnection_No',

#  'Channel1_No',

#  'Channel1_No tv connection',

#  'Channel1_Yes',

#  'Channel2_No',

#  'Channel2_No tv connection',

#  'Channel2_Yes',

#  'Channel3_No',

#  'Channel3_No tv connection',

#  'Channel3_Yes',

#  'Channel4_No',

#  'Channel4_No tv connection',

#  'Channel4_Yes',

#  'Channel5_No',

#  'Channel5_No tv connection',

#  'Channel5_Yes',

#  'Channel6_No',

#  'Channel6_No tv connection',

#  'Channel6_Yes',

#  'AddedServices_No',

#  'AddedServices_Yes',

#  'Subscription_Annually',

#  'Subscription_Biannually',

#  'Subscription_Monthly',

#  'PaymentMethod_Bank transfer',

#  'PaymentMethod_Cash',

#  'PaymentMethod_Credit card',

#  'PaymentMethod_Net Banking']

# ]

# train_clusters = df_train_ohe[['custId', 'Satisfied']]



# test_df = df_test_ohe[['SeniorCitizen',

#  'tenure',

#  'MonthlyCharges',

#  'TotalCharges',

#  'Married_No',

#  'Married_Yes',

#  'Children_No',

#  'Children_Yes',

#  'TVConnection_Cable',

#  'TVConnection_DTH',

#  'TVConnection_No',

#  'Channel1_No',

#  'Channel1_No tv connection',

#  'Channel1_Yes',

#  'Channel2_No',

#  'Channel2_No tv connection',

#  'Channel2_Yes',

#  'Channel3_No',

#  'Channel3_No tv connection',

#  'Channel3_Yes',

#  'Channel4_No',

#  'Channel4_No tv connection',

#  'Channel4_Yes',

#  'Channel5_No',

#  'Channel5_No tv connection',

#  'Channel5_Yes',

#  'Channel6_No',

#  'Channel6_No tv connection',

#  'Channel6_Yes',

#  'AddedServices_No',

#  'AddedServices_Yes',

#  'Subscription_Annually',

#  'Subscription_Biannually',

#  'Subscription_Monthly',

#  'PaymentMethod_Bank transfer',

#  'PaymentMethod_Cash',

#  'PaymentMethod_Credit card',

#  'PaymentMethod_Net Banking']

# ]



train_df = df_train_ohe[['SeniorCitizen',

 'tenure',

 'MonthlyCharges',

 'TotalCharges',

 'Married_No',

 'Married_Yes',

 'Children_No',

 'Children_Yes',

 'TVConnection_Cable',

 'TVConnection_DTH',

 'TVConnection_No',

 'Channel1_No',

 'Channel1_No tv connection',

 'Channel1_Yes',

 'Channel2_No',

 'Channel2_No tv connection',

 'Channel2_Yes',

 'Channel3_No',

 'Channel3_No tv connection',

 'Channel3_Yes',

 'Channel4_No',

 'Channel4_No tv connection',

 'Channel4_Yes',

 'Channel5_No',

 'Channel5_No tv connection',

 'Channel5_Yes',

 'Channel6_No',

 'Channel6_No tv connection',

 'Channel6_Yes',

 'AddedServices_No',

 'AddedServices_Yes',

 'Subscription_Annually',

 'Subscription_Biannually',

 'Subscription_Monthly',]

]

train_clusters = df_train_ohe[['custId', 'Satisfied']]



test_df = df_test_ohe[['SeniorCitizen',

 'tenure',

 'MonthlyCharges',

 'TotalCharges',

 'Married_No',

 'Married_Yes',

 'Children_No',

 'Children_Yes',

 'TVConnection_Cable',

 'TVConnection_DTH',

 'TVConnection_No',

 'Channel1_No',

 'Channel1_No tv connection',

 'Channel1_Yes',

 'Channel2_No',

 'Channel2_No tv connection',

 'Channel2_Yes',

 'Channel3_No',

 'Channel3_No tv connection',

 'Channel3_Yes',

 'Channel4_No',

 'Channel4_No tv connection',

 'Channel4_Yes',

 'Channel5_No',

 'Channel5_No tv connection',

 'Channel5_Yes',

 'Channel6_No',

 'Channel6_No tv connection',

 'Channel6_Yes',

 'AddedServices_No',

 'AddedServices_Yes',

 'Subscription_Annually',

 'Subscription_Biannually',

 'Subscription_Monthly']

]
from sklearn.cluster import KMeans

from sklearn.metrics import roc_auc_score
km1 = KMeans(n_clusters=2)
km1.fit(train_df)
tmp = km1.predict(train_df)
print(np.sum(tmp), len(tmp))
print(np.sum(np.asarray(train_clusters['Satisfied'])))
tmp = km1.predict(test_df)
sub1 = pd.merge(left=df_test_ohe[['custId']], right=pd.DataFrame(data=tmp, columns=['Satisfied']), left_index=True, right_index=True)

# pd.merge([df_test_ohe[['custId']], pd.DataFrame(data=tmp, columns=['Satisfied'])])
pd.DataFrame.to_csv(sub1, 'sub1.csv', index=False)
roc_auc_score(train_clusters['Satisfied'], km1.predict(train_df))