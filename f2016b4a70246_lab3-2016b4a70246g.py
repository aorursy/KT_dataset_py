import numpy as np

import pandas as pd



from sklearn import cluster, datasets

from sklearn.preprocessing import StandardScaler,MinMaxScaler



np.random.seed(0)
data = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")
def encode(data):

    data['gender'].replace("Male",1,inplace=True)

    data['gender'].replace("Female",0,inplace=True)

    data['Married'].replace("No",0,inplace=True)

    data['Married'].replace("Yes",1,inplace=True)

    data['Children'].replace("No",0,inplace=True)

    data['Children'].replace("Yes",1,inplace=True)

    data['Internet'].replace("No",0,inplace=True)

    data['Internet'].replace("Yes",1,inplace=True)

    data['AddedServices'].replace("No",0,inplace=True)

    data['AddedServices'].replace("Yes",1,inplace=True)

    data['Subscription'].replace("Monthly",12,inplace=True)

    data['Subscription'].replace("Biannually",6,inplace=True)

    data['Subscription'].replace("Annually",1,inplace=True)

    highspeed_map = {'No internet':0,'No':1,'Yes':2}

    channel_map = {'No tv connection':0,'No':1,'Yes':2}

    channels=['Channel1','Channel2','Channel3','Channel4','Channel5','Channel6']

    for ch in channels:

        data.replace({ch:channel_map},inplace=True)

    data.replace({'HighSpeed':highspeed_map},inplace=True)

    
encode(data)

encode(test)
mean=0

num=0

for i in range(len(data['TotalCharges'])):

    if(data['TotalCharges'][i]!=' '):

        mean+=float(data['TotalCharges'][i])

        num+=1

mean=(float(mean))/num

mean

for i in range(len(data['TotalCharges'])):

    if(data['TotalCharges'][i]==' '):

        data['TotalCharges'][i]=mean



data['TotalCharges']=[float(i) for i in data['TotalCharges']]
for i in range(len(test['TotalCharges'])):

    if(test['TotalCharges'][i]==' '):

        test['TotalCharges'][i]=mean

        print(test['TotalCharges'][i])



test['TotalCharges']=[float(i) for i in test['TotalCharges']]
numerical_features=['MonthlyCharges','TotalCharges']

categorical_features=['TVConnection','PaymentMethod']

others=['gender','SeniorCitizen','Married','Internet','Children','AddedServices','tenure','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','HighSpeed']
X=data.loc[:,'gender':'TotalCharges']

y=data.loc[:,["Satisfied"]]

X_test=test.loc[:,'gender':'TotalCharges']

y.shape
comp_X=pd.concat([X,X_test])

comp_X.info()
arr = X_test

comp_X_num = arr[numerical_features]

scaler = MinMaxScaler()

comp_X_num = scaler.fit_transform(comp_X_num)

comp_X_num.shape
comp_X_enc= np.array(pd.get_dummies(arr[categorical_features]))

comp_X_enc.shape
X_new = np.concatenate([comp_X_num,comp_X_enc],axis=1)
X_new = np.concatenate([X_new,np.array(arr[others])],axis=1)
kmeans = cluster.KMeans(n_clusters=2)

new_cluster_indexes=kmeans.fit_predict(X_new)

new_cluster_indexes.shape
out = [[test['custId'][i],new_cluster_indexes[i]] for i in range(m)]
out_df = pd.DataFrame(data=out,columns=['custId','Satisfied'])

out_df.to_csv(r'out_4_4.csv',index=False)