# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
filename_train = '/kaggle/input/eval-lab-3-f464/train.csv'

filename_test = '/kaggle/input/eval-lab-3-f464/test.csv'
df_train = pd.read_csv(filename_train)

df_test = pd.read_csv(filename_test)
df_train.head(1)
def detail_df(df):

    data_type = pd.concat([df.dtypes,df.nunique(),df.isnull().sum()],axis=1)

    data_type.columns = ["dtype", "unique","no of null"]

    return data_type

df_detail = detail_df(df_train)

df_detail
#columns to be dropped:

drop_columns = []

#drop_columns = ['custId','Married','Children','PaymentMethod']



#dropping columns

df_proc1_train = df_train.drop(labels = drop_columns,axis =1,inplace = False)



df_proc1_test = df_test.drop(labels = drop_columns,axis =1,inplace = False)

print(df_test.__len__(),df_proc1_test.__len__())
#splitting into numerical and categorical feature set





numerical_features = ['MonthlyCharges','TotalCharges']

categorical_features = ['PaymentMethod','Children','Married','SeniorCitizen','gender','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription']

X_train = df_proc1_train[numerical_features+categorical_features+['Satisfied']]





X_test = df_proc1_test[numerical_features+categorical_features]

print(df_test.__len__(),df_proc1_test.__len__(),X_test.__len__())
# Function to process the data

def categorical_data_processing(data,excep):

    data["gender"][data["gender"] == "Male"] = 1

    data["gender"][data["gender"] == "Female"] = 0

    

    data["Married"][data["Married"] == "Yes"] = 1

    data["Married"][data["Married"] == "No"] = 0

    

    data["Children"][data["Children"] == "Yes"] = 1

    data["Children"][data["Children"] == "No"] = 0

    

    data["TVConnection"][data["TVConnection"] == "DTH"] = 1

    data["TVConnection"][data["TVConnection"] == "Cable"] = 0.5

    data["TVConnection"][data["TVConnection"] == "No"] = 0

    

    for ch_no in range(1, 7):

        data["Channel"+str(ch_no)][data["Channel"+str(ch_no)] == "Yes"] = 1

        data["Channel"+str(ch_no)][data["Channel"+str(ch_no)] == "No"] = 0.5

        data["Channel"+str(ch_no)][data["Channel"+str(ch_no)] == "No tv connection"] = 0

    

    data["Internet"][data["Internet"] == "Yes"] = 1

    data["Internet"][data["Internet"] == "No"] = 0

    

    data["HighSpeed"][data["HighSpeed"] == "Yes"] = 1

    data["HighSpeed"][data["HighSpeed"] == "No"] = 0.5

    data["HighSpeed"][data["HighSpeed"] == "No internet"] = 0

    

    data["AddedServices"][data["AddedServices"] == "Yes"] = 1

    data["AddedServices"][data["AddedServices"] == "No"] = 0

    

    data["Subscription"][data["Subscription"] == "Monthly"] = 1

    data["Subscription"][data["Subscription"] == "Biannually"] = 0.5

    data["Subscription"][data["Subscription"] == "Annually"] = 0

    

    data["PaymentMethod"][data["PaymentMethod"] == "Cash"] = 0

    data["PaymentMethod"][data["PaymentMethod"] == "Bank transfer"] = 0.33

    data["PaymentMethod"][data["PaymentMethod"] == "Net Banking"] = 0.67

    data["PaymentMethod"][data["PaymentMethod"] == "Credit card"] = 1

    

    data["TotalCharges"][data["TotalCharges"] == " "] = -1

    

    for col in data.columns:

        if col != excep:

            data[col] = data[col].astype(np.float)

            data[col] -= data[col].mean()

            data[col] /= data[col].std()

            print("processed " + str(col))

        

    return data
X_train = categorical_data_processing(X_train,'Satisfied')

X_test = categorical_data_processing(X_test,'Satisfied')

X_train['TotalCharges'][X_train['TotalCharges'] == -1] = X_train['TotalCharges'][X_train['TotalCharges'] != -1].mean()

X_test['TotalCharges'][X_test['TotalCharges'] == -1] = X_test['TotalCharges'][X_test['TotalCharges'] != -1].mean()

print(df_test.__len__(),df_proc1_test.__len__(),X_test.__len__())
from matplotlib import pyplot as plt

from sklearn.cluster import SpectralClustering,KMeans

no_of_clusters = 20

#model = SpectralClustering(n_clusters=no_of_clusters, affinity='rbf',assign_labels='kmeans')

#labels = model.fit_predict(X_train.drop(labels = "Satisfied",axis =1,inplace = False))

model = KMeans(n_clusters=no_of_clusters).fit(X_train.drop(labels = "Satisfied",axis =1,inplace = False))

labels = model.predict(X_train.drop(labels = "Satisfied",axis =1,inplace = False))



centres = []

assosciated_satisfied_value = []

for cluster_no in range(no_of_clusters):

    print(X_train[labels == cluster_no].__len__(),':',sum(X_train['Satisfied'][labels == cluster_no] == 1)/X_train[labels == cluster_no].__len__())

    centres.append(X_train[labels == cluster_no].drop(labels = 'Satisfied',axis =1).mean())

    if sum(X_train['Satisfied'][labels == cluster_no] == 1)/X_train[labels == cluster_no].__len__() > 0.61:

        assosciated_satisfied_value.append(1)

    else:

        assosciated_satisfied_value.append(0)

print(df_test.__len__(),df_proc1_test.__len__(),X_test.__len__())
from sklearn.metrics import pairwise_distances_argmin

#Predicting cluster

cluster = []

test_satisfied_values = []



for i in range(X_test.__len__()):

    print(i)

    min_dis = 10000

    centre_i = 0

    for c in range(centres.__len__()):

        temp_dis = 0

        for col in X_test.columns:

            temp_dis += abs(centres[c][col] - X_test.iloc[i][col])

        if temp_dis < min_dis:

            min_dis = temp_dis

            centre_i = c

    cluster.append(centre_i)

    test_satisfied_values.append(assosciated_satisfied_value[centre_i])



    

print(df_test.__len__(),df_proc1_test.__len__(),X_test.__len__())

    

answer_df = pd.DataFrame(df_test["custId"])[0:2113]

answer_df['Satisfied'] = test_satisfied_values



print(df_test.__len__(),df_proc1_test.__len__(),X_test.__len__(),answer_df.__len__())
answer_df.head(100)
answer_df.to_csv("submission.csv",index=False)