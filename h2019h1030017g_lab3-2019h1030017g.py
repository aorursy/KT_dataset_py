import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv(r"../input/eval-lab-3-f464/train.csv")

test = pd.read_csv(r"../input/eval-lab-3-f464/test.csv")



train.head()

test.head()
train.isnull().head(4)

missing_count = train.isnull().sum()

missing_count[missing_count > 0]
train.replace("No tv connection", np.nan, inplace = True)

train.head(5)

test.head()

test.replace("No tv connection", np.nan, inplace = True)

test.head(5)
train.isnull().head(5)
train.isnull().head(4)

missing_count = train.isnull().sum()

missing_count[missing_count > 0]
train_dtype_nunique = pd.concat([train.dtypes, train.nunique()],axis=1)

train_dtype_nunique.columns = ["dtype","unique"]

train_dtype_nunique
test.isnull().head(4)

missing_count = test.isnull().sum()

missing_count[missing_count > 0]
train.head()

train['Channel1'].value_counts()
train['Channel1'].replace(np.nan, "No", inplace=True)
train['Channel2'].value_counts()
train['Channel3'].value_counts()
train['Channel4'].value_counts()
train['Channel5'].value_counts()
train['Channel6'].value_counts()
train['Channel2'].replace(np.nan, "No", inplace=True)
train['Channel3'].replace(np.nan, "No", inplace=True)
train['Channel4'].replace(np.nan, "No", inplace=True)
train['Channel5'].replace(np.nan, "No", inplace=True)
train['Channel6'].replace(np.nan, "No", inplace=True)
train.isnull().head(4)

missing_count = train.isnull().sum()

missing_count[missing_count > 0]

train.head()
test['Channel1'].value_counts()
test['Channel2'].value_counts()
test['Channel3'].value_counts()
test['Channel4'].value_counts()
test['Channel5'].value_counts()
test['Channel6'].value_counts()
test['Channel1'].replace(np.nan, "No", inplace=True)
test['Channel2'].replace(np.nan, "No", inplace=True)
test['Channel3'].replace(np.nan, "No", inplace=True)
test['Channel4'].replace(np.nan, "No", inplace=True)
test['Channel5'].replace(np.nan, "No", inplace=True)
test['Channel6'].replace(np.nan, "No", inplace=True)
test.isnull().head(4)

missing_count = test.isnull().sum()

missing_count[missing_count > 0]
test.head(2)
train.replace("No internet", np.nan, inplace = True)

train.head()

test.replace("No internet", np.nan, inplace = True)

test.head(2)

train['HighSpeed'].value_counts()
train['HighSpeed'].replace(np.nan, "No", inplace=True)
test['HighSpeed'].value_counts()
test['HighSpeed'].replace(np.nan, "No", inplace=True)
train.head(2)
temp_code = {'Female':0,'Male':1}

train["gender"] = train["gender"].map(temp_code)
married_num = {'No':0,'Yes':1}

train["Married"] = train["Married"].map(married_num)
children_num = {'No':0,'Yes':1}

train["Children"] = train["Children"].map(children_num)

test.head(2)
TVConnection_num = {'No':0,'DTH':1, 'Cable':2}

train["TVConnection"] = train["TVConnection"].map(TVConnection_num)
Channel1_num = {'No':0,'Yes':1}

train["Channel1"] = train["Channel1"].map(Channel1_num)
Channel2_num = {'No':0,'Yes':1}

train["Channel2"] = train["Channel2"].map(Channel2_num)
Channel3_num = {'No':0,'Yes':1}

train["Channel3"] = train["Channel3"].map(Channel3_num)
Channel4_num = {'No':0,'Yes':1}

train["Channel4"] = train["Channel4"].map(Channel4_num)
Channel5_num = {'No':0,'Yes':1}

train["Channel5"] = train["Channel5"].map(Channel5_num)
Channel6_num = {'No':0,'Yes':1}

train["Channel6"] = train["Channel6"].map(Channel6_num)
train.head()
Internet_num = {'No':0,'Yes':1}

train["Internet"] = train["Internet"].map(Internet_num)
HighSpeed_num = {'No':0,'Yes':1}

train["HighSpeed"] = train["HighSpeed"].map(HighSpeed_num)
AddedServices_num = {'No':0,'Yes':1}

train["AddedServices"] = train["AddedServices"].map(AddedServices_num)
Subscription_num = {'Monthly':0, 'Annually':1, 'Biannually':2}

train["Subscription"] = train["Subscription"].map(Subscription_num)
train.head()
test.head()



train["TotalCharges"].replace(' ', np.nan, inplace=True)
PaymentMethod_num = {'Cash':0, 'Bank transfer':1, 'Net Banking':2, 'Credit card':3}

train["PaymentMethod"] = train["PaymentMethod"].map(PaymentMethod_num)

train.head()
train.dropna(subset=['TotalCharges'], inplace=True)
numerical_features = ["TotalCharges"]

train[numerical_features] = train[numerical_features].astype("float")

test.head()
temp_test = {'Female':0,'Male':1}

test["gender"] = test["gender"].map(temp_test)

married_test = {'No':0,'Yes':1}

test["Married"] = test["Married"].map(married_test)

children_test = {'No':0,'Yes':1}

test["Children"] = test["Children"].map(children_test)

TVConnection_test = {'No':0,'DTH':1, 'Cable':2}

test["TVConnection"] = test["TVConnection"].map(TVConnection_test)

Channel1_test = {'No':0,'Yes':1}

test["Channel1"] = test["Channel1"].map(Channel1_test)

Channel2_test = {'No':0,'Yes':1}

test["Channel2"] = test["Channel2"].map(Channel2_test)

Channel3_test = {'No':0,'Yes':1}

test["Channel3"] = test["Channel3"].map(Channel3_test)

Channel4_test = {'No':0,'Yes':1}

test["Channel4"] = test["Channel4"].map(Channel4_test)

Channel5_test = {'No':0,'Yes':1}

test["Channel5"] = test["Channel5"].map(Channel5_test)

Channel6_test = {'No':0,'Yes':1}

test["Channel6"] = test["Channel6"].map(Channel6_test)

Internet_test = {'No':0,'Yes':1}

test["Internet"] = test["Internet"].map(Internet_test)

HighSpeed_test = {'No':0,'Yes':1}

test["HighSpeed"] = test["HighSpeed"].map(HighSpeed_test)

AddedServices_test = {'No':0,'Yes':1}

test["AddedServices"] = test["AddedServices"].map(AddedServices_test)

Subscription_test = {'Monthly':0, 'Annually':1, 'Biannually':2}

test["Subscription"] = test["Subscription"].map(Subscription_test)

PaymentMethod_test = {'Cash':0, 'Bank transfer':1, 'Net Banking':2, 'Credit card':3}

test["PaymentMethod"] = test["PaymentMethod"].map(PaymentMethod_test)

test.head()
test['TotalCharges'].replace(' ', np.nan, inplace=True)

test.head()
test['TotalCharges'].value_counts().idxmax()

test["TotalCharges"].replace(np.nan, 20.2, inplace=True)

test.head()




numerical_features = ["TotalCharges"]

test[numerical_features] = test[numerical_features].astype("float")
test.head()
# standardizing the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_scaled = scaler.fit_transform(train)

test_scaled = scaler.fit_transform(test)



# statistics of scaled data

pd.DataFrame(train_scaled).describe()
test.head()
del test['gender']

del test['SeniorCitizen']

del test['Married']

del test['Children']

del test['Internet']

del test['HighSpeed']

del test['tenure']

del test['MonthlyCharges']



del train['gender']

del train['SeniorCitizen']

del train['Married']

del train['Children']

del train['Internet']

del train['HighSpeed']

del train['tenure']

del train['MonthlyCharges']

del train['Satisfied']





test.head()
train.head()

from sklearn.cluster import KMeans

# defining the kmeans function with initialization as k-means++

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter = 300)



# fitting the k means algorithm on scaled data

kmeans.fit(train)



#from sklearn.cluster import MiniBatchKMeans

#batch_size = 50

#mbk = MiniBatchKMeans(init='k-means++', n_clusters=2, batch_size=batch_size,

#                      n_init=10, max_no_improvement=10, verbose=0)

#mbk.fit(train)
# inertia on the fitted data

kmeans.inertia_
kmeans = KMeans(n_jobs = 1, n_clusters = 2, init='k-means++')

kmeans.fit(train)

pred = kmeans.predict(test)

#pred = mbk.predict(test)

pred


test = pd.read_csv(r"../input/eval-lab-3-f464/test.csv")



submission = pd.DataFrame({'custId': test['custId'], 'satisfied': pred })

submission.to_csv("submission19.csv", index=False)

