#try other clustering

#try one hod encoding

#try adding payment method
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.utils import resample

from sklearn.metrics import accuracy_score

%matplotlib inline



pd.set_option('display.max_columns', 100)
#getting data

testdf = pd.read_csv('"../input/eval-lab-3-f464/test.csv"')

df = pd.read_csv('"../input/eval-lab-3-f464/train.csv"')

[df.shape, testdf.shape]
df.head();
#try other clustering

#try one hot encoding

#try adding payment method

from sklearn.datasets import make_classification

from imblearn.under_sampling import ClusterCentroids

xi, yi = make_classification(n_classes=2, class_sep=2,

  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,

 n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

cc = ClusterCentroids(random_state=42)

X_res, y_res = cc.fit_resample(xi, yi)
df['gender'] = df['gender'].apply(lambda x: 0 if x=='Male' else 1)

df['Married'] = df['Married'].apply(lambda x: 0 if x=='No' else 1)

df['Children'] = df['Children'].apply(lambda x: 0 if x=='No' else 1)

df['TVConnection'] = df['TVConnection'].apply(lambda x: 1 if x=='Cable' else 2 if x=='DTH' else 3)

df['Channel1'] = df['Channel1'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

df['Channel2'] = df['Channel2'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

df['Channel3'] = df['Channel3'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

df['Channel4'] = df['Channel4'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

df['Channel5'] = df['Channel5'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

df['Channel6'] = df['Channel6'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

df['Internet'] = df['Internet'].apply(lambda x: 0 if x=='No' else 1)

df['HighSpeed'] = df['HighSpeed'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

df['Subscription'] = df['Subscription'].apply(lambda x: 1 if x=='Monthly' else 2 if x=='Biannually' else 3) 

df['AddedServices'] = df['AddedServices'].apply(lambda x: 0 if x=='No' else 1)

df['NetB'] = df['PaymentMethod'].apply(lambda x: 1 if x=='Net Banking' else 0) 

#

testdf['gender'] = testdf['gender'].apply(lambda x: 0 if x=='Male' else 1)

testdf['Married'] = testdf['Married'].apply(lambda x: 0 if x=='No' else 1)

testdf['Children'] = testdf['Children'].apply(lambda x: 0 if x=='No' else 1)

testdf['TVConnection'] = testdf['TVConnection'].apply(lambda x: 1 if x=='Cable' else 2 if x=='DTH' else 3)

testdf['Channel1'] = testdf['Channel1'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

testdf['Channel2'] = testdf['Channel2'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

testdf['Channel3'] = testdf['Channel3'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

testdf['Channel4'] = testdf['Channel4'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

testdf['Channel5'] = testdf['Channel5'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

testdf['Channel6'] = testdf['Channel6'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

testdf['Internet'] = testdf['Internet'].apply(lambda x: 0 if x=='No' else 1)

testdf['HighSpeed'] = testdf['HighSpeed'].apply(lambda x: -1 if x=='No' else 1 if x=='Yes' else 0)

testdf['Subscription'] = testdf['Subscription'].apply(lambda x: 1 if x=='Monthly' else 2 if x=='Biannually' else 3) 

testdf['AddedServices'] = testdf['AddedServices'].apply(lambda x: 0 if x=='No' else 1)

testdf['NetB'] = testdf['PaymentMethod'].apply(lambda x: 1 if x=='Net Banking' else 0) 

df.head();
encoded = pd.get_dummies(df['PaymentMethod'])

df = pd.concat([df,encoded],axis=1)

encodedtest = pd.get_dummies(testdf['PaymentMethod'])

testdf = pd.concat([testdf,encodedtest],axis=1)

# x = df.copy()

# y = df["rating"].copy()

# x=x.drop(['rating','type'],axis=1);



# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# X_scaled = scaler.fit_transform(x)



# x = np.concatenate([X_scaled,encoded.values],axis=1)

df.head()
# Compute the correlation matrix

corr = df.corr()

corr['Satisfied']
# from mpl_toolkits import mplot3d

# import matplotlib.pyplot as plt

# fig = plt.figure()

# ax = plt.axes(projection='3d')

# ax.scatter(df['Bank transfer'], df['Cash'],df['Channel5'], c=df['Satisfied'], cmap='viridis', linewidth=0.5);
#Constructing 4 sets of data, (x,y)-> (x_test,~); (x_train,y_train);(x_cv,y_cv)

#columns = ['SeniorCitizen','tenure','Married','AddedServices','MonthlyCharges','TVConnection','Subscription','Channel6','Channel5','Channel4','Channel3']

#columns = ['custId','Channel6','Subscription','tenure','MonthlyCharges']

columns = ['Married','Channel6','TVConnection','tenure','MonthlyCharges']

x = df[columns].copy()

y = df["Satisfied"].copy()

cid =df["custId"].copy()

#x = x.drop(['id'],axis=1);

#no need to scale in xgboost

#test = testdf.drop(['id'],axis=1);

test = testdf[columns].copy()

testcid = testdf['custId'].copy()

test.shape
from sklearn.preprocessing import StandardScaler,RobustScaler

scaler = RobustScaler()

x = scaler.fit_transform(x)

test=scaler.fit_transform(test)

from sklearn.model_selection import train_test_split

x_train,x_cv,y_train,y_cv,cid_train,cid_cv = train_test_split(x,y,cid,test_size=0.33)

[x_train.shape, y_train.shape, x.shape, y.shape, x_cv.shape, y_cv.shape]
from sklearn import cluster 

n=10

#model = cluster.KMeans(n_clusters=n)

model = cluster.Birch(n_clusters=n)

model.fit(x_train)



# predict the target on the train dataset

predict_train = model.fit_predict(x_train)



# predict the target on the test dataset

y_predict_cv = model.fit_predict(x_cv)
map_class= np.ndarray((n,),int);

for i in range(n):

    class_sat = np.average(y_train[predict_train==i])

    map_class[i]= 0 if class_sat<0.6 else 1   
map_class
predict_train= map_class[predict_train]

y_predict_cv=map_class[y_predict_cv]

predict_train

test.shape
# Accuray Score on train dataset

from sklearn.metrics import roc_auc_score

accuracy_train = roc_auc_score(y_train,predict_train)

print('\naccuracy_score on train dataset : ', accuracy_train)



# Accuracy Score on test dataset

accuracy_test = roc_auc_score(y_cv,y_predict_cv)

print('\naccuracy_score on test dataset : ', accuracy_test)
plt.scatter(y_predict_cv,y_cv,alpha = 0.1)

test.shape
# fit the model with the whole data

model.fit(x)



# predict the target on the train dataset

y_test = model.fit_predict(x)



accuracy_score# predict the target on the test dataset

y_pred = model.fit_predict(test)
map_classf= np.ndarray((n,),int);

for i in range(n):

    class_sat = np.average(y[y_test==i]).round()

    map_classf[i]=0 if class_sat <0.7 else 1

map_classf
y_pred.shape, y_test.shape, test.shape
y_test= map_classf[y_test]

y_pred=map_classf[y_pred]

np.average(y_pred), y_pred.size, y_test.size
# Accuray Score on train dataset

accuracy_train = roc_auc_score(y_test,y)

print('\naccuracy_score on train dataset : ', accuracy_train)
#storing y_test in reuired format

ID = testdf['custId']

#y_test= y_test.reshape(len(y_test),1)

ans = pd.concat([ID,pd.DataFrame(y_pred)],axis=1)

ans[0].value_counts()
#check the things

ans.astype('int64')

ans.dtypes
ans.info

#store in csv

ans.to_csv("submit7.csv",index=None,header=["custId","Satisfied"])
np.average(y_pred)