import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler



sns.set() # for plot styling
df = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')
df.head(40)
df.corr()
# df['TotalCharges']=df['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(float)

# df['TotalCharges'] = df["TotalCharges"].astype('float')

# df['MonthlyCharges'] = df["MonthlyCharges"].astype('float')



df['gender']=df['gender'].astype('category').cat.codes

df['Married']=df['Married'].astype('category').cat.codes

df['Children']=df['Children'].astype('category').cat.codes

df['TVConnection']=df['TVConnection'].astype('category').cat.codes

df['Channel1']=df['Channel1'].astype('category').cat.codes

df['Channel2']=df['Channel2'].astype('category').cat.codes

df['Channel3']=df['Channel3'].astype('category').cat.codes

df['Channel4']=df['Channel4'].astype('category').cat.codes

df['Channel5']=df['Channel5'].astype('category').cat.codes

df['Channel6']=df['Channel6'].astype('category').cat.codes

df['Internet']=df['Internet'].astype('category').cat.codes

df['HighSpeed']=df['HighSpeed'].astype('category').cat.codes

df['AddedServices']=df['AddedServices'].astype('category').cat.codes

df['Subscription']=df['Subscription'].astype('category').cat.codes

df['PaymentMethod']=df['PaymentMethod'].astype('category').cat.codes

df['TotalCharges']=df['TotalCharges'].replace(r'^\s+$', np.nan, regex=True).astype(np.float64)

df = df[np.isfinite(df['TotalCharges'])]

df.isnull().any()
X = df[['SeniorCitizen','Married','Children','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','TVConnection','AddedServices','Subscription','PaymentMethod','tenure','MonthlyCharges']].copy()

y = df[['Satisfied']].copy()
X = np.array(X)

y = np.array(y)

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split



from sklearn import metrics

from sklearn.metrics import accuracy_score



X_train,X_val,y_train,y_val = train_test_split(X_scaled,y,test_size=0.10,random_state=42)
from sklearn.cluster import KMeans,DBSCAN

# from sklearn.cluster import Birch

# brc = Birch(branching_factor=50, n_clusters=None, threshold=1.5)

# kmeans = DBSCAN(eps=0.3, min_samples=10).fit(X)

kmeans = KMeans(n_clusters=2,n_init = 10, max_iter=200, algorithm = 'auto')

kmeans.fit(X_train,y_train)

y_pred=kmeans.predict(X_val)

y_pred=pd.DataFrame(y_pred)
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred,y_val)

print(accuracy)
predict = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

predict['TotalCharges']=predict['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(float)

predict['TotalCharges'] = predict["TotalCharges"].astype('float')

predict['MonthlyCharges'] = predict["MonthlyCharges"].astype('float')

predict.head()
predict['gender']=predict['gender'].astype('category').cat.codes

predict['Married']=predict['Married'].astype('category').cat.codes

predict['Children']=predict['Children'].astype('category').cat.codes

predict['TVConnection']=predict['TVConnection'].astype('category').cat.codes

predict['Channel1']=predict['Channel1'].astype('category').cat.codes

predict['Channel2']=predict['Channel2'].astype('category').cat.codes

predict['Channel3']=predict['Channel3'].astype('category').cat.codes

predict['Channel4']=predict['Channel4'].astype('category').cat.codes

predict['Channel5']=predict['Channel5'].astype('category').cat.codes

predict['Channel6']=predict['Channel6'].astype('category').cat.codes

predict['Internet']=predict['Internet'].astype('category').cat.codes

predict['HighSpeed']=predict['HighSpeed'].astype('category').cat.codes

predict['AddedServices']=predict['AddedServices'].astype('category').cat.codes

predict['Subscription']=predict['Subscription'].astype('category').cat.codes

predict['PaymentMethod']=predict['PaymentMethod'].astype('category').cat.codes

predict['TotalCharges']=predict['TotalCharges'].replace(r'^\s+$', 0, regex=True).astype(np.float64)

predict['TotalCharges']=predict['TotalCharges'].replace(r'^\s+$', np.nan, regex=True).astype(np.float64)

predict = predict[np.isfinite(predict['TotalCharges'])]

X_test = predict[['SeniorCitizen','Married','Children','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','TVConnection','AddedServices','Subscription','PaymentMethod','tenure','MonthlyCharges']].copy()

X_test.head()
X_test = np.array(X_test)

X_scaled_test = scaler.fit_transform(X_test)

# X_scaled_test = X_test
y_pred_test=kmeans.predict(X_scaled_test)

y_pred_test=pd.DataFrame(y_pred_test)
predict['Satisfied'] = y_pred_test
ans = predict[["custId","Satisfied"]].copy()
ans.to_csv('ans.csv',index=False,encoding ='utf-8' )
ans.head()