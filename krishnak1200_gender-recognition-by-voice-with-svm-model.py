import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
#import category_encoders as ce #encoding
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA #dim red
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 



%matplotlib inline
GenReg_ds = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-3/master/Projects/gender_recognition_by_voice.csv')
GenReg_ds.head(10)
GenReg_ds.shape
GenReg_ds.info()
GenReg_ds.describe()
GenReg_ds.isnull().sum()
GenReg_ds.select_dtypes(include=['object']).head()
print("Total number of labels : {} ".format(GenReg_ds.shape[0]))
print("Total number of males : {}".format(GenReg_ds[GenReg_ds.label=='male'].shape[0]))
print("Total number of females : {}".format(GenReg_ds[GenReg_ds.label=='female'].shape[0]))
GenReg_ds.corr()
sb.heatmap( GenReg_ds.corr());
GenReg_ds.drop(['centroid'], axis=1, inplace=True)
GenReg_ds.drop(['dfrange'], axis=1, inplace=True)
GenReg_ds.columns
GenReg_ds['label'].value_counts().plot(kind='bar',figsize = (12,5),fontsize = 14,colormap='Dark2', yticks=np.arange(0, 19, 2))
plt.xlabel('Gender')
plt.ylabel('No. of persons')
Male_df = GenReg_ds.loc[GenReg_ds.label == "male"]
Female_df = GenReg_ds.loc[GenReg_ds.label == "female"]
print(Male_df.shape)
print(Female_df.shape)
Male_df['meanfreq'].plot(kind='line', figsize=(12,5), color='blue', fontsize=13, linestyle='-.')
plt.ylabel('Meanfreq')
plt.title('Mean Frequency for Male persons')
Female_df['meanfreq'].plot(kind='line', figsize=(12,5), color='blue', fontsize=13, linestyle='-.')
plt.ylabel('Meanfreq')
plt.title('Mean Frequency for Female persons')
stdcol = GenReg_ds['modindx'].std()==0     
stdcol
X = GenReg_ds.loc[:,GenReg_ds.columns != 'label']
X.head()
drop_cols=[]
for cols in X.columns:
    if X[cols].std()==0:
        drop_cols.append(cols)
print("Number of constant columns to be dropped: ", len(drop_cols))
print(drop_cols)
X.drop(drop_cols,axis=1, inplace = True)
Y = GenReg_ds.loc[:,GenReg_ds.columns == 'label']
Y.head()
gender_encoder = LabelEncoder()
Y = gender_encoder.fit_transform(Y)
Y
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 1)
X_train
from sklearn.svm import SVC
from sklearn import metrics
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test,Y_pred))
svc = SVC(kernel='linear')
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test,Y_pred))
svc=SVC(kernel='rbf')
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(Y_test,Y_pred))
svc=SVC(kernel='poly')
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
print('Accuracy Score : ')
print(metrics.accuracy_score(Y_test,Y_pred))
Y
from sklearn.model_selection import cross_val_score
svc=SVC(kernel='linear')
scores = cross_val_score(svc, X, Y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)
print(scores.mean())
from sklearn.model_selection import cross_val_score
svc=SVC(kernel='rbf')
scores = cross_val_score(svc, X, Y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)
print(scores.mean())
from sklearn.model_selection import cross_val_score
svc=SVC(kernel='poly')
scores = cross_val_score(svc, X, Y, cv=10, scoring='accuracy') #cv is cross validation
print(scores)
print(scores.mean())
