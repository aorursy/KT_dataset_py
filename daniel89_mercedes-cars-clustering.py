import numpy as np

import pandas as pd

from sklearn.cluster import KMeans

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import cross_val_score,cross_val_predict

train = pd.read_csv('../input/train.csv')

X_train = train.drop(['y'],axis=1)

y_train = train['y']

X_test = pd.read_csv('../input/test.csv')



#

#   Here we drop columns with zero std

#



zero_std = X_train.std()[X_train.std()==0].index

X_train = X_train.drop(zero_std,axis=1)

X_test = X_test.drop(zero_std,axis=1)
sns.distplot(y_train[y_train<170],bins=100,kde=False)
class cluster_target_encoder:

    def make_encoding(self,df):

        self.encoding = df.groupby('X')['y'].mean()

    def fit(self,X,y):

        df = pd.DataFrame(columns=['X','y'],index=X.index)

        df['X'] = X

        df['y'] = y

        self.make_encoding(df)

        clust = KMeans(4,random_state=0)

        labels = clust.fit_predict(self.encoding[df['X'].values].values.reshape(-1,1))

        df['labels'] = labels

        self.clust_encoding = df.groupby('X')['labels'].median()

    def transform(self,X):

        res = X.map(self.clust_encoding).astype(float)

        return res

    def fit_transform(self,X,y):

        self.fit(X,y)

        return self.transform(X)
enc1 = cluster_target_encoder()

labels_train = enc1.fit_transform(X_train['X0'],train['y'])

labels_test = enc1.transform(X_test['X0'])

%pylab inline

plt.figure(figsize(10,5))

plt.hist(y_train.values[labels_train==0],bins=70,label='cluster 0')

plt.hist(y_train.values[labels_train==1],bins=100,label='cluster 1')

plt.hist(y_train.values[labels_train==2],bins=70,label='cluster 2')

plt.hist(y_train.values[labels_train==3],bins=70,label='cluster 3')

plt.legend()

plt.title('Train targets distribution for all clusters')

plt.xlim((60,170))

plt.show()
labels_test[np.isnan(labels_test)].shape
cross_val_score(

    X = X_train.select_dtypes(include=[np.number]),

    y = labels_train,

    estimator = xgb.XGBClassifier(),

    cv = 5,

    scoring = 'accuracy')
est = xgb.XGBClassifier()

est.fit(X_train.select_dtypes(include=[np.number]),labels_train)

labels_test[np.isnan(labels_test)] = est.predict(

    X_test.select_dtypes(include=[np.number]))[np.isnan(labels_test)]

np.isnan(labels_test).any()
y_pred = cross_val_predict(

    X = X_train.select_dtypes(include=[np.number]),

    y = y_train,

    estimator = xgb.XGBRegressor(),

    cv = 5)

plt.figure(figsize(10,5))

plt.hist(y_pred[labels_train==0],bins=70,label='cluster 0')

plt.hist(y_pred[labels_train==1],bins=100,label='cluster 1')

plt.hist(y_pred[labels_train==2],bins=70,label='cluster 2')

plt.hist(y_pred[labels_train==3],bins=70,label='cluster 3')

plt.legend()

plt.title('Cross_val_predict distribution for all clusters')

plt.show()