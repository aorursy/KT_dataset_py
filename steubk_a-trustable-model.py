import numpy as np  

import pandas as pd  



from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



pd.options.mode.chained_assignment = None  





def rmse(y_true, y_pred) :

    return np.sqrt( mean_squared_error(y_true, y_pred ) ) 



class AverageModel():

        

    def fit (self,X):

        self.df_mean = X[['X0','y']].groupby(['X0']).mean().reset_index()

        self.df_mean.columns = ['X0','avg']

        self.y_mean = X['y'].mean()

        return self



    def predict(self, X):

        X = X.copy().merge(self.df_mean,on=['X0'],how="left")

        X['avg'].fillna(self.y_mean,inplace=True)

        return X['avg']

    

    

df_train = pd.read_csv('../input/train.csv', usecols=['ID','X0','y'])

df_test = pd.read_csv('../input/test.csv', usecols=['ID','X0'])



model=AverageModel()





print("Train with outlier:")

model.fit(df_train)

y_pred = model.predict(df_train)

r2_=r2_score(df_train['y'],y_pred)

rmse_ = rmse(df_train['y'],y_pred)

print("r2:", r2_  )

print("rmse:", rmse_ )

print("")

print("Train without outlier:")

df_train=df_train [df_train['y']<250]

model.fit(df_train)

y_pred = model.predict(df_train)

r2_=r2_score(df_train['y'],y_pred)

rmse_ = rmse(df_train['y'],y_pred)

print("r2:", r2_  )

print("rmse:", rmse_ )



y_test=model.predict(df_test)

df_test['y']=y_test

df_test[['ID','y']].to_csv("avg_sub.csv",index=False)

import matplotlib.pyplot as plt



%matplotlib inline



plt.figure(figsize=(7,5))

ax = plt.subplot(111)

plt.scatter(df_train['y'], y_pred )

plt.xlabel('y_train', fontsize=12)

plt.ylabel('y_pred', fontsize=12)

ax.set_aspect('equal' )

[x1,x2] = ax.get_xlim()

plt.plot( [x1+10,x2-10], [x1+10,x2-10], ls="--" )

plt.show()
