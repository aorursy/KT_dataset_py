import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv('../input/bits-f464-l1/train.csv')
uniq = df.nunique().sort_values()

cols_with_1 = uniq[uniq == 1].index

cols_with_1
df.drop(cols_with_1,axis=1,inplace=True)

missing_count=df.isnull().sum()

missing_count[missing_count>0]



drop_columns=['time']



df=df.drop(drop_columns,axis=1)

label=df['label']







# Compute the correlation matrix

corr = df[df.columns[1:]].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(36, 27))



# Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()

df_x=df.drop(['label'],axis=1)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val = train_test_split(df_x,label,test_size=0.25,random_state=42)
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor



reg_rf = RandomForestRegressor(n_estimators = 150,max_depth = 50)

reg_rf.fit(X_train,y_train)
y_pred_rf = reg_rf.predict(X_val)



mse_rf = mean_squared_error(y_pred_rf,y_val)



print("Mean Squared Error of RandomForestRegressor: {}".format(mse_rf))
df_test=pd.read_csv('../input/bits-f464-l1/train.csv')

df_test.head()
x_test=df_test[df_x.columns]
x_test.info()
y_pred_test=reg_rf.predict(x_test)

df_sub = pd.DataFrame({'id':x_test['id'],'label':y_pred_test})

df_sub.to_csv('sub.csv',index=False)
df_sub.info()