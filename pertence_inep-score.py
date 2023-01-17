import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, linear_model, metrics
#read structured data
df_2014 = pd.read_csv('../input/2014.csv')
df_2015 = pd.read_csv('../input/2015.csv')
df_2016 = pd.read_csv('../input/2016.csv')
#select data
df_sel_2014 = df_2014[['nt_ger', 'enem_nt_cn', 'enem_nt_ch', 'enem_nt_lc', 'enem_nt_mt']]
df_sel_2015 = df_2015[['nt_ger', 'enem_nt_cn', 'enem_nt_ch', 'enem_nt_lc', 'enem_nt_mt']]
df_sel_2016 = df_2016[['nt_ger', 'enem_nt_cn', 'enem_nt_ch', 'enem_nt_lc', 'enem_nt_mt']]
df_con = pd.concat([df_sel_2014, df_sel_2015, df_sel_2016])
df_na = df_con.dropna()
df = df_na[(df_na.T != 0).all()]
#df.shape
#df_con.plot.hexbin('enem_nt_lc', 'nt_ger')
#df_con.plot.box()
#print(df_con.corr())
#plt.matshow(df_con.corr())
#Pre-processing
X = df[['enem_nt_ch', 'enem_nt_cn', 'enem_nt_lc', 'enem_nt_mt']]
y = df[['nt_ger']]
X = preprocessing.scale(X)
#Split the data for training and testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#Training 
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
#Prediction
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
#Model testing
print('Coefficients:', regr.coef_)
print("Mean squared error: %.4f" % metrics.mean_squared_error(y_test, y_pred))
print('Variance score: %.4f' % metrics.r2_score(y_test, y_pred))
