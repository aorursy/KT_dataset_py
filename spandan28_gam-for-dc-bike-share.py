from pygam import LinearGAM
import pandas as pd
import torch as t
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

day_data = pd.read_csv("../input/bike_sharing_daily.csv")
feature =['cnt','season','windspeed','atemp','hum','yr','mnth']
Daily_bike_share= pd.DataFrame()
Daily_bike_share = day_data[feature]
Daily_bike_share.head(10)
correlation = Daily_bike_share.corr()
fig, axes  = plt.subplots(figsize=(10,8))
sns.heatmap(correlation,annot=True,vmin=-1,vmax=1,center=0,ax=axes)
Train_data, test_data = train_test_split(Daily_bike_share, test_size = 0.2)

X_train = Train_data.drop(columns = ["cnt"])
y_train = Train_data[['cnt']]
########################################
X_test = test_data.drop(columns = ["cnt"])
y_test = test_data[['cnt']]



gam = LinearGAM(n_splines=10).gridsearch(X_train, y_train)

XX = gam.generate_X_grid()
fig, axs = plt.subplots(1,6, figsize=(20,4))
titles = feature[1:]

for i, ax in enumerate(axs):
    pdep, confi = gam.partial_dependence(XX, feature=i, width=.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], *confi, c='r', ls='--')
    ax.set_title(titles[i])
    

gam.summary()
plt.plot(gam.predict(XX), 'r--')
plt.plot(gam.prediction_intervals(XX, width=.95), color='b', ls='--')

#plt.plot(y_train, facecolor='gray', edgecolors='none')
plt.title('95% prediction interval')
plt.plot(y_test, gam.predict(X_test),"*")
plt.xlabel("Predicted Value")
plt.ylabel("Actual value")
from sklearn.metrics import mean_absolute_error,r2_score
print("Mean absolute error >> " + str(mean_absolute_error(y_test, gam.predict(X_test))))
print("Coefficient of determination >> " + str(r2_score(y_test, gam.predict(X_test))))
