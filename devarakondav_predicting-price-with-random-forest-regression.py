%matplotlib inline
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(9,6)})
#Load data

data = pd.read_csv("../input/renfe.csv")

#Lets drop missing values for EDA

data = data.dropna(axis=0,how='any')
data['duration'] = (pd.to_datetime(data.end_date)-pd.to_datetime(data.start_date)).apply(lambda x: (x.seconds)/60)

#Some fares are too small and redundent combine them to for other

data.fare.replace(['Individual-Flexible','Mesa','Grupos Ida'],'Other',inplace=True)

data['route'] = data.origin+data.destination

# One hot encode routes and add to data frame

route_names = {}

i = 1

for route in data.route.unique().tolist():

    route_names["route_"+route] = "route"+str(i)

    i = i+1
#add month

data['month'] = pd.to_datetime(data.start_date).apply(lambda x: x.month)

data['day'] = pd.to_datetime(data.start_date).apply(lambda x: x.day)

data['dayname'] = pd.to_datetime(data.start_date).apply(lambda x: x.day_name())
data = data.rename(index=str,columns=route_names)

data.head()
# Distribution of prices and train duration

fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(15,5))

_ = sns.distplot(data.price,ax=axs[0])

_ = sns.distplot(data.duration,ax=axs[1])
# Lets check the count fo the categorical variables

fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(15,15))

_ = sns.countplot(x=data.route,ax=axs[0,0]).set_xticklabels(rotation=90,labels=data.route.unique())

_ = sns.countplot(x=data.train_class,ax=axs[0,1]).set_xticklabels(rotation=90,labels=data.train_class.unique())

_ = sns.countplot(x=data.train_type,ax=axs[1,0]).set_xticklabels(rotation=90,labels=data.train_type.unique())

_ = sns.countplot(x=data.fare,ax=axs[1,1]).set_xticklabels(rotation=90,labels=data.fare.unique())

fig.subplots_adjust(hspace=.9)
# Lets check how they might effect the price

fig,axs = plt.subplots(ncols=2,nrows=4,figsize=(15,15))

_ = sns.violinplot(x=data.route,y=data.price,ax=axs[0,0]).set_xticklabels(rotation=90,labels=data.route.unique())

_ = sns.violinplot(x=data.train_class,y=data.price,ax=axs[0,1]).set_xticklabels(rotation=90,labels=data.train_class.unique())

_ = sns.violinplot(x=data.train_type,y=data.price,ax=axs[1,0]).set_xticklabels(rotation=90,labels=data.train_type.unique())

_ = sns.violinplot(x=data.fare,y=data.price,ax=axs[1,1]).set_xticklabels(rotation=90,labels=data.fare.unique())

_ = sns.violinplot(x=data.month,y=data.price,ax=axs[2,0]).set_xticklabels(rotation=90,labels=data.month.unique())

_ = sns.violinplot(x=data.dayname,y=data.price,ax=axs[2,1]).set_xticklabels(rotation=90,labels=data.dayname.unique())

_ = sns.violinplot(x=data.day,y=data.price,ax=axs[3,0]).set_xticklabels(rotation=90,labels=data.day.unique())

fig.subplots_adjust(hspace=1.5)

fig.delaxes(axs[3,1])
# Lets check the data ranges

fig,axs = plt.subplots(ncols=2,nrows=4,figsize=(15,15))

_ = sns.boxplot(x=data.route,y=data.price,ax=axs[0,0]).set_xticklabels(rotation=90,labels=data.route.unique())

_ = sns.boxplot(x=data.train_class,y=data.price,ax=axs[0,1]).set_xticklabels(rotation=90,labels=data.train_class.unique())

_ = sns.boxplot(x=data.train_type,y=data.price,ax=axs[1,0]).set_xticklabels(rotation=90,labels=data.train_type.unique())

_ = sns.boxplot(x=data.fare,y=data.price,ax=axs[1,1]).set_xticklabels(rotation=90,labels=data.fare.unique())

_ = sns.boxplot(x=data.month,y=data.price,ax=axs[2,0]).set_xticklabels(rotation=90,labels=data.month.unique())

_ = sns.boxplot(x=data.dayname,y=data.price,ax=axs[2,1]).set_xticklabels(rotation=90,labels=data.dayname.unique())

_ = sns.boxplot(x=data.day,y=data.price,ax=axs[3,0]).set_xticklabels(rotation=90,labels=data.day.unique())

fig.subplots_adjust(hspace=1.5)

fig.delaxes(axs[3,1])
fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(15,15))

_ = sns.lineplot(x='dayname',y='price',data=data,ax=axs[0,0])

_ = sns.lineplot(x='month',y='price',data=data,ax=axs[0,1])

_ = sns.lineplot(x='day',y='price',data=data,ax=axs[1,0])

fig.delaxes(axs[1,1])
data_i = data[['route','month','dayname','train_class','train_type','fare','duration','day','price']]

data_i.price = (data_i.price - data_i.price.mean())/data_i.price.std()

data_i.duration = (data_i.duration - data_i.duration.mean())/data_i.duration.std()
data_i.head()
formula = "price~C(route)+C(month)+C(dayname)+C(train_class)+C(train_type)+C(fare)+duration+day"

#Due to multico had to drop train_type and day

formula = "price~C(route)+C(month)+C(dayname)+C(train_class)+C(fare)+duration"



# Unable to run due to issues with statsmodel

# import statsmodels.formula.api as smf

# reg = smf.ols(formula = formula,data=data_i).fit()
#reg.summary()
#_ = sns.distplot(reg.resid)
skdata  = data[['route','month','dayname','train_class','fare','price','duration']]

skdata.duration = (skdata.duration - skdata.duration.mean())/skdata.duration.std()

skdata = pd.get_dummies(skdata,columns=['route','month','dayname','train_class','fare'])
skdata.head()
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor





linreg = LinearRegression()

rfreg = RandomForestRegressor(n_estimators=10)

nn = MLPRegressor(hidden_layer_sizes=(100,),

                 activation='relu',

                 solver='sgd',

                 learning_rate='adaptive',

                 learning_rate_init=.001,

                 verbose=True)

y = skdata.price

x = skdata[skdata.keys()[1:]]



print("Linear Regression R-Squared: ",cross_val_score(linreg,cv=10,X=x,y=y))

print("RF Regression R-Squared: ",cross_val_score(rfreg,cv=10,X=x,y=y))
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split



y = skdata.price

x = skdata[skdata.keys()[1:]]



X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=.3)



rf = RandomForestRegressor(n_estimators=10,

                           n_jobs=4)

cv  = KFold(n_splits=10,shuffle=True)



train_r = []

test_r = []

test_mse = []

i = 0

for train_idx,test_idx in cv.split(X=X_train,y=y_train):

        i+=1

        print("CV: ",i)

        

        

        # Random forest regression

        rf.fit(X=X_train.iloc[train_idx,:],y=y_train[train_idx])

        train_r.append(rf.score(X_train.iloc[train_idx,:],y=y_train[train_idx]))

        preds = rf.predict(X=X_train.iloc[test_idx,:])

        test_r.append(r2_score(y_train.iloc[test_idx],preds))

        test_mse.append(mean_squared_error(y_train.iloc[test_idx],preds))
test_p = rf.predict(X_test)

restest = y_test - test_p

px = np.arange(0,len(test_p))



fig,axs = plt.subplots(ncols=2,nrows=1,figsize=(15,5))

ax = sns.scatterplot(px[:100],test_p[:100],label="Predicted",ax=axs[0])

ax = sns.scatterplot(px[:100],y_test[:100],label="Truth",ax=axs[0])

ax = sns.scatterplot(px[:100],restest[:100],label="Res",ax=axs[0])

ax.set(title="Comparing Some Predicted, Truth and the Residuals")

ax2 = sns.distplot(restest,label="Residual Distribution")

_ = ax2.set(title="Residual Distribtion")
from scipy.stats import normaltest

from sklearn.metrics import explained_variance_score

print("PREDICTED MEAN SQUARED ERROR: ",mean_squared_error(y_test,test_p))

print("R2: ",r2_score(y_test,test_p))