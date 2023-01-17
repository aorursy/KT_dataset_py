import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

mpl.style.use('ggplot')
data=pd.read_excel('../input/analyst/Data Analyst Assignment (1).xlsx')

data.head()
data.isna().sum()
data.describe()
for i in range(1,7):

    print(data[data.columns[i]].unique())


fig,a =  plt.subplots(2,2,figsize=(15,15))



a[0][0].bar(data.groupby('campaign_platform').sum().index,data.groupby('campaign_platform').sum()['spends'])

a[0][0].set_title('Spends')

a[0][1].bar(data.groupby('campaign_platform').sum().index,data.groupby('campaign_platform').sum()['impressions'])

a[0][1].set_title('Impressions')

a[1][0].bar(data.groupby('campaign_platform').sum().index,data.groupby('campaign_platform').sum()['clicks'])

a[1][0].set_title('Clicks')

a[1][1].bar(data.groupby('campaign_platform').sum().index,data.groupby('campaign_platform').sum()['link_clicks'])

a[1][1].set_title('link_clicks')

    



    


fig,a =  plt.subplots(2,2,figsize=(15,15))



a[0][0].bar(data.groupby('campaign_type').sum().index,data.groupby('campaign_type').sum()['spends'])

a[0][0].set_title('Spends')

a[0][1].bar(data.groupby('campaign_type').sum().index,data.groupby('campaign_type').sum()['impressions'])

a[0][1].set_title('Impressions')

a[1][0].bar(data.groupby('campaign_type').sum().index,data.groupby('campaign_type').sum()['clicks'])

a[1][0].set_title('Clicks')

a[1][1].bar(data.groupby('campaign_type').sum().index,data.groupby('campaign_type').sum()['link_clicks'])

a[1][1].set_title('link_clicks')

    
corrMat=data.corr()
corrMat
sns.heatmap(corrMat)

plt.show()
data.head()
data['age'].unique()
age_low=[]

age_high=[]

split=[]

for age in data['age']:

    if age=='Undetermined':

        age_low.append('0')

        age_high.append('0')

    elif age=='65 or more':

        age_low.append('65')

        age_high.append('100')

    else:    

        split=age.split('-')

        age_low.append(split[0])

        age_high.append(split[1])

    

        
data['age_low']=age_low

data['age_high']=age_high
data.head()
data['total clicks']=data['clicks']+data['link_clicks']

data.head()
data[['age_low','age_high']]=data[['age_low','age_high']].astype(float)
data.dtypes
data['age_low']=data['age_low'].replace(0,data['age_low'].mean())

data['age_high']=data['age_high'].replace(0,data['age_high'].mean())
date_df=data.groupby('Date')

date_df.head()
data_date= data.groupby('Date').sum()['spends']

data_date
data_date.plot(kind='line')
data_refined=data[['campaign_platform','campaign_type','communication_medium','subchannel','age_low','age_high','spends','impressions','total clicks']]

data_final=pd.get_dummies(data_refined)

data_final.head()
train=data_final.drop(['total clicks'], axis=1)

train.head()
target=data_final['total clicks']
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

train = scaler.fit_transform(train)
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp=imp.fit(train)

train=imp.transform(train)
train.shape
target=target.values

target=target.reshape(-1,1)

imp=SimpleImputer()

#imp=imp.fit(y_train)

target=imp.fit_transform(target)
from sklearn.model_selection import KFold

from sklearn import linear_model

from sklearn.metrics import make_scorer

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import svm

from sklearn.metrics import r2_score

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV
def lets_try(train,labels):

    results={}

    def test_model(clf):

        

        cv = KFold(n_splits=5,shuffle=True,random_state=45)

        r2 = make_scorer(r2_score)

        r2_val_score = cross_val_score(clf, train, labels, cv=cv,scoring=r2)

        scores=[r2_val_score.mean()]

        return scores



    clf = linear_model.LinearRegression()

    results["Linear"]=test_model(clf)

    

    clf = linear_model.Ridge()

    results["Ridge"]=test_model(clf)

    

    clf = linear_model.BayesianRidge()

    results["Bayesian Ridge"]=test_model(clf)

    

    clf = linear_model.HuberRegressor()

    results["Hubber"]=test_model(clf)

    

    clf = linear_model.Lasso(alpha=1e-4)

    results["Lasso"]=test_model(clf)

    

    clf = BaggingRegressor()

    results["Bagging"]=test_model(clf)

    

    clf = RandomForestRegressor()

    results["RandomForest"]=test_model(clf)

    

    clf = AdaBoostRegressor()

    results["AdaBoost"]=test_model(clf)

    

    clf = svm.SVR()

    results["SVM RBF"]=test_model(clf)

    

    clf = svm.SVR(kernel="linear")

    results["SVM Linear"]=test_model(clf)

    

    results = pd.DataFrame.from_dict(results,orient='index')

    results.columns=["R Square Score"] 

    

    results.plot(kind="bar",title="Model Scores")

    axes = plt.gca()

    axes.set_ylim([0.5,1])

    return results



lets_try(train,target.ravel())