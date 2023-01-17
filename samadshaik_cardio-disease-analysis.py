import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('/kaggle/input/cardio_train2.csv')

df.head()
df.shape
df.isnull().sum()
import math

df['age']=round(df['age']//365)
# list of numerical variables

numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']



print('Number of numerical variables: ', len(numerical_features))



# visualise the numerical variables

df[numerical_features].head()
discrete_feature=[feature for feature in numerical_features if len(df[feature].unique())<100]

print("Discrete Variables Count: {}".format(len(discrete_feature)))

df[discrete_feature]
discrete_feature

#count_plot
#continuos_features='height','weight','ai_hi','ap_lo'

#histogram
df.weight.unique

fig = df.weight.hist(bins=50)

fig.set_title('weights')

fig.set_xlabel('weight')

fig.set_ylabel('Number of weights')
fig = df.height.hist(bins=50)

fig.set_title('heights')

fig.set_xlabel('height')

fig.set_ylabel('Number of heights')
fig=df.ap_lo.hist(bins=10)

fig.set_title('ap_lo')

fig.set_xlabel('ap_lo')

fig.set_ylabel('Number of ap_lo')
df[discrete_feature]


catagorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']



print('Number of catagorical_features: ', len(catagorical_features))



#feature engineering
feature_scale=[feature for feature in df.columns if feature not in ['id','cardio']]



from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(df[feature_scale])

data = pd.concat([df[['id', 'cardio']].reset_index(drop=True),

                    pd.DataFrame(scaler.transform(df[feature_scale]), columns=feature_scale)], axis=1)

data
dfc=data.copy()
#outliers
import seaborn as sns

sns.distplot(dfc.weight)
fig = dfc.weight.hist(bins=50)

fig.set_title('weight Distribution')

fig.set_xlabel('weight')

fig.set_ylabel('Number of people')
fig = df.boxplot(column='weight')

fig.set_title('')

fig.set_xlabel('')

fig.set_ylabel('weight')
import seaborn as sns

sns.distplot(dfc.height)
import seaborn as sns

sns.distplot(dfc.ap_hi)
dataset=sorted(dfc['weight'])

quantile1, quantile3= np.percentile(dataset,[25,75])

print(quantile1,quantile3)
## Find the IQR



iqr=quantile3-quantile1

print(iqr)


lower_bound_val = quantile1 -(1.5 * iqr) 

upper_bound_val = quantile3 +(1.5 * iqr)

print(lower_bound_val,upper_bound_val)