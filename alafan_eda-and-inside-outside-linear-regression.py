import numpy as np 

import pandas as pd 



from datetime import datetime

from sklearn.linear_model import LinearRegression



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# import Dataset

df = pd.read_csv('../input/temperature-readings-iot-devices/IOT-temp.csv', parse_dates=['noted_date'])

df.head()
df['room_id/id'].value_counts()
# dropping columns

cols_drop = ['id', 'room_id/id']

df = df.drop(cols_drop, axis=1)
print("the dataset has shape = {}".format(df.shape))
df.describe()

# duplicate rows have been dropped
# building new features for time stamp.

df['measure_hour'] = df.noted_date.apply(lambda x:datetime.strftime(x,'%Y-%m-%d %H:00:00'))
data = df.groupby(['measure_hour','out/in']).temp.mean().reset_index()

data = data.pivot(index = 'measure_hour',columns = 'out/in', values = 'temp').reset_index().dropna()

data.head()
fig, ax = plt.subplots(figsize = (6,4))

g = sns.distplot(data.In, label = 'In')

g = sns.distplot(data.Out, label = 'Out')

plt.legend()

g.set_xlabel('Temperature')
sns.scatterplot(x =data.Out, y = data.In)

sns.scatterplot(x =data[data.Out>35].Out, y = data[data.Out>35].In)
sns.scatterplot(x =data[data.Out<=35].Out, y = data[data.Out<=35].In)
linear = data[(pd.notna(data.Out))&(pd.notna(data.In)) & (data.Out<35)]
#removing oultliers

linear = linear.drop(index = linear[((linear.Out>32)&(linear.In<30)|(linear.Out<25))].index, axis = 0)

sns.scatterplot(x = linear.Out, y = linear.In)
#Linear regression building

model = LinearRegression()

model.fit(linear[['Out']],linear.In)
k,b = model.coef_[0],model.intercept_

print(k,b)

sns.scatterplot(x = linear.Out, y = linear.In)

reg_line = np.linspace(25,35,100)

plt.plot(reg_line, reg_line*k + b)