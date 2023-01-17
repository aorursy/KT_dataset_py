import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
df.info()
df.describe()
df.isnull().any()
df['total'] = (df['math score']+df['reading score']+df['writing score'])/3
df["gender"] = df["gender"].astype('category')

df["race/ethnicity"] = df["race/ethnicity"].astype('category')

df["parental level of education"] = df["parental level of education"].astype('category')

df["lunch"] = df["lunch"].astype('category')

df["test preparation course"] = df["test preparation course"].astype('category')
df["gender"] = df["gender"].cat.codes

df["race/ethnicity"] = df["race/ethnicity"].cat.codes

df["parental level of education"] = df["parental level of education"].cat.codes

df["lunch"] = df["lunch"].cat.codes

df["test preparation course"] = df["test preparation course"].cat.codes
df.dtypes
df2 = df.drop(['math score','reading score','writing score'], axis = 1)
df3 = df2.drop(['total'], axis=1)
X = df3.values

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
X
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents1 = pca.fit_transform(X)
principalComponents1
PCA_dataset1 = pd.DataFrame(data = principalComponents1, columns = ['component1', 'component2'] )

PCA_dataset1.head()
principal_component1 = PCA_dataset1['component1']

principal_component2 = PCA_dataset1['component2']
plt.figure()

plt.figure(figsize=(10,10))

plt.xlabel('Component 1')

plt.ylabel('Component 2')

plt.title('2 Component PCA')

plt.scatter(PCA_dataset1['component1'], PCA_dataset1['component2'])
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

train_x,test_x,train_y,test_y = train_test_split(X,df2['total'],test_size = 0.05)

model = XGBRegressor(max_depth = 6)

model.fit(train_x,train_y)

target = model.predict(test_x)

mean_squared_error(target,test_y)
len(target)
test_y[:5].values
target[:5]