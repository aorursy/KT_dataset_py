%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/diabetes.csv')
df.head()
#Check null values & how many numerical and categorical features
df.info()
#the min and max values
df.describe()

sns.pairplot(df,hue= 'Outcome')
#Build a histogram of all KPIs together
_=df.hist(figsize=(12,10))
#Find the correation between the variables
sns.heatmap(df.corr(),annot=True)
#Since variables have varied scale, we would rescale all of them, using Standard Scalar
#StandardScalar subtracts mean from all the values and divide by SD, so we scale all the values to mean = 0 and sd = 1

from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
sc= StandardScaler()
x = sc.fit_transform(df.drop('Outcome',axis =1))
Y = df['Outcome'].values
y_cat = to_categorical(Y)
x
x.shape
y_cat
y_cat.shape
