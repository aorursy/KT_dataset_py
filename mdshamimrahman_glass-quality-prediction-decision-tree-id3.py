
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder #if data is string thn use it to preprocess the data
from sklearn.tree import DecisionTreeClassifier


df=pd.read_csv('../input/glass/glass.csv')
df.head(5)
df.info()
df.describe()
df.shape
x=df.iloc[:,:-1]
x
x.head(5)
y=df.iloc[:,9]
y.head(5)
dt=DecisionTreeClassifier()
dt
dt.fit(x.iloc[:,0:8],y)
xinput=np.array([[4,7000,8000,93,200,1000,2,8145]])
y_result=dt.predict(xinput)
print('The quality of glass is Type no', y_result)
