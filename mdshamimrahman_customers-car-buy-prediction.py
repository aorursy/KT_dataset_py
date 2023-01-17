
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv('../input/super-shop-data/shop data.csv')
df.head(5)
df.info()
df.shape
df.describe()
df.head(5)
##spilit the data

x=df.iloc[:,:-1]
x.head(5)
y=df.iloc[:,4]
y
#Now need to  convert string into int..cz algo can't understand string obj
le=LabelEncoder()
le
x=x.apply(LabelEncoder().fit_transform)
x
df
##Train the data

dt=DecisionTreeClassifier()
dt
dt.fit(x.iloc[:,0:4],y)
dt.score(x.iloc[:,0:4],y) #accuracy 
xinput=np.array([[2,0,1,1]]) ##Input Value
result=dt.predict(xinput) #Test
print('The Customar will buy the Car ? ->', result)
