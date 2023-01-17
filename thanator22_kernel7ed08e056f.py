import numpy as np

import pandas as pd

from sklearn.cluster import DBSCAN

from collections import Counter

from sklearn.cluster import DBSCAN

from sklearn import metrics



from sklearn.cluster import DBSCAN 

from sklearn.preprocessing import StandardScaler 

from sklearn.preprocessing import normalize

from sklearn.preprocessing import StandardScaler

df=pd.read_csv('../input/mobiledata/12_18.csv')
df.head()
df.columns=['Operator', 'Technology', 'Test_Type', 'Data_Speed','Signal_Strength', 'LSA']

df['Test_Type']=df['Test_Type'].str.lower()

df['LSA']=df['LSA'].fillna(method='ffill')

df['Signal_Strength']=df['Signal_Strength'].dropna()

df['Signal_Strength']=df['Signal_Strength'].transform(lambda value : -1*value if (value>0) else value)

df=df.dropna()

df=df.drop(['Operator', 'Technology', 'Test_Type','LSA'], axis=1)

df.head()
import matplotlib.pyplot as plt

plt.matshow(df.corr())

plt.show()

smpl = df.sample(frac =.0001)

smpl.shape
X=smpl.iloc[:,:].values

X = StandardScaler().fit_transform(X)

X_normalized = X

  

# Converting the numpy array into a pandas DataFrame 

X_normalized = pd.DataFrame(X_normalized)

X_normalized.columns = ['P1', 'P2'] 

print(X_normalized.head()) 


db = DBSCAN(eps=0.4, min_samples=9).fit(X_normalized)

labels=db.labels_

labels.max()
# Building the label to colour mapping 

colours = {} 

colours[0] = 'r'

colours[1] = 'g'

colours[2] = 'b'

colours[3] = 'y'

colours[-1] = 'k'

  

# Building the colour vector for each data point 

cvec = [colours[label] for label in labels] 

  

# For the construction of the legend of the plot 

r = plt.scatter(X_normalized['P1'], X_normalized['P2'], color ='r'); 

g = plt.scatter(X_normalized['P1'], X_normalized['P2'], color ='g'); 

b = plt.scatter(X_normalized['P1'], X_normalized['P2'], color ='b');

y = plt.scatter(X_normalized['P1'], X_normalized['P2'], color ='y');

k = plt.scatter(X_normalized['P1'], X_normalized['P2'], color ='k');  



# Plotting P1 on the X-Axis and P2 on the Y-Axis  

# according to the colour vector defined 

plt.figure(figsize =(15,15)) 

plt.scatter(X_normalized['P1'], X_normalized['P2'],c = cvec)

# Building the legend 

plt.legend((r, g, b,y, k), ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label -1')) 

plt.show() 