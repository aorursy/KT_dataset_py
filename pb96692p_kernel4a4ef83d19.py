import pandas as pd,seaborn as sns, numpy as np, matplotlib.pyplot as plt # for data preprocessing and visualization











import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/ph-recognition/ph-data.csv") # load train data frame in pandas 
df.head()
ax = sns.pairplot(df)
list1 = df.label

phbalance = []

for i in  list1:

    if i < 7:

        phbalance.append(0)

    elif i > 7: 

        phbalance.append(1)

    else :

        phbalance.append(2)

    
df['phbalance'] = phbalance

df.head()
ax = sns.countplot(data = df, x = phbalance)
sns.relplot(data=df,x="red",y="green",hue="phbalance")

import pandas_profiling as pp
pp.ProfileReport(df)
import plotly.express as px

fig = px.scatter_3d(df, x='red', y='green', z='blue',  

              color='label', symbol ='phbalance' )

fig.show()
from sklearn.svm import SVC

clf = SVC(gamma='auto')

from sklearn.model_selection import train_test_split
x = df[['blue','green', 'red',]]

x.head()
y = df['phbalance']

y.head()
#Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(x,y , test_size=0.25, random_state=True)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_pred
clf.score(X_test,y_test)