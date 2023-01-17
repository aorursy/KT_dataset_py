# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv')

df.head()
df.shape
df.describe()
df['target'].value_counts()
df.info()
#Checking Data

pd.DataFrame(df.isnull().sum()).T
#Try to visualise correlation 
import matplotlib as mpl

df_discover=df.iloc[:,1:58]

d = df_discover.values

covariance = np.corrcoef(d.T)

mpl.rc("figure", figsize=(28,28))

g=sns.heatmap(covariance)
#Random Forest Algorithm 
##if (df[df['ps_ind_07_cat']>1].id.count() == 0):   
df1=df.iloc[:,1]

df1.head()

df1.shape

df2=df.iloc[:,2:59]

df2.shape

df2.head()
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

X_train, X_test, y_train, y_test = train_test_split(df2,df1)

L=[]

for k in range(30,45):

    clf = RandomForestClassifier(max_depth=k, random_state=0)

    clf.fit(X_train,y_train )

    t=clf.score(X_test,y_test)

    L.append(t)

    

x=np.arange(30,45)   

mpl.rc("figure", figsize=(15,10))

plt.plot(x,L)

plt.show()
#Studying Correlation -> PCA -> lOGISTIC Regression
#Correlation
from collections import defaultdict

dfc=df.iloc[:,2:58]

x = dfc.values

correlation_matrix = np.corrcoef(x.T)

correlation_matrix.shape

nb=0

d = defaultdict(list)

for i in range(56):

    #print("------colonne----------")

    #print(i)

    for j in range(56):

          if (abs(correlation_matrix[i][j])>0.3 and correlation_matrix[i][j] < 0.99999):

                print ("correlation entre"+str(i)+"et"+str(j))

                print(correlation_matrix[i][j])

                d[i].append(j)

                nb=nb+1;

                print("nb= "+str(nb))
d
#PCA
from sklearn.decomposition import PCA as sklearnPCA

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from plotly.graph_objs import *

import plotly.plotly as py

X_std = StandardScaler().fit_transform(dfc)

sklearn_pca = sklearnPCA(n_components=28)

Y_sklearn = sklearn_pca.fit_transform(X_std)

t=sklearn_pca.explained_variance_

print (t)

#trace1 = Bar(

#       x=['PC %s' %i for i in range(1,28)],

#        y=t,

#        showlegend=False)

#data = Data([trace1])

#layout=Layout(

#       yaxis=YAxis(title='Explained variance'),

#        title='Explained variance by different principal components')

#fig = Figure(data=data, layout=layout)

#py.iplot(fig)

x = [i + 0.1 for i, _ in enumerate(t)]

plt.ylabel("explained variance")

plt.title("Info")

l=np.arange(28)

plt.xticks([i + 0.5 for i, _ in enumerate(t)], l )

x=np.arange(28)

plt.bar(x,t)

X=Y_sklearn

X
X.shape
Y=df.iloc[:,1]

Y.head()
import seaborn as sns

dfx = pd.DataFrame(X)

dfx.head()



dfx.shape
#sns.pairplot(dfx, hue=Y)

#plt.show()
#Logistic Regression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

logistic = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X,Y)

logistic.fit(X_train, y_train )

logistic.score(X_test,y_test)
df_test=pd.read_csv('../input/test.csv')

df_test.head()