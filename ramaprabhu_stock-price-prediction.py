import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import seaborn as sns
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import metrics
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )
df = pd.read_csv("../input/fundamentals.csv")
df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.replace(',', '_')
df.columns = df.columns.str.replace('/', '_')
df.columns = df.columns.str.replace('___', '_')
df.columns = df.columns.str.replace('__', '_')
df.columns = df.columns.str.replace('&', '_and_')
df.columns
df.drop(columns = ['Unnamed:_0'], inplace = True)
tkrsym = df.groupby(['Ticker_Symbol']).mean()
tkrsym.head()
df.dropna()
df.describe()
df[["Accounts_Payable", "Accounts_Receivable",  "Total_Current_Assets", "Total_Current_Liabilities", "Total_Equity", "Total_Liabilities", "Total_Liabilities__and__Equity", "Total_Revenue", "Treasury_Stock" 
]].describe().plot(kind='line')
dataset=pd.read_csv("../input/prices.csv")
dataset=dataset.sample(5000)
dataset.describe()
x=dataset[["high","low","open","volume"]].values
y=dataset["close"].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)
predicted=regressor.predict(x_test)
print(predicted)
dframe = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':predicted.flatten()})
dframe.head(25)
graph = dframe.head(25)
graph.plot(kind='bar')
dframe.describe()
sns.barplot( data=dframe)
plt.show()
sns.jointplot(dframe.Actual, dframe.Predicted, kind = 'reg')
clf = KMeans(n_clusters = 2)
clf.fit(x_train)
visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(x_train)       
visualizer.show()    
from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(clf)
visualizer.fit(x_train)        # Fit the data to the visualizer
visualizer.show()    
sns.scatterplot(dframe.Actual, dframe.Predicted, )
dfsec=pd.read_csv("../input/securities.csv")
df.columns
dfsec.columns = dfsec.columns.str.replace(' ', '_')
dfsec.columns = dfsec.columns.str.replace(',', '_')
dfsec.columns = dfsec.columns.str.replace('/', '_')
dfsec.columns = dfsec.columns.str.replace('___', '_')
dfsec.columns = dfsec.columns.str.replace('__', '_')
dfsec.columns = dfsec.columns.str.replace('&', '_and_')
dfsec.columns
