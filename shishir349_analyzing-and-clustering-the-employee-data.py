import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/employee-analysis/employee.csv")
data.head()
data.tail()
data.describe()
data.info()
data.isnull().sum()
data[(data["Gender"]=="Female") 

     & (data["MonthlyIncome"]>10000) 

     & (data["PerformanceRating"]<4)

    ][["Age"]].agg("mean")
y=data.groupby(["EducationField"]).agg(["mean","min","max"])[["Age"]]

print(round(y,2))
y=data[["Age","MonthlyIncome","EducationField"]].groupby(["EducationField"]).agg("max")



y.sort_values(by='MonthlyIncome')
x=data.groupby(["Gender"]).agg("mean").T

x.style.background_gradient(cmap='Wistia')
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



x = data.loc[:, ['Age',

       'MonthlyRate']].values



# let's check the shape of x

print(x.shape)
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)





plt.rcParams['figure.figsize'] = (15, 8)

plt.style.use('fivethirtyeight')



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')

#plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

#plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')

#plt.scatter(x[y_means == 5, 0], x[y_means == 5, 1], s = 100, c = 'gray', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('JobSatisfaction')

plt.ylabel('MonthlyRate')

plt.legend()

plt.grid()

plt.show()
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



x = data.loc[:, ['Age',

       'MonthlyRate']].values



# let's check the shape of x

print(x.shape)



from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff



x = data[['Age', 'MonthlyRate', 'PerformanceRating']].values

km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

km.fit(x)

labels = km.labels_

centroids = km.cluster_centers_

data['labels'] =  labels

trace1 = go.Scatter3d(

    x= data['Age'],

    y= data['MonthlyRate'],

    z= data['PerformanceRating'],

    mode='markers',

     marker=dict(

        color = data['labels'], 

        size= 10,

        line=dict(

            color= data['labels'],

            width= 12

        ),

        opacity=0.8

     )

)

df = [trace1]



layout = go.Layout(

    title = 'Character vs Gender vs Alive or not',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    ),

    scene = dict(

            xaxis = dict(title  = 'Age'),

            yaxis = dict(title  = 'MonthlyRate'),

            zaxis = dict(title  = 'PerformanceRating')

        )

)



fig = go.Figure(data = df, layout = layout)

py.iplot(fig)