# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import cv2

import numpy as np
cd /kaggle/input/stanford-dogs-dataset/images/Images
ls
lists_dog = ['n02091134-whippet','n02093754-Border_terrier','n02094258-Norwich_terrier','n02099712-Labrador_retriever','n02099849-Chesapeake_Bay_retriever','n02100583-vizsla'] 
import os

def feature_generation(lists_dog):

    my_lists = {key:[] for key in lists_dog}

    for file in lists_dog:

        list_files = os.listdir(file)

        os.chdir(file)

        for files in list_files:

            my_lists[str(file)].append(cv2.imread(files))

            

        os.chdir('..')       

    return my_lists
lis = feature_generation(lists_dog)



lists_n = [*lis]

for file in lists_n:

    val = []

    for fil in lis[file]:

        rgb = [np.average(fil[:,:,2]),np.average(fil[:,:,1]),np.average(fil[:,:,0])]

        val.append(rgb)

    lis[file] = val
lis["n02099712-Labrador_retriever"] [:4]
lists=list(lis.values())
lower_bound = []

upper_bound= []

lower = 0

upper = 0

for num in lists:

    upper  = (len(num))+upper

    lower_bound.append(lower)

    upper_bound.append(upper)

    lower = (len(num))+lower
lower_bound
upper_bound
# lists


lists = list(lists)

lists = np.array(lists)

lists.shape

import itertools

lists

lists = list(itertools.chain.from_iterable(lists))

lists = np.array(lists)

lists.shape


x = []

y = []

z = []

for i in range(0,lists.shape[0]):

    x.append(lists[i][0])

    y.append(lists[i][1])

    z.append(lists[i][2])
import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



data = []

for i in range(0, len(lists_dog)):

    data.append(go.Scatter3d(

        x=x[lower_bound[i]:upper_bound[i]-1],

        y=y[lower_bound[i]:upper_bound[i]-1],

        z=z[lower_bound[i]:upper_bound[i]-1],

        mode='markers',

        marker=dict(

            size=12,

            line=dict(

                color='rgba(217, 217, 217, 0.14)',

                width=0.5

            ),

            opacity=1

        ),

        name = lists_dog[i]

    ))



layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
from scipy.stats import multivariate_normal







class Expectation_Maximization:

    def __init__(self, num_cluster, max_iter=5):

        self.num_cluster = num_cluster

        self.max_iter = int(max_iter)



    def initialize(self, X):

        self.shape = X.shape

        self.n, self.m = self.shape



        self.phi = np.full(shape=self.num_cluster, fill_value=1/self.num_cluster)              # Initializing scales for all clusters

        self.weights = np.full( shape=self.shape, fill_value=1/self.num_cluster)               # Initializing weights for all points

        

        random_row = np.random.randint(low=0, high=self.n, size=self.num_cluster)              # Setting the size of initial clusters randomly      

        self.mu = [  X[row_index,:] for row_index in random_row ]                              # Initializing the mean 

        self.sigma = [ np.cov(X.T) for _ in range(self.num_cluster) ]                          # Initializing the variance



    def e_step(self, X):

        self.weights = self.predict_proba(X)                                                   # Updadting weights

        self.phi = self.weights.mean(axis=0)                                                   # Updating phi

        # here mu and sigma is constant

    

    def m_step(self, X):

        # Updating mu and sigma but weight and phi is constant

        for i in range(self.num_cluster):                      

            weight = self.weights[:, [i]]

            total_weight = weight.sum()

            self.mu[i] = (X * weight).sum(axis=0) / total_weight

            self.sigma[i] = np.cov(X.T, 

                aweights=(weight/total_weight).flatten(), 

                bias=True)

     

    def fit(self, X):                                                                         # fit the model

        self.initialize(X)

        

        for iteration in range(self.max_iter):

            self.e_step(X)

            self.m_step(X)

            

    def predict_proba(self, X):                                                               # Function for calculating pdf

        likelihood = np.zeros( (self.n, self.num_cluster) )

        for i in range(self.num_cluster):

            distribution = multivariate_normal(

                mean=self.mu[i], 

                cov=self.sigma[i])

            likelihood[:,i] = distribution.pdf(X)

        

        numerator = likelihood * self.phi

        denominator = numerator.sum(axis=1)[:, np.newaxis]

        weights = numerator / denominator

        return weights

    

    def predict(self, X):                                                                     # Predict the cluster

        weights = self.predict_proba(X)

        return np.argmax(weights, axis=1)
X = lists
np.random.seed(31)

expm = Expectation_Maximization(num_cluster=6, max_iter=12)

expm.fit(X)
lists_new = list(lis.values())
lists_new = list(itertools.chain.from_iterable(lists_new))

lists_new = np.array(lists)

lists_new.shape
dicts = {}



for i in range(0,lists_new.shape[0]):

        dicts.setdefault(int(np.unique(expm.predict(lists[i]))),[]).append(lists[i])


py.init_notebook_mode(connected=True)



data = []

for j in range(0,6):

    x=[]

    y=[]

    z=[]

    for i in range(0, len(dicts[j])):

        

        x.append(dicts[j][i][0])

        y.append(dicts[j][i][1])

        z.append(dicts[j][i][2])

    data.append(go.Scatter3d(

        x=x,

        y=y,

        z=z,

        mode='markers',

        marker=dict(

            size=12,

            line=dict(

                color='rgba(217, 217, 217, 0.14)',

                width=0.5

                ),

            opacity=1

            ),

            name = j

            ))



layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)