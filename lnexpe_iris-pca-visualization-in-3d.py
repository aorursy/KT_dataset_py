import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
data.head()
data.dtypes
import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

plt.style.use('fivethirtyeight')  

import warnings

warnings.filterwarnings('ignore') 

counts = data['Species'].value_counts()

plt.bar(counts.index, counts)
data = data.drop('Id', axis=1)

data.head()
import scipy.stats as stat



def plot_hists(data, target, feature, num_bins):

    counts = data[target].value_counts()

    fig, ax = plt.subplots(1, 3, figsize = [12, 4], sharey=True)

    for axis, index in zip(ax, counts.index):

        column = data[data[target]==index][feature]

        n, bins, patches = axis.hist(column, num_bins, density=True, edgecolor='black')

        print(bins)

        mean = column.mean()

        std = column.std()

        axis.axvline(mean, color='#444444', label='Mean', linewidth=2)

        y = stat.norm.pdf(bins, mean, std)

        axis.plot(bins, y, '--')

        title = index + (' mu=%.2f'% mean)

        axis.set_title(title)



    plt.tight_layout()

    plt.show()
plot_hists(data, 'Species', 'SepalLengthCm', 8)
plot_hists(data, 'Species', 'SepalWidthCm', 8)
plot_hists(data, 'Species', 'PetalLengthCm', 8)
plot_hists(data, 'Species', 'PetalWidthCm', 8)
from sklearn.decomposition import PCA
training = data.drop('Species', axis=1)
pca = PCA(n_components=3)

pca.fit(training)
print(pca.components_)
print(pca.explained_variance_)
training_pca = pca.transform(training)
training_pca = pd.DataFrame({'First': training_pca[:, 0], 'Second': training_pca[:, 1], 'Third': training_pca[:, 2]})
training_pca = training_pca.join(data['Species'])
plot_hists(training_pca, 'Species', 'First', 10)
plot_hists(training_pca, 'Species', 'Second', 10)
plot_hists(training_pca, 'Species', 'Third', 10)
%matplotlib inline

from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(10,10))

ax = plt.axes(projection='3d')



for i in training_pca['Species'].unique():

    spec = training_pca[training_pca['Species']==i]

    xdata = spec['First']

    zdata = spec['Second']

    ydata = spec['Third']

    ax.scatter3D(xdata, ydata, zdata, label=i)

    

ax.legend()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(training_pca['Species'])

encoded_column = le.transform(training_pca['Species'])

training_pca['encoded'] = encoded_column
logtrain = training_pca[training_pca['encoded']<2]
logtrain.drop('Species', axis=1, inplace=True)
logtrain.head()
logtrain.describe()
x_train = logtrain.drop('encoded', axis=1)

y_train = logtrain['encoded']
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression().fit(x_train, y_train)
log_reg.score(x_train, y_train) # as axpected really easy to fit a hyperplane between blue and red dots