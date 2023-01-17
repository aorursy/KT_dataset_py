# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# load iris database

data = pd.read_csv('../input/Iris.csv')

data.sample(n=5)
data.describe()
# simple visualization to show how the inputs compare against each other

sns.pairplot( data=data, vars=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'), hue='Species' )
df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

df_norm.sample(n=5)
df_norm.describe()
target = data[['Species']].replace(['Iris-setosa','Iris-versicolor','Iris-virginica'],[0,1,2])

target.sample(n=5)
df = pd.concat([df_norm, target], axis=1)

df.sample(n=5)
train_test_per = 90/100.0

df['train'] = np.random.rand(len(df)) < train_test_per

df.sample(n=5)
train = df[df.train == 1]

train = train.drop('train', axis=1).sample(frac=1)

train.sample(n=5)
test = df[df.train == 0]

test = test.drop('train', axis=1)

test.sample(n=5)
X = train.values[:,:4]

X[:5]
targets = [[1,0,0],[0,1,0],[0,0,1]]

y = np.array([targets[int(x)] for x in train.values[:,4:5]])

y[:5]
num_inputs = len(X[0])

hidden_layer_neurons = 5

np.random.seed(4)

w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1

w1
num_outputs = len(y[0])

w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1

w2
# taken from> https://gist.github.com/craffel/2d727968c3aaebd10359

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):

    '''

    Draw a neural network cartoon using matplotilb.

    

    :usage:

        >>> fig = plt.figure(figsize=(12, 12))

        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    

    :parameters:

        - ax : matplotlib.axes.AxesSubplot

            The axes on which to plot the cartoon (get e.g. by plt.gca())

        - left : float

            The center of the leftmost node(s) will be placed here

        - right : float

            The center of the rightmost node(s) will be placed here

        - bottom : float

            The center of the bottommost node(s) will be placed here

        - top : float

            The center of the topmost node(s) will be placed here

        - layer_sizes : list of int

            List of layer sizes, including input and output dimensionality

    '''

    n_layers = len(layer_sizes)

    v_spacing = (top - bottom)/float(max(layer_sizes))

    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    # Nodes

    for n, layer_size in enumerate(layer_sizes):

        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.

        for m in range(layer_size):

            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,

                                color='w', ec='k', zorder=4)

            ax.add_artist(circle)

    # Edges

    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):

        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.

        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.

        for m in range(layer_size_a):

            for o in range(layer_size_b):

                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],

                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')

                ax.add_artist(line)
fig = plt.figure(figsize=(12, 12))

ax = fig.gca()

ax.axis('off')

draw_neural_net(ax, .1, .9, .1, .9, [4, 5, 3])
# sigmoid function representation

_x = np.linspace( -5, 5, 50 )

_y = 1 / ( 1 + np.exp( -_x ) )

plt.plot( _x, _y )
learning_rate = 0.2 # slowly update the network

for epoch in range(50000):

    l1 = 1/(1 + np.exp(-(np.dot(X, w1)))) # sigmoid function

    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

    er = (abs(y - l2)).mean()

    l2_delta = (y - l2)*(l2 * (1-l2))

    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))

    w2 += l1.T.dot(l2_delta) * learning_rate

    w1 += X.T.dot(l1_delta) * learning_rate

print('Error:', er)
X = test.values[:,:4]

y = np.array([targets[int(x)] for x in test.values[:,4:5]])



l1 = 1/(1 + np.exp(-(np.dot(X, w1))))

l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))



np.round(l2,3)
yp = np.argmax(l2, axis=1) # prediction

res = yp == np.argmax(y, axis=1)

correct = np.sum(res)/len(res)



testres = test[['Species']].replace([0,1,2], ['Iris-setosa','Iris-versicolor','Iris-virginica'])



testres['Prediction'] = yp

testres['Prediction'] = testres['Prediction'].replace([0,1,2], ['Iris-setosa','Iris-versicolor','Iris-virginica'])



print(testres)

print('Correct:',sum(res),'/',len(res), ':', (correct*100),'%')