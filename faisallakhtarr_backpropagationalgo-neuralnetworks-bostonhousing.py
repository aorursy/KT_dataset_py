#import all the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
%matplotlib inline
#load the dataset
df = datasets.load_boston(return_X_y=False)
df.sample(n=5)
#check the dataset
df.describe()
#plot a pairplot
sns.pairplot(data=df)
#get the names of the columns
df.columns
#normalise the dataset
df_norm = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)
#check the normalised dataset
df_norm.describe()
#create the target column to have integer 0/1
target = df['target']
target.sample(n=5)
#add the target column to the normalised dataset
df = pd.concat([df_norm, target], axis=1)
df.sample(n=5)
#mark some data to test as unseen data
train_test_per = 75/100.0
df['train'] = np.random.rand(len(df)) < train_test_per
df.sample(n=5)
#seperate train data
train = df[df.train == 1]
train = train.drop('train', axis=1).sample(frac=1)
train.sample(n=5)
#seperate test data
test = df[df.train == 0]
test = test.drop('train', axis=1)
test.sample(n=5)
X = train.values[:,:13]
X[:14]
targets = [[1,0],[0,1]]
y = np.array([targets[int(x)] for x in train.values[:,13:14]])
y[:14]
#create backpropogation neural network
num_inputs = len(X[0])
hidden_layer_neurons = 14
np.random.seed(13)
w1 = 2*np.random.random((num_inputs, hidden_layer_neurons)) - 1
w1
#connect hidden layer and input layer
num_outputs = len(y[0])
w2 = 2*np.random.random((hidden_layer_neurons, num_outputs)) - 1
w2
#train the neural network by updating weights
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
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
learning_rate = 0.28 # slowly update the network
for epoch in range(50000):
    l1 = 1/(1 + np.exp(-(np.dot(X, w1)))) # sigmoid function
    l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))
    er = (abs(y - l2)).mean()
    l2_delta = (y - l2)*(l2 * (1-l2))
    l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
    w2 += l1.T.dot(l2_delta) * learning_rate
    w1 += X.T.dot(l1_delta) * learning_rate
    print('Error:', er)
#test the network for accuracy
X = test.values[:,:13]
y = np.array([targets[int(x)] for x in test.values[:,13:14]])

l1 = 1/(1 + np.exp(-(np.dot(X, w1))))
l2 = 1/(1 + np.exp(-(np.dot(l1, w2))))

np.round(l2,3)
#make the predictions
yp = np.argmax(l2, axis=1) # prediction
res = yp == np.argmax(y, axis=1)
correct = np.sum(res)/len(res)

testres = test[['target']].replace([0,1], ['heart-disease','healthy'])

testres['Prediction'] = yp
testres['Prediction'] = testres['Prediction'].replace([0,1], ['heart-disease','healthy',])

print(testres)
print('Accuracy is : ',sum(res),'/',len(res), ':', (correct*100),'%')