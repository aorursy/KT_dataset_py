from sklearn import datasets
from sklearn.manifold import TSNE
import keras
import numpy as np
import pandas as pd
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation, Dense

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import logging
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans.train import train
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import BasicIterativeMethod

import matplotlib.pyplot as plt
from cleverhans.dataset import MNIST
tf.reset_default_graph()
# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session and set as Keras backend session
sess = tf.Session()
print("Created TensorFlow session.")

set_log_level(logging.DEBUG)

# Get MNIST test data
mnist = MNIST(train_start=0, train_end=60000,
            test_start=0, test_end=10000)
x_train, y_train = mnist.get_set('train')
x_test, y_test = mnist.get_set('test')

# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                    nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

nb_filters = 64
# Define TF model graph
model = ModelBasicCNN('model1', nb_classes, nb_filters)
preds = model.get_logits(x)
loss = CrossEntropy(model, smoothing=0.1)
print("Defined TensorFlow model graph.")

###########################################################################
# Training the model using TensorFlow
###########################################################################
NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = .001
SOURCE_SAMPLES = 10

# Train an MNIST model
train_params = {
  'nb_epochs': NB_EPOCHS,
  'batch_size': BATCH_SIZE,
  'learning_rate': LEARNING_RATE
}
sess.run(tf.global_variables_initializer())
rng = np.random.RandomState([2017, 8, 30])
train(sess, loss, x_train, y_train, args=train_params, rng=rng)

# Evaluate the accuracy of the MNIST model on legitimate test examples
eval_params = {'batch_size': BATCH_SIZE}
accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
assert x_test.shape[0] == 10000 , x_test.shape
print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
report.clean_train_clean_eval = accuracy

#creating y_target : each image should be transformed into an adversarial 3
y_targets=np.zeros((100,10))
y_targets[:,3]=1
y_targets[:5]
def get_bim_params(y_targets=None,clip_min=0.,clip_max=1., eps_iter = 0.01, nb_iter= 100):
    bim_params = {'eps_iter': eps_iter,
                  'nb_iter': nb_iter,
                  'clip_min': clip_min,
                  'clip_max': clip_max,
                  'y_target': y_targets}

    return bim_params
bim_op = BasicIterativeMethod(model, sess=sess)
advs_on_attacks=[]
accuracies_on_advs=[]
for i in range(10):
    #advs = bim_op.generate_np(x_test[i*100:(i+1)*100], **get_bim_params(nb_iter=i*2, eps_iter=0.01))
    advs = bim_op.generate_np(x_test[0:100], **get_bim_params(y_targets=y_targets,nb_iter=i*10, eps_iter=0.01))

    advs_on_attacks.append(advs)
    pred_class_adv=model_argmax(sess, x, preds, advs)
    res_advs = np.array(pred_class_adv == np.argmax(y_test[0:100],axis=1)).astype(int)
    accuracy_advs=res_advs.sum()/len(res_advs)
    accuracies_on_advs.append(accuracy_advs)
plt.plot(accuracies_on_advs)
y_advs=[i*np.ones((100,1)) for i in range(10)]
y_advs=np.array(y_advs).reshape(1000)
X_advs=np.array(advs_on_attacks).reshape(1000,784)
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(X_advs)
#target_ids = range(len(digits.target_names))

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
plt.title("distribution t-sne on iteration from 0 to 10 with epsilon_iter 0.01")
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', '#ff66cc', 'orange', 'purple'
for i, c, label in zip(range(10), colors, range(10)):
    plt.scatter(X_2d[y_advs == i, 0], X_2d[y_advs == i, 1], c=c, label=label)
plt.legend()
plt.show()
plt.figure(figsize=(25,30))
for i in range(10):
    pred_class_adv=model_argmax(sess, x, preds, X_advs[i*100].reshape(1,28,28,1))
    plt.title(pred_class_adv)
    plt.subplot(4,3,i+1)
    plt.imshow(X_advs[i*100].reshape(28,28))
plt.show()
X_test_2d = tsne.fit_transform(x_test[:1000].reshape(1000,784))
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
plt.title("T-sne clean mnist")
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', '#ff66cc', 'orange', 'purple'
for i, c, label in zip(range(10), colors, range(10)):
    plt.scatter(X_test_2d[np.argmax(y_test[:1000],axis=1) == i, 0], X_test_2d[np.argmax(y_test[:1000],axis=1) == i, 1], c=c, label=label)
plt.legend()
plt.show()
X_test_advs=np.append(X_advs[:100],x_test[:1000])
y_test_advs=np.append(np.ones((100,1))*10,np.argmax(y_test[:1000],axis=1))
X_test_advs_2d = tsne.fit_transform(X_test_advs.reshape(1100,784))
plt.figure(figsize=(10, 10))
plt.title("T-sne on clean mnist + adversarial with eps ")
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', '#ff66cc', 'orange', 'purple','#00ff00'
for i, c, label in zip(range(11), colors, range(11)):
    plt.scatter(X_test_advs_2d[y_test_advs == i, 0], X_test_advs_2d[y_test_advs == i, 1], c=c, label=label)
plt.legend()
plt.show()
X_test_advs2=np.append(X_advs[900:1000],X_test_advs)
y_test_advs2=np.append(np.ones((100,1))*11,y_test_advs)
X_test_advs_2d_2 = tsne.fit_transform(X_test_advs2.reshape(1200,784))
X_test_advs_2d_2.shape
plt.figure(figsize=(10, 10))
plt.title("T-sne on clean mnist + adversarial targeted on 3 (10:iter0, 11:iter10)")
colors = 'k', 'k', 'k', 'r', 'k', 'k', 'k', 'k', 'k', 'k','#00ff00','#fdff00'
for i, c, label in zip(range(12), colors, range(12)):
    plt.scatter(X_test_advs_2d_2[y_test_advs2 == i, 0], X_test_advs_2d_2[y_test_advs2 == i, 1], c=c, label=label)
plt.legend()
plt.show()
import plotly.plotly 
import plotly.graph_objs as go
np.unique(y_test_advs2)
scale = np.linspace(0, 1, 11)

colors=['#b86b77','#f7f2b8','#cc99a5','#a590f0','#4a570c','#9c84ef','#e1ad46','#d1a589','#c19e87','#00ff00','#fdff00']
colors2=['#000000','#000000','#000000','#FF0000','#000000','#000000','#000000','#000000','#000000','#00ff00','#fdff00']
colorscale=[ [sc,colors[i]] for i,sc in enumerate(scale)]
colorscale2=[ [sc,colors2[i]] for i,sc in enumerate(scale)]

tsne_3d = TSNE(n_components=3, random_state=0)
X_3d = tsne_3d.fit_transform(X_test_advs2.reshape(1200,784))
#'r', 'g', 'b', 'c', 'm', 'y', 'k', '#ff66cc', 'orange', 'purple','#00ff00'
#colorscale=[[0, 'r'],[0.1, 'green'], [0.2, 'green'], [0.3, 'green'],[0.4, 'green'],[0.5,'green'],[0.6,'green'],[0.7,'green'],[0.8,'green'],[0.9,'green'],[1.0,'red']]
# Configure Plotly to be rendered inline in the notebook.
plotly.offline.init_notebook_mode()

# Configure the trace.
trace = go.Scatter3d(
    x=X_3d[:,0],  
    y=X_3d[:,1],  
    z=X_3d[:,2],  
    mode='markers',
    marker=dict(
        size=6,
        color=y_test_advs2,                # set color to an array/list of desired values
        colorscale=colorscale,   # choose a colorscale
        opacity=0.8,
        showscale=True
    )
)

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
plotly.offline.iplot(plot_figure)
# Configure Plotly to be rendered inline in the notebook.
plotly.offline.init_notebook_mode()

# Configure the trace.
trace = go.Scatter3d(
    x=X_3d[:,0],  
    y=X_3d[:,1],  
    z=X_3d[:,2],  
    mode='markers',
    marker=dict(
        size=6,
        color=y_test_advs2,                # set color to an array/list of desired values
        colorscale=colorscale2,   # choose a colorscale
        opacity=0.8,
        showscale=True
    )
)

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
plotly.offline.iplot(plot_figure)
