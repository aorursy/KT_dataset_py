import pandas as pd

import numpy as np



# for reproducibility

import os

os.environ['PYTHONHASHSEED'] = str(0)

np.random.seed(5)



# options

pd.set_option('display.max_columns', 28)



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.shape, test.shape
train.isnull().sum().sum(), test.isnull().sum().sum()
train.values.min(), train.values.max(), test.values.min(), test.values.max()
#heatmap

to_heatmap = []

for i in train.sample(n = 10, random_state = 5).index:

    new_map = pd.DataFrame(np.array(train.drop(columns = 'label').iloc[i])

              .reshape(28, 28)[::-1])

    to_heatmap.append(new_map)
import plotly.graph_objects as go

from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=5)



for i in range(len(to_heatmap)):

    if i < 5:

        fig.add_trace(go.Heatmap(z = to_heatmap[i],

                                 colorscale = 'Greys',

                                 showscale = False),

                             row = 1,

                             col = i+1)

    else:

        fig.add_trace(go.Heatmap(z = to_heatmap[i],

                                 colorscale = 'Greys',

                                 showscale = False),

                             row = 2,

                             col = i-4)

fig.update_layout(title_text = 'Figure 1: Random Training Numbers')

fig.show()
import plotly.express as px



plot_df = pd.concat([train['label'],train.drop(columns = 'label')],

                     axis = 1)

plot_df = plot_df.sort_values('label', axis = 0)

plot_df['label'] = plot_df['label'].transform(lambda x: x.astype(object))

fig = px.scatter_3d(plot_df, 

                 x = 'pixel387', 

                 y = 'pixel397',

                 z = 'pixel402',

                 color = 'label',

                 title = 'Figure 2: Plot of Class Groupings by Pixels')

fig.show()
# LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



response = train['label']



train_predictors, test_predictors, train_response, test_response = train_test_split(train.drop(columns = 'label'),

                                                                response,

                                                                train_size = 0.7,

                                                                shuffle = True)
# reduce

lda = LinearDiscriminantAnalysis()

train_lda = lda.fit_transform(train_predictors, train_response)

test_lda = lda.transform(test_predictors)
np.cumsum(pd.Series(lda.explained_variance_ratio_))[np.cumsum(lda.explained_variance_ratio_) > 0.95]
lda_df = pd.concat([train_response.reset_index(), 

                    pd.DataFrame(train_lda)], 

                    axis = 1).drop(columns = 'index')

lda_df = lda_df.sort_values('label', axis = 0)

labels = ['label']

for i in range(len(lda_df.columns)-1):

    labels.append(f'Linear Discriminant {i+1}')

lda_df.columns = labels

lda_df['label'] = lda_df['label'].transform(lambda x: x.astype(object))

lda_df.describe()
fig = px.scatter_3d(lda_df, 

                 x = 'Linear Discriminant 1', 

                 y = 'Linear Discriminant 2', 

                 z = 'Linear Discriminant 3',

                 color = 'label',

                 title = 'Figure 3: Training Data Projected Onto LD1, LD2, and LD3')

fig.show()
def normalize(data):

    if np.std(data) == 0:

        return data

    else:

        return (data - np.mean(data))/np.std(data)



train_predictors = train_predictors.transform(lambda x: normalize(x))
from sklearn.decomposition import PCA

pca = PCA(random_state = 5)

train_pca = pca.fit_transform(train_predictors)
# number of entries for > 0.999% variance is explained

np.cumsum(pd.Series(pca.explained_variance_ratio_))[np.cumsum(pd.Series(pca.explained_variance_ratio_)) > 0.95][:1]
# number of entries for > 0.999% variance is explained

np.cumsum(pd.Series(pca.explained_variance_ratio_))[np.cumsum(pd.Series(pca.explained_variance_ratio_)) > 0.99][:1]
pca = PCA(n_components = 525,

          random_state = 5)

train_pca = pca.fit_transform(train_predictors)
pca_df = pd.concat([train_response.reset_index(), 

                    pd.DataFrame(train_pca)], 

                    axis = 1).drop(columns = 'index')

pca_df = pca_df.sort_values('label', axis = 0)
labels = ['label']

for i in range(len(pca_df.columns)-1):

    labels.append(f'Principal Component {i+1}')

pca_df.columns = labels

pca_df['label'] = pca_df['label'].transform(lambda x: x.astype(object))



fig = px.scatter_3d(pca_df, 

                 x = 'Principal Component 1', 

                 y = 'Principal Component 2', 

                 z = 'Principal Component 3',

                 color = 'label',

                 title = 'Figure 4: Training Data Projected Onto PC1, PC2, and PC3')

fig.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 10,

                random_state = 5)

train_kmeans = kmeans.fit_predict(train_pca)
kmeans_df = pd.concat([pd.Series(train_kmeans).rename('cluster'),

                       pd.DataFrame(train_pca)], 

                       axis = 1)

kmeans_df = kmeans_df.sort_values('cluster', axis = 0)
labels = ['cluster']

for i in range(len(kmeans_df.columns)-1):

    labels.append(f'Principal Component {i+1}')

kmeans_df.columns = labels

kmeans_df['cluster'] = kmeans_df['cluster'].transform(lambda x: x.astype(object))



fig = px.scatter_3d(kmeans_df, 

                 x = 'Principal Component 1', 

                 y = 'Principal Component 2',

                 z = 'Principal Component 3',

                 color = 'cluster',

                 title = 'Figure 5: Training Data Projected Onto PC1, PC2, and PC3 with K-Means Clustering')

fig.show()
train_lda = lda.fit_transform(train, response)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.constraints import MaxNorm

from tensorflow.keras.optimizers import RMSprop

import time
callback = [EarlyStopping(monitor = 'loss',

                          min_delta = 0.001,

                          patience = 5)]

optim = RMSprop(lr = 0.05)



model = Sequential()

model.add(Dense(2048, input_dim = 9, kernel_constraint = MaxNorm(4)))

model.add(Activation('softmax'))

model.add(Dropout(0.5))

model.add(Dense(2048))

model.add(Activation('softmax'))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation('softmax'))







model.compile(optimizer = optim,

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy'])
start_time = time.time()



history = model.fit(train_lda, 

                    response, 

                    batch_size = 200, 

                    epochs = 20,

                    verbose = 0,

                    callbacks = callback,

                    shuffle = True,

                    validation_split = 0.1)



print(f'Runtime {time.time() - start_time} seconds')
history_plot = history.history

history_plot.update({'epoch' : history.epoch})
fig = make_subplots(rows = 1,

                    cols = 1,

                    x_title = 'Epoch')



trace1 = go.Scatter(x = history_plot['epoch'], 

                    y = history_plot['acc'],

                    name = 'Training Accuracy')



trace2 = go.Scatter(x = history_plot['epoch'], 

                    y = history_plot['loss'],

                    name = 'Training Loss')



trace3 = go.Scatter(x = history_plot['epoch'], 

                    y = history_plot['val_acc'],

                    name = 'Validation Accuracy')



trace4 = go.Scatter(x = history_plot['epoch'], 

                    y = history_plot['val_loss'],

                    name = 'Validation Loss')



fig.add_traces([trace1, trace2, trace3, trace4])

fig.update_layout(title = 'Figure 6: Model Accuracy and Loss per Epoch')

fig.show()