import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report,confusion_matrix

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

import time

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_excel("/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx","Data")

df.head()
percent_missing = df.isnull().sum() * 100 / len(df)

missing_values = pd.DataFrame({'percent_missing': percent_missing})

missing_values.sort_values(by ='percent_missing' , ascending=False)
X = df.drop(['Personal Loan'], axis=1)

X = StandardScaler().fit_transform(X)

Y = df['Personal Loan'].values

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)
clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_Train, Y_Train)

y_pred = clf.predict(X_Test)

accuracy_score(Y_Test, y_pred)
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])

PCA_df = pd.concat([PCA_df, df['Personal Loan']], axis = 1)

PCA_df.head()
X = PCA_df.drop(['Personal Loan'], axis=1)

X = StandardScaler().fit_transform(X)

Y = PCA_df['Personal Loan'].values

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)
clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_Train, Y_Train)

y_pred = clf.predict(X_Test)

accuracy_score(Y_Test, y_pred)
import matplotlib.gridspec as gridspec

import itertools

gs = gridspec.GridSpec(2, 2)



fig = plt.figure(figsize=(10,8))



clf1 = LogisticRegression(random_state=1,

                          solver='newton-cg',

                          multi_class='multinomial')

clf2 = RandomForestClassifier(random_state=1, n_estimators=100)

clf3 = GaussianNB()

clf4 = SVC(gamma='auto')



labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'SVM']

for clf, lab, grd in zip([clf1, clf2, clf3, clf4],

                         labels,

                         itertools.product([0, 1], repeat=2)):



    clf.fit(X_Train, Y_Train)

    ax = plt.subplot(gs[grd[0], grd[1]])

    fig = plot_decision_regions(X_Train, Y_Train, clf=clf, legend=2)

    plt.title(lab)



plt.savefig("boundaries.png", dpi=300)

plt.show()
from yellowbrick.contrib.classifier import DecisionViz

from sklearn.neighbors import KNeighborsClassifier



viz = DecisionViz(

    KNeighborsClassifier(3), title="Nearest Neighbors",

    features=['PCA1', 'PCA2'], classes=['0', '1']

)

viz.fit(X_Train, Y_Train)

viz.draw(X_Test, Y_Test)

viz.show()
viz = DecisionViz(

    RandomForestClassifier(3), title="Random Forest",

    features=['PCA1', 'PCA2'], classes=['0', '1']

)

viz.fit(X_Train, Y_Train)

viz.draw(X_Test, Y_Test)

viz.show()
from itertools import product



clf1 = RandomForestClassifier(max_depth=4)

clf2 = KNeighborsClassifier(n_neighbors=7)

clf3 = SVC(gamma=.1, kernel='rbf', probability=True)

eclf = VotingClassifier(estimators=[('rf', clf1), ('knn', clf2),

                                    ('svc', clf3)],

                        voting='soft', weights=[2, 1, 2])



clf1.fit(X_Train, Y_Train)

clf2.fit(X_Train, Y_Train)

clf3.fit(X_Train, Y_Train)

eclf.fit(X_Train, Y_Train)



# Plotting decision regions

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                     np.arange(y_min, y_max, 0.1))



f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))



for idx, clf, tt in zip(product([0, 1], [0, 1]),

                        [clf1, clf2, clf3, eclf],

                        ['Random Forest (depth=4)', 'KNN (k=7)',

                         'Kernel SVM', 'Soft Voting']):



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)

    axarr[idx[0], idx[1]].scatter(X_Train[:, 0], X_Train[:, 1], c=Y_Train,

                                  s=20, edgecolor='k')

    axarr[idx[0], idx[1]].set_title(tt)



plt.show()
# Following https://www.kaggle.com/arthurtok/decision-boundaries-visualised-via-python-plotly

from plotly import tools

import plotly.offline as py



trees = RandomForestClassifier(max_depth=4, n_estimators=20, random_state=0)

trees.fit(X_Train, Y_Train)

h = 0.02



trees_overfit = RandomForestClassifier(max_depth=50, n_estimators=5, random_state=0)

trees_overfit.fit(X_Train, Y_Train)



x_min, x_max = X_Train[:, 0].min() - 1, X_Train[:, 0].max() + 1

y_min, y_max = X_Train[:, 1].min() - 1, X_Train[:, 1].max() + 1



xx, yy = np.meshgrid(np.arange(x_min, x_max, h)

                     , np.arange(y_min, y_max, h))

y_ = np.arange(y_min, y_max, h)



Z = trees.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

#Z = Z.reshape((xx.shape[0], xx.shape[1], 3))



fig = tools.make_subplots(rows=1, cols=2,

                          subplot_titles=("Random Forest (Depth = 4)",

                                          "Random Forest (Depth = 50)")

                         )



trace1 = go.Heatmap(x=xx[0], y=y_, z=Z,

                  colorscale='Viridis',

                  showscale=False)



trace2 = go.Scatter(x=X_Train[:, 0], y=X_Train[:, 1], 

                    mode='markers',

                    showlegend=False,

                    marker=dict(size=10,

                                color=Y_Train, 

                                colorscale='Viridis',

                                line=dict(color='black', width=1))

                    )

                  

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)



Z = trees_overfit.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)





trace3 = go.Heatmap(x=xx[0], y=y_, 

                    z=Z,

                    colorscale='Viridis',

                    showscale=True)



trace4 = go.Scatter(x=X_Train[:, 0], y=X_Train[:, 1],

                    mode='markers',

                    showlegend=False,

                    marker=dict(size=10,

                                color=Y_Train, 

                                colorscale='Viridis',

                                line=dict(color='black', width=1))

                   )

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)



for i in map(str, range(1, 3)):

    x = 'xaxis' + i

    y2 = 'yaxis' + i

    fig['layout'][x].update(showgrid=False, zeroline=False,

                                   showticklabels=False, ticks='', autorange=True)

    fig['layout'][y2].update(showgrid=False, zeroline=False,

                                   showticklabels=False, ticks='', autorange=True)



py.iplot(fig)
def forest_test(X, Y):

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.30, random_state = 101)

    start = time.process_time()

    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)

    print(time.process_time() - start)

    predictionforest = trainedforest.predict(X_Test)

    print(confusion_matrix(Y_Test,predictionforest))

    print(classification_report(Y_Test,predictionforest))
def complete_test_2D(X, Y, plot_name = ''):

    Small_df = pd.DataFrame(data = X, columns = ['C1', 'C2'])

    Small_df = pd.concat([Small_df, df['Personal Loan']], axis = 1)

    Small_df['Personal Loan'] = LabelEncoder().fit_transform(Small_df['Personal Loan'])

    forest_test(X, Y)

    data = []

    for clas, col, name in zip((1, 0), ['red', 'darkblue'], ['0', '1']):



        trace = dict(

            type='scatter',

            x= Small_df.loc[Small_df['Personal Loan'] == clas, 'C1'],

            y= Small_df.loc[Small_df['Personal Loan'] == clas, 'C2'],

            mode= 'markers',

            name= name,

            marker=dict(

                color=col,

                size=12,

                line=dict(

                    color='rgba(217, 217, 217, 0.14)',

                    width=0.5),

                opacity=0.8)

        )

        data.append(trace)



    layout = dict(

            title= plot_name + ' 2D Dimensionality Reduction',

            xaxis=dict(title='C1', showline=False),

            yaxis=dict(title='C2', showline=False)

    )

    fig = dict(data=data, layout=layout)

    iplot(fig)
complete_test_2D(X_pca, Y, 'PCA')
import keras

from keras.models import Sequential

from keras.layers import Dense



input_shape2 = [X_Train.shape[1]]

model = Sequential()

model.add(Dense(50, input_shape=input_shape2, activation='relu'))

model.add(Dense(20, activation='relu'))

model.add(Dense(1, activation='tanh'))

model.compile(loss='squared_hinge', optimizer='adam', metrics=['accuracy'])

model.fit(X_Train, Y_Train, epochs=20, batch_size=64, verbose=2, validation_split=0.3)

model.evaluate(X_Test, Y_Test, verbose=1)
plot_decision_regions(X_Train, Y_Train, clf=model, legend=2)



# Working with more than 2 dimensions

# value=1.5

# width=0.75

# plot_decision_regions(X_Train, Y_Train, clf=model, legend=2, feature_index=[0,2],

#                       filler_feature_values={1: value, 3:value, 4: value, 5:value, 

#                                              6: value, 7:value, 8: value, 9:value, 

#                                              10: value, 11: value, 12:value},

#                       filler_feature_ranges={1: width, 3: width, 4: width, 5: width, 

#                                              6: width, 7: width, 8: width, 9: width, 

#                                              10: width, 11: width, 12: width})



plt.show()
plot_decision_regions(X_Test, Y_Test, clf=model, legend=2)

plt.show()
# Following https://towardsdatascience.com/simple-method-of-creating-animated-graphs-127c11f58cc5

# boundary of the graph

GRID_X_START = -7

GRID_X_END = 9

GRID_Y_START = -1.0

GRID_Y_END = 8.5

# output directory (the folder must be created on the drive)

OUTPUT_DIR = "./"

grid = np.mgrid[GRID_X_START:GRID_X_END:100j,GRID_X_START:GRID_Y_END:100j]

grid_2d = grid.reshape(2, -1).T

XX, YY = grid
# the function making up the graph of a dataset

def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):

    if (dark):

        plt.style.use('dark_background')

    else:

        sns.set_style("whitegrid")

    plt.figure(figsize=(16,12))

    axes = plt.gca()

    axes.set(xlabel="$X_1$", ylabel="$X_2$")

    plt.title(plot_name, fontsize=30)

    plt.subplots_adjust(left=0.20)

    plt.subplots_adjust(right=0.80)

    if(XX is not None and YY is not None and preds is not None):

        plt.contourf(XX, YY, preds[:,0].reshape(XX.shape), 25, alpha = 1, cmap=cm.Spectral)

        plt.contour(XX, YY, preds[:,0].reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')

    if(file_name):

        plt.savefig(file_name)

        plt.close()
def make_gif(input_folder, save_filepath):

    episode_frames = []

    time_per_step = 0.25

    for root, _, files in os.walk(input_folder):

        file_paths = [os.path.join(root, file) for file in files]

        #sorted by modified time

        file_paths = sorted(file_paths, key=lambda x: os.path.getmtime(x))

        episode_frames = [imageio.imread(file_path) for file_path in file_paths if file_path.endswith('.png')]

    episode_frames = np.array(episode_frames)

    imageio.mimsave(save_filepath, episode_frames, duration=time_per_step)
import seaborn as sns

from matplotlib import cm



def callback_keras_plot(epoch, logs):

    plot_title = "Keras Model - It: {:05}".format(epoch)

    file_name = "keras_model_{:05}.png".format(epoch)

    file_path = os.path.join(OUTPUT_DIR, file_name)

    prediction_probs = model.predict_proba(grid_2d, batch_size=32, verbose=0)

    make_plot(X_Test, Y_Test, plot_title, file_name=file_path, XX=XX, YY=YY, preds=prediction_probs)

    

# Adding callback functions that they will run in every epoch

testmodelcb = keras.callbacks.LambdaCallback(on_epoch_end=callback_keras_plot)



# Building a model

model = Sequential()

model.add(Dense(25, input_dim=2,activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(25, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])



# Training

history = model.fit(X_Train, Y_Train, epochs= 118, callbacks=[testmodelcb])
prediction_probs = model.predict_proba(grid_2d, batch_size=32, verbose=0)

make_plot(X_Test, Y_Test, "Keras Model", file_name=None, XX=XX, YY=YY, preds=prediction_probs)
import imageio



input_folder = './'

save_filepath = './Keras_gif.gif'

make_gif(input_folder, save_filepath)