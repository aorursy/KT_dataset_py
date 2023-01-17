import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from sklearn.utils import shuffle

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import time

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

pd.options.display.max_columns = None

df.head()
percent_missing = df.isnull().sum() * 100 / len(df)

missing_values = pd.DataFrame({'percent_missing': percent_missing})

missing_values.sort_values(by ='percent_missing' , ascending=False)
sns.set(style="ticks")

f = sns.countplot(x="class", data=df, palette="bwr")

plt.show()
df['class'].value_counts()
X = df.drop(['class'], axis = 1)

Y = df['class']
X = pd.get_dummies(X, prefix_sep='_')

X.head()
len(X.columns)
Y = LabelEncoder().fit_transform(Y)

#np.set_printoptions(threshold=np.inf)

Y
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier



X = StandardScaler().fit_transform(X)
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

    Small_df = pd.concat([Small_df, df['class']], axis = 1)

    Small_df['class'] = LabelEncoder().fit_transform(Small_df['class'])

    forest_test(X, Y)

    data = []

    for clas, col, name in zip((1, 0), ['red', 'darkblue'], ['Poisonous', 'Edible']):



        trace = dict(

            type='scatter',

            x= Small_df.loc[Small_df['class'] == clas, 'C1'],

            y= Small_df.loc[Small_df['class'] == clas, 'C2'],

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
def complete_test_3D(X, Y, plot_name = ''):

    Small_df = pd.DataFrame(data = X, columns = ['C1', 'C2', 'C3'])

    Small_df = pd.concat([Small_df, df['class']], axis = 1)

    Small_df['class'] = LabelEncoder().fit_transform(Small_df['class'])

    forest_test(X, Y)

    data = []

    for clas, col, name in zip((1, 0), ['red', 'darkblue'], ['Poisonous', 'Edible']):



        trace = dict(

            type='scatter3d',

            x= Small_df.loc[Small_df['class'] == clas, 'C1'],

            y= Small_df.loc[Small_df['class'] == clas, 'C2'],

            z= Small_df.loc[Small_df['class'] == clas, 'C3'],

            mode= 'markers',

            name= name

        )

        data.append(trace)



    layout = {

        "scene": {

          "xaxis": {

            "title": "C1", 

            "showline": False

          }, 

          "yaxis": {

            "title": "C2", 

            "showline": False

          }, 

          "zaxis": {

            "title": "C3", 

            "showline": False

          }

        }, 

        "title": plot_name + ' 3D Dimensionality Reduction'

    }

    fig = dict(data=data, layout=layout)

    iplot(fig)
forest_test(X, Y)
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])

PCA_df = pd.concat([PCA_df, df['class']], axis = 1)

PCA_df['class'] = LabelEncoder().fit_transform(PCA_df['class'])

PCA_df.head()
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')



classes = [1, 0]

colors = ['r', 'b']

for clas, color in zip(classes, colors):

    plt.scatter(PCA_df.loc[PCA_df['class'] == clas, 'PC1'], PCA_df.loc[PCA_df['class'] == clas, 'PC2'], c = color)

    

plt.xlabel('Principal Component 1', fontsize = 12)

plt.ylabel('Principal Component 2', fontsize = 12)

plt.title('2D PCA', fontsize = 15)

plt.legend(['Poisonous', 'Edible'])

plt.grid()
pca.explained_variance_ratio_
complete_test_2D(X_pca, Y, 'PCA')
var_ratio = pca.explained_variance_ratio_

cum_var_ratio = np.cumsum(var_ratio)



trace1 = dict(

    type='bar',

    x=['PC %s' %i for i in range(1,5)],

    y=var_ratio,

    name='Individual'

)



trace2 = dict(

    type='scatter',

    x=['PC %s' %i for i in range(1,5)], 

    y=cum_var_ratio,

    name='Cumulative'

)



data = [trace1, trace2]



layout=dict(

    title='Explained variance Ratio by each principal components',

    yaxis=dict(

        title='Explained variance ratio in percent'

    ),

    annotations=list([

        dict(

            x=1.16,

            y=1.05,

            xref='paper',

            yref='paper',

            showarrow=False,

        )

    ])

)



fig = dict(data=data, layout=layout)

iplot(fig)
from itertools import product



X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_pca, Y, test_size = 0.30, random_state = 101)

trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Reduced,Y_Reduced)



x_min, x_max = X_Reduced[:, 0].min() - 1, X_Reduced[:, 0].max() + 1

y_min, y_max = X_Reduced[:, 1].min() - 1, X_Reduced[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = trainedforest.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)

plt.scatter(X_Reduced[:, 0], X_Reduced[:, 1], c=Y_Reduced, s=20, edgecolor='k')

plt.xlabel('Principal Component 1', fontsize = 12)

plt.ylabel('Principal Component 2', fontsize = 12)

plt.title('Random Forest', fontsize = 15)

plt.show()
pca = PCA(n_components=3)

X_pca = pca.fit_transform(X)

complete_test_3D(X_pca, Y, 'PCA')
var_ratio = pca.explained_variance_ratio_

cum_var_ratio = np.cumsum(var_ratio)



trace1 = dict(

    type='bar',

    x=['PC %s' %i for i in range(1,5)],

    y=var_ratio,

    name='Individual'

)



trace2 = dict(

    type='scatter',

    x=['PC %s' %i for i in range(1,5)], 

    y=cum_var_ratio,

    name='Cumulative'

)



data = [trace1, trace2]



layout=dict(

    title='Explained variance Ratio by each principal components',

    yaxis=dict(

        title='Explained variance ratio in percent'

    ),

    annotations=list([

        dict(

            x=1.16,

            y=1.05,

            xref='paper',

            yref='paper',

            showarrow=False,

        )

    ])

)



fig = dict(data=data, layout=layout)

iplot(fig)
from sklearn.manifold import TSNE



time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

X_tsne = tsne.fit_transform(X)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
sns.scatterplot(

    x=X_tsne[:,0], y=X_tsne[:,1],

    hue=Y,

    palette=sns.color_palette("hls", 2),

    data=df,

    legend="full",

    alpha=0.3

)
complete_test_2D(X_tsne, Y, 't-SNE')
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)

X_tsne = tsne.fit_transform(X)

complete_test_3D(X_tsne, Y, 't-SNE')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis(n_components=1)



# run an LDA and use it to transform the features

X_lda = lda.fit(X, Y).transform(X)

print('Original number of features:', X.shape[1])

print('Reduced number of features:', X_lda.shape[1])
lda.explained_variance_ratio_
forest_test(X_lda, Y)
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y, test_size = 0.30, random_state = 101)



start = time.process_time()

lda = LinearDiscriminantAnalysis().fit(X_Reduced,Y_Reduced)

print(time.process_time() - start)

predictionlda = lda.predict(X_Test_Reduced)

print(confusion_matrix(Y_Test_Reduced,predictionlda))

print(classification_report(Y_Test_Reduced,predictionlda))
LDA_df = pd.DataFrame(data = X_lda, columns = ['LDA1'])

LDA_df = pd.concat([LDA_df, df['class']], axis = 1)

LDA_df['class'] = LabelEncoder().fit_transform(LDA_df['class'])



figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

sns.distplot(LDA_df.loc[LDA_df['class'] == 0]['LDA1'], label = 'Edible', hist=True, kde=True, rug=True)

sns.distplot(LDA_df.loc[LDA_df['class'] == 1]['LDA1'], label = 'Poisonous', hist=True, kde=True, rug=True)

plt.legend(loc='upper right')
sns.jointplot(x="LDA1", y="class", data=LDA_df, kind="kde")
grid = sns.JointGrid(x='LDA1', y='class', data=LDA_df)



g = grid.plot_joint(sns.scatterplot, hue='class', data=LDA_df)

sns.kdeplot(LDA_df.loc[LDA_df['class']== 0, 'LDA1'], ax=g.ax_marg_x, legend=False)

sns.kdeplot(LDA_df.loc[LDA_df['class']== 1, 'LDA1'], ax=g.ax_marg_x, legend=False)

sns.kdeplot(LDA_df.loc[LDA_df['class']== 0, 'LDA1'], ax=g.ax_marg_y, vertical=True, legend=False)

sns.kdeplot(LDA_df.loc[LDA_df['class']== 1, 'LDA1'], ax=g.ax_marg_y, vertical=True, legend=False)
from sklearn.decomposition import FastICA



ica = FastICA(n_components=2)

X_ica = ica.fit_transform(X)

print('Original number of features:', X.shape[1])

print('Reduced number of features:', X_ica.shape[1])
complete_test_2D(X_ica, Y, 'ICA')
ica = FastICA(n_components=3)

X_ica = ica.fit_transform(X)



complete_test_3D(X_ica, Y, 'ICA')
from sklearn.manifold import LocallyLinearEmbedding



embedding = LocallyLinearEmbedding(n_components=2)

X_lle = embedding.fit_transform(X)

print('Original number of features:', X.shape[1])

print('Reduced number of features:', X_lle.shape[1])
complete_test_2D(X_lle, Y, 'LLE')
embedding = LocallyLinearEmbedding(n_components=3)

X_lle = embedding.fit_transform(X)

complete_test_3D(X_lle, Y, 'LLE')
from keras.layers import Input, Dense

from keras.models import Model



input_layer = Input(shape=(X.shape[1],))

encoded = Dense(2, activation='relu')(input_layer)

decoded = Dense(X.shape[1], activation='softmax')(encoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')



X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=101)



autoencoder.fit(X1, Y1,

                epochs=100,

                batch_size=300,

                shuffle=True,

                verbose = 0,

                validation_data=(X2, Y2))



encoder = Model(input_layer, encoded)

X_ae = encoder.predict(X)
complete_test_2D(X_ae, Y, 'AE')
input_layer = Input(shape=(X.shape[1],))

encoded = Dense(3, activation='relu')(input_layer)

decoded = Dense(X.shape[1], activation='softmax')(encoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')



X1, X2, Y1, Y2 = train_test_split(X, X, test_size=0.3, random_state=101)



autoencoder.fit(X1, Y1,

                epochs=100,

                batch_size=300,

                shuffle=True,

                verbose = 0,

                validation_data=(X2, Y2))



encoder = Model(input_layer, encoded)

X_ae = encoder.predict(X)
complete_test_3D(X_ae, Y, 'AE')