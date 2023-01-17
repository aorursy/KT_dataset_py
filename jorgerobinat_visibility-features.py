import time

import matplotlib.pyplot as plt

import pandas as pd 

import seaborn as sns

import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

sample_size=60000

master=pd.read_csv("../input/meteorological-model-versus-real-data/vigo_model_vs_real.csv",index_col="datetime",parse_dates=True).sample(sample_size)

X=master[[ "cape_4K", "cfh_4K", "cft_4K", "lhflx_4K", "pbl_height_4K", "prec_4K", "rh_4K", "shflx_4K","visibility_4K",]]

threshold=1000

Y=["Fog" if c<=threshold else "Clear" for c in master.visibility_o]

display=X.copy()

display["visibility_o"]=Y
g=sns.pairplot(display[[ "cape_4K", "cfh_4K", "cft_4K", "lhflx_4K","visibility_o"]].sample(500), hue = 'visibility_o', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)
g=sns.pairplot(display[["pbl_height_4K", "prec_4K", "rh_4K", "shflx_4K","visibility_4K","visibility_o"]].sample(50000), hue = 'visibility_o', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)
fig = px.scatter_3d(display.sample(5000), x='cfh_4K', y='rh_4K', z='lhflx_4K',

              color='visibility_o')

fig.show()

X=StandardScaler().fit_transform(X) 
def forest_test(X, Y):

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, 

                                                        test_size = 0.30, 

                                                        random_state = 101)

    start = time.process_time()

    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train,Y_Train)

    print("process time=",time.process_time() - start)

    predictionforest = trainedforest.predict(X_Test)

    print(confusion_matrix(Y_Test,predictionforest))

    print(classification_report(Y_Test,predictionforest))

forest_test(X, Y)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

 

X_pca = pca.fit_transform(X)

PCA_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2',"PC3"])

PCA_df["visibility_o"]=Y

g=sns.pairplot(PCA_df.sample(10000), hue = 'visibility_o', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)
fig = px.scatter_3d(PCA_df.sample(500), x='PC1', y='PC2', z='PC3',

              color='visibility_o')

fig.show()
forest_test(X_pca, Y)
from sklearn.decomposition import FastICA

ica = FastICA(n_components=3)

X_ica = ica.fit_transform(X)

ICA_df = pd.DataFrame(data = X_ica, columns = ['ICA1', 'ICA2',"ICA3"])

ICA_df["visibility_o"]=Y

fig = px.scatter_3d(ICA_df.sample(1000), x='ICA1', y='ICA2', z='ICA3',

              color='visibility_o')

fig.show()
sns.pairplot(ICA_df.sample(1000), hue = 'visibility_o', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)
forest_test(X_ica, Y)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



lda = LinearDiscriminantAnalysis(n_components=1)



# run an LDA and use it to transform the features

X_lda = lda.fit(X, Y).transform(X)

lda_df = pd.DataFrame(data = X_lda, columns = ['LDA1',])

lda_df["visibility_o"]=Y
forest_test(X_lda, Y)
sns.pairplot(lda_df.sample(1000), hue = 'visibility_o', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             height = 4)
X_Reduced, X_Test_Reduced, Y_Reduced, Y_Test_Reduced = train_test_split(X_lda, Y, 

                                                                        test_size = 0.30, 

                                                                        random_state = 101)



start = time.process_time()

lda = LinearDiscriminantAnalysis().fit(X_Reduced,Y_Reduced)

print("process time=",time.process_time() - start)

predictionlda = lda.predict(X_Test_Reduced)

print(confusion_matrix(Y_Test_Reduced,predictionlda))

print(classification_report(Y_Test_Reduced,predictionlda))
from sklearn.manifold import LocallyLinearEmbedding

embedding = LocallyLinearEmbedding(n_components=3)

X_lle = embedding.fit_transform(X)

lle_df = pd.DataFrame(data = X_lle, columns = ['LLE1',"LLE2","LLE3"])

lle_df["visibility_o"]=Y



fig = px.scatter_3d(lle_df.sample(1000), x='LLE1', y='LLE2', z='LLE3',

              color='visibility_o')

fig.show()
forest_test(X_lle, Y)
from sklearn.manifold import TSNE



tsne = TSNE(n_components=3,  )

X_tsne = tsne.fit_transform(X)

tsne_df = pd.DataFrame(data = X_tsne, columns = ['TSNE1',"TSNE2","TSNE3"])

tsne_df["visibility_o"]=Y



fig = px.scatter_3d(tsne_df.sample(1000), x='TSNE1', y='TSNE2', z='TSNE3',

              color='visibility_o')

fig.show()
from keras.layers import Input, Dense

from keras.models import Model



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

                verbose = 30,

                validation_data=(X2, Y2))



encoder = Model(input_layer, encoded)

X_ae = encoder.predict(X)
ae_df = pd.DataFrame(data = X_ae, columns = ['AE1',"AE2","AE3"])

ae_df["visibility_o"]=Y

fig = px.scatter_3d(ae_df.sample(1000), x='AE1', y='AE2', z='AE3',

              color='visibility_o')

fig.show()
forest_test(X_ae, Y)