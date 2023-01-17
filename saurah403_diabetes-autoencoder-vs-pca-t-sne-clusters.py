import pandas as pd
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
np.random.seed(203)
from sklearn.decomposition import PCA
train = pd.read_csv('../input/diabetes-data-set/diabetes-dataset.csv')
train.head()
target = train.Outcome
data = train.drop('Outcome', axis=1)
X = data.values

# Invoke the PCA method. Since this is a binary classification problem
# let's call n_components = 2
pca = PCA(n_components=2)
pca_2d = pca.fit_transform(X)

# Invoke the TSNE method
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
tsne_results = tsne.fit_transform(X)
import seaborn as sns
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18,6), dpi=100)
sns.scatterplot(pca_2d[:,0],pca_2d[:,1], hue=target, ax=axes[0])
axes[0].set_title('PCA_PLOT')
sns.scatterplot(tsne_results[:,0],tsne_results[:,1], hue=target, ax=axes[1])
axes[1].set_title('Tsne_plot')
# Calling Sklearn scaling method
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
pca_2d_std = pca.fit_transform(X_std)

# Invoke the TSNE method
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
tsne_results_std = tsne.fit_transform(X_std)
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18,6), dpi=100)
sns.scatterplot(pca_2d_std[:,0],pca_2d_std[:,1], hue=target, ax=axes[0])
axes[0].set_title('Normalize_PCA_PLOT')
sns.scatterplot(tsne_results_std[:,0],tsne_results_std[:,1], hue=target, ax=axes[1])
axes[1].set_title('Normalize_Tsne_plot')
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
pca = PCA(n_components=5)
X_pca_model = pca.fit_transform(X_std)

train_pca_x, val_pca_x, train_pca_y, val_pca_y = train_test_split(X_pca_model, y, stratify=y ,shuffle =True , test_size=0.25)
clf = SVC(kernel='rbf').fit(train_pca_x, train_pca_y)
pred_y = clf.predict(val_pca_x)

print (classification_report(val_pca_y, pred_y))
print (accuracy_score(val_pca_y, pred_y))
input_layer = Input(shape=(X.shape[1],))
encoded = Dense(100, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(50, activation='relu')(encoded)
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='relu')(decoded)
output_layer = Dense(X.shape[1], activation ='relu')(decoded)

autoencoder = Model(input_layer,output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
y=target.values
scaler = preprocessing.MinMaxScaler()
scaler.fit(X)
X_scale = scaler.transform(X)


x_perished, x_survived = X_scale[y == 0], X_scale[y == 1]
autoencoder.fit(x_perished, x_perished, epochs = 20, shuffle = True, validation_split = 0.25)
autoencoder.layers[0]
hidden_repr = Sequential()
hidden_repr.add(autoencoder.layers[0])
hidden_repr.add(autoencoder.layers[1])
hidden_repr.add(autoencoder.layers[2])

No_diabetes_hid_rep = hidden_repr.predict(x_perished)
diabetes_hid_rep = hidden_repr.predict(x_survived)

rep_x = np.append(No_diabetes_hid_rep, diabetes_hid_rep, axis = 0)
y_n = np.zeros(No_diabetes_hid_rep.shape[0])
y_f = np.ones(diabetes_hid_rep.shape[0])
rep_y = np.append(y_n, y_f)
from sklearn.svm import SVC
train_x, val_x, train_y, val_y = train_test_split(rep_x, rep_y, stratify=rep_y ,shuffle =True , test_size=0.25)
clf = SVC(kernel='rbf').fit(train_x, train_y)
pred_y = clf.predict(val_x)

print (classification_report(val_y, pred_y))
print (accuracy_score(val_y, pred_y))
