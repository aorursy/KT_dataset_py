import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_context('paper')

sns.set_style('whitegrid')
df = pd.read_csv('/kaggle/input/train.csv')

df.head()
df.info()
y = df['label']

X = df.drop(columns=['label'])
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)





plt.figure(figsize=(8, 8))

sns.scatterplot(X_pca[:,0], X_pca[:, 1], hue=y, palette=sns.color_palette("bright"), legend='full')

plt.xlabel('1st component')

plt.ylabel('2nd component')

plt.title('PCA projection of the dataset')

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier(hidden_layer_sizes=(512, 512), solver='adam', activation='relu', random_state=12, max_iter=5000)

mlp.fit(X_train, y_train)
from sklearn.metrics import classification_report



y_predict = mlp.predict(X_test)

print(classification_report(y_test, y_predict))
X_eval = pd.read_csv('/kaggle/input/test.csv')

Y_eval = mlp.predict(X_eval)



df_eval = pd.DataFrame(Y_eval)

df_eval.index += 1



df_eval.index.name = 'ImageId'

df_eval.columns = ['Label']



df_eval.to_csv('results.csv', sep=',')