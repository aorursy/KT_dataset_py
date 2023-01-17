import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pylab as plt

import plotly.graph_objects as go

import seaborn as sns



from sklearn.decomposition import PCA

data = pd.read_csv("/kaggle/input/sandp500/all_stocks_5yr.csv")

data = data.dropna()

data.head(5)
data_np = data.to_numpy()

# print(data_np)



X = []

labels = []



for row in data_np:

    if row[-1] not in labels:

        X.append(row[1:-1])

        labels.append(row[-1])



X = np.array(X, dtype="int")

labels = np.array(labels)

        

print(X)

print(labels)
pca = PCA(n_components=2)

X = pca.fit_transform(X)

print(X)
# Create dataframe

df = pd.DataFrame({

'x': X[:, 0],

'y': X[:, 1],

'labels': labels

})



fig = go.Figure(data=go.Scatter(x=df['x'],

                                y=df['y'],

                                mode='markers',

                                text=df['labels'])) # hover text goes here



fig.update_layout(title='Stocks Graphed')

fig.show()


