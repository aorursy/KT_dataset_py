# import numpy and pandas
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
iris_dataset = pd.read_csv('../input/Iris.csv')
iris_dataset.head()
iris_features_df = iris_dataset[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
iris_labels_df = iris_dataset[["Species"]]
iris_labels_df.head()
iris_labels_df["Species"].unique()
iris_df_corr = iris_features_df.corr(method="pearson")
iris_df_corr
import networkx as nx
iris_df_corr_df = iris_df_corr.stack().reset_index()
iris_df_corr_df.columns = ["Dim1","Dim2", "Corr"]
iris_df_corr_df

network_df=iris_df_corr_df[iris_df_corr_df['Corr'] >= 0.8]
graph = nx.from_pandas_edgelist(network_df, 'Dim1', 'Dim2', edge_attr=True)
nx.draw_circular(graph, with_labels=True, node_color='red', node_size=200, edge_color='blue', linewidths=1, font_size=12)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_features_df, iris_labels_df, test_size=.25)

X_train = X_train.reset_index().drop(['index'],axis=1)
X_test  = X_test.reset_index().drop(['index'],axis=1)
y_train = y_train.reset_index().drop(['index'],axis=1)
y_test  = y_test.reset_index().drop(['index'],axis=1)

# Print the size of training and test data
print(f"Train Data Size = {X_train.shape[0]}\nTest Data Size = {X_test.shape[0]}\n ")

# X_train.head()
# y_train.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]).head()

from sklearn.decomposition import PCA as sklearnPCA

pca = sklearnPCA(n_components=2) #2-dimensional PCA
X_train_scaled_2D = pd.DataFrame(pca.fit_transform(X_train_scaled))
X_test_scaled_2D = pd.DataFrame(pca.fit_transform(X_test_scaled))

X_train_scaled_2D = pd.DataFrame(X_train_scaled_2D)
X_test_scaled_2D = pd.DataFrame(X_test_scaled_2D)
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_notebook, show
from bokeh.palettes import inferno as palette
import itertools
output_notebook()

# Combine the data for plotting
data_combine = pd.concat([X_train_scaled_2D, y_train], axis=1)
# data_combine[1]


plot = figure(x_axis_label='PCA Dim 1', y_axis_label='PCA Dim 2', title='IRIS Dataset in 2D')
unique_species = list(data_combine['Species'].unique())

# colors = itertools.cycle(palette(len(unique_species)))
colors = ['Red', 'Green', 'Blue']


for species_name, color in zip(unique_species, colors):
        data = data_combine[data_combine['Species'] == species_name]
        plot.circle(x=data[0], y=data[1], legend=species_name , color=color, line_width=2)
show(plot, notebook_handle=True)
# X_train_scaled_2D, y_train
# X_test_scaled_2D, y_test

from sklearn.svm import LinearSVC
clf = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)
clf.fit(X_train_scaled, y_train)
print(f'Coefficients = {clf.coef_}')
print(f'Intercept = {clf.intercept_}')
predicted = clf.predict(X_test_scaled)
# predicted
y_predicted = np.array(list(predicted))
y_actual = np.array(y_test['Species'].values.tolist())

count = 0
for p, a in zip(y_predicted, y_actual):
    if p == a:
        count += 1
accuracy = count*100/len(y_predicted)
accuracy
