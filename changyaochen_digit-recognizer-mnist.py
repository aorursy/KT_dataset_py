# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/train.csv')
import numpy as np

# separate X and Y

Y = df.iloc[:, 0].values  # target label, np.ndarray

X = df.iloc[:, 1:].values.astype(float)  # inputs, n x 284 size, np.ndarray



from sklearn.preprocessing import StandardScaler

# standarize the input

scaler = StandardScaler().fit(X)  # for later scaling of test data

X_std = StandardScaler().fit_transform(X)

# to get the covariant matrix, for the later PCA

cov = np.cov(X_std.T)

# to get the eigen-vectors and eigen-values

eig_vals, eig_vecs = np.linalg.eig(cov)
# sort the eigen-value, eigen-vector pairs

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(eig_vals.shape[0])]

eig_pairs.sort(key = lambda x: x[0], reverse=True) 
# calculate sum of eigen-values

eig_vals = np.abs(eig_vals)

eig_vals = -np.sort(-eig_vals)

eig_cum = 100*np.cumsum(eig_vals)/sum(eig_vals)

eig_df = pd.DataFrame(eig_vals, columns=['eigen-value'])

eig_df['cum_per'] = eig_cum



# plotting

import bokeh.charts as bkc

import bokeh.plotting as bkp

from bokeh.models import LinearAxis, Range1d, HoverTool



bkp.output_notebook()

source = bkc.ColumnDataSource(eig_df)

# Create a HoverTool: hover

p1 = bkp.figure(plot_height = 300, plot_width = 700, 

               x_axis_label = 'eigen-value number', y_axis_label = 'eigen-value',

              tools = ["pan, box_zoom, reset, resize, hover"], toolbar_location="above")

p1.vbar(x = 'index', top = 'eigen_value', width = 0.5, source = source, 

       color = 'blue', alpha = 0.5)

hover = p1.select(dict(type = HoverTool))

hover.tooltips = dict([("Index","@index"), ("Value", "@eigen_value{0.00}")])

# Setting the second y axis range name and range

#p.extra_y_ranges = {"cum_per": Range1d(start=0, end=100)}

# Adding the second axis to the plot.  

#p.add_layout(LinearAxis(y_range_name="cum_per_axis"), 'right')

# Using the aditional y range 

#p.line(x = 'index', y = 'cum_per', source=source,

#         color="green", y_range_name="cum_per_axis")



bkp.show(p1)



p2 = bkp.figure(plot_height = 300, plot_width = 700, 

               x_axis_label = 'eigen-value number', y_axis_label = 'cum sum (%)',

              tools = ["pan, box_zoom, reset, resize, hover"], toolbar_location="above")

p2.circle(x = 'index', y = 'cum_per', source = source, alpha = 0.5, color = 'red')

hover = p2.select(dict(type = HoverTool))

hover.tooltips = dict([("Index","@index"), ("Value", "@cum_per")])

bkp.show(p2)
# find the 90% index

d = eig_df['cum_per'].searchsorted(90,side='right')[0]
# perform PCA

from sklearn.decomposition import PCA as sklearnPCA

sklearn_pca = sklearnPCA(n_components=d)

pca = sklearn_pca.fit(X_std)

X_sklearn = sklearn_pca.fit_transform(X_std)
# split the training set

mask = [True,]*int(0.8*X_sklearn.shape[0])

mask.extend([False,]*(X_sklearn.shape[0] - int(0.8*X_sklearn.shape[0])))

mask = np.random.permutation(mask)

X_train, Y_train = X_sklearn[mask], Y[mask] 

X_xv, Y_xv = X_sklearn[~mask], Y[~mask] 



# train a Neural Network

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,20), max_iter=50, alpha=1e-4,

                    solver='sgd', verbose=10, tol=1e-4, random_state=1,

                    learning_rate_init=.1)



mlp.fit(X_train, Y_train)

print("\nTraining set score: %f" % mlp.score(X_train, Y_train))

print("Test set score: %f" % mlp.score(X_xv, Y_xv))
# read the test data

df_test = pd.read_csv('../input/test.csv')

X_test = df_test.iloc[:, :].values.astype(float)  # inputs, n x 284 size, np.ndarray



# perform the scaling and the PCA on the test data

X_test_std = scaler.transform(X_test)

X_final = pca.transform(X_test_std)



# predict with the train NN

Y_final = mlp.predict(X_final)



# save the output

df_output = pd.DataFrame(np.arange(1, Y_final.shape[0]+1), columns=['ImageId'])

df_output['Label'] = Y_final

df_output.to_csv('output.csv', index=False, header=False)