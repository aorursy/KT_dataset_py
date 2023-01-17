import numpy as np  # nd array manipulation library
import pandas as ps  # focused on processing 2d arrays for data analysis

csv = ps.read_csv('../input/concrete-strength-reduced-feature-dataset/concrete.csv')  # load as a pandas array;
display(csv[0:5])
Xy = csv.values
X = Xy[:, 0:2]  # numpy matrix of inputs
y = Xy[:, 2]  # numpy vector of outputs

print('Type of X', type(X))
print('Type of y', type(y))

# visualize rows 1 ... 3, and corresponding outputs
print(X[1:4])
print(y[1:4])

print('Shape of X:', X.shape)
print('Shape of y:', y.shape)
# this is a set of helper functions, that are used to make visualizations
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(True)

def plot_of_data(X, y):
    """Visualizes data points"""
    return go.Scatter3d(
        x = X[:,0], y = X[:,1], z = y, 
        mode='markers', marker={'size': 3})

def make_layout(title='Dataset'):
    """Creates a layout for a 3d plot"""
    return go.Layout(
        title=title,
        autosize=True,
        height=800,
        margin=dict(l=0,r=0,b=0,t=0),
        scene=go.Scene(
            xaxis=dict(title=csv.columns[0]),
            yaxis=dict(title=csv.columns[1]),
            zaxis=dict(title=csv.columns[-1]),
        )
    )

# a few helper functions to evaluate the models
def render_model(model, X, y, resolution = 37):
    """Evaluates model on a grid over the domain of dataset features"""
    X1 = np.linspace(min(X[:, 0]), max(X[:, 0]), resolution)
    X2 = np.linspace(min(X[:, 1]), max(X[:, 1]), resolution)

    X1, X2 = np.meshgrid(X1, X2)
    Z = X1 * 0.0

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i,j] = model.predict([[X1[i,j], X2[i,j]]])[0]

    return go.Surface(
        x=X1, y=X2, z=Z
    )
# visualize the dataset itself
fig = go.Figure(data=[plot_of_data(X, y)], layout=make_layout())
py.iplot(fig)
from sklearn.linear_model import LinearRegression

# fitting of the linear function to the data
lin_model = LinearRegression()
lin_model.fit(X, y)

# make estimations with fitted function
y_true = y[:5]
y_pred = lin_model.predict(X[:5])  # note - this also takes as input a matrix!
print("Estimated: ", np.round(y_pred))
print("Actual:    ", np.round(y_true))

# Task: make predictions by hand!
y_subset = y[:500]
X_subset = X[:500]

# evaluate how good the model is on data subset
my_score = lin_model.score(X_subset, y_subset)
print(my_score)
# Task: import the `KNeighborsRegressor` class from sklearn.neighbors, fit KNN model to the X and y, and print its score on a subset of test values.
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=1)
knn_model.fit(X, y)
print(knn_model.score(X_subset, y_subset))
# render the KNN model
fig = go.Figure(
    data=[
        plot_of_data(X, y), 
        render_model(knn_model, X, y)
    ], 
    layout=make_layout()
)
py.iplot(fig)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Train KNN on train partition, and evaluate on test. Try different values of n_neighbors.
for k in [1, 2, 4, 10, 20, 40, 80, 160, 320]:
    knn_model = KNeighborsRegressor(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    score = knn_model.score(X_test, y_test)
    print('k=',k,'score: ', score)
import pandas as pd

housing = pd.read_csv('../input/boston-housing-data/housing.csv')
Xy = housing.values
X = Xy[:, :-1]  # all but last column are features
y = Xy[:, -1]  # this takes vector of last column

display(housing.head())
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# use a separate set for estimation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

rows = []

best_model = None
best_score = 0
for a in 10.0 ** np.arange(-9, 6):
    pass
    # Task: implement fitting of Lasso model with various alpha values
    # then use Ridge model
    
    #model = Ridge(alpha=a)  # sum of sq. values of w as regularization
    model = Lasso(alpha=a)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    
    if score > best_score:
        best_score = score
        best_model = model
    
    row = [score, a] + model.coef_.tolist()
    rows.append(row)

# Task: select the best model, and evaluate it on the test set. 
print(best_model.score(X_test, y_test))

# once done with adding rows, uncomment below
cols = ['Validation score', 'Alpha'] + housing.columns[:-1].tolist()
df = pd.DataFrame(data=rows, columns=cols)
display(df)
# a dummy dataset
X = np.array([
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.0]
])
y = np.array([0.0, 1.0, 0.0, 1.0])

# loss function and regularizations
f_loss = lambda w: np.mean((np.dot(X, w) - y)**2)
f_l2 = lambda w: np.mean(np.power(w,2))
f_l1 = lambda w: np.mean(np.abs(w))
# some helper functions to make visualization
def z_on_grid(f):
    """Calculates function over a rectangular grid"""
    rng = np.linspace(-0.8, 0.8, 53)  # make n sized vector of range of values
    z = np.zeros((53, 53))  # this will hold function values
    for i, y in enumerate(rng):
        for j, x in enumerate(rng):
            z[i, j] = f([x, y])  # evaluate function for different x and y
    return z, rng, rng  # return z and ranges for x and y values
# evaluate the functions on a grid
loss, xr, yr = z_on_grid(f_loss)
l1, xr, yr = z_on_grid(f_l1)
l2, xr, yr = z_on_grid(f_l2)

# find a minimum with L1 and L2 regularizations
from scipy.optimize import minimize

# this calculates minimum
w = minimize(lambda w: f_loss(w), x0=[0.0, 0.0]).x
w1 = minimize(lambda w: f_loss(w) + 0.46*f_l1(w), x0=[0.0, 0.0]).x
w2 = minimize(lambda w: f_loss(w) + f_l2(w), x0=[0.0, 0.0]).x

# plot the functions and their minimizing points
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(True)

fig = tls.make_subplots(rows=1, cols=3, subplot_titles=('Loss', 'L1 Regularization', 'L2 Regularization'))

# plot loss function
fig.append_trace(go.Contour(x=xr, y=yr, z=loss, showscale=False), 1, 1)
fig.append_trace(go.Scatter(x=[w[0]], y=[w[1]], name='Loss minimum'), 1, 1)
print('Loss minimum', w)

# plot L1 regularization and corresponding optimum
fig.append_trace(go.Contour(x=xr, y=yr, z=l1, showscale=False), 1, 2)
fig.append_trace(go.Scatter(x=[w1[0]], y=[w1[1]], name='Loss + L1 minimum'), 1, 2)
print('Loss + L1 minimum', w1)

# same for L2 regularization
fig.append_trace(go.Contour(x=xr, y=yr, z=l2, showscale=False), 1, 3)
fig.append_trace(go.Scatter(x=[w2[0]], y=[w2[1]], name='Loss + L2 minimum'), 1, 3)
print('Loss + L2 minimum', w2)

py.iplot(fig)