# visualization functionality
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
py.init_notebook_mode(True)
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC

# a dummy dataset
X = np.array([0, 0.5, 1, 1.5, 3, 4, 5, 6])[:, np.newaxis]
y = np.array([-1, -1, -1, 1, 1, 1, 1, 1])

# fit regression and classification models
ridge = Ridge().fit(X, y)
svc = LinearSVC(C=10000.0).fit(X, y)

# this will be used to get predictions of models on a range of [-1, 7]
X_test = np.linspace(-1, 7)[:, np.newaxis]

# visualize outputs of linear functions trained with different losses
py.iplot(go.Figure(
    data=[
        go.Scatter(x=X[:, 0], y=y, name='Data', mode='markers'),
        go.Scatter(x=X_test[:, 0], y=svc.decision_function(X_test), name='Hinge loss', mode='lines'),
        go.Scatter(x=X_test[:, 0], y=ridge.predict(X_test), name='Sq. Loss', mode='lines')
    ],
    layout=go.Layout(
        yaxis=dict(range=[-2, 3], title="Linear model output"),
        xaxis=dict(title="Input value"),
    )
))
# this is a set of helper functions, that are used to make visualizations
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(True)

def plot_of_data(X, y):
    """Visualizes data points with binary labels"""
    I = y < 0
    return [
        go.Scatter3d(
            x = X[I,0], y = X[I,1], z = y[I], 
            mode='markers', marker={'size': 3}
        ), # negative labels
        go.Scatter3d(
            x = X[~I,0], y = X[~I,1], z = y[~I], 
            mode='markers', marker={'size': 3}
        )  # positive labels
    ]

def make_layout(axis_names, title='Dataset'):
    """Creates a layout for a 3d plot"""
    return go.Layout(
        title=title, autosize=True, height=600,
        margin=dict(l=0,r=0,b=0,t=0),
        scene=go.Scene(
            xaxis=dict(title=axis_names[0]),
            yaxis=dict(title=axis_names[1]),
            zaxis=dict(title=axis_names[-1]),
        )
    )

# a few helper functions to evaluate the models
def render_model(model, X, y, resolution = 53):
    """Evaluates model on a grid over the domain of dataset features"""
    X1 = np.linspace(min(X[:, 0]), max(X[:, 0]), resolution)
    X2 = np.linspace(min(X[:, 1]), max(X[:, 1]), resolution)
    
    # make a grid of input values
    X1, X2 = np.meshgrid(X1, X2)
    Z = X1 * 0.0

    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i,j] = model.decision_function([[X1[i,j], X2[i,j]]])[0]

    return go.Surface(x=X1, y=X2, z=Z)

# ignore ugly warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(True)
X = X[y < 2, :2]  # select first 2 features, and only 2 classes
y = y[y < 2]*2-1  # select elements of 2 classes only, and convert their labels to {-1, 1}

# fit the SVM model with Gaussian kernel; Try different parameters here!
model = SVC(gamma=100, C=0.1)
model.fit(X, y)

py.iplot(go.Figure(
    data=plot_of_data(X, y) + [render_model(model, X, y)], 
    layout=make_layout(['f1', 'f2', 'y'])
))