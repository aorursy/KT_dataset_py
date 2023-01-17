import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import ipywidgets as widgets

from IPython.display import IFrame
IFrame('https://app.powerbi.com/view?r=eyJrIjoiN2E2M2RkZGMtNDVmMC00ODJiLTlkYTUtMzU3NzM0MjViMmVlIiwidCI6ImUwNDhkZjczLWU1MDMtNDdmNC04ZWMxLWQ4YmM1NGI3NzNhNSIsImMiOjF9', width=800, height=600)
dataset=pd.read_csv('../input/Energy_Dataset.csv')
dataset=dataset.iloc[:,2:]

dataset.head()
toggle_button = widgets.ToggleButtons(

            options= dataset.columns[0:4],

            description='Response Variable:',

            disabled=False,

            button_style='')

toggle_button
variable = toggle_button.value
var_res=[]

var_res.append(variable)

for i in list(dataset.columns[4:]):

    var_res.append(i)

    

res_pred = dataset.loc[:,var_res]

res_pred.describe()
res_pred.corr()[variable]
res_var = widgets.SelectMultiple(

    options=list(res_pred.corr()[[variable]].index[1:]),

    value=(),

    rows=15,

    description='Predictors',

    disabled=False

)

res_var 
predictors = list(res_var.value)

y = res_pred[variable]

X = res_pred[predictors]



cols = 4

if len(predictors)<=cols:

    rows = 1

    cols = len(predictors)

    

else:

    if len(predictors)%cols == 0:

        rows = int(len(predictors)/cols)

    

    else:

        rows = int(len(predictors)/cols) + 1

               

plt.style.use('ggplot')

plt.figure(figsize=(cols*4,2*rows + 5))

plt.tight_layout()

for index in range(len(predictors)):

    plt.subplot(rows,cols, index+1)

    plt.scatter(y, X.iloc[:,index])

    plt.title('Predictor: '+ predictors[index])

plt.show()
if len(predictors)==2:

    import plotly.express as px

    fig = px.scatter_3d(dataset, x=predictors[0], y=predictors[1], z=variable)

    fig.show()
predictors = list(res_var.value)

corr_matrix = res_pred.loc[:,predictors].corr()

corr_matrix[corr_matrix == 1] = np.nan



plt.figure(figsize=(8, 7))

sns.heatmap(corr_matrix, annot=True, linewidths= 2, cmap='coolwarm')

plt.show()
sns.pairplot(res_pred.loc[:,predictors])

plt.show()
toggle_button2 = widgets.ToggleButtons(

            options= ['Statmodel', 'SciKitLearn', 'Gradient Descent'],

            description='Model:',

            disabled=False,

            button_style='')

toggle_button2
model = toggle_button2.value



if model == 'Statmodel':

    import statsmodels.api as sm

    

    y = res_pred[variable]

    X = res_pred[predictors]

    

    X = sm.add_constant(X)

    rmodel=sm.regression.linear_model.OLS(y,X)

    linear_regression=rmodel.fit()

    print(linear_regression.summary())

    parameters = linear_regression.params

    

elif model=='SciKitLearn':

    from sklearn import linear_model

    

    y = res_pred[variable]

    X = res_pred[predictors]

    

    rmodel = linear_model.LinearRegression()

    rmodel  = rmodel.fit(X,y)

    print(rmodel.intercept_)

    print(rmodel.coef_)

    

elif model == 'Gradient Descent':

    

    from sklearn.preprocessing import StandardScaler



    X = res_pred[predictors]

    elements=len(X)

    standardization=StandardScaler()

    Xst = standardization.fit_transform(X)

    original_means=standardization.mean_

    original_stds=standardization.var_**0.5

    Xst = np.column_stack((Xst, np.ones(elements)))

    y = res_pred[variable]

    

    import random

    def random_w(p):

        return np.array([np.random.normal() for j in range(p)])



    def hypothesis(X,w): 

        return np.dot(X,w) # X is the matrix ()X and ()w are the coeficients (w only one column)

                         # if we multiply X.w we get the matrix y(one column)



    def loss(X,w,y): # 

        return hypothesis(X,w) - y # this is the loss function (Xw -y)



    def squared_loss(X,w,y): #(Xw -y)*2

        return loss(X,w,y)**2 



    def gradient(X,w,y): # the loop 

        gradients = list() # defining a list

        n = float(len( y )) # the lenght of the y

        for j in range(len(w)):

            gradients.append(np.sum(loss(X,w,y) * X[:,j]) / n) # generating coefficients

        return gradients #everytime we loop through we generate a coefficient and we add it to the gradient



    def update(X,w,y, alpha=0.01): # alpha value by default

        return [t - alpha*g for t, g in zip(w, gradient(X,w,y))]



    def optimize(X, y, alpha = 0.01, eta = 10*-12, iterations = 1000):

        w=random_w(X.shape[1])

        path = list()

        for k in range(iterations):

            SSL = np.sum(squared_loss(X, w, y))

            new_w = update(X, w, y, alpha= alpha)

            new_SSL= np.sum(squared_loss(X, new_w,y))

            w = new_w

            if k >= 5 and (new_SSL -SSL <= eta and new_SSL - SSL > -eta):

                path.append(new_SSL)

                return w, path

            if k%(iterations /20)==0:

                path.append(new_SSL)

        return w, path

    

    alpha = 0.02

    w, path = optimize(Xst, y, alpha, eta = 10**-12, iterations=20000)

    

    unstandardized_betas = w[:-1] / original_stds

    unstandardized_bias = w[-1]-np.sum((original_means /

    original_stds) * w[:-1])

    print ('%8s: %8.4f' % ('bias', unstandardized_bias))

    for beta,varname in zip(unstandardized_betas, predictors):

        print ('%8s: %8.4f' % (varname, beta))
if len(predictors)==2 and model == 'Statmodel':

    x_min = res_pred[predictors[0]].min()

    x_max = res_pred[predictors[0]].max()



    y_min = res_pred[predictors[1]].min()

    y_max = res_pred[predictors[1]].max()



    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))



    zz = parameters.const + xx*parameters[predictors[0]] + yy*parameters[predictors[1]]

    

    import plotly.graph_objects as go

    import plotly.express as px

    

    trace1 = trace1 = go.Scatter3d(

        x=dataset[predictors[0]],

        y=dataset[predictors[1]],

        z=dataset[variable],

        mode='markers'

    )

    

    

    trace2 = go.Surface(z=zz, x=xx, y=yy, colorscale='Greys', opacity=0.75)

    

    data_test1 = go.Data([trace1, trace2])

    

    fig = go.Figure(data=data_test1)

    

    fig.show()   