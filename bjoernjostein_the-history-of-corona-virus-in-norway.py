import numpy as np

import pandas as pd

from sklearn import preprocessing

import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import Ridge, Lasso

from sklearn.preprocessing import PolynomialFeatures

from random import random, seed 

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt 

import ipywidgets as widgets

from IPython.display import display

%matplotlib inline
data=pd.read_csv("/kaggle/input/corona-virus-in-norway/coronaNorway.csv")
def get_data(days,infected):

    X=days

    y=infected

    X=np.asarray(X)

    y=np.asarray(y)

    X=X.reshape(-1,1)

    y=y.reshape(-1,1)

    return X,y



def PolynomialRidge(pol,alphafactor, **kwargs):

    return make_pipeline(PolynomialFeatures(degree=pol),

                         Ridge(**kwargs, alpha=alphafactor))



def PolynomialLasso(pol,alphafactor, **kwargs):

    return make_pipeline(PolynomialFeatures(degree=pol),

                         Lasso(**kwargs, alpha=alphafactor))



def PolynomialRegression(pol, **kwargs):

    return make_pipeline(PolynomialFeatures(degree=pol),

                         LinearRegression(**kwargs))



def fit_lasso_model(pol,X,y,alphafactor):

    lassomodel = PolynomialLasso(pol,alphafactor)

    lassomodel.fit(X,y)

    return lassomodel



def fit_ridge_model(pol,X,y,alphafactor):

    ridgemodel = PolynomialRidge(pol,alphafactor)

    ridgemodel.fit(X,y)

    return ridgemodel



def fit_reg_model(pol,X,y):

    regmodel = PolynomialRegression(pol)

    regmodel.fit(X,y)

    return regmodel

    

def create_test_data(start_day, end_day,steps):

    X_test=np.linspace(start_day, end_day, steps)[:, None]

    return X_test



def ridge_pred(ridgemodel,X_test):

    y_test = ridgemodel.predict(X_test)

    return X_test,y_test



def lasso_pred(lassomodel, X_test):

    y_test = lassomodel.predict(X_test)

    return X_test,y_test



def reg_pred(regmodel, X_test):

    y_test = regmodel.predict(X_test)

    return X_test,y_test
def execute_ridge_model(days,infected,pol,alphafactor,start_test,end_test,steps):

    X , y = get_data(days,infected)

    ridgemodel = fit_ridge_model(pol,X,y,alphafactor)

    X_test, y_test = ridge_pred(ridgemodel,create_test_data(start_test,end_test,steps))

    return X , y , X_test , y_test, ridgemodel



def execute_lasso_model(days,infected,pol,alphafactor,start_test,end_test,steps):

    X , y = get_data(days,infected)

    lassomodel = fit_lasso_model(pol,X,y,alphafactor)

    X_test, y_test = lasso_pred(lassomodel,create_test_data(start_test,end_test,steps))

    return X , y , X_test , y_test, lassomodel



def execute_reg_model(days,infected,pol,start_test,end_test,steps):

    X , y = get_data(days,infected)

    regmodel = fit_reg_model(pol,X,y)

    X_test, y_test = reg_pred(regmodel,create_test_data(start_test,end_test,steps))

    return X , y , X_test , y_test, regmodel
def plot_pred(X,y,X_test,y_test,model):

    plt.style.use('ggplot')

    mse=mean_squared_error(model.predict(X), y)

    r2=r2_score(model.predict(X),y)

    print ("{} mean Square Error: ".format(model.get_params().get('steps')[1][0]),mse)

    print ("{} R2-score: ".format(model.get_params().get('steps')[1][0]),r2 )

    print('Mean absolute error: %.2f' % mean_absolute_error(model.predict(X), y))

    print ("------------------------------------")

    plt.plot(X_test.ravel(), y_test, color='C1',label='predicted')

    plt.plot(X,y, 'ro', label='infected', color='C7')

    plt.xlabel(r'$x$') #Setter navn på x-akse

    plt.ylabel(r'$y$')

    plt.ylim(bottom=0)

    plt.title(r' {}model - {}.order polynomial, alpha = {}'.format(model.get_params().get('steps')[1][0],model.get_params().get('steps')[0][1].get_params().get('degree'),model.get_params().get('steps')[1][1].get_params().get('alpha')))

    plt.legend()

w = widgets.IntSlider(

    value=5,

    min=0,

    max=10,

    step=1,

    description='Polynom:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='d'

)

v = widgets.FloatSlider(

    value=0.7,

    min=0,

    max=1.0,

    step=0.1,

    description='Alpha:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='.1f',

)

o = widgets.IntSlider(

    value=4,

    min=0,

    max=10,

    step=1,

    description='Polynom:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='d'

)

p = widgets.FloatSlider(

    value=0.2,

    min=0,

    max=1.0,

    step=0.1,

    description='Alpha:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='.1f',

)

q = widgets.IntSlider(

    value=3,

    min=0,

    max=10,

    step=1,

    description='Polynom:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='d'

)

s = widgets.IntSlider(

    value=0,

    min=0,

    max=100,

    step=1,

    description='Start test set',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='d'

)

e = widgets.IntSlider(

    value=50,

    min=0,

    max=100,

    step=1,

    description='End test set',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='d'

)

print("Ridge Parameters:")

display(w)

display(v)

print("Lasso Parameters:")

display(o)

display(p)

print("Polynomial Regression Parameters:")

display(q)

print("Test set:")

display(s)

display(e)



button = widgets.Button(description="Run prediction")

output = widgets.Output()



display(button, output)



def on_button_clicked(b):

    with output:

        X_r,y_r,X_test_r,y_test_r,ridgemodel = execute_ridge_model(data.iloc[0:,0],data['Infected'],int(w.value),float(v.value),int(s.value),int(e.value),int(e.value)-int(s.value))

        fig = plt.figure(figsize=(30,5))

        fig.add_subplot(131)

        plot_pred(X_r,y_r,X_test_r,y_test_r,ridgemodel)

        

        

        X_l,y_l,X_test_l,y_test_l,lassomodel = execute_lasso_model(data.iloc[0:,0],data['Infected'],int(o.value),float(p.value),int(s.value),int(e.value),int(e.value)-int(s.value))

        fig.add_subplot(132)

        plot_pred(X_l,y_l,X_test_l,y_test_l,lassomodel)

        

        X_r,y_r,X_test_r,y_test_r,regmodel = execute_reg_model(data.iloc[0:,0],data['Infected'],int(q.value),int(s.value),int(e.value),int(e.value)-int(s.value))

        fig.add_subplot(133)

        plot_pred(X_r,y_r,X_test_r,y_test_r,regmodel)

        plt.suptitle("Regression methods for predicting Corona Virus in Norway",fontsize=18 )

        plt.show()

        

        %matplotlib inline



button.on_click(on_button_clicked)