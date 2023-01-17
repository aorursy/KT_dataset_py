#Loading dependencies 

import numpy as np

import scipy as sp

import pandas as pd

import sklearn as sk

import matplotlib.pyplot as plot

from sklearn.neighbors import KNeighborsRegressor as NN

from sklearn.preprocessing import StandardScaler as SS

from mpl_toolkits.mplot3d import Axes3D
#Loading data

hp = pd.read_csv("../input/london-house-prices/hpdemo.csv")



hp
#Scaling the data



#Initailise scaling object and preliminary calculation

scaler = SS()

scaler.fit(hp[['east','north','fl_area']]) #This only computes the mean and std to be used for scaling later.



#Scaling the data

hp_sc = scaler.transform(hp[['east','north','fl_area']])

#Creating the regression model object with arbitrary parameters

mod1 = NN(n_neighbors=6,weights='uniform',p=2)



#Fitting the regressors and response variable to model

price = hp['price']/1000.0 #seperate response from regressors

mod1.fit(hp_sc,price)

#Assessing model preformance 



#Initailise scoring object

mae = sk.metrics.make_scorer(sk.metrics.mean_absolute_error, greater_is_better=False)



#Create gridseracher to iterate through all parameters of model

mod_list = sk.model_selection.GridSearchCV(estimator=NN(),scoring=mae,param_grid= {'n_neighbors':range(1,35),

                                                                                 'weights':['uniform','distance'],

                                                                                 'p':[1,2]})

#Fit the data to the gridsearcher

mod_list.fit(hp[['east','north','fl_area']],price)



#Show parameters of the best model found

print_summary(mod_list)
#Preping data for plotting



#Creating x and y axis meshs

east_mesh, north_mesh = np.meshgrid(np.linspace(505000, 555800, 100,),

                                    np.linspace(158400, 199900, 100))

#Create empty z-axis meshs that are the same size as x and y mesh

fl_mesh = np.zeros_like(east_mesh) 

fl_mesh2 = np.zeros_like(east_mesh) 

fl_mesh3 = np.zeros_like(east_mesh) 



#Fill every z-axis cell with the average floor area of dataset and two other floor sizes

fl_mesh[:,:] = np.mean(hp['fl_area'])

fl_mesh2[:,:] = 75

fl_mesh3[:,:] = 125



#Preping the data for predictions 



#Need to unravel all 2d arrays in 1d vector for the predict function

regressor_df = np.array([np.ravel(east_mesh),np.ravel(north_mesh),np.ravel(fl_mesh)]).T

regressor_df2 = np.array([np.ravel(east_mesh),np.ravel(north_mesh),np.ravel(fl_mesh2)]).T

regressor_df3 = np.array([np.ravel(east_mesh),np.ravel(north_mesh),np.ravel(fl_mesh3)]).T
#Make predictions (this prediction assumes an average floor area for every case)

hp_pred = mod_list.predict(regressor_df)

hp_pred2 = mod_list.predict(regressor_df2)

hp_pred3 = mod_list.predict(regressor_df3)



#Shape the 1d vector of predictions into 2d array for z-axis of the plot

hp_mesh = hp_pred.reshape(east_mesh.shape)

hp_mesh2 = hp_pred2.reshape(east_mesh.shape)

hp_mesh3 = hp_pred3.reshape(east_mesh.shape)
#Plot1 

fig = plot.figure()

ax = Axes3D(fig)

ax.plot_surface(east_mesh, north_mesh, hp_mesh, rstride=1, cstride=1, cmap='YlOrBr',lw=0.01)

plot.title('London House Prices')

ax.set_xlabel('Easting')

ax.set_ylabel('Northing')

ax.set_zlabel('Price at Mean Floor Area')

plot.show()
#Plot2 

fig = plot.figure()

ax = Axes3D(fig)

ax.plot_surface(east_mesh, north_mesh, hp_mesh2, rstride=1, cstride=1, cmap='YlOrBr',lw=0.01)

plot.title('London House Prices')

ax.set_xlabel('Easting')

ax.set_ylabel('Northing')

ax.set_zlabel('Price at 75m Floor Area')

plot.show()
#Plot3 

fig = plot.figure()

ax = Axes3D(fig)

ax.plot_surface(east_mesh, north_mesh, hp_mesh3, rstride=1, cstride=1, cmap='YlOrBr',lw=0.01)

plot.title('London House Prices')

ax.set_xlabel('Easting')

ax.set_ylabel('Northing')

ax.set_zlabel('Price at 125m Floor Area')

plot.show()