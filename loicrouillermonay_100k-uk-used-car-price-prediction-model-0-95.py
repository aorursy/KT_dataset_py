# Machine learning library
!pip install pycaret
# Python 3 environment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O

# Plotly libraries
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots

# Machine learning library
from pycaret.regression import *
# data 
audi = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/audi.csv')
bmw = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/bmw.csv')
cclass = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/cclass.csv')
focus = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/focus.csv')
ford = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/ford.csv')
hyundai = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/hyundi.csv')
merc = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/merc.csv')
skoda = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/skoda.csv')
toyota = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/toyota.csv')
vauxhall = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/vauxhall.csv')
vw = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/vw.csv')
# data summary
audi.info()
audi['manufacturer'] = 'Audi'
audi['has_tax'] = 1

bmw['manufacturer'] = 'BMW'
bmw['has_tax'] = 1

cclass['manufacturer'] = 'Mercedes-Benz'
cclass['has_tax'] = 0
cclass['tax'] = np.nan

focus['manufacturer'] = 'Ford'
focus['has_tax'] = 0
focus['tax'] = np.nan

hyundai['manufacturer'] = 'Hyundai Motor'
hyundai['has_tax'] = 1
hyundai = hyundai.rename(columns={"tax(£)": "tax"})

merc['manufacturer'] = 'Mercedes'
merc['has_tax'] = 1

skoda['manufacturer'] = 'Skoda'
skoda['has_tax'] = 1

toyota['manufacturer'] = 'Toyota'
toyota['has_tax'] = 1

vauxhall['manufacturer'] = 'Vauxhall'
vauxhall['has_tax'] = 1

vw['manufacturer'] = 'Volkswagen'
vw['has_tax'] = 1
# aggregation
cars = pd.concat([audi, bmw, cclass, focus, hyundai, merc, skoda, toyota, vauxhall, vw], ignore_index=True)

# dimensionality
cars.shape
# data overview
cars
fig = px.histogram(cars, x="manufacturer", hover_data=cars.columns)
fig.update_layout(title='Quantitative representation of the number of vehicles per manufacturer')
fig.show()
fig = px.histogram(cars, x="price", marginal="box")
fig.update_layout(title='Statistical Distribution of Price')
fig.show()
fig = px.histogram(cars, x="year", marginal="box")
fig.update_layout(title='Statistical Distribution of Year')
fig.show()
fig = px.histogram(cars, x="mpg", marginal="box")
fig.update_layout(title='Statistical Distribution of Miles Per Gallon')
fig.show()
fig = px.histogram(cars, x="mileage", marginal="box")
fig.update_layout(title='Statistical Distribution of Mileage')
fig.show()
fig = px.histogram(cars, x="tax", marginal="box")
fig.update_layout(title='Statistical Distribution of Tax')
fig.show()
fig = px.histogram(cars, x="engineSize", marginal="box")
fig.update_layout(title='Statistical Distribution of Engine Size')
fig.show()
fig = px.scatter(cars, x='manufacturer', y='price', color='price')
fig.update_layout(title='Sales Price VS Manufacturer',xaxis_title="Manufacturer",yaxis_title="Price")
fig.show()
fig = px.scatter(cars, x='mileage', y='price', color='price')
fig.update_layout(title='Sales Price VS Mileage',xaxis_title="Mileage",yaxis_title="Price")
fig.show()
fig = px.scatter(cars, x='transmission', y='price', color='price')
fig.update_layout(title='Sales Price VS Transmission System',xaxis_title="Transmission",yaxis_title="Price")
fig.show()
env = setup(cars, target='price', normalize=True, transformation=True, transform_target=True, polynomial_features=True, polynomial_degree=2, sampling=False, silent=True,  session_id=707)
compare_models(blacklist=['svm', 'tr', 'ransac', 'huber', 'lar', 'llar', 'lr'])
rf = create_model('rf')
plot_model(rf, plot='parameter')
plot_model(rf, plot='feature')
rf_final = finalize_model(rf)
save_model(rf_final, 'lrm_rf_model_28072020')