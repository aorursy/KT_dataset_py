# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



data = pd.read_csv("/kaggle/input/dsia19-california-housing/housing.csv")
import seaborn as sns

from matplotlib import pyplot as plt
data.head(3)
data.describe()
data.info()
data.isna().any()
data.corr()
sns.pairplot(data.select_dtypes(include=["number"]).dropna())
sns.clustermap(data.select_dtypes(include=["number"]).dropna().corr())
data.hist(bins=50, figsize=(20,8))

plt.show()
import matplotlib.image as mpimg

import matplotlib.cm as cm



data_frame_plot = data[[

    'longitude',

    'population',

    'latitude',

    'median_house_value'

]]



plt.figure(figsize=(10,10))



plt.scatter(

    x=data_frame_plot.longitude,

    y=data_frame_plot.latitude,

    s=data_frame_plot.population/109,

    c=data_frame_plot.median_house_value,

    alpha=0.4,

    cmap='plasma',

    marker="o",

)



plt.imshow(

    plt.imread('/kaggle/input/dsia19-california-housing/california.png'),

    extent=[-124.55, -113.80, 32.45, 42.05],

    alpha=0.9

)



plt.tight_layout()

plt.show()
from sklearn.model_selection import train_test_split
# trennen in input und output Features

input_feature_names = [

    'longitude',

    'latitude',

    'housing_median_age',

    'total_rooms',

    'total_bedrooms',

    'population',

    'households',

    'median_income',

    'ocean_proximity'

]



output_feature_names = ["median_house_value"]



# Datenset horizontal in input/output features splitten

X = data[input_feature_names]

y = data[output_feature_names]
# trennen in nummerische und nicht-numerische Features

numeric_feature_names = list(X.select_dtypes(include=['number']).columns)

nonnumeric_feature_names = list(X.select_dtypes(exclude=['number']).columns)



# train/test Daten aufteilen

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.3,

    random_state=1239

)
data.isna().any()
data.total_bedrooms.isna().sum()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.pipeline import Pipeline, FeatureUnion
class FeatureSelector:

    

    """This transformer lets you pick columns from a pandas dataset based on name

    

    :param features: List of feature names to select

    :type features: List of strings

    :param debug: Switch to send output to debug console

    :type debug: Boolean

    :raises: ValueError if features is not of type list

    """

    

    def __init__(self, features=[]):

        

        if type(features) != list:

            raise ValueError("Input features must be of type List.")

        

        self.c = features



    def fit(self, X, y=None):

        """This method passes-on the object as no fitting is required

        

        :param X: Input matrix

        :type X: Numpy matrix

        :param y: Output vector

        :type y: Numpy array

        :returns: self

        """

        return self



    def transform(self, X):

        """This method transforms the input data by selecting the features

        

        :param X: Input matrix

        :type X: Numpy matrix

        :param y: Output vector

        :type y: Numpy array

        :returns: Selected colums as Numpy matrix

        """

        return X[self.c]
X_pipeline = Pipeline([

    ("union", FeatureUnion([

        ("numeric", Pipeline([

            ("select_numeric_features", FeatureSelector(features=numeric_feature_names)),

            ("replacing_missing_values", SimpleImputer(strategy="mean")),

            ("scale_values", MinMaxScaler())

        ])),

        ("non-numeric", Pipeline([

            ("select_non-numeric_features", FeatureSelector(features=nonnumeric_feature_names)),

            ("replacing_missing_values", SimpleImputer(strategy="constant", fill_value="missing")),

            ("encode_values", OneHotEncoder())

        ]))

    ]))

])
y_pipeline = Pipeline([

    ("scale", MinMaxScaler())

])
X_pipeline.fit(X_train)

X_train_processed = X_pipeline.transform(X_train)

X_test_processed = X_pipeline.transform(X_test)
y_pipeline.fit(y_train)

y_train_processed = y_pipeline.transform(y_train)

y_test_processed = y_pipeline.transform(y_test)
from sklearn.linear_model import LinearRegression



lr_model = LinearRegression()

lr_model.fit(X_train_processed, y_train_processed)

lr_model.score(X_test_processed, y_test_processed)
from sklearn.linear_model import Ridge



rr_model = Ridge()

rr_model.fit(X_train_processed, y_train_processed)

rr_model.score(X_test_processed, y_test_processed)
from sklearn.tree import DecisionTreeRegressor



dt_model = DecisionTreeRegressor()

dt_model.fit(X_train_processed, y_train_processed)

dt_model.score(X_test_processed, y_test_processed)
from sklearn.ensemble import RandomForestRegressor



rf_model = DecisionTreeRegressor()

rf_model.fit(X_train_processed, y_train_processed)

rf_model.score(X_test_processed, y_test_processed)
from bokeh.io import show, output_notebook

from bokeh.plotting import figure

from bokeh.transform import linear_cmap

from bokeh.palettes import Spectral6



output_notebook()



def plot_predictions_bokeh(model, X_test_processed, y_test, y_pipeline):



    prediction = model.predict(X_test_processed)

    prediction_rev = y_pipeline.inverse_transform(prediction.reshape(-1,1))

    

    red_color = '#d5042a'

    orange_color = '#ED7D31'

    blue_color = '#43bed8'

    lightgreen_color = '#98c235'

    darkgreen_color = '#0b8f6a'

    darkblue_color = '#0062A7'

    lightblue_color = '#4DBED3'



    r_min = y_test.min()[0]

    r_max = y_test.max()[0]



    plot = figure(

        title="Prediction accuracy",

        x_axis_label="actual",

        y_axis_label="prdiction",

        x_range=[r_min, r_max],

        y_range=[r_min, r_max]

    )





    plot.circle(

        y=prediction_rev.ravel(), 

        x=y_test.values.ravel(),

        alpha=0.2,

        color=lightgreen_color

    )



    plot.line(

        x=[r_min,r_max], 

        y=[r_min,r_max],

        color=red_color

    )



    show(plot)
from sklearn.metrics import mean_squared_error



def plot_predictions_multi(models, X_test_processed, y_test, y_pipeline):

    

    """Method plots prediction accuracy of passed models

    

    :param models: List of fitted models

    :param X_test_processed: Processed input data

    :param y_test: Unprocessed output data matching input data

    :param y_pipeline: y Pipeline for inverse transformation

    :type models: List of objects

    :type X_test_processed: Input matrix

    :type y_test: Output vector

    :type y_pipeline: Fitted pipeline object

    """

    

    fig, ax = plt.subplots(1, len(models), figsize=(20,5))

    i = 0

    

    for model in models:



        prediction = model.predict(X_test_processed)

        prediction_rev = y_pipeline.inverse_transform(

            prediction.reshape(-1,1)

        )

        

        me = round(

            (mean_squared_error(

                y_test, prediction_rev

            ))**0.5, 2)



        ax[i].scatter(

            x=y_test,

            y=prediction_rev,

            color="k",

            alpha=0.3,

            label=f"prediction with me={me}"

        )

        

        ax[i].plot(

            [

                y_test.min()[0],

                y_test.max()[0]

            ], 

            [

                y_test.min()[0],

                y_test.max()[0]

            ],

            color="red",

            ls="-", 

            lw=4

        )

    

        ax[i].set_xlabel("acutal data")

        ax[i].set_ylabel("predicted data")

        ax[i].legend(loc=0)

        

        i += 1
plot_predictions_bokeh(

    model=rf_model,

    X_test_processed=X_test_processed,

    y_test=y_test,

    y_pipeline=y_pipeline

)
plot_predictions_multi(

    models=[rf_model, lr_model, dt_model],

    X_test_processed=X_test_processed,

    y_test=y_test,

    y_pipeline=y_pipeline

)
from dill import dump



output_file = "lr_model.pk"



# open the dedicated output file in write-binary mode

with open(output_file, "wb") as dump_file:

    dump(lr_model, dump_file)
output_file = "X_pipeline.pk"



# open the dedicated output file in write-binary mode

with open(output_file, "wb") as dump_file:

    dump(X_pipeline, dump_file)
output_file = "y_pipeline.pk"



# open the dedicated output file in write-binary mode

with open(output_file, "wb") as dump_file:

    dump(y_pipeline, dump_file)