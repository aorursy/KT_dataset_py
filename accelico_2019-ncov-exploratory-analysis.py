# Basic libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno



# Visualization libraries

import altair as alt

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



# Plotly visualization

import chart_studio.plotly as py

import plotly.tools as tls





# Any results you write to the current directory are saved as output.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Display markdown formatted output like bold, italic bold etc.

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv") # Loading Day level information on 2019-nCoV affected cases to "data"

# Splits our data into train, valid and test. 10% validation, 10% testing, 80% for training



head = data.head()

tail = data.tail()



conc_data_row = pd.concat([head,tail], axis =0, ignore_index =True)

conc_data_row
# Simple statistics on this dataset

data.describe()
# This is information about the dataset itself

data.info()
# Converting Date and Last Update objects to datetime

data['Last Update'] = data['Last Update'].apply(pd.to_datetime)

data['Date'] = data['Date'].apply(pd.to_datetime)

data.drop(['Sno'],axis=1,inplace=True)

data = data.replace("nan", np.nan)

data.head()
# Creating a data-dense display to visualize patterns in data completion

# As you can see, this dataset is very well put together with complete fields

msno.matrix(data)
bold("**Areas where deaths occurred**")

from datetime import date

data_dates = data[data['Date'] > pd.Timestamp(date(2020,2,11))]

data_deaths = data_dates[data_dates['Deaths'] > 1]

data_deaths



data_deaths.groupby(['Country', 'Province/State']).sum()
from sklearn.model_selection import train_test_split



# Assigning X as our data

X = data.copy()



# Remove rows with missing tartget, seperate target from predictors

X.dropna(axis=0, subset=['Confirmed'], inplace=True)

y = X.Confirmed

X.drop(['Confirmed'], axis=1, inplace=True)



# Create validation set from training data

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                     train_size=0.8, test_size=0.2,

                                                     random_state=0)



# Selecting categorical columns with cardinality in mind

categorical_cols = [cname for cname in X_train.columns if

                   X_train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if

                X_train[cname].dtype in ['int64', 'float64']]



#Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train[my_cols].copy()

X_test = X_test[my_cols].copy()



print(categorical_cols)

print(numerical_cols)

print("")



X_train.head()
bold("**Setting up pipeline and getting scores**")

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score



# preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Run cross_validation since this is a small dataset

# Multiply by -1 since sklearn calculates *negative* MAE

def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                             ('model', RandomForestRegressor(n_estimators, random_state=0))

                             ])

    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,

                                 cv=5,

                                 scoring='neg_mean_absolute_error')

    return scores.mean()



results = {}

for i in range(1,20):

    results[50*i] = get_score(50*i)

    

print(results)
bold("**Visualizing best model to use**")

plt.plot(results.keys(), results.values())

plt.show
bold("**Evaluating the model yet for predicting confirmed cases of 2019-nCoV**")

bold("**The number below outputs our MAE score for the model**")

key_max = max(results.keys(), key=(lambda k: results[k]))

key_min = min(results.keys(), key=(lambda k: results[k]))

print('Train MAE:', get_score(key_min))



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                             ('model', RandomForestRegressor(n_estimators=50, random_state=0))

                             ])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_test)

score = mean_absolute_error(y_test, preds)

print('MAE:', score)