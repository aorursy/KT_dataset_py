# Import libraries

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

from sklearn.metrics import roc_auc_score

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore")
# Load data using pandas

data = pd.read_csv("../input/2016 County Election Data.csv")
# View column names

data.columns
# Dataframe shape

data.shape
# Take a glimpse of the data

data.head()
data.info()
data.describe()
# Distribution of results

print('Trump Won Counties: ', len(data.loc[data['Clinton-lead'] < 0]), ', Mean Trump County Pop.: ', round(data.loc[data['Clinton-lead'] < 0]['Population'].mean(), 1))

print('Clinton Won Counties: ', len(data.loc[data['Clinton-lead'] > 0]), ', Mean Clinton County Pop.: ', round(data.loc[data['Clinton-lead'] > 0]['Population'].mean(), 1))

# Put the election results into bins

data['Results_binned'] = pd.cut(data['Clinton-lead'], bins = np.linspace(-100, 100, num = 7))

data.head()
# Group by the bin and review averages

data.groupby('Results_binned').mean()
data.corr()['Clinton-lead'].sort_values()
data.corr()['Clinton-lead'].abs().sort_values(ascending = False)
# Correlation Heatmap

plt.figure(figsize=(10,10))

mask = np.triu(np.ones_like(data.corr(), dtype=np.bool))

#heatmap = sns.heatmap(data.corr(), mask=mask, cmap=plt.cm.RdBu_r, annot=True)

heatmap = sns.heatmap(data.corr(), mask=mask, cmap=sns.color_palette("RdBu_r", 7), annot=True)

heatmap.set_title('Correlation Heatmap')

plt.show()
# Create histograms for each data column

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (14,14)

data.hist(density=True, edgecolor='k');
# use the column that contains Clinton's margin of victory to create a column for our labels (Trump or Clinton victory)

data.loc[data["Clinton-lead"] > 0, "Label"] = "CLINTON"

data.loc[data["Clinton-lead"] <  0, "Label"] = "TRUMP"



# we no longer need the binned results column and can drop it

data.drop(["Results_binned"], axis=1, inplace=True)
sns.distributions._has_statsmodels = False

num_cols = len(data.columns[1:-1])

plt.figure(figsize=(10, num_cols*8))



for i, col in enumerate(data.columns[1:-2]):

  plt.subplot(num_cols, 1, i + 1)

  sns.kdeplot(data.loc[data['Label'] == 'CLINTON', col], label='Clinton Counties', shade=True)

  sns.kdeplot(data.loc[data['Label'] == 'TRUMP', col], label='Trump Counties', shade = True)

  plt.ylabel('Density')

  plt.xlabel('%s' % col)

  plt.title('Distribution of %s' % col)
# Simple functions we can reuse to return a trained random forest model and to split our data into training and testing datasets

def trained_model(X, y):

  model = RandomForestClassifier(random_state=7)

  model.fit(X, y)

  return model



def split_data(X, y, test_size=.2):

  # Split the data into training and test sets

  return train_test_split(X, y, test_size=test_size, random_state=5)
y = data['Label'].values

# Create a dataframe with the county features (X values). We exclude the county name, Clinton-lead, and Label columns.

features = data[data.columns[1:-2]]
# Train the model using only the features that are in the original data

X_train, X_test, y_train, y_test = split_data(features, y)

model_original_feat = trained_model(X_train, y_train)

print('Accuracy score using original features only: ', round(model_original_feat.score(X_test, y_test) * 100, 2))

print('ROC AUC score using original features: ', round(roc_auc_score(y_test, model_original_feat.predict_proba(X_test)[:, 1]) * 100, 2))
# Create the polynomial object and fit using the original features

poly_transformer = PolynomialFeatures(degree=3, include_bias=False)

poly_transformer.fit(features)

poly_features = poly_transformer.transform(features)

feature_names = poly_transformer.get_feature_names(input_features=features.columns)

poly_features.shape
X_train, X_test, y_train, y_test = split_data(poly_features, y)

model_poly_feat = trained_model(X_train, y_train)

print('Score using polynomial features: ', round(model_poly_feat.score(X_test, y_test) * 100, 2))

print('ROC AUC score using polynomial features: ', round(roc_auc_score(y_test, model_poly_feat.predict_proba(X_test)[:, 1]) * 100, 2))
scaler = MinMaxScaler()



X = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = split_data(X, y)

model_scaled_feat = trained_model(X_train, y_train)

print('Score using MinMaxScaler on original features: ', round(model_scaled_feat.score(X_test, y_test) * 100, 2))

print('ROC AUC score using MinMaxScaler original features: ', round(roc_auc_score(y_test, model_scaled_feat.predict_proba(X_test)[:, 1]) * 100, 2))
def predict_winner(county, model):

    X = data[data['County'] == county][data.columns[1:-2]]

    y = data[data['County'] == county]['Clinton-lead'].values[0]

    prediction  = model.predict(X)[0]

    print(county)

    print("Predicted Winner: {0}\nClinton margin of victory: {1}\n".format(prediction, y))
# Select 15 random counties

random_counties = data.sample(n=15)

random_counties
# Predict the election winner for each county in the counties list

for county in random_counties['County']:

  predict_winner(county, model_original_feat)