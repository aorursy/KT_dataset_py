# Bruno Viera Ribeiro - 09/2020
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px
pd.set_option('display.max_colwidth', None)

desc_df = pd.read_csv('../input/covid19-healthy-diet-dataset/Supply_Food_Data_Descriptions.csv', index_col = 'Categories')

desc_df
kg_df_full = pd.read_csv('../input/covid19-healthy-diet-dataset/Food_Supply_Quantity_kg_Data.csv')

kg_df_full.head()
kg_df_full.columns
kg_df_full.columns.size
# Let's drop the last column as it is just a unit information

kg_df = kg_df_full.drop('Unit (all except Population)', axis = 1)

kg_df.head()
kg_df.isnull().sum()
kg_df.head()
kg_df = kg_df.dropna()
kg_df.info()
kg_df['Undernourished'][:20]
kg_df['Undernourished'][0]
kg_df.loc[kg_df['Undernourished'] == '<2.5', 'Undernourished'] = '2.0'
kg_df['Undernourished'][:20]
kg_df['Undernourished'] = pd.to_numeric(kg_df['Undernourished'])
kg_df.info()
fig = px.scatter(kg_df, x="Confirmed", y = "Deaths",size = "Active", hover_name='Country', log_x=False,

                 size_max=30, trendline = "ols", marginal_x = "box",marginal_y = "violin", template="simple_white")

fig.show()
kg_df.columns
kg_df['Animal Products'] + kg_df['Vegetal Products']
(kg_df['Animal Products'] + kg_df['Vegetal Products']).mean()
kg_df.iloc[:, 1:24].sum(axis=1)
kg_df.iloc[:,1:24] = kg_df.iloc[:, 1:24] * 2
(kg_df['Animal Products'] + kg_df['Vegetal Products']).round(1)
(kg_df['Animal Products'] + kg_df['Vegetal Products']).mean()
(kg_df['Confirmed'] - (kg_df['Deaths'] + kg_df['Recovered'] + kg_df['Active'])).round(2)
kg_df['Mortality'] = kg_df['Deaths']/kg_df['Confirmed']
kg_df['Mortality']
# Distributions

fig = px.bar(kg_df, x = "Country", y ="Confirmed").update_xaxes(categoryorder="total descending")

fig.show()
# Distributions

fig = px.bar(kg_df, x = "Country", y ="Deaths").update_xaxes(categoryorder="total descending")

fig.show()
# Distributions

fig = px.bar(kg_df, x = "Country", y ="Active").update_xaxes(categoryorder="total descending")

fig.show()
# Distributions

fig = px.bar(kg_df, x = "Country", y ="Mortality").update_xaxes(categoryorder="total descending")

fig.show()
kg_df[kg_df.Country == 'Yemen']['Deaths']
fig = px.scatter(kg_df[kg_df.Country != 'Yemen'], x="Mortality", y = "Obesity", size = "Active", hover_name='Country', log_x=False,

                 size_max=30, template="simple_white")



fig.add_shape(

        # Line Horizontal

            type="line",

            x0=0,

            y0=kg_df[kg_df.Country != 'Yemen']['Obesity'].mean(),

            x1=kg_df[kg_df.Country != 'Yemen']['Mortality'].max(),

            y1=kg_df[kg_df.Country != 'Yemen']['Obesity'].mean(),

            line=dict(

                color="crimson",

                width=4

            ),

    )





fig.show()
fig = px.scatter(kg_df, x="Mortality", y = "Obesity", size = "Active", hover_name='Country', log_x=False,

                 size_max=30, template="simple_white")



fig.add_shape(

        # Line Horizontal

            type="line",

            x0=0,

            y0=kg_df['Obesity'].mean(),

            x1=kg_df['Mortality'].max(),

            y1=kg_df['Obesity'].mean(),

            line=dict(

                color="crimson",

                width=4

            ),

    )





fig.show()
fig = px.scatter(kg_df, x="Deaths", y = "Obesity", size = "Mortality",

                 hover_name='Country', log_x=False, size_max=30, template="simple_white")



fig.add_shape(

        # Line Horizontal

            type="line",

            x0=0,

            y0=kg_df['Obesity'].mean(),

            x1=kg_df['Deaths'].max(),

            y1=kg_df['Obesity'].mean(),

            line=dict(

                color="crimson",

                width=4

            ),

    )



fig.show()
kg_df[kg_df.Obesity < kg_df['Obesity'].mean()].shape
kg_df[kg_df.Obesity > kg_df['Obesity'].mean()].shape
df_high_ob = kg_df[kg_df.Obesity > kg_df['Obesity'].mean()]

df_low_ob = kg_df[kg_df.Obesity <= kg_df['Obesity'].mean()]
kg_df['ObesityAboveAvg'] = (kg_df["Obesity"] > kg_df['Obesity'].mean()).astype(int)
fig = px.histogram(kg_df, x = "Animal Products", nbins=50, color = "ObesityAboveAvg", marginal="rug")



fig.add_shape(

        # Mean value of Animal Products intake in low obesity countries

            type="line",

            x0=df_low_ob['Animal Products'].median(),

            y0=0,

            x1=df_low_ob['Animal Products'].median(),

            y1=12,

            line=dict(

                color="darkblue",

                width=4

            ),

    )



fig.add_shape(

        # Mean value of Animal Products intake in high obesity countries

            type="line",

            x0=df_high_ob['Animal Products'].median(),

            y0=0,

            x1=df_high_ob['Animal Products'].median(),

            y1=12,

            line=dict(

                color="crimson",

                width=4

            ),

    )







fig.show()
fig = px.histogram(kg_df, x = "Vegetal Products", nbins=50, color = "ObesityAboveAvg", marginal="rug")



fig.add_shape(

        # Mean value of Vegetal Products intake in low obesity countries

            type="line",

            x0=df_low_ob['Vegetal Products'].median(),

            y0=0,

            x1=df_low_ob['Vegetal Products'].median(),

            y1=12,

            line=dict(

                color="darkblue",

                width=4

            ),

    )



fig.add_shape(

        # Mean value of Vegetal Products intake in high obesity countries

            type="line",

            x0=df_high_ob['Vegetal Products'].median(),

            y0=0,

            x1=df_high_ob['Vegetal Products'].median(),

            y1=12,

            line=dict(

                color="crimson",

                width=4

            ),

    )



fig.show()
fig = px.bar(kg_df, x = "Country", y ="Deaths", facet_col = "ObesityAboveAvg")

fig.update_xaxes(matches=None,categoryorder="total descending")

fig.show()
kg_df.columns
animal_features = ['Animal fats', 'Aquatic Products, Other', 'Eggs', 'Fish, Seafood', 'Meat',

                   'Milk - Excluding Butter', 'Offals']

vegetal_features = ['Alcoholic Beverages', 'Cereals - Excluding Beer', 'Fruits - Excluding Wine', 'Miscellaneous', 'Oilcrops', 'Pulses',

                    'Spices', 'Starchy Roots', 'Stimulants', 'Sugar & Sweeteners', 'Sugar Crops', 'Treenuts',

                    'Vegetable Oils', 'Vegetables']
# Sanity check

kg_df[animal_features + vegetal_features].sum(axis=1).round(2)
df_high_ob.mean()
fig = px.pie(values = df_high_ob[animal_features].mean().tolist(), names = animal_features,

             title='Mean food intake by Animal products groups - High Obesity Countries')

fig.show()
fig = px.pie(values = df_low_ob[animal_features].mean().tolist(), names = animal_features,

             title='Mean food intake by Animal products groups - Low Obesity Countries')

fig.show()
fig = px.pie(values = df_high_ob[vegetal_features].mean().tolist(), names = vegetal_features,

             title='Mean food intake by Vegetal products groups - High Obesity Countries')

fig.show()



fig = px.pie(values = df_low_ob[vegetal_features].mean().tolist(), names = vegetal_features,

             title='Mean food intake by Vegetal products groups - Low Obesity Countries')

fig.show()
fig = px.scatter(kg_df, x = 'Animal Products', y ='Vegetal Products',

                 color='ObesityAboveAvg', hover_name = 'Country')

fig.show()
df_ob = kg_df[animal_features+vegetal_features+['ObesityAboveAvg']]

df_ob.head()
df_ob.describe()
df_ob.corr()
ob_features = df_ob.columns.drop('ObesityAboveAvg')

ob_target = 'ObesityAboveAvg'



print('Model features: ', ob_features)

print('Model target: ', ob_target)
from sklearn.model_selection import train_test_split



train_data, test_data = train_test_split(df_ob, test_size = 0.2, shuffle = True, random_state = 28)
print('Training set shape:', train_data.shape)



print('Class 0 samples in the training set:', sum(train_data[ob_target] == 0))

print('Class 1 samples in the training set:', sum(train_data[ob_target] == 1))



print('Class 0 samples in the test set:', sum(test_data[ob_target] == 0))

print('Class 1 samples in the test set:', sum(test_data[ob_target] == 1))
from sklearn.utils import shuffle



class_0_no = train_data[train_data[ob_target] == 0]

class_1_no = train_data[train_data[ob_target] == 1]



upsampled_class_0_no = class_0_no.sample(n=len(class_1_no), replace=True, random_state=42)



train_data = pd.concat([class_1_no, upsampled_class_0_no])

train_data = shuffle(train_data)
print('Training set shape:', train_data.shape)



print('Class 1 samples in the training set:', sum(train_data[ob_target] == 1))

print('Class 0 samples in the training set:', sum(train_data[ob_target] == 0))
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline



## Defining the pipeline



classifier = Pipeline([

    ('scaler', MinMaxScaler()),

    ('estimator', KNeighborsClassifier(n_neighbors = 3))

])



# Visualize the pipeline

from sklearn import set_config

set_config(display='diagram')

classifier
# Get train data

X_train = train_data[ob_features]

y_train = train_data[ob_target]



# Fit the classifier

classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score



# Using the fitted model to make predicitions on the training set



train_preds = classifier.predict(X_train)



print('Model performance on the train set:')

print(confusion_matrix(y_train, train_preds))

print(classification_report(y_train, train_preds))

print("Train accuracy:", accuracy_score(y_train, train_preds))
from sklearn.metrics import plot_confusion_matrix



disp = plot_confusion_matrix(classifier, X_train, y_train)



disp.ax_.set_title('Confusion matrix for train set');
# Get data to test classifier

X_test = test_data[ob_features]

y_test = test_data[ob_target]



test_preds = classifier.predict(X_test)



print('Model performance on the test set:')

print(confusion_matrix(y_test, test_preds))

print(classification_report(y_test, test_preds))

print("Test accuracy:", accuracy_score(y_test, test_preds))
disp = plot_confusion_matrix(classifier, X_test, y_test)



disp.ax_.set_title('Confusion matrix for test set');
# Setting k values to try on our validation performance

k_values = list(range(1,11))



# Creating a validation set within the train set

sub_train_data, val_data = train_test_split(train_data, test_size = 0.2, shuffle = True, random_state = 28)



# Upsampling to fix imbalance

class_0_no = sub_train_data[sub_train_data[ob_target] == 0]

class_1_no = sub_train_data[sub_train_data[ob_target] == 1]



upsampled_class_0_no = class_0_no.sample(n=len(class_1_no), replace=True, random_state=42)



sub_train_data = pd.concat([class_1_no, upsampled_class_0_no])

sub_train_data = shuffle(sub_train_data, random_state = 28)



# Creating training and validation sets

X_sub_train = sub_train_data[ob_features]

y_sub_train = sub_train_data[ob_target]



X_val = val_data[ob_features]

y_val = val_data[ob_target]
# Searching for best performing K value

for k in k_values:

    classifier = Pipeline([

    ('scaler', MinMaxScaler()),

    ('estimator', KNeighborsClassifier(n_neighbors = k))

    ])

    

    classifier.fit(X_sub_train, y_sub_train)

    val_preds = classifier.predict(X_val)

    print(f"K = {k} -- Test accuracy: {accuracy_score(y_val, val_preds)}")
# Build the classifier

classifier = Pipeline([

    ('scaler', MinMaxScaler()),

    ('estimator', KNeighborsClassifier(n_neighbors = 2))

])



# Fit the classifier

classifier.fit(X_train, y_train)



# Making predictions on test set

test_preds = classifier.predict(X_test)



print('Model performance on the test set:')

print(confusion_matrix(y_test, test_preds))

print(classification_report(y_test, test_preds))

print("Test accuracy:", accuracy_score(y_test, test_preds))



disp = plot_confusion_matrix(classifier, X_test, y_test)



disp.ax_.set_title('Confusion matrix for test set - k = 2');
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



knn = KNeighborsClassifier()



# Creating a dictionary of all values to test

param_grid = {'n_neighbors': np.arange(2,10)}



# Use grid search to test all values

knn_gscv = GridSearchCV(knn, param_grid, cv = 5)



# Fit the model to data

knn_gscv.fit(X_train, y_train)



# Check for best parameter

knn_gscv.best_params_
# Accuracy when at best parameters

knn_gscv.best_score_
kg_df.columns
# df_mort = kg_df[animal_features+vegetal_features+['Obesity','Mortality']]

df_mort = kg_df[kg_df.Country != 'Yemen'][animal_features+vegetal_features+['Obesity','Mortality']]

# df_mort = kg_df[['Animal Products','Vegetal Products','Obesity','Mortality']]



df_mort = shuffle(df_mort)



mort_features = df_mort.columns.drop('Mortality')

mort_target = 'Mortality'



print('Model features: ', mort_features)

print('Model target: ', mort_target)



X = df_mort[mort_features]

y = df_mort[mort_target]
train_data, test_data = train_test_split(df_mort, test_size = 0.2, shuffle = True, random_state = 28)
df = df_mort[['Meat', 'Milk - Excluding Butter', 'Fish, Seafood',

                         'Cereals - Excluding Beer', 'Obesity','Mortality']]

g = sns.PairGrid(df)

g.map(plt.scatter)
df_mort.corr().tail()
df_mort.corr().loc['Mortality'].sort_values()
# Get train data

X_train = train_data[mort_features]

y_train = train_data[mort_target]
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score



## Defining the pipeline



regressor = Pipeline([

    ('scaler', StandardScaler()),

    ('estimator', Ridge(random_state=28))

])



# Visualize the pipeline

from sklearn import set_config

set_config(display='diagram')

regressor
# Training

regressor.fit(X_train, y_train)
# Scoring the training set



train_preds = regressor.predict(X_train)

regressor.score(X_train, y_train)
# Cross validate

cv_score = cross_val_score(regressor, X_train, y_train, cv = 10)

print(cv_score)

print(cv_score.mean())
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Create function to evaluate model on a few different scores

def show_scores(model, X_train, X_test, y_train, y_test):    

    train_preds = model.predict(X_train)

    test_preds = model.predict(X_test)

    scores = {'Training MAE': mean_absolute_error(y_train, train_preds),

              'Test MAE': mean_absolute_error(y_test, test_preds),

              'Training MSE': mean_squared_error(y_train, train_preds),

              'Test MSE': mean_squared_error(y_test, test_preds),

              'Training R^2': r2_score(y_train, train_preds),

              'Test R^2': r2_score(y_test, test_preds)}

    return scores
# Get data to test model

X_test = test_data[mort_features]

y_test = test_data[mort_target]



show_scores(regressor, X_train, X_test , y_train, y_test)
test_plot = X_test.copy()

test_plot['Mortality'] = y_test

test_plot['Mortality_pred'] = regressor.predict(X_test)



test_plot.head()
# fig = px.scatter(test_plot, x = 'Animal fats', y = ['Mortality','Mortality_pred'],

#                  trendline = "ols")





# fig.show()
fig, ax = plt.subplots(figsize=[10,8])



sns.regplot(x = 'Animal fats', y = 'Mortality', data = test_plot, ax = ax, label='Mortality')

sns.regplot(x = 'Animal fats', y = 'Mortality_pred', data = test_plot, ax = ax, label='Mortality_pred')



plt.legend();
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor



# First, we create a dict with our desired models

models = {'Ridge':Ridge(random_state=28),

          'SVR':SVR(),

          'RandomForest':RandomForestRegressor(),

          'XGBoost':XGBRegressor(n_estimators = 1000, learning_rate = 0.05)}



# Now to build the function that tests each model

def model_build(model, X_train, y_train, X_test, y_test, scale=True):

    

    if scale:

        regressor = Pipeline([

            ('scaler', StandardScaler()),

            ('estimator', model)

        ])

    

    else:

        regressor = Pipeline([

            ('estimator', model)

        ])



    # Training

    regressor.fit(X_train, y_train)



    # Scoring the training set



    train_preds = regressor.predict(X_train)

    print(f"R2 on single split: {regressor.score(X_train, y_train)}")



    # Cross validate

    cv_score = cross_val_score(regressor, X_train, y_train, cv = 10)



    print(f"Cross validate R2 score: {cv_score.mean()}")



    # Scoring the test set

    for k, v in show_scores(regressor, X_train, X_test , y_train, y_test).items():

        print("     ", k, v)
for name, model in models.items():

    print(f"==== Scoring {name} model====")

    

    if name == 'RandomForest' or name == 'XGBoost':

        model_build(model, X_train, y_train, X_test, y_test, scale=False)

    else:

        model_build(model, X_train, y_train, X_test, y_test,)

    print()

    print(40*"=")

        
xgb = XGBRegressor()



parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:squarederror'],

              'learning_rate': [.03, 0.05, .07], #so called `eta` value

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500, 1000]}
# from sklearn.model_selection import GridSearchCV



# xgb_grid = GridSearchCV(xgb, parameters, cv = 5, n_jobs = 4, verbose = True)



# xgb_grid.fit(X_train, y_train)



# print(xgb_grid.best_score_)

# print(xgb_grid.best_params_)



## RAN AND GOT THE PARAMETERS USED BELLOW
xgb_best = XGBRegressor(colsample_bytree = 0.7,

                        learning_rate = 0.05,

                        max_depth = 6,

                        min_child_weight = 4,

                        n_estimators = 500,

                        nthread = 4,

                        objective = 'reg:squarederror',

                        subsample = 0.7)
model_build(xgb_best, X_train, y_train, X_test, y_test, scale=False)
df_mort2 = kg_df[kg_df.Country != 'Yemen'][['Animal Products','Vegetal Products','Obesity','Mortality']]





df_mort2 = shuffle(df_mort2)



mort2_features = df_mort2.columns.drop('Mortality')

mort2_target = 'Mortality'



print('Model features: ', mort2_features)

print('Model target: ', mort2_target)



X = df_mort2[mort2_features]

y = df_mort2[mort2_target]
df_mort2.head()
dummie = df_mort2.copy()

dummie['Mortality'] = dummie['Mortality']*1000





plt.figure(figsize=(10,10))

sns.boxplot(data = dummie, palette = 'rainbow');
train_data, test_data = train_test_split(df_mort2, test_size = 0.2, shuffle = True, random_state = 28)



# Get train data

X_train = train_data[mort2_features]

y_train = train_data[mort2_target]



# Get data to test model

X_test = test_data[mort2_features]

y_test = test_data[mort2_target]
# First, we create a dict with our desired models

models = {'Ridge':Ridge(random_state=28),

          'SVR':SVR(),

          'RandomForest':RandomForestRegressor(),

          'XGBoost':XGBRegressor(n_estimators = 1000, learning_rate = 0.05)}
for name, model in models.items():

    print(f"==== Scoring {name} model====")

    

    if name == 'RandomForest' or name == 'XGBoost':

        model_build(model, X_train, y_train, X_test, y_test, scale=False)

    else:

        model_build(model, X_train, y_train, X_test, y_test,)

    print()

    print(40*"=")
model = RandomForestRegressor()
model.fit(X_train, y_train)
test_preds = model.predict(X_test)



test_plot = X_test.copy()

test_plot['Mortality'] = y_test

test_plot['Mortality_pred'] = test_preds



test_plot.head()
def plotTest(col, target, data):

    fig, ax = plt.subplots(figsize=[10,8])



    sns.regplot(x = col, y = target, data = data, ax = ax, label=target)

    sns.regplot(x = col, y = target+'_pred', data = data, ax = ax, label=target+'_pred')



    plt.legend();
plotTest('Animal Products', 'Mortality', test_plot)
plotTest('Vegetal Products', 'Mortality', test_plot)
plotTest('Obesity', 'Mortality', test_plot)
df_obes = kg_df[animal_features+vegetal_features+['Obesity']]



df_obes = shuffle(df_obes)



obes_features = df_obes.columns.drop('Obesity')

obes_target = 'Obesity'



print('Model features: ', obes_features)

print('Model target: ', obes_target)



X = df_obes[obes_features]

y = df_obes[obes_target]
df_obes.corr().loc['Obesity'].sort_values()
train_data, test_data = train_test_split(df_obes, test_size = 0.2, shuffle = True, random_state = 28)



# Get train data

X_train = train_data[obes_features]

y_train = train_data[obes_target]



# Get data to test model

X_test = test_data[obes_features]

y_test = test_data[obes_target]
# First, we create a dict with our desired models

models = {'Ridge':Ridge(random_state=28),

          'SVR':SVR(),

          'RandomForest':RandomForestRegressor(),

          'XGBoost':XGBRegressor(n_estimators = 1000, learning_rate = 0.05)}
for name, model in models.items():

    print(f"==== Scoring {name} model====")

    

    if name == 'RandomForest' or name == 'XGBoost':

        model_build(model, X_train, y_train, X_test, y_test, scale=False)

    else:

        model_build(model, X_train, y_train, X_test, y_test,)

    print()

    print(40*"=")
model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)
model.fit(X_train, y_train)
test_preds = model.predict(X_test)



test_plot = X_test.copy()

test_plot['Obesity'] = y_test

test_plot['Obesity_pred'] = test_preds



test_plot.head()
# def plotTest(col, target, data):

#     fig, ax = plt.subplots(figsize=[10,8])



#     sns.regplot(x = col, y = target, data = data, ax = ax, label=target)

#     sns.regplot(x = col, y = target+'_pred', data = data, ax = ax, label=target+'_pred')



#     plt.legend();
plotTest('Cereals - Excluding Beer', 'Obesity', test_plot)
plotTest('Meat', 'Obesity', test_plot)
X = kg_df[kg_df.Country != 'Yemen'][['Obesity', 'Mortality']]



X.head()
scaler = StandardScaler()



# Fit the scaler

scaler.fit(X)
# Transform our data

X_scaled = scaler.transform(X)



# Sanity checks

print(X_scaled.mean(axis = 0))



print(X_scaled.std(axis=0))
from sklearn.cluster import KMeans



# Instantiate the model

kmeans = KMeans(n_clusters = 3)



# Fit the model

kmeans.fit(X_scaled)



# Make predictions

preds = kmeans.predict(X_scaled)



print(preds)
# Amount of countries in each cluster



unique_countries, counts_countries = np.unique(preds, return_counts=True)

print(unique_countries)

print(counts_countries)
df_vis = kg_df[kg_df.Country != 'Yemen'].copy()

df_vis['cluster'] = [str(i) for i in preds]



df_vis.head()
fig = px.scatter(df_vis, x = 'Mortality', y = 'Obesity', color = 'cluster', hover_name = 'Country')

fig.show()
# Calculate inertia for a range of clusters number

inertia = []



for i in np.arange(1,11):

    km = KMeans(n_clusters = i)

    km.fit(X_scaled)

    inertia.append(km.inertia_)

    

# Plotting

plt.plot(np.arange(1,11), inertia, marker = 'o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.grid()

plt.show();
def cluster_preds(df, feat1, feat2, k):

    X = df[[feat1, feat2]]



    # Scaling

    scaler = StandardScaler()



    # Fit the scaler

    scaler.fit(X)



    # Transform our data

    X_scaled = scaler.transform(X)



    # Instantiate the model

    kmeans = KMeans(n_clusters = k)



    # Fit the model

    kmeans.fit(X_scaled)



    # Make predictions

    preds = kmeans.predict(X_scaled)



    # Visualizing

    df_vis = df.copy()

    df_vis['cluster'] = [str(i) for i in preds]



    fig = px.scatter(df_vis, x = feat1, y = feat2, color = 'cluster', hover_name = 'Country')

    fig.show()
cluster_preds(kg_df, 'Animal Products', 'Obesity', 3)
cluster_preds(kg_df, 'Confirmed', 'Deaths', 3)