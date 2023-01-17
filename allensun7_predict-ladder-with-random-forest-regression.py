import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



from sklearn import metrics
df = pd.read_csv('../input/world-happiness-report-2019/world-happiness-report-2019.csv')

df
df.info()
df.isnull().sum()
# replace nan with mean

df.fillna(df.mean(), inplace=True)

df.isnull().sum()
#heatmap using seaborn

#set the context for plotting 

sns.set(context="paper", font="monospace")

#set the matplotlib figure

fig, axe = plt.subplots(figsize=(12,8))

#Generate color palettes 

cmap = sns.diverging_palette(220, 10, center="light", as_cmap=True)

#draw the heatmap

sns.heatmap(df.corr(), vmax=1, square =True, cmap=cmap, annot=True ) 
X = df.drop(['Ladder', 'SD of Ladder', 'Country (region)'], axis=1)

y = df['Ladder']

X.head()
#data split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

print("X_train shape {} and size {}".format(X_train.shape,X_train.size))

print("X_test shape {} and size {}".format(X_test.shape,X_test.size))

print("y_train shape {} and size {}".format(y_train.shape,y_train.size))

print("y_test shape {} and size {}".format(y_test.shape,y_test.size))
#Standardize training and test datasets.

#==============================================================================

# Feature scaling is to bring all the independent variables in a dataset into

# same scale, to avoid any variable dominating  the model. Here we will not 

# transform the dependent variables.

#==============================================================================

independent_scaler = StandardScaler()

X_train = independent_scaler.fit_transform(X_train)

X_test = independent_scaler.transform(X_test)

print(X_train[0:5,:])

print("test data")

print(X_test[0:5,:])
# One Hot Encoding

feature = pd.get_dummies(X)

# List of features for later use

feature_list = list(feature.columns)

features_num = np.size(feature_list)

# Convert to numpy arrays

features = np.array(feature)

print("features numbers: ", features_num)
# model apply

ntree_list = [10, 20, 50, 100, 200, 500, 1000]

mtry_list = [int(0.5*features_num**0.5),

             int(features_num**0.5), int(2*features_num**0.5)]

best_ntree = 0

best_mtry = 0

best_error = 9999999999999

best_model = None

best_y_pred = 0

count = 0

total_models = len(ntree_list) * len(mtry_list)

for ntree in ntree_list:

    for mtry in mtry_list:

        count += 1

        print("Training model %i out of %i..." % (count, total_models))

        print("ntree: %i, mtry: %i" % (ntree, mtry))

        rfg = RandomForestRegressor(n_estimators=ntree,

                                    max_features=mtry,

                                    bootstrap=True,

                                    random_state=0)

        rfg.fit(X_train, y_train)

        # predict the test dataset

        y_pred = rfg.predict(X_test)

        # compute square root error

        error = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        if error < best_error:

            best_ntree = ntree

            best_mtry = mtry

            best_error = error

            best_model = rfg

            print("Found new optimal model")

            print(best_model)

            print("The error of model RFR : %6f" % best_error)

            print("best_ntree: %i, best_mtry: %i" % (best_ntree, best_mtry))

            print(

                "========================================================================")

# print optimal results

print("========================================================================")

print('Finished tuning model')

print('Optimal model')

print(best_model)

print("The error of model RFR : %6f" % best_error)

print("best_ntree: %i, best_mtry: %i" % (best_ntree, best_mtry))
# predict the test dataset

y_pred = best_model.predict(X_test)

test = pd.DataFrame({'Predicted':y_pred, 

                     'Actual':y_test})

fig= plt.figure(figsize=(16,8))

test = test.reset_index()

test = test.drop(['index'],axis=1)

plt.plot(test[:50])

plt.legend(['Actual','Predicted'])

sns.jointplot(x='Actual',y='Predicted',data=test,kind="reg", joint_kws={'line_kws':{'color':'red'}})
# Get numerical feature importances

importances = list(best_model.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse=True)

# Print out the feature and importances 

[print('Variable: {:20}    Importance: {}'.format(*pair)) for pair in feature_importances]
def features(feature_importances):

    # sorted importances of features

    feature_list = [x[0] for x in feature_importances][::-1]

    importances = [x[1] for x in feature_importances][::-1]

    # list of x locations for plotting

    y_values = list(range(len(importances)))

    # Make a bar chart

    plt.barh(y_values, importances, orientation = 'horizontal', color = 'r', edgecolor = 'k', linewidth = 1.2)

    # Tick labels for x axis

    plt.yticks(y_values, feature_list, rotation='horizontal')

    # Axis labels and title

    plt.xlabel('Importance') 

    plt.ylabel('Feature')

    plt.title('Feature Importance')

    plt.show()
# feature importance

features(feature_importances)