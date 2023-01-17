# imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV





from IPython.display import SVG

from graphviz import Source

from IPython.display import display



import os

print(os.listdir("../input/forest-cover-type-kernels-only"))
# new dataframe onthe training dataset

train = pd.read_csv('../input/forest-cover-type-kernels-only/train.csv')

train.columns
train
train.shape
train.describe()
train[train.columns[1:11]].hist(figsize = (20,15))
cols = train.columns[1:11]

fig, axes = plt.subplots(nrows = 1, ncols = len(cols), figsize = (30,5))

for i, ax in enumerate(axes):

    sns.distplot(train[cols[i]], ax=ax)

    sns.despine()
cols = train.columns[1:11]

fig, axes = plt.subplots(nrows = 1, ncols = len(cols), figsize = (30,5))

for i, ax in enumerate(axes):

    sns.violinplot(data=train, x = "Cover_Type", y = cols[i],ax=ax)

    sns.despine()

plt.tight_layout()
areas_list  = [ 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']

train['Wilderness_Area'] = train.Wilderness_Area1 * 1 + train.Wilderness_Area2 * 2 + train.Wilderness_Area3 * 3 + train.Wilderness_Area4 *4
train['Soil_Type'] = (train.Soil_Type1 * 1 + 

                    train.Soil_Type2 * 2 + 

                    train.Soil_Type3 * 3 + 

                    train.Soil_Type4 * 4 + 

                    train.Soil_Type5 * 5 + 

                    train.Soil_Type6 * 6 + 

                    train.Soil_Type7 * 7 + 

                    train.Soil_Type8 * 8 + 

                    train.Soil_Type9 * 9 + 

                    train.Soil_Type10 * 10 + 

                    train.Soil_Type11 * 11 + 

                    train.Soil_Type12 * 12 + 

                    train.Soil_Type13 * 13 + 

                    train.Soil_Type14 * 14 + 

                    train.Soil_Type15 * 15 + 

                    train.Soil_Type16 * 16 + 

                    train.Soil_Type17 * 17 + 

                    train.Soil_Type18 * 18 + 

                    train.Soil_Type19 * 19 + 

                    train.Soil_Type20 * 20 + 

                    train.Soil_Type21 * 21 + 

                    train.Soil_Type22 * 22 + 

                    train.Soil_Type23 * 23 + 

                    train.Soil_Type24 * 24 + 

                    train.Soil_Type25 * 25 + 

                    train.Soil_Type26 * 26 + 

                    train.Soil_Type27 * 27 + 

                    train.Soil_Type28 * 28 + 

                    train.Soil_Type29 * 29 + 

                    train.Soil_Type30 * 30 + 

                    train.Soil_Type31 * 31 + 

                    train.Soil_Type32 * 32 + 

                    train.Soil_Type33 * 33 + 

                    train.Soil_Type34 * 34 + 

                    train.Soil_Type35 * 35 + 

                    train.Soil_Type36 * 36 + 

                    train.Soil_Type37 * 37 + 

                    train.Soil_Type38 * 38 + 

                    train.Soil_Type39 * 39 + 

                    train.Soil_Type40 * 40)
# this is a useful plot for categorical variables

cols = train.columns[-2:]

fig, axes = plt.subplots(ncols = 1, nrows = len(cols), figsize = (20,10))

for i, ax in enumerate(axes):

    sns.barplot(data=train.groupby(by = [cols[i],"Cover_Type"]).Id.count().reset_index(),

                  x=cols[i], y="Id", hue="Cover_Type", ax=ax)

    sns.despine()

plt.tight_layout()
# Make sure I'm getting the right columns

train.columns[1:-3]
labels = train.columns[1:-3]

y = train.Cover_Type

X = train[labels]
X
estimator = tree.DecisionTreeClassifier(max_depth = 10)

estimator.fit(X, y)



graph = Source(tree.export_graphviz(estimator, out_file=None

   , feature_names= labels, class_names = ['0', '1', '2', '3', '4' ,'5','6']

   , filled = True))



display(SVG(graph.pipe(format='svg')))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



estimator = tree.DecisionTreeClassifier(max_depth = 5)

estimator.fit(X_train, y_train)



#calculate the percent correct

sum(estimator.predict(X_test)==y_test)/len(y_test)
depths = []

performance = []

for depth in range(2,50):

    estimator = tree.DecisionTreeClassifier(max_depth = depth)

    estimator.fit(X_train, y_train)

    correct = sum(estimator.predict(X_test)==y_test)/len(y_test)

    #print('Depth = ',depth,' correct = ',correct)

    depths.append(depth)

    performance.append(correct)

results =  pd.DataFrame()

results['tree_depths'] = depths

results['performance'] = performance

results.plot(x = 'tree_depths', y = 'performance')
depths = []

performance = []

for depth in range(0,50):

    fraction = depth/100

    estimator = tree.DecisionTreeClassifier(min_weight_fraction_leaf = fraction)

    estimator.fit(X_train, y_train)

    correct = sum(estimator.predict(X_test)==y_test)/len(y_test)

    #print('Depth = ',depth,' correct = ',correct)

    depths.append(fraction)

    performance.append(correct)

results =  pd.DataFrame()

results['tree_depths'] = depths

results['performance'] = performance

results.plot(x = 'tree_depths', y = 'performance')
estimator = tree.DecisionTreeClassifier(max_depth = 15)

estimator.fit(X_train, y_train)

#calculate the percent correct

sum(estimator.predict(X_test)==y_test)/len(y_test)
conf_mx = confusion_matrix(y_test, estimator.predict(X_test))



ax = sns.heatmap(conf_mx, annot = True, fmt = 'd')

ax.set(xlabel='Predicted', ylabel='Actual')
row_sums = conf_mx.sum(axis=1, keepdims = True)

norm_conf_mx = conf_mx/row_sums

np.fill_diagonal(norm_conf_mx, 0)

ax = sns.heatmap(norm_conf_mx, annot = True)#, fmt = 'd')

ax.set(xlabel='Predicted', ylabel='Actual')
# test my formula - use np instead of math

for deg in range(0,370,30):

    print (deg, np.sin(deg*np.pi/180),np.cos(deg*np.pi/180))
train['Aspect_N_S'] = np.cos(train.Aspect*np.pi/180)

train['Aspect_E_W'] = np.sin(train.Aspect*np.pi/180)

train[['Aspect', 'Aspect_N_S', 'Aspect_E_W']]
# new column names of interest.  

X_col_names = [train.columns[1]]+train.columns[3:-5].tolist()+train.columns[-2:].tolist()

X_col_names
y = train.Cover_Type

X = train[X_col_names]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



estimator = tree.DecisionTreeClassifier(max_depth = 15)

estimator.fit(X_train, y_train)



#calculate the percent correct

sum(estimator.predict(X_test)==y_test)/len(y_test)
scaler = StandardScaler()





estimator = tree.DecisionTreeClassifier(max_depth = 15)

estimator.fit(scaler.fit_transform(X_train), y_train)



#calculate the percent correct

sum(estimator.predict(scaler.transform(X_test))==y_test)/len(y_test)
# logistic regression code for comparison

estimator = LogisticRegression()

estimator.fit(X_train, y_train)



#calculate the percent correct

print('unscaled = ',sum(estimator.predict(X_test)==y_test)/len(y_test))



# logistic regression code for comparison

estimator = LogisticRegression()

estimator.fit(scaler.fit_transform(X_train), y_train)



#calculate the percent correct

print('scaled = ',sum(estimator.predict(scaler.transform(X_test))==y_test)/len(y_test))
estimator = RandomForestClassifier()

estimator.fit(X_train, y_train)



#calculate the percent correct

sum(estimator.predict(X_test)==y_test)/len(y_test)
pd.DataFrame(index = X_col_names, columns  = ['feature_importance'], data = estimator.feature_importances_).sort_values(ascending = False,by='feature_importance')
# new column names of interest.  

X2_col_names = [train.columns[1]]+train.columns[3:11].tolist()+train.columns[-2:].tolist()

X2_col_names
y2 = train.Cover_Type

X2 = train[X2_col_names]



X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=43)
# logistic regression code for comparison

estimator = LogisticRegression()

estimator.fit(X2_train, y2_train)



#calculate the percent correct

print('unscaled = ',sum(estimator.predict(X2_test)==y2_test)/len(y2_test))



# logistic regression code for comparison

estimator = LogisticRegression()

estimator.fit(scaler.fit_transform(X2_train), y2_train)



#calculate the percent correct

print('scaled = ',sum(estimator.predict(scaler.transform(X2_test))==y2_test)/len(y2_test))
wilderness_area_lookup = {}

for n in range(5):

    binstr =format(n, '03b')

    vals = [int(binstr[i]) for i in range(len(binstr))]

    wilderness_area_lookup[n] = vals

wilderness_area_lookup
# looping is a slow way to do this but it is adequate 

for row in train.index:

    bin_list = wilderness_area_lookup[train.loc[row,'Wilderness_Area']]

    train.loc[row,'Wilderness_Area_bin0'] = bin_list[0]

    train.loc[row,'Wilderness_Area_bin1'] = bin_list[1]

    train.loc[row,'Wilderness_Area_bin2'] = bin_list[2]

train.head()
soil_type_lookup = {}

for n in range(41):

    binstr =format(n, '06b')

    vals = [int(binstr[i]) for i in range(len(binstr))]

    soil_type_lookup[n] = vals

soil_type_lookup
# looping is a slow way to do this but it is adequate 

for row in train.index:

    bin_list = soil_type_lookup[train.loc[row,'Soil_Type']]

    train.loc[row,'Soil_Type_bin0'] = bin_list[0]

    train.loc[row,'Soil_Type_bin1'] = bin_list[1]

    train.loc[row,'Soil_Type_bin2'] = bin_list[2]

    train.loc[row,'Soil_Type_bin3'] = bin_list[3]

    train.loc[row,'Soil_Type_bin4'] = bin_list[4]

    train.loc[row,'Soil_Type_bin5'] = bin_list[5]

train.head()
train.columns
# new column names of interest.  

X_col_names = [train.columns[1]]+train.columns[3:11].tolist()+train.columns[-11:].tolist()

X_col_names
y = train.Cover_Type

X = train[X_col_names]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



estimator = tree.DecisionTreeClassifier(max_depth = 15)

estimator.fit(X_train, y_train)



#calculate the percent correct

sum(estimator.predict(X_test)==y_test)/len(y_test)
estimator = RandomForestClassifier()

estimator.fit(X_train, y_train)



#calculate the percent correct

sum(estimator.predict(X_test)==y_test)/len(y_test)
pd.DataFrame(index = X_col_names, columns  = ['feature_importance'], data = estimator.feature_importances_).sort_values(ascending = False,by='feature_importance')
train['Aspect_N_S_Slope'] = train['Aspect_N_S'] * train['Slope'] 

train['Aspect_E_W_Slope'] = train['Aspect_E_W'] * train['Slope'] 
# new column names of interest.  

X_col_names = [train.columns[1]]+train.columns[3:11].tolist()+train.columns[-13:].tolist()

X_col_names
y = train.Cover_Type

X = train[X_col_names]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)



estimator = tree.DecisionTreeClassifier(max_depth = 15)

estimator.fit(X_train, y_train)



#calculate the percent correct

sum(estimator.predict(X_test)==y_test)/len(y_test)
estimator = RandomForestClassifier()

estimator.fit(X_train, y_train)



#calculate the percent correct

sum(estimator.predict(X_test)==y_test)/len(y_test)
pd.DataFrame(index = X_col_names, columns  = ['feature_importance'], 

             data = estimator.feature_importances_

            ).sort_values(ascending = False,by='feature_importance')
factors = [0,1,2,4,6,8,10,20,100, 1000, 10]

for factor in factors:

    train['Elev_Asp_Slope'] = train['Aspect_N_S_Slope'] * factor +  train['Elevation'] 

    X_col_names = [train.columns[1]]+train.columns[3:11].tolist()+train.columns[-14:].tolist()

    y = train.Cover_Type

    X = train[X_col_names]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

    estimator = RandomForestClassifier()

    estimator.fit(X_train, y_train)

    pct = sum(estimator.predict(X_test)==y_test)/len(y_test)

    print('factor = ',factor, '; percent correct = ', pct) 
pd.DataFrame(index = X_col_names, columns  = ['feature_importance'], data = estimator.feature_importances_).sort_values(ascending = False,by='feature_importance')
# logistic regression code for comparison

estimator = LogisticRegression()

estimator.fit(scaler.fit_transform(X_train), y_train)

pct = sum(estimator.predict(scaler.transform(X_test))==y_test)/len(y_test)

pct
# Cross validation

estimator = RandomForestClassifier()

scores = cross_val_score(estimator, X_train, y_train, cv=10)



pd.Series(scores).describe()
estimator.get_params()
# Number of trees in random forest

n_estimators = [3, 5, 10, 50, 100]

# Number of features to consider at every split

max_features = ['auto', None]

# Maximum number of levels in tree

max_depth = [3, 5, 10, 50, 100, None]

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4, 10]

# Method of selecting samples for training each tree

bootstrap = [True, False]# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

random_search = RandomizedSearchCV(estimator, param_distributions = random_grid, 

                               n_iter = 20, cv = 3, verbose=2, random_state=42)# Fit the random search model

random_search.fit(X_train, y_train)
random_search.best_params_
random_search.best_score_
random_search.best_estimator_
random_search.best_estimator_
# see how it performs on the test set

pct = sum(random_search.predict(X_test)==y_test)/len(y_test)

pct