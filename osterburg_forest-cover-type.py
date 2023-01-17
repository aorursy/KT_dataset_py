import pandas as pd

import numpy as np

import math



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score



from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from sklearn.metrics import roc_curve, auc



from sklearn.neighbors import KNeighborsClassifier



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.metrics import confusion_matrix



import warnings

warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/covtype.csv')

df.head()
print('Features: {} \nObservations: {}'.format(df.shape[1], df.shape[0]))
print(df.columns)
print(df.info())
df.isnull().values.any()
# Handling Duplicates

df.drop_duplicates(keep='first')

df.shape
# Create different datasets by type and area

cont_df = df.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points']

cat_df  = df.loc[:,'Wilderness_Area1':'Soil_Type40']

wild_df = df.loc[:,'Wilderness_Area1': 'Wilderness_Area4']

soil_df = df.loc[:,'Soil_Type1':'Soil_Type40']

target  = df['Cover_Type']
# pick number of columns

ncol = 2

# make sure enough subplots

nrow = math.floor((len(cont_df.columns) + ncol - 1) / ncol)

# create the axes

height = 6 * nrow

fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))



# go over a linear list of data

for i, col in enumerate(cont_df.columns):

    # compute an appropriate index (1d or 2d)

    ix = np.unravel_index(i, ax.shape) 



    sns.distplot(cont_df[col], ax=ax[ix])



plt.tight_layout()

plt.show();
cont_df.describe()
print(cont_df.skew())
skew_df = pd.DataFrame(cont_df.skew(), columns=['Skewness'])



sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(16, 8))

sns.barplot(data=skew_df, x=skew_df.index, y='Skewness')

plt.xticks(rotation=90)

plt.show();
for c in wild_df.columns:

    print('{}: {}'.format(c, wild_df[c].value_counts()[1]))
tmpList = []

for c in wild_df.columns:

    tmpList += [str(c)] * wild_df[c].value_counts()[1]



se = pd.Series(tmpList)

df['Wilderness_Types'] = se.values
plt.figure(figsize=(16, 8))

sns.countplot(data=df, x='Wilderness_Types', hue='Cover_Type')

plt.show();
for c in soil_df.columns:

    print('{}: {}'.format(c, soil_df[c].value_counts()[1]))
tmpList = []

for c in soil_df.columns:

    tmpList += [str(c)] * soil_df[c].value_counts()[1]



se = pd.Series(tmpList)

df['Soil_Types'] = se.values
plt.figure(figsize=(16, 8))

sns.countplot(data=df, x='Soil_Types', hue='Cover_Type')

plt.title('Number of Observation by Cover Type')

plt.xticks(rotation=90)

plt.show();
soil_df['Soil_Type29'].describe()
soil_df['Soil_Type29'].value_counts()
# sum Soil data values, and pass it as a series 

soil_sum = pd.Series(soil_df.sum())



# will sort values in descending order

soil_sum.sort_values(ascending = False, inplace = True)



# plot horizontal bar with given size using color defined

soil_sum.plot(kind='barh', figsize=(16, 12))



# horizontal bar flips columns in ascending order, this will filp it back in descending order

plt.gca().invert_yaxis()



plt.title('No. of observations of Soil Types')

plt.xlabel('No.of Observation')

plt.ylabel('Soil Types')



plt.xticks(rotation = 'horizontal')

plt.show();
soil_sum
print(cat_df.skew())
skew_df = pd.DataFrame(cat_df.skew(), columns=['Skewness'])



sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(16, 8))

sns.barplot(data=skew_df, x=skew_df.index, y='Skewness')

plt.xticks(rotation=90)

plt.show();
# pick number of columns

ncol = 4

# make sure enough subplots

nrow = math.floor((len(cat_df.columns) + ncol - 1) / ncol)

# create the axes

height = 4 * nrow

fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))



# go over a linear list of data

for i, col in enumerate(cat_df.columns):

    # compute an appropriate index (1d or 2d)

    ix = np.unravel_index(i, ax.shape) 



    sns.distplot(cat_df[col], ax=ax[ix])



plt.tight_layout()

plt.show();
covertype_df = pd.DataFrame(df.groupby('Cover_Type').size(), columns=['Size'])



plt.figure(figsize=(16, 8))

sns.barplot(data=covertype_df, x=covertype_df.index, y='Size')

plt.show();
# grouping by forest cover type and calculate the total occurance

df.groupby('Cover_Type').size()
t = df.groupby('Cover_Type').size()

print('Cover_Type 1 and 2 in percent: {:.2f}%'.format((t.values[0] + t.values[1]) / (df.shape[0] / 100)))
# Box and whiskers plot

# Spread of numerical features



sns.set_style("whitegrid")

plt.subplots(figsize=(16, 12))



# Using seaborn to plot it horizontally

sns.boxplot(data=cont_df, orient='h', palette='pastel')



plt.title('Spread of data in Numerical Features')

plt.xlabel('Observation Distribution')

plt.ylabel('Features')

plt.show();
print('Elevation min/max: {} - {} meters'.format(df['Elevation'].min(), df['Elevation'].max()))
# pick number of columns

ncol = 2

# make sure enough subplots

nrow = math.floor((len(cont_df.columns) + ncol - 1) / ncol)

# create the axes

height = 6 * nrow

fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))



# go over a linear list of data

for i, col in enumerate(cont_df.columns):

    # compute an appropriate index (1d or 2d)

    ix = np.unravel_index(i, ax.shape) 



    sns.violinplot(data=df, x=df['Cover_Type'], y=col, ax=ax[ix], palette="coolwarm")



plt.tight_layout()

plt.show();
# pick number of columns

ncol = 2

# make sure enough subplots

nrow = math.floor((len(wild_df.columns) + ncol - 1) / ncol)

# create the axes

height = 6 * nrow

fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))



# go over a linear list of data

for i, col in enumerate(wild_df.columns):

    # compute an appropriate index (1d or 2d)

    ix = np.unravel_index(i, ax.shape) 



    sns.violinplot(data=df, x=df['Cover_Type'], y=col, ax=ax[ix], palette="coolwarm")



plt.tight_layout()

plt.show();
# pick number of columns

ncol = 2

# make sure enough subplots

nrow = math.floor((len(soil_df.columns) + ncol - 1) / ncol)

# create the axes

height = 6 * nrow

fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))



# go over a linear list of data

for i, col in enumerate(soil_df.columns):

    # compute an appropriate index (1d or 2d)

    ix = np.unravel_index(i, ax.shape) 



    sns.violinplot(data=df, x=df['Cover_Type'], y=col, ax=ax[ix], palette="coolwarm")



plt.tight_layout()

plt.show();
# Set the style of the visualization

sns.set_style('white')



# Create a convariance matrix

corr = cont_df.corr()



# Generate a mask the size of our covariance matrix

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = None



# Set up the matplotlib figure

fig, ax = plt.subplots(figsize=(16, 12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(240, 10, sep=20, n=9, as_cmap=True)



# Draw the heatmapwith the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, mask=mask, square=True, annot=True)



plt.show();
corr_list = [['Hillshade_3pm', 'Aspect', ], 

             ['Hillshade_9am', 'Aspect', ], 

             ['Hillshade_3pm', 'Hillshade_9am'], 

             ['Hillshade_3pm', 'Hillshade_Noon'],

             ['Hillshade_Noon', 'Slope'],  

             ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']]
%%time

# pick number of columns

ncol = 2

# make sure enough subplots

nrow = math.floor((len(corr_list) + ncol - 1) / ncol)

# create the axes

height = 10 * nrow

fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))



k=0

for i, j in corr_list:

    # compute an appropriate index (1d or 2d)

    ix = np.unravel_index(k, ax.shape) 

    

    sns.scatterplot(data=df, x = i, y = j, hue="Cover_Type", ax=ax[ix], 

                    legend = 'full', palette='coolwarm')



    k += 1



plt.tight_layout()

plt.show();
#Load the data

cm_df = pd.read_csv('../input/covtype.csv')

# cm_df.drop('Id', axis=1, inplace=True)



#Define appropriate X and y

y = cm_df['Cover_Type']

X = cm_df.drop('Cover_Type', axis=1)



#Normalize the Data

for col in cm_df.columns:

    cm_df[col] = (cm_df[col]-min(cm_df[col]))/ (max(cm_df[col]) - min(cm_df[col]))



# Split the data into train and test sets.

X_cm_train, X_cm_test, y_cm_train, y_cm_test = train_test_split(X, y, random_state=0)



#Fit a model

classifier = KNeighborsClassifier(weights='distance', n_jobs=-1)

y_pred = classifier.fit(X_cm_train, y_cm_train).predict(X_cm_test)



#Create confusion matrix

cnf_matrix = confusion_matrix(y_pred, y_cm_test)
# def plot_confusion_matrix(cm, classes,

#                           normalize=False,

#                           title='Confusion matrix',

#                           cmap=plt.cm.Blues):

#     """

#     This function prints and plots the confusion matrix.

#     Normalization can be applied by setting `normalize=True`.

#     """

#     if normalize:

#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#         print("Normalized confusion matrix")

#     else:

#         print('Confusion matrix, without normalization')



#     print(cm)



#     plt.imshow(cm, interpolation='nearest', cmap=cmap)

#     plt.title(title)

#     plt.colorbar()

#     tick_marks = np.arange(len(classes))

#     plt.xticks(tick_marks, classes, rotation=45)

#     plt.yticks(tick_marks, classes)



#     fmt = '.2f' if normalize else 'd'

#     thresh = cm.max() / 2.

#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

#         plt.text(j, i, format(cm[i, j], fmt),

#                  horizontalalignment="center",

#                  color="white" if cm[i, j] > thresh else "black")



#     plt.ylabel('True label')

#     plt.xlabel('Predicted label')

#     plt.tight_layout()





# # Compute confusion matrix

# cnf_matrix = confusion_matrix(y_cm_test, y_pred)

# np.set_printoptions(precision=2)



# # Plot non-normalized confusion matrix

# plt.figure(figsize=(12, 12))

# plot_confusion_matrix(cnf_matrix, classes=class_names,

#                       title='Confusion matrix, without normalization')



# # Plot normalized confusion matrix

# plt.figure(figsize=(12, 12))

# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

#                       title='Normalized confusion matrix')



# plt.show();



#----------------------------------------------

# Can be replaced by pandas_ml
wild_df.sum().sum() == df.shape[0]
soil_df.sum().sum() == df.shape[0]
# Drop added features first

df.drop(['Wilderness_Types', 'Soil_Types'], axis=1, inplace=True)
# Create a target (y) and feature (X) set

y = df['Cover_Type']

X = df.drop('Cover_Type', axis=1)
# Create an empty dataframe to hold our findings for feature_importances_

ranking_df = pd.DataFrame()
%%time

RFC_model = RandomForestClassifier(random_state=0, n_jobs=-1)

RFC_model.fit(X, y)



importances = RFC_model.feature_importances_

indices = np.argsort(importances)[::-1]



# Get feature name

rfc_list = [X.columns[indices[f]] for f in range(X.shape[1])]

ranking_df['RFC'] = rfc_list



# Get feature importance

rfci_list = [importances[indices[f]] for f in range(X.shape[1])]

ranking_df['RFC importance'] = rfci_list
%%time

ABC_model = AdaBoostClassifier(random_state=0)

ABC_model.fit(X, y)



importances = ABC_model.feature_importances_

indices = np.argsort(importances)[::-1]



abc_list = [X.columns[indices[f]] for f in range(X.shape[1])]

ranking_df['ABC'] = abc_list



abci_list = [importances[indices[f]] for f in range(X.shape[1])]

ranking_df['ABC importance'] = abci_list
%%time

GBC_model = GradientBoostingClassifier(random_state=0)

GBC_model.fit(X, y)



importances = GBC_model.feature_importances_

indices = np.argsort(importances)[::-1]



gbc_list = [X.columns[indices[f]] for f in range(X.shape[1])]

ranking_df['GBC'] = gbc_list



gbci_list = [importances[indices[f]] for f in range(X.shape[1])]

ranking_df['GBC importance'] = gbci_list
%%time

ETC_model = ExtraTreesClassifier(random_state=0, n_jobs=-1)

ETC_model.fit(X, y)



importances = ETC_model.feature_importances_

indices = np.argsort(importances)[::-1]



etc_list = [X.columns[indices[f]] for f in range(X.shape[1])]

ranking_df['ETC'] = etc_list



etci_list = [importances[indices[f]] for f in range(X.shape[1])]

ranking_df['ETC importance'] = etci_list
ranking_df.head(25)
ranking_df[['RFC','ETC']].head(25)
sample_df = df[['Elevation', 

                'Aspect', 

                'Slope', 

                'Horizontal_Distance_To_Hydrology',

                'Vertical_Distance_To_Hydrology', 

                'Horizontal_Distance_To_Roadways',

                'Horizontal_Distance_To_Fire_Points', 

                'Hillshade_9am', 

                'Hillshade_Noon',

                'Hillshade_3pm', 

                'Wilderness_Area1', 

                'Wilderness_Area3', 

                'Wilderness_Area4', 

                'Soil_Type2',

                'Soil_Type4', 

                'Soil_Type10', 

                'Soil_Type22', 

                'Soil_Type23', 

                'Soil_Type29',

                'Soil_Type39', 

                'Cover_Type']]
y = df['Cover_Type']

X = sample_df.drop('Cover_Type', axis=1)



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

print(X_train.shape, X_test.shape)
%%time

clf = KNeighborsClassifier(weights='distance', n_jobs=-1)

clf.fit(X_train, y_train)
%%time

accuracy = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy', n_jobs=-1)

f1_score = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'f1_macro', n_jobs=-1)



acc_mean = np.round(accuracy.mean() * 100, 2)

f1_mean = np.round(f1_score.mean() * 100, 2)

    

print('accuracy: {}%'.format(acc_mean))

print('f1_score: {}%'.format(f1_mean))
y = df['Cover_Type']

X = df.drop('Cover_Type', axis=1)



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
pd.DataFrame(data=X_scaled, columns=X.columns).head()
%%time

pca = PCA(n_components=20)

X_pca = pca.fit_transform(X_scaled)
pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',

      'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20']



X_pca = pd.DataFrame(data=X_pca, columns=pc)

X_pca.head()
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=0)



print(X_train.shape, X_test.shape)
%%time

clf = KNeighborsClassifier(weights='distance', n_jobs=-1)

clf.fit(X_train, y_train)
%%time

accuracy = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy', n_jobs=-1)

f1_score = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'f1_macro', n_jobs=-1)
acc_mean = np.round(accuracy.mean() * 100, 2)

f1_mean = np.round(f1_score.mean() * 100, 2)

    

print('accuracy: {}%'.format(acc_mean))

print('f1_score: {}%'.format(f1_mean))
predict = clf.predict(X_test)
# calculating accuracy

accuracy = accuracy_score(y_test, predict)



print('KNeighbors Classifier model')

print('Accuracy: {:.2f}%'.format(accuracy * 100))
knn_classification_report = classification_report(y_test, predict)

print(knn_classification_report)
%%time

rf_clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)

rf_clf.fit(X_train, y_train)
%%time

accuracy = cross_val_score(rf_clf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)

f1_score = cross_val_score(rf_clf, X_train, y_train, cv=10, scoring='f1_macro', n_jobs=-1)
print('accuracy: {:.2f}%'.format(accuracy.mean() * 100))

print('f1_score: {:.2f}%'.format(f1_score.mean() * 100))
%%time

tree_clf = ExtraTreesClassifier(n_estimators=50, random_state=0, n_jobs=-1)

tree_clf.fit(X_train, y_train)
%%time

accuracy = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)

f1_score = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring='f1_macro', n_jobs=-1)
print('accuracy: {:.2f}%'.format(accuracy.mean() * 100))

print('f1_score: {:.2f}%'.format(f1_score.mean() * 100))
%%time

grad_clf = GradientBoostingClassifier(n_estimators=50, random_state=0)

grad_clf.fit(X_train, y_train)
%%time

accuracy = cross_val_score(grad_clf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)

f1_score = cross_val_score(grad_clf, X_train, y_train, cv=10, scoring='f1_macro', n_jobs=-1)
print('accuracy: {:.2f}%'.format(accuracy.mean() * 100))

print('f1_score: {:.2f}%'.format(f1_score.mean() * 100))
from sklearn.metrics import precision_recall_fscore_support
est = [10, 25, 50, 100, 150, 200, 250]



for e in est:

    clf = RandomForestClassifier(n_estimators=e, random_state=0, n_jobs=-1)

    clf = clf.fit(X_train, y_train)

    

    predict = clf.predict(X_test)

    

    print('n_estimators={}'.format(e))

    

    accuracy = accuracy_score(y_test, predict)

    print('Accuracy: {:.2f}%'.format(accuracy * 100))



    p, r, f, s = precision_recall_fscore_support(y_test, predict, average='weighted')

    print('fscore: {:.2f}%'.format(f*100))

    

    print()
clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)

clf = clf.fit(X_train, y_train)
# predicting unseen data

predict = clf.predict(X_test)
# calculating accuracy

accuracy = accuracy_score(y_test, predict)



print('Random Forest Classifier model')

print('Accuracy: {:.2f}%'.format(accuracy * 100))
print(classification_report(y_test, predict))
train_df = pd.read_csv('../input/data/train.csv')

train_df.head()
train_df.drop('Id', axis=1, inplace=True)

train_df.shape
test_df = pd.read_csv('../input/data/covtype.csv')

test_df.head()
test_df.shape
y = train_df['Cover_Type']

X = train_df.drop('Cover_Type', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



# Create the pipeline

pipe = Pipeline([('scl', StandardScaler()),

                 ('pca', PCA(iterated_power=7)),

                 ('clf', ExtraTreesClassifier(random_state=0, n_jobs=-1))])



param_range = [1, 2, 3, 4, 5]



# Create the grid parameter

grid_params = [{'pca__n_components': [10, 15, 20, 25, 30],

                'clf__criterion': ['gini', 'entropy'],

                'clf__min_samples_leaf': param_range,

                'clf__max_depth': param_range,

                'clf__min_samples_split': param_range[1:]}]



# Create the grid, with "pipe" as the estimator

gridsearch = GridSearchCV(estimator=pipe,

                          param_grid=grid_params,

                          scoring='accuracy',

                          cv=3)



# Fit using grid search

gridsearch.fit(X_train, y_train)
# Best accuracy

print('Best accuracy: %.3f' % gridsearch.best_score_)



# Best params

print('\nBest params:\n', gridsearch.best_params_)
pipe = Pipeline([('scl', StandardScaler()),

                 ('pca', PCA(n_components=25)),

                 ('tree', ExtraTreesClassifier(criterion='entropy',

                                               max_depth=5, 

                                               min_samples_leaf=1, 

                                               min_samples_split=4,

                                               random_state=42, 

                                               n_jobs=-1))])
y = test_df['Cover_Type']

X = test_df.drop('Cover_Type', axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)

print('Test Accuarcy: {:.2f}%'.format(score * 100))
cross_val_score(pipe, X_train, y_train)
y_pred = pipe.predict(X_test)

print("Testing Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
pipe = Pipeline([('scl', StandardScaler()),

                 ('pca', PCA(n_components=20)),

                 ('tree', ExtraTreesClassifier(random_state=42, n_jobs=-1))])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)

print('Test Accuarcy: {:.2f}%'.format(score * 100))
cross_val_score(pipe, X_train, y_train)
y_pred = pipe.predict(X_test)

print("Testing Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))