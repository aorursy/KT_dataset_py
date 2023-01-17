# Load all the necessary packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from mlxtend.plotting import plot_confusion_matrix
# Load data

test = pd.read_csv("../input/learn-together/test.csv", index_col = "Id")

train = pd.read_csv("../input/learn-together/train.csv", index_col = "Id")

sample = pd.read_csv("../input/learn-together/sample_submission.csv", index_col = "Id")
# define X,Y for modelling

Y_train = train.Cover_Type

X_train = train.drop(['Cover_Type'], axis=1)
print("shape: ", train.shape, "\ncolumn names: \n", train.columns,

     "\nNaN values: ", train.isnull().sum().sum(),

"\nNA values:", train.isna().sum().sum())
df_train = X_train.copy()

train_cols = df_train.columns.tolist()

df_train[train_cols[10:]].sum()#/df_train.shape[0]*100
train_sub = train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 54]]

train_sub.head()
train_sub.describe()
corr = train.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
train_no_soil = train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 54]]

corr = train_no_soil.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
train_soil = train.iloc[:, 14:]

corr = train_soil.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
fig, ax =plt.subplots(3,3, figsize=(20,10))

sns.boxplot("Cover_Type", "Elevation", data=train, ax=ax[0][0])

sns.boxplot("Cover_Type", "Aspect", data=train, ax=ax[0][1])

sns.boxplot("Cover_Type", "Slope", data=train, ax=ax[0][2])

sns.boxplot("Cover_Type", "Horizontal_Distance_To_Hydrology", data=train, ax=ax[1][0])

sns.boxplot("Cover_Type", "Vertical_Distance_To_Hydrology", data=train, ax=ax[1][1])

sns.boxplot("Cover_Type", "Horizontal_Distance_To_Roadways", data=train, ax=ax[1][2])

sns.boxplot("Cover_Type", "Hillshade_9am", data=train, ax=ax[2][0])

sns.boxplot("Cover_Type", "Hillshade_Noon", data=train, ax=ax[2][1])

sns.boxplot("Cover_Type", "Hillshade_3pm", data=train, ax=ax[2][2])

fig.show()
# simple random forest with CV

clf = RandomForestClassifier()

param_grid = {

                 'n_estimators': [300],

                 'max_depth': [2, 4, 6, 8]

             }





grid_clf = GridSearchCV(clf, param_grid, cv=10)

grid_clf.fit(X_train, Y_train)
grid_clf.cv_results_['mean_test_score'] 
# simple random forest with CV

clf2 = RandomForestClassifier()

param_grid2 = {

                 'n_estimators': [300],

                 'max_depth': [10, 12, 14, 16, 18, 20]

             }





grid_clf2 = GridSearchCV(clf2, param_grid2, cv=10)

grid_clf2.fit(X_train, Y_train)
grid_clf2.cv_results_['mean_test_score'] 
# simple random forest with CV

base_model = RandomForestClassifier(n_estimators = 2000, max_depth = 18)

base_model.fit(X_train, Y_train)
# Predict train labels matrix

Y_train_pred = base_model.predict(X_train)
cfm = confusion_matrix(Y_train, Y_train_pred)

fig, ax = plot_confusion_matrix(conf_mat=cfm,

                                colorbar=True,

                                show_absolute=True,

                                show_normed=True, figsize=(20,10))

plt.show()
# Here I use a slightly modified function by Georg Fischer, which can be found in his kernel: 

# https://www.kaggle.com/grfiv4/plotting-feature-importances

def plot_feature_importances(clf, X_train, y_train=None, 

                             top_n=10, figsize=(8,8), title="Feature Importances"):

    '''

    plot feature importances of a tree-based sklearn estimator

    

    Note: X_train and y_train are pandas DataFrames

    

    Note: Scikit-plot is a lovely package but I sometimes have issues

              1. flexibility/extendibility

              2. complicated models/datasets

          But for many situations Scikit-plot is the way to go

          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

    

    Parameters

    ----------

        clf         (sklearn estimator) if not fitted, this routine will fit it

        

        X_train     (pandas DataFrame)

        

        y_train     (pandas DataFrame)  optional

                                        required only if clf has not already been fitted 

        

        top_n       (int)               Plot the top_n most-important features

                                        Default: 10

                                        

        figsize     ((int,int))         The physical size of the plot

                                        Default: (8,8)

        

        

    Returns

    -------

        the pandas dataframe with the features and their importance

        

    Author

    ------

        George Fisher

    '''

    

    __name__ = "plot_feature_importances"

    

    import pandas as pd

    import numpy  as np

    import matplotlib.pyplot as plt

    

    from xgboost.core     import XGBoostError

    from lightgbm.sklearn import LightGBMError

    

    try: 

        if not hasattr(clf, 'feature_importances_'):

            clf.fit(X_train.values, y_train.values.ravel())



            if not hasattr(clf, 'feature_importances_'):

                raise AttributeError("{} does not have feature_importances_ attribute".

                                    format(clf.__class__.__name__))

                

    except (XGBoostError, LightGBMError, ValueError):

        clf.fit(X_train.values, y_train.values.ravel())

            

    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    

    feat_imp['feature'] = X_train.columns

    feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    feat_imp = feat_imp.iloc[:top_n]

    

    feat_imp.sort_values(by='importance', inplace=True)

    feat_imp = feat_imp.set_index('feature', drop=True)

    feat_imp.plot.barh(title=title, figsize=figsize)

    plt.xlabel('Feature Importance Score')

    plt.show()

   
plot_feature_importances(base_model, X_train, Y_train, top_n= X_train.shape[1], 

                         title=base_model.__class__.__name__, figsize = (12, 12))
Y_test_pred = base_model.predict(test)
# Save test predictions to file

output = pd.DataFrame({'Id': test.index,

                       'Cover_Type': Y_test_pred})

output.to_csv('submission.csv', index=False)