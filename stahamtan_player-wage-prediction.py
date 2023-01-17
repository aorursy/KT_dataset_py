import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import re
data = pd.read_csv('../input/CompleteDataset.csv', low_memory=False)
data = data.drop(data.columns[0], axis = 1)

data.head()
# Extract numeric vales of the wage and value

data['Wage(TEUR)'] = data['Wage'].map(lambda x : re.sub('[^0-9]+', '', x)).astype('float64')

data['Value(MEUR)'] = data['Value'].map(lambda x : re.sub('[^0-9]+', '', x)).astype('float64')
reordered_cols = []

personal_cols = []

personal_cols = ['ID', 'Name', 'Photo', 'Club', 'Club Logo', 'Preferred Positions', 'Wage', 'Value',

                 'Nationality', 'Flag']

reordered_cols = personal_cols + [col for col in data if (col not in personal_cols)]

data = data[reordered_cols]
country_data = data.iloc[:, 8:].apply(pd.to_numeric, errors='coerce')

price_pred_cols = list(country_data.columns[-2:]) + list(country_data.columns[2:40])

price_pred_data = country_data[price_pred_cols]

price_pred_data.head()
corr = price_pred_data.corr()
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(24, 18))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
price_pred_data.isnull().any()[price_pred_data.isnull().any()==True]
col_name_missing = price_pred_data.isnull().any()[price_pred_data.isnull().any()==True].index

col_index_missing = [price_pred_data.columns.get_loc(x) for x in col_name_missing]

print(col_index_missing)
X = price_pred_data.iloc[:, 2:].values

y = price_pred_data.iloc[:, :2].values
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X)

X = imputer.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
y_w_train = y_train[:,0]

y_v_train = y_train[:,1]

y_w_test = y_test[:,0]

y_v_test = y_test[:,1]
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')

regressor.fit(X_train, y_w_train)

regressor.score(X_test, y_w_test)
from sklearn.model_selection import GridSearchCV

parameters = [{'kernel': ['rbf'],

               'epsilon': [0.1, 0.2, 0.5],

               #'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],

               'C': [1, 10, 100]

              },

              #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

             ]
svr_cv = SVR()

regressor_cv = GridSearchCV(svr_cv, param_grid = parameters)

regressor_cv.fit(X_train, y_w_train)
cv_df = pd.DataFrame(regressor_cv.cv_results_)
#'mean_test_score','mean_train_score', 'param_C', 'param_epsilon','rank_test_score'

score_df = cv_df[['rank_test_score','param_C','param_epsilon','mean_test_score','mean_train_score']]

score_df = pd.melt(score_df, id_vars=['rank_test_score','param_C','param_epsilon'],

                   value_vars=['mean_test_score','mean_train_score'],

                   var_name="score_name",

                   value_name="score")
score_df[score_df['rank_test_score']==1]
g = sns.FacetGrid(score_df, col="param_C", row="param_epsilon", hue="rank_test_score", margin_titles=True)

g.map(sns.barplot, "score_name", "score")

g.add_legend()

#Rotate x-axis labels

for ax in g.axes.flat:

    for label in ax.get_xticklabels():

        label.set_rotation(10)
# Principle Component Analysis

from sklearn.decomposition import PCA

pca = PCA(n_components = None)

pca.fit(X_train)

explained_variance = pca.explained_variance_ratio_
f, ax = plt.subplots(figsize=(10, 5))

sns.barplot(x=np.arange(explained_variance.size)+1,

            y=np.cumsum(explained_variance),

            color="b"

)

ax.set(ylabel="Cumulative Explained Variance",

       xlabel="Number of Principal Component")
pca = PCA(n_components = 4)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)
regressor_pca = SVR(kernel='rbf')

regressor_pca.fit(X_train_pca, y_w_train)

regressor_pca.score(X_test_pca, y_w_test)
parameters = [{'kernel': ['rbf'],

               'epsilon': [0.1, 0.2, 0.5],

               #'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],

               'C': [1, 10, 100, 1000]

              },

              #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

             ]
svr_pca_cv = SVR()

regressor_pca_cv = GridSearchCV(svr_pca_cv, param_grid = parameters, return_train_score=True)

regressor_pca_cv.fit(X_train_pca, y_w_train)
pca_cv_df = pd.DataFrame(regressor_pca_cv.cv_results_)

pca_score_df = pca_cv_df[['rank_test_score','param_C','param_epsilon','mean_test_score','mean_train_score']]

pca_score_df = pd.melt(pca_score_df, id_vars=['rank_test_score','param_C','param_epsilon'],

                   value_vars=['mean_test_score','mean_train_score'],

                   var_name="score_name",

                   value_name="score")
pca_score_df[pca_score_df['rank_test_score']==1]
g = sns.FacetGrid(pca_score_df, col="param_C", row="param_epsilon",

                  hue="rank_test_score", margin_titles=True,

                  palette=(sns.color_palette("coolwarm", 12)))

g.map(sns.barplot, "score_name", "score")

g.add_legend()

#Rotate x-axis labels

for ax in g.axes.flat:

    for label in ax.get_xticklabels():

        label.set_rotation(10)