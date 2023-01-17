import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
print('Importing Finished')
#Reading in the data
df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
df.head()
#Cleaning column names
cleaned_cols = [col.replace("-", "_") for col in df.columns]
df = pd.DataFrame(data = df.values,
                 columns = cleaned_cols,
                 index = df.index)

X = df.iloc[:,1:]
y = df.iloc[:,0]

df.head()
df.dtypes

#All columns are categorical
df.describe()

#Binomial outcome (edible or poisonous), relatively low cardinality of features
#Some features are ordinal and others are not
#Dropping veil_type as it is uninformative (all same value)
df = df.drop(columns = 'veil_type')
X = X.drop(columns = 'veil_type')
df.head()
#Distribution of values within each feature
for col in df.columns:
    print(df[col].value_counts())
    
#We see that there is a ? for stalk root, indicating it is missing data
#Most features are well-disbursed but many (cap-shape, cap-surface, cap-color, stalk-color-above-ring, veil-color) have some values 
#that appear very infrequently (< 20 / 8124)
#Features with rare values
rare_valued_cols = [col for col in df.columns if df[col].value_counts().min() <= 20]
print(rare_valued_cols)

#These rare values could present issues in training a model as they could effectively act as noise in the data. 
#If problematic, we may need to treat them as outliers and remove.
fig = plt.figure(figsize=(16,30))
for i in range(len(X.columns)):
    fig.add_subplot(7,3,i+1)
    col = X.columns[i]
    a = df.groupby(col).apply(lambda df: df['class'] == 'p').reset_index()
    a = a.groupby(col)['class'].agg({'sum', 'count'}).reset_index().rename(columns = {'count':'total', 'sum':'total_poisonous'})
    a['percent_poisonous'] = round(a['total_poisonous'] *100 / a['total'], 2)
#     plt.title('Percent of Edible Mushrooms by Subcategory in {}'.format(df.columns[1]))
    ax = sns.barplot(x = a[col], y = a['percent_poisonous'], color = 'b')
    ax2 = ax.twinx()
    sns.lineplot(x = a[col], y = a['total'], ax = ax2, color = 'r')
    
plt.show()

#Some variables are clearly more predictive of poisonous/edible such as odor, stalk_color_below_ring, spore_print_color whereas
#others are poor predictors as they split close to 50% (stalk_shape, cap_color, stalk_surface_above_ring)
#The next step to exploring the data is understanding the relationships between the features.
#Because all of the data is categorical, we need another measure or method to determine similarity outside of correlation.
#Clustering can highlight distinguishing features of the mushrooms as well as those that permeate across many types.  
#Splitting data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,train_size=0.8, random_state = 0)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size = 0.25,train_size =0.75, random_state = 0)
print("""Train dataset size: {},
Validation dataset size: {},
Test dataset size: {}""".format(X_train.shape, X_val.shape, X_test.shape))
#Transforming target labels
y_le = LabelEncoder()
y_train = pd.Series(y_le.fit_transform(y_train), index = X_train.index)
y_val = pd.Series(y_le.transform(y_val), index = X_val.index)
y_test = pd.Series(y_le.transform(y_test), index = X_test.index)

y_le.classes_
#Kmodes Clustering

from kmodes.kmodes import KModes

#performing clustering on training and validation sets
X_val_train = pd.concat([X_val, X_train])

#imputing missing values (which are '?') and encoding the data
clustering_imputer = SimpleImputer(missing_values = '?', strategy = 'most_frequent')
clustering_ord_enc = OrdinalEncoder()

clustering_transformer = Pipeline(steps = [
    ('imputer', clustering_imputer),
    ('ord_enc', clustering_ord_enc)
])

kmodes_df = pd.DataFrame(data = clustering_transformer.fit_transform(X_val_train), columns = X.columns)

#clustering mushrooms into 2 to 10 clusters
k_list = [x+2 for x in range(9)]

km = {}
km_clusters = {}

for k in k_list:
    km[k] = KModes(n_clusters = k, n_init = 10, verbose = 0)
    km_clusters[k] = km[k].fit_predict(kmodes_df)

km_clusters
#Graphing the clustering cost for each k
km_cost = [km[k].cost_ for k in k_list]
sns.lineplot(k_list, km_cost, color = 'b')

#There are "elbows" at k=4,7,9 but for interpretability, we'll choose 4 clusters
#The 4 clusters' centroids, or modes
final_clusters = pd.DataFrame(data = clustering_ord_enc.inverse_transform(pd.DataFrame(km[4].cluster_centroids_)), columns = X.columns, index = [[0,1,2,3]] )
final_clusters
#Graphing the features of the mushrooms by what cluster they fall into
df_val_train = X_val_train
df_val_train['cluster'] = pd.Series(km_clusters[4], index = df_val_train.index)

fig = plt.figure(figsize=(16,26))
for i in range(len(X.columns)):
    fig.add_subplot(7,3,i+1)
    col = X.columns[i]
    sns.countplot(x = col, data = df_val_train, hue = 'cluster')
    
plt.show()

#Nearly all of cluster 2 mushrooms have gill color b (b stands for buff) and gill color b is exclusively comprised of cluster 2 mushrooms. 
#This indicates that a gill color of buff is a defining characteristic of cluster 2 mushrooms. 
#Looking at the graphs, we can see that gill_color, odor, spore_print_color among others are defining characteristics for some of the clusters
#whereas cap_shape, for example, is not a differentiating feature between the clusters
#Graphing clusters by their class
y_val_train = pd.Series(pd.concat([y_val, y_train]))
df_val_train['class'] = y_val_train
sns.countplot('cluster', data = df_val_train, hue = 'class')

#Cluster 0 and 3 are predominantly of edible mushrooms and clusters 1 and 2 are made up of nearly all poisonous mushrooms.
#Cluster 0 seems to be a catch-all category as evidenced by the greater number of mushrooms in its cluster, its diversity of 
#mushroom class, and its diversity of features as seen in the graphs above

#This graph indicates that mushrooms of similar features are likely of the same class. In other words, at least some of these 
#features, likely the defining ones mentioned above, have the potential to help us in classifying mushrooms with high accuracy.
#Preprocessing pipeline for logistic regression

#categorizing features as ordinal or nominal
#features with 2 unique values in the training set are categorized as ordinal (and ring_number), all others are nominal
ord_cols = [col for col in X.columns if X[col].nunique() == 2]
nom_cols = [col for col in X.columns if col not in ord_cols]
nom_cols.remove('ring_number')

#imputer for any columns with missing values (only stalk_root)
imputer = SimpleImputer(missing_values = '?', strategy = 'most_frequent')

#onehot encoder for nominal variables
encoder_nom = OneHotEncoder(handle_unknown = 'ignore')

#Nominal variables should in general be one-hot encoded as there is no known or implied ordering between the values
#Logistic regression with a penalty is adept at handling one-hot encoded variables and the penalty will account for the collinearity 
#deriving from one-hot encoding

#ordinal encoder for columns with inferred ordering (ring_number)
encoder_ord1 = OrdinalEncoder(categories=[['n', 'o', 't']])

#n, o, and t stand for none, one, and two, respectively. This is a natural ordering that we know so therefore it was hardcoded in the encoder

#ordinal encoder for all other ordinal variables
encoder_ord2 = OrdinalEncoder()

nom_transformer = Pipeline(steps = [
    ('imputer', imputer),
    ('onehot', encoder_nom)
])

ord_transformer = Pipeline(steps = [
    ('imputer', imputer),
    ('ord_enc', encoder_ord2)
])

preprocessor_log = ColumnTransformer(
    transformers = [
        ('nom', nom_transformer, nom_cols),
        ('ord1', encoder_ord1, ['ring_number']),
        ('ord2', ord_transformer, ord_cols)
])
print(ord_cols)
#Building logistic model with penalty
penalty = ['l1','l2']
C = [.01, 0.1, 1, 5, 10, 100]

logistic = {}
for p in penalty:
    for c in C:
        logistic[(c,p)] = LogisticRegression(C=c, penalty=p, solver='liblinear', random_state=0)
# for key in logistic.keys():
#     print(key)
#Bundling pipeline (preprocessing and modeling) for logistic regression with penalty
bundled_pipeline_log = {}
for key in logistic.keys():
    bundled_pipeline_log[key] = Pipeline(steps = [
        ('preprocessing', preprocessor_log),
        ('model', logistic[key])
    ])
#Fit, predict, and calculate loss for the logistic models of different parameters
pred_log = {}
pred_proba_log = {}
accuracy_log = {}

for key in bundled_pipeline_log.keys():
    bundled_pipeline_log[key].fit(X_train, y_train)

    pred_log[key] = bundled_pipeline_log[key].predict(X_val)
    pred_proba_log[key] = bundled_pipeline_log[key].predict_proba(X_val)

    accuracy_log[key] = bundled_pipeline_log[key].score(X_val, y_val)

accuracy_log
#Visualizing logistic models' accuracies on validation set as a heatmap
accuracy_log_series = pd.Series(list(accuracy_log.values()),
                  index=pd.MultiIndex.from_tuples(accuracy_log.keys()))
accuracy_log_df = accuracy_log_series.unstack()
sns.heatmap(accuracy_log_df)
#Displaying feature importance through coefficients of logistic regression
pd.set_option('display.max_rows', 150)

#choosing one of the many perfectly accurate logistic models
best_log = bundled_pipeline_log[(1, 'l1')]
nom_cols_expanded = list(best_log['preprocessing'].transformers_[0][1]['onehot'].get_feature_names(nom_cols))

pipeline_cols_log = nom_cols_expanded + ['ring_number'] + ord_cols
coef_data_log = best_log['model'].coef_.reshape((108,1))
coef_df_log = pd.DataFrame(data = coef_data_log, index = pipeline_cols_log).rename(columns = {0: 'coef_value'})

#filtering out zero-valued coefficients
coef_df_log.loc[coef_df_log.coef_value != 0].coef_value.sort_values(ascending=True)

#Note: Negative indicates a prediction of edible and positive for poisonous. 
#We can use coefficient magnitudes as a proxy for feature importance. 
#Odor, spore_print_color, gill_size, and population are some of the most impactful features (if that characteristic of the mushroom
# is present).  
#Ordinal column labels
for i in range(len(ord_cols)):
    print('Ordering of levels within {} column: {}'.format(ord_cols[i], best_log['preprocessing'].transformers_[2][1][1].categories_[i]))
#Preprocessing Decision Tree / Random Forest Classifier
preprocessor_dt = ColumnTransformer(
    transformers = [
        ('ord1', encoder_ord1, ['ring_number']),
        ('ord2', ord_transformer, ord_cols + nom_cols)
    ])

#Using the same preprocessor for DTs and RFs. Unlike logistic regression, these tree-based models perform better with ordinal encoding,
#even when the variable is nominal. This is because one-hot encoding variables in these models generally devalues the splits of these
#variables and leads to sparse trees
#Creating DTs and RFs with various hyperparameter values
max_depth = [1, 5, 10, 20]
max_features = [.1, .25, .5, .75, "auto"]
n_estimators = [10, 25, 50]

dt = {}
rf = {}
for d in max_depth:
    for f in max_features:
        dt[(f,d)] = DecisionTreeClassifier(max_depth = d, max_features = f, random_state=0)

for d in max_depth:
    for f in max_features:
        for e in n_estimators:
            rf[(f,d,e)] = RandomForestClassifier(n_estimators = e, max_depth = d, max_features = f, random_state=0)
#Bundling pipeline (preprocessing and modeling) for logistic regression with penalty
bundled_pipeline_dt = {}
for key in dt.keys():
    bundled_pipeline_dt[key] = Pipeline(steps = [
        ('preprocessing', preprocessor_dt),
        ('model', dt[key])
    ])
    
bundled_pipeline_rf = {}
for key in rf.keys():
    bundled_pipeline_rf[key] = Pipeline(steps = [
        ('preprocessing', preprocessor_dt),
        ('model', rf[key])
    ])

#Fit, predict, and calculate loss for DT model
pred_dt = {}
pred_proba_dt = {}
accuracy_dt = {}

for key in bundled_pipeline_dt.keys():
    bundled_pipeline_dt[key].fit(X_train, y_train)

    pred_dt[key] = bundled_pipeline_dt[key].predict(X_val)
    pred_proba_dt[key] = bundled_pipeline_dt[key].predict_proba(X_val)

    accuracy_dt[key] = bundled_pipeline_dt[key].score(X_val, y_val)

accuracy_dt
#Fit, predict, and calculate loss for RF model
pred_rf = {}
pred_proba_rf = {}
accuracy_rf = {}

for key in bundled_pipeline_rf.keys():
    bundled_pipeline_rf[key].fit(X_train, y_train)

    pred_rf[key] = bundled_pipeline_rf[key].predict(X_val)
    pred_proba_rf[key] = bundled_pipeline_rf[key].predict_proba(X_val)

    accuracy_rf[key] = bundled_pipeline_rf[key].score(X_val, y_val)

accuracy_rf
#Displaying feature importance through coefficients of DTs
accuracy_dt_series = pd.Series(list(accuracy_dt.values()),
                  index=pd.MultiIndex.from_tuples(accuracy_dt.keys()))
accuracy_dt_df = accuracy_dt_series.unstack()
sns.heatmap(accuracy_dt_df)
#Displaying feature importance through coefficients of RFs
accuracy_rf_series = pd.Series(list(accuracy_rf.values()),
                  index=pd.MultiIndex.from_tuples(accuracy_rf.keys()))
accuracy_rf_df = accuracy_rf_series.unstack()
sns.heatmap(accuracy_rf_df)
#Visualizing feature importance in the DT

#choosing one of the many perfectly accurate DTs
best_dt = bundled_pipeline_dt[(0.5, 20)]

#features are in the order they were preprocessed in the pipeline
dt_features = ['ring_number'] + ord_cols + nom_cols

for i,f in enumerate(best_dt['model'].feature_importances_):
    print('{} has feature importance score: {}'.format(dt_features[i], round(f, 2)))

sns.barplot(best_dt['model'].feature_importances_, dt_features, color = 'b')

#Note: Changing order of columns with set random seed affects importance levels. 
#Therefore, random forest using a random forest with many trees is likely a better indication of feature importances.
#Visualizing feature importance in the RF

#choosing one of the many perfectly accurate RFs
best_rf = bundled_pipeline_rf[(0.5, 20, 25)]

rf_features = ['ring_number'] + ord_cols + nom_cols

for i,f in enumerate(best_rf['model'].feature_importances_):
    print('{} has feature importance score: {}'.format(rf_features[i], round(f, 2)))
    
sns.barplot(best_rf['model'].feature_importances_, rf_features, color = 'b')

#Gill_color, spore_print_color, odor, gill_size, and population are the most important features of the RF. This is mostly in line
#with what was observed in the DT and logistic regression model. Gill_color was not as emphasized in the logistic regression model
#as it is in the DT and RF, potentially because of the differences in preprocessing, the well-disbursed nature and high cardinality 
#of gill_color that would favor tree-based methods, and/or due to underlying similarities between features
#Fitting chosen models with training and validation data and then predicting on test data

#Fitting
best_log.fit(X_val_train, y_val_train)
best_dt.fit(X_val_train, y_val_train)
best_rf.fit(X_val_train, y_val_train)

#Predicting
log_test_score = best_log.score(X_test, y_test)
dt_test_score = best_dt.score(X_test, y_test)
rf_test_score = best_rf.score(X_test, y_test)

print("""
Logistic regression accuracy on test set: {},
Pruned decision tree accuracy on test set: {},
Random Forest accuracy on test set: {},
""".format(dt_test_score, pruned_dt_test_score, rf_test_score))