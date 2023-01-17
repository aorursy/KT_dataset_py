import pandas as pd # importing Pandas library for performing dataframe related operations
import numpy as np  # importing numpy for performing numeric array related operations
import matplotlib.pyplot as plt # importing matplotlib.pyplot for basic plotting operations
import seaborn as sns  # importing seaborn for advanced data visualization
# Below is the magic function to display and save graphs/figures in the output cells
%matplotlib inline   
from sklearn.model_selection import train_test_split # For train-test split
# For standardizing/normalizing the data (let's import many and see which suits the best)
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler 
from sklearn.preprocessing import PolynomialFeatures # To create polynomial features
import warnings # Import warnings module
warnings.filterwarnings('ignore') # Ignore the warnings
from sklearn.feature_selection import f_regression,SelectKBest,mutual_info_regression,RFE # For feature selection
# Let's import the various supervised ML models
from sklearn.linear_model import LinearRegression # Linear Regression model
from sklearn.neighbors import KNeighborsRegressor # K-NN Regressor Model
from sklearn.svm import SVR # Support vector regressor
from sklearn.tree import DecisionTreeRegressor # Decision tree regressor
# Let's import the ensemble regressor models
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor
from sklearn.pipeline import Pipeline # Import sklearn pipeline
from sklearn.model_selection import cross_val_score, KFold # Cross validation
from sklearn.metrics import explained_variance_score # Metric used to evaluate the regression models
from scipy.stats import zscore # zscore normalization from scipy.stats
from sklearn.utils import resample # Used to find the bootstrapping confidence interval
df_orig = pd.read_csv('../input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv')
# Creating a copy of the original dataframe
df = df_orig.copy()
df.head()
# Renaming the column names
df.columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
# Lets see the dataframe once again..
df.head()
# Let's see shape of the dataset
df.shape
# Let's see datatypes of the attributes
df.info()
# Let's see the names of the independent columns
for col in df.columns:
    if col!='strength': # Print all column names except target column
        print(col)
# Let's see the five point summary of the columns
df.describe()
# Let us check the missing values..
df.isnull().sum()
# Let's us only plot the distributions of independent attributes
df.drop('strength',axis=1).hist(figsize=(12,16),layout=(4,2));
# Let's check the skewness values quantitatively
df.skew().sort_values(ascending=False)
# Let us check presence of outliers
plt.figure(figsize=(18,14))
box = sns.boxplot(data=df)
box.set_xticklabels(labels=box.get_xticklabels(),rotation=90);
# Let us see how many of them are correlated..
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True, cmap="YlGnBu")
# Let us see the significant correlation either negative or positive among independent attributes..
c = df.drop('strength',axis=1).corr().abs() # Since there may be positive as well as -ve correlation
s = c.unstack() # 
so = s.sort_values(ascending=False) # Sorting according to the correlation
so=so[(so<1) & (so>0.3)].drop_duplicates().to_frame() # Due to symmetry.. dropping duplicate entries.
so.columns = ['correlation']
so
sns.pairplot(df,diag_kind='kde');
# let us remove the outliers
for column in df.columns.tolist():
    Q1 = df[column].quantile(.25) # 1st quartile
    Q3 = df[column].quantile(.75) # 3rd quartile
    IQR = Q3-Q1 # get inter quartile range
    # Replace elements of columns that fall below Q1-1.5*IQR and above Q3+1.5*IQR
    df[column].replace(df.loc[(df[column] > Q3+1.5*IQR)|(df[column] < Q1-1.5*IQR), column], df[column].median(),inplace=True)
# Let us check presence of outliers
plt.figure(figsize=(18,14))
box = sns.boxplot(data=df)
box.set_xticklabels(labels=box.get_xticklabels(),rotation=90);
# Let's add this new composite feature before target attribute.
df.insert(8,'water/cement',df['water']/df['cement'])
# Let's check whether the feature is added properly or not?
df.head()
df.corr()
poly3 = PolynomialFeatures(degree = 3, interaction_only=True)
poly3_ft = poly3.fit_transform(df.drop('strength',axis=1))
df_poly3= pd.DataFrame(poly3_ft,columns=['feat_'+str(x) for x in range(poly3_ft.shape[1])])
df_poly3.head()
# Let us create the dataframe with all features
df_feat = df.drop('strength',axis=1).join(df_poly3)
df_feat['strength'] = df['strength']
print(df_feat.shape)
df_feat.head()
df_feat.head()
from sklearn.linear_model import Lasso
X = df_feat.drop('strength',axis=1)
y = df_feat['strength']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lasso = Lasso() # Since it uses L1 reglarization features with zero coefficients will be insignificant.
lasso.fit(X_train,y_train)
print ("Lasso model:", (lasso.coef_))
# Let's us get the features selected by Lasso
lasso_feat = X_train.columns[lasso.coef_!=0].tolist() # Dropping the features with 0 coefficient value
print(lasso_feat) # Features selected using LASSO regularization
print("Out of total {} independent features, number of features selected by LASSO regularization are {} ".format(X_train.shape[1],len(lasso_feat)))
df_feat = df_feat[lasso_feat] # Select independent features 
df_feat.head()
from sklearn.cluster import KMeans
df_z = df_orig.apply(zscore) # Get the normalized dataframe
cluster_range = range( 1, 15 )
cluster_errors = []
for num_clusters in cluster_range:
  clusters = KMeans( num_clusters, n_init = 10 )
  clusters.fit(df_z)
  labels = clusters.labels_
  centroids = clusters.cluster_centers_
  cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:14]
# Elbow plot
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
kmeans = KMeans(n_clusters= 7)
kmeans.fit(df_z)
labels = kmeans.labels_
counts = np.bincount(labels[labels>=0])
print(counts)
## creating a new dataframe only for labels and converting it into categorical variable
cluster_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))
cluster_labels['labels'] = cluster_labels['labels'].astype('category')
df_labeled = df_orig.join(cluster_labels)
df_labeled.boxplot(by = 'labels',  layout=(3,3), figsize=(30, 20));
df_orig.columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age', 'strength']
# Let's create train and test sets
X = df_orig.drop('strength',axis=1)
y = df_orig['strength']
# Let's split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create empty dataframe to store the results
df_result_raw_train = pd.DataFrame({'Regressor':[],'ExplVarianceScore':[],'StdDev':[]})
# We will use the pipeline approach
pipelines = []
pipelines.append(('Linear Regression',Pipeline([('scaler',RobustScaler()),('LR',LinearRegression())])))
pipelines.append(('KNN Regressor',Pipeline([('scaler',RobustScaler()),('KNNR',KNeighborsRegressor())])))
pipelines.append(('SupportVectorRegressor',Pipeline([('scaler',RobustScaler()),('SVR',SVR())])))
pipelines.append(('DecisionTreeRegressor',Pipeline([('scaler',RobustScaler()),('DTR',DecisionTreeRegressor())])))
pipelines.append(('AdaboostRegressor',Pipeline([('scaler',RobustScaler()),('ABR',AdaBoostRegressor())])))
pipelines.append(('RandomForestRegressor',Pipeline([('scaler',RobustScaler()),('RBR',RandomForestRegressor())])))
pipelines.append(('BaggingRegressor',Pipeline([('scaler',RobustScaler()),('BGR',BaggingRegressor())])))
pipelines.append(('GradientBoostRegressor',Pipeline([('scaler',RobustScaler()),('GBR',GradientBoostingRegressor())])))
# Let's find and store the cross-validation score for each pipeline for training data with raw features.
for ind, val in enumerate(pipelines):
    # unpack the val
    name, pipeline = val
    kfold = KFold(n_splits=10,random_state=2020) 
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='explained_variance')
    df_result_raw_train.loc[ind] = [name,cv_results.mean()*100,cv_results.std()*100]
# Let's check the training results with raw features 
df_result_raw_train
# Let's find and store the cross-validation score for each pipeline for test data with raw features.
df_result_raw_test = pd.DataFrame({'Regressor':[],'ExplVarianceScore':[]})
for ind, val in enumerate(pipelines):
    # unpack the val
    name, pipeline = val
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    df_result_raw_test.loc[ind] = [name,explained_variance_score(y_test,y_pred)*100]
# Let's check the test results with raw features
df_result_raw_test
df_feat.head()
# Let's create train and test sets from modified dataframe with raw as well as new features.
X = df_feat
y = df['strength']
# Let's split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Let's create dataframe to store the results.
df_result_mod_train = pd.DataFrame({'Regressor':[],'ExplVarianceScore':[],'StdDev':[]})
for ind, val in enumerate(pipelines):
    # unpack the val
    name, pipeline = val
    kfold = KFold(n_splits=10,random_state=2020) 
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='explained_variance')
    df_result_mod_train.loc[ind] = [name,cv_results.mean()*100,cv_results.std()*100]
# Let's check the training results with raw features  as well as new features
df_result_mod_train
# Let's find and store the cross-validation score for each pipeline for training data with raw as well as new features.
df_result_mod_test = pd.DataFrame({'Regressor':[],'ExplVarianceScore':[]})
for ind, val in enumerate(pipelines):
    # unpack the val
    name, pipeline = val
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_test)
    df_result_mod_test.loc[ind] = [name,explained_variance_score(y_test,y_pred)*100]
# Let's check the test results with raw features  as well as new features
df_result_mod_test
# Separate target and independent features
X = df_orig.drop('strength',axis=1)
y = df_orig['strength']
# Let's split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating pipelines for 3 best models.
pipe_rf = Pipeline([('scaler',RobustScaler()),('RBR',RandomForestRegressor())])
pipe_br = Pipeline([('scaler',RobustScaler()),('BGR',BaggingRegressor())])
pipe_gbr = Pipeline([('scaler',RobustScaler()),('GBR',GradientBoostingRegressor())])
# Initalize the empty dataframes to capture the feature importances given by these models..
df_featImp_rf = df_featImp_br = df_featImp_gbr = pd.DataFrame({'Features':[], 'Importance':[]})
# feature importance given by random forest regressor
pipe_rf.fit(X_train,y_train)
featImp_rf = pipe_rf.steps[1][1].feature_importances_
df_featImp_rf['Features'] = X_train.columns
df_featImp_rf['Importance'] = featImp_rf
# Feature importance given by Random Forest Regressor
df_featImp_rf.sort_values(by='Importance', ascending=False)
# feature importance given by Gradient Boost Regressor
pipe_gbr.fit(X_train,y_train)
featImp_gbr = pipe_gbr.steps[1][1].feature_importances_
df_featImp_gbr['Features'] = X_train.columns
df_featImp_gbr['Importance'] = featImp_gbr
# Feature importance given by Random Forest Regressor
df_featImp_gbr.sort_values(by='Importance', ascending=False)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# Separate target and independent features
X = df_orig.drop('strength',axis=1)
y = df_orig['strength']
# Let's split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe_gbr = Pipeline([('scaler',RobustScaler()),('GBR',GradientBoostingRegressor())])
# Let's see what are the hyper parameters for gradient boosting regressor model
pipe_gbr.steps[1][1]
param_grid=[{'GBR__n_estimators':[100,500,1000], 'GBR__learning_rate': [0.1,0.05,0.02,0.01], 'GBR__max_depth':[4,6], 
            'GBR__min_samples_leaf':[3,5,9,17], 'GBR__max_features':[1.0,0.3,0.1] }]
search = GridSearchCV(pipe_gbr, param_grid, cv = kfold, scoring = 'explained_variance', n_jobs=-1)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
y_pred_train = search.predict(X_train)
y_pred_test = search.predict(X_test)
from sklearn.metrics import explained_variance_score,r2_score
print('Testing Explained Variance Score is  {}'.format(explained_variance_score(y_test,y_pred_test)))
print('Testing R2 Score is  {}'.format(r2_score(y_test,y_pred_test)))
random_grid={'GBR__n_estimators':[100,500,1000], 'GBR__learning_rate': [0.1,0.05,0.02,0.01], 'GBR__max_depth':[4,6], 
            'GBR__min_samples_leaf':[3,5,9,17], 'GBR__max_features':[1.0,0.3,0.1] }
search = RandomizedSearchCV(estimator=pipe_gbr, param_distributions=random_grid, n_iter = 5, cv = kfold, scoring = 'explained_variance', n_jobs=-1)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
y_pred_train = search.predict(X_train)
y_pred_test = search.predict(X_test)
from sklearn.metrics import explained_variance_score,r2_score
print('Testing Explained Variance Score is  {}'.format(explained_variance_score(y_test,y_pred_test)))
print('Testing R2 Score is  {}'.format(r2_score(y_test,y_pred_test)))

