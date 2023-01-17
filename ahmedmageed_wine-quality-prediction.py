# Import Modules 

import pandas as pd

from pandas import set_option



import matplotlib.pyplot as plt

from matplotlib import cm



import seaborn as sns



import numpy as np

from scipy import stats



from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import learning_curve, GridSearchCV



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score 

from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.preprocessing import Normalizer, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.feature_selection import SelectKBest, chi2

from sklearn import datasets



import warnings

warnings.filterwarnings("ignore")

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm

from matplotlib import rcParams

rcParams['xtick.major.pad'] = 1

rcParams['ytick.major.pad'] = 1
dataset_pd = pd.read_csv('../input/winequality-red.csv')

dataset_pd.head()
dataset_pd.describe()
def count_missing(dataframe):   

    total = dataframe.isnull().sum().sort_values(ascending=False)

    percent = (dataframe.isnull().sum()/dataframe.isnull().count()*100).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data

count_missing(dataset_pd)
dataset_pd.groupby('quality').mean()
features_pd = dataset_pd.iloc[:, :-1]

features_np = features_pd.values

goal_pd = dataset_pd.iloc[:, -1]

goal_np = goal_pd.values
## Numerical Values 

# Selecting the most relevant features according to the chi2 

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(features_np,goal_np)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(features_pd.columns)

#concat two dataframes for better visualization 

numericalFeatureScores = pd.concat([dfcolumns,dfscores],axis=1)

numericalFeatureScores.columns = ['Specs','Chi2-Score']  #naming the dataframe columns

print(numericalFeatureScores.nlargest(15,'Chi2-Score'))  #print 10 best features
# Correlation Heatmap 

corrmat = dataset_pd.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(10,10))

#plot heat map

g=sns.heatmap(dataset_pd[top_corr_features].corr(),annot=True,cmap="RdYlGn")
fig = plt.figure(figsize=(15,15))



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(dataset_pd.columns[0:-1])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(6,2,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    plt.boxplot(x=dataset_pd[b])

    

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   
fig = plt.figure(figsize=(15,15))



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(dataset_pd.columns[0:-1])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(6,2,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    plt.scatter(dataset_pd[b],dataset_pd['quality'])

    

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   
# Instantiate a figure object for OOP figure manipulation.



fig = plt.figure(figsize=(15,15)) # start a figure with certain size 



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(dataset_pd.columns[0:-1])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(6,2,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    sns.distplot(dataset_pd[b], kde=True)

    

   

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   
dataset_pd.skew()
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLars
# The Regression model 

reg_models = []



RR_params = {'alpha':[10,100,1000]}

Lasso_params = {'alpha':[10,100,1000]}

Elastic_params = {'alpha':[10,100,1000]}

LassoLars_params = {'alpha':[10,100,1000]}

ForestRegressor_params = {'n_estimators': (10, 50, 100, 1000)}
reg_models.append(('RidgeRegression', Ridge(), RR_params))

reg_models.append(('LASSO', Lasso(), Lasso_params))

reg_models.append(('ElasticNet', ElasticNet(),Elastic_params))

reg_models.append(('LassoLars', LassoLars(), LassoLars_params))

reg_models.append(('Random Forest Regressor', RandomForestRegressor(random_state=0), ForestRegressor_params))
results = []

names = []



def data_divider(Data, goal, goal_index, train_size):

    X_train, X_test, y_train, y_test = train_test_split(Data.iloc[:,:goal_index], Data[goal], train_size = train_size )

    return X_train, X_test, y_train, y_test

    



def estimator_function(models_dict, data, scoring = 'r2'):

    

    X_train, X_test, y_train, y_test = data_divider(data, 'quality', -1, 0.8)

    

    # The Linear Regression model 

    LR = LinearRegression()

    LR.fit(X_train, y_train)

    

    print(f"The Linear Regression Model R2 : {LR.score(X_test, y_test)}")

    

    for name, model, params in models_dict:

    

        kfold = KFold( n_splits=5 ,  random_state=0, shuffle=True)



        model_grid = GridSearchCV(model, params)



        cv_results = cross_val_score(model_grid, X_train, y_train, cv = kfold, scoring=scoring)



        results.append(cv_results)



        names.append(name)



        msg = "Cross Validation Accuracy %s: R2: %f " % (name, cv_results.mean())



        print(msg)
estimator_function(reg_models, dataset_pd)
features_pd = np.log10(features_pd+1)
pd.DataFrame(stats.skew(features_pd), index=features_pd.columns, columns=['Skewnes'])
features_np = features_pd.values
data_clean = pd.concat([pd.DataFrame(features_np, columns=features_pd.columns), goal_pd], axis=1)

data_clean
# Instantiate a figure object for OOP figure manipulation.



fig = plt.figure(figsize=(15,15)) # start a figure with certain size 



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(data_clean.columns[0:-1])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(6,2,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    sns.distplot(data_clean[b], kde=True)

    

   

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   
# standardizing the data 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

features_np = scaler.fit_transform(features_np)

features_np
data_clean = pd.concat([pd.DataFrame(features_np, columns=features_pd.columns), goal_pd], axis=1)

data_clean.head()
# Instantiate a figure object for OOP figure manipulation.



fig = plt.figure(figsize=(15,15)) # start a figure with certain size 



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(data_clean.columns[0:-1])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(6,2,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    sns.distplot(data_clean[b], kde=True)

    

   

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   
estimator_function(reg_models, data_clean)
type(data_clean)
data_clean.quality = pd.cut(data_clean.quality, bins=[0,3,5,8,10], labels=['very bad', 'bad', 'good', 'very good'])

data_clean
data_clean_np = data_clean.values

goal_pd = data_clean.iloc[:,-1]

goal_np = goal_pd.values



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder = LabelEncoder()

data_clean_np[:,-1] = encoder.fit_transform(data_clean_np[:,-1])

pd.DataFrame(data_clean_np)
SVM_params = {'C':[0.001, 0.1, 10, 100], 'kernel':['rbf' ,'linear', 'poly', 'sigmoid']}

LR_params = {'C':[0.001, 0.1, 1, 10, 100]}

LDA_params = {'n_components':[None, 1,2,3], 'solver':['svd'], 'shrinkage':[None]}

KNN_params = {'n_neighbors':[1,5,10,20, 50], 'p':[2], 'metric':['minkowski']}

RF_params = {

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
classification_models = []



classification_models.append(('LR', LogisticRegression(), LR_params))

classification_models.append(('LDA', LinearDiscriminantAnalysis(), LDA_params))

classification_models.append(('KNN', KNeighborsClassifier(),KNN_params))

classification_models.append(('SVM', SVC(), SVM_params))

classification_models.append(('RF',RandomForestClassifier(),RF_params))
results = []

names = []





def estimator_function_classification(models_dict, data, scoring = 'accuracy'):

    

    X_train, X_test, y_train, y_test = data_divider(data, 'quality', -1, 0.8)



    for name, model, params in models_dict:

    

        kfold = KFold( n_splits=5 ,  random_state=2, shuffle=True)



        model_grid = GridSearchCV(model, params)



        cv_results = cross_val_score(model_grid, X_train, y_train, cv = kfold, scoring=scoring)

        

        model_grid.fit(X_train, y_train)



        results.append(cv_results)



        names.append(name)



        msg = "Cross Validation Accuracy %s: Accarcy: %f SD: %f" % (name, cv_results.mean(), cv_results.std())



        print(msg)

    print(f'The Best Estimator {model_grid.best_estimator_}')

    print(f'Best Parameters {model_grid.best_params_}')

        
estimator_function_classification(classification_models, data_clean)