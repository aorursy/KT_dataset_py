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
# import pandas as pd

# sample_submission = pd.read_csv("../input/sample_submission.csv")

# submission = pd.read_csv("../input/submission.csv")

# test = pd.read_csv("../input/test.csv")

# train = pd.read_csv("../input/train.csv")



data_dirty = pd.read_csv("../input/train.csv")

data_dirty['MSSubClass'] = data_dirty['MSSubClass'].astype('str')
data_dirty.head()
test_set_dirty = pd.read_csv('../input/test.csv')

test= test_set_dirty.iloc[1]

test
features = data_dirty.iloc[:,:-1].drop(['Id'], axis=1)
goal = data_dirty.iloc[:,-1]
# dividing features to categorical and numeric 

numerical_features= features.select_dtypes(exclude='object')

numerical_features.head()
categorical_featres = features.select_dtypes(include='object').astype('str')

categorical_featres.head()
# count the missing values 

def count_missing(dataframe):   

    total = dataframe.isnull().sum().sort_values(ascending=False)

    percent = (dataframe.isnull().sum()/dataframe.isnull().count()*100).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data

count_missing(numerical_features)
numerical_columns = numerical_features.columns

columns = data_dirty.columns

categorical_columns = categorical_featres.columns



# imputing missing data 

from sklearn.preprocessing import Imputer 

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

numerical_features = imputer.fit_transform(numerical_features)

numerical_features
## Numerical Values 

# Selecting the most relevant features according to the chi2 

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(numerical_features,goal)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(numerical_columns)

#concat two dataframes for better visualization 

numericalFeatureScores = pd.concat([dfcolumns,dfscores],axis=1)

numericalFeatureScores.columns = ['Specs','Chi2-Score']  #naming the dataframe columns

print(numericalFeatureScores.nlargest(15,'Chi2-Score'))  #print 10 best features
## Numerical 

# Feature Importance using extra tree classifier 

model = ExtraTreesClassifier()

model.fit(numerical_features,goal)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

numerical_feat_importances = pd.Series(model.feature_importances_, index=numerical_columns)

numerical_feat_importances.nlargest(10).plot(kind='barh')

plt.show()
# labeling the Categorical Data

from sklearn.preprocessing import LabelEncoder

encode = LabelEncoder()

for i in categorical_columns :

    categorical_featres[i] = encode.fit_transform(categorical_featres[i])

categorical_featres.head()    
# Selecting the most relevant features according to the chi2 

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(categorical_featres,goal)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(categorical_columns)

#concat two dataframes for better visualization 

categoricalFeatureScores = pd.concat([dfcolumns,dfscores],axis=1)

categoricalFeatureScores.columns = ['Specs','Chi2-Score']  #naming the dataframe columns

print(categoricalFeatureScores.nlargest(15,'Chi2-Score'))  #print 10 best features
# Correlation Heatmap 

corrmat = pd.concat([categorical_featres,pd.DataFrame(numerical_features, columns=numerical_columns),goal],axis=1).corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(data_dirty[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# Get the highest 10 features 

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(data_dirty[cols].values.T)

hm = sns.heatmap(cm, annot=True, square=True, yticklabels=cols.values, xticklabels=cols.values)

sns.set(font_scale=0.9)

plt.show()

corr_feats = cols.drop('SalePrice').values

corr_feats
chi2_score = pd.concat([numericalFeatureScores,categoricalFeatureScores])
chi2_feat = chi2_score.nlargest(10,'Chi2-Score')['Specs'].values
chi2_feat
features[corr_feats]
features[chi2_feat]
X_dirty = pd.concat([features[corr_feats],features[chi2_feat]] , axis=1)
X_dirty
X_dirty.drop(columns= 'GrLivArea', inplace=True)
GrLivArea = features.GrLivArea
X_dirty = pd.concat([X_dirty, pd.DataFrame(GrLivArea)], axis=1)
X_dirty
count_missing(X_dirty)
X = X_dirty.drop(columns='MasVnrArea')
X
X_feat = X.columns.values

X_feat
numerical_columns.values
np.intersect1d(X_feat, categorical_columns.values)
data_clean = pd.concat([X,pd.DataFrame(goal)], axis=1)


fig = plt.figure(figsize=(30,20))



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(data_clean.columns[0:5])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(2,3,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    plt.boxplot(x=data_clean[b])

    

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   


fig = plt.figure(figsize=(30,60))



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(data_clean.columns[0:5])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(5,1,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    plt.scatter(data_clean[b],data_clean['SalePrice'])

    

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   
data_clean.drop(data_clean[data_clean['GarageArea']>1200].index, inplace=True)

data_clean.drop(data_clean[data_clean['1stFlrSF']>2900].index, inplace=True)


fig = plt.figure(figsize=(30,20))



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(data_clean.columns[5:10])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(2,3,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    plt.boxplot(x=data_clean[b])

    

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   


fig = plt.figure(figsize=(30,60))



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(data_clean.columns[5:10])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(5,1,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    plt.scatter(data_clean[b],data_clean['SalePrice'])

    

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   
data_clean.drop(data_clean[data_clean['LotArea']>50000].index, inplace=True)

data_clean.drop(data_clean[data_clean['MiscVal']>=2000].index, inplace=True)
data_clean


fig = plt.figure(figsize=(30,20))



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(data_clean.columns[10:15])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(2,3,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    plt.boxplot(x=data_clean[b])

    

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   


fig = plt.figure(figsize=(30,60))



# Create 'for loop' to enerate though tumor features and compare with histograms

for i,b in enumerate(list(data_clean.columns[10:15])):

    

    # Enumerate starts at index 0, need to add 1 for subplotting

    i +=1

    

    # Create axes object for position i

    ax = fig.add_subplot(5,1,i)

    

    # Plot via histogram tumor charateristics using stacked and alpha parameters for..

    # comparisons.

    plt.scatter(data_clean[b],data_clean['SalePrice'])

    

    ax.set_title(b)



sns.set_style("whitegrid")

plt.tight_layout()

plt.legend()

plt.show()   
y = data_clean['SalePrice'].values

y
data_clean.drop('SalePrice', axis=1, inplace=True)
X = data_clean.values

X

len(data_clean.columns)
# standardizing the data 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean=False)

X = scaler.fit_transform(X)

X
# Split data into testing and training set. Use 80% for training

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8, random_state=0)
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLars
LR = LinearRegression()

LR.fit(X_train,y_train)
pred_train_LR= LR.predict(X_train)

print("Train Scores")

print(np.sqrt(mse(y_train,pred_train_LR)))

print(r2_score(y_train, pred_train_LR))



pred_test_LR= LR.predict(X_test)

print("\nTest Scores")

print(np.sqrt(mse(y_test,pred_test_LR))) 

print(r2_score(y_test, pred_test_LR))
import math



#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

def rmsle(y, y_pred): 

    assert len(y) == len(y_pred)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
rmsle(y_test, LR.predict(X_test))
kfold = KFold(n_splits=5, random_state=2, shuffle=True)

cv_results = cross_val_score(LR, X_train, y_train, cv=kfold, scoring='r2')
cv_results.mean()
# Define parameters for optimization using dictionaries {parameter name: parameter list}



RR_params = {'alpha':[10,100,1000]}

Lasso_params = {'alpha':[10,100,1000]}

Elastic_params = {'alpha':[10,100,1000]}

LassoLars_params = {'alpha':[10,100,1000]}

ForestRegressor_params = {'n_estimators': (10, 50, 100, 1000)}
# Append list of models with parameter dictionaries



models_opt = []



models_opt.append(('RidgeRegression', Ridge(), RR_params))

models_opt.append(('LASSO', Lasso(), Lasso_params))

models_opt.append(('ElasticNet', ElasticNet(),Elastic_params))

models_opt.append(('LassoLars', LassoLars(), LassoLars_params))

models_opt.append(('Random Forest Regressor', RandomForestRegressor(random_state=0), ForestRegressor_params))

results = []

names = []





def estimator_function(parameter_dictionary, scoring = 'r2'):

    

    

    for name, model, params in models_opt:

    

        kfold = KFold( n_splits=5 ,  random_state=0, shuffle=True)



        model_grid = GridSearchCV(model, params)



        cv_results = cross_val_score(model_grid, X_train, y_train, cv = kfold, scoring=scoring)



        results.append(cv_results)



        names.append(name)



        msg = "Cross Validation Accuracy %s: R2: %f " % (name, cv_results.mean())



        print(msg)
estimator_function(models_opt)
RFR = RandomForestRegressor(random_state=0)

kfold = KFold( n_splits=5 ,  random_state=0, shuffle=False)



model_grid = GridSearchCV(RFR, ForestRegressor_params)



cv_results = cross_val_score(model_grid, X_train, y_train, cv = kfold, scoring='r2')
cv_results.mean()
RFR.fit(X_train, y_train)
pred_train_RFR= RFR.predict(X_train)

print("Train Scores")

print(np.sqrt(mse(y_train,pred_train_RFR)))

print(r2_score(y_train, pred_train_RFR))



pred_test_RFR= RFR.predict(X_test)

print("\nTest Scores")

print(np.sqrt(mse(np.log(y_test),np.log(pred_test_RFR)))) 

print(r2_score(y_test, pred_test_RFR))
rmsle(y_test, RFR.predict(X_test))