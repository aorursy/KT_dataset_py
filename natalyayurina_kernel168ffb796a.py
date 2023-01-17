!pip install mlens
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np

! pip uninstall --yes pandas pandas-datareader pandas-gbq pandas-profiling sklearn-pandas
! pip install sklearn-pandas==1.8.0 pandas==1.0.5 pandas-datareader==0.8.1 pandas-gbq==0.11.0 pandas-profiling==1.4.1



import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from xgboost import XGBRegressor

# Modelling Helpers
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import  Normalizer , scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.feature_selection import RFECV
from mlens.ensemble import SuperLearner
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import auc, roc_curve, roc_auc_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report


# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
from pylab import *
! pip install google-colab
def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( data ):
    corr = data.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))

def score(clf, x_train, x_test, y_train, y_test):
    try:
        y_pred_train, y_pred_test = clf.predict_proba(x_train)[:, 1], clf.predict_proba(x_test)[:, 1]
        print(f'Train-test roc auc: {roc_auc_score(y_train.astype(bool), y_pred_train)}, {roc_auc_score(y_test.astype(bool), y_pred_test)}')
    except AttributeError:
        y_pred_train, y_pred_test = clf.predict(x_train), clf.predict(x_test)
        print(f'Train-test r2 score: {r2_score(y_train, y_pred_train)}, {r2_score(y_test, y_pred_test)}')
      
data = pd.read_csv('../input/wine-quality/winequalityN.csv')
data.head()
data.info()
data.describe().transpose()
plot_correlation_map( data )
plt.figure(figsize=(19,2))
sns.boxplot(x=data['alcohol'])
# Plot distributions of Alcohol and wine's quality
plot_distribution( data , var = 'alcohol' , target = 'quality' , row = 'type' )
# Plot quality rate by type
plot_categories( data , cat = 'type' , target = 'quality' )
cat_feat_data = data[['type']].apply(LabelEncoder().fit_transform)
num_feat_data = data.drop(['type'], axis=1)
data = pd.concat([num_feat_data, cat_feat_data], axis=1)
cat_feat = cat_feat_data.columns
num_feat = num_feat_data.columns
data.head()
data.isnull().sum()
print('Для фиксированной кислотности пустых строк ' + str( len( data[ pd.isnull( data['fixed acidity'] ) ] ) ))
print('Для летучей кислотности пустых строк ' + str( len( data[ pd.isnull( data['volatile acidity'] ) ] ) ))
print('Для лимонной кислоты пустых строк ' + str( len( data[ pd.isnull( data['citric acid'] ) ] ) ))
print('Для остаточного сахара пустых строк ' + str( len( data[ pd.isnull( data['residual sugar'] ) ] ) ))
print('Для хлоридов пустых строк ' + str( len( data[ pd.isnull( data['chlorides'] ) ] ) ))
print('Для Ph пустых строк ' + str( len( data[ pd.isnull( data['pH'] ) ] ) ))
print('Для сульфатов пустых строк ' + str( len( data[ pd.isnull( data['sulphates'] ) ] ) ))
print('Всего строк в наборе ' + str( len( data ) ))
data.corrwith(data['fixed acidity']).sort_values(ascending=False)
data[data['fixed acidity'].isnull()].groupby('density').head()
((data.groupby('fixed acidity')['density'].value_counts()).sort_values(ascending=False))
data.loc[data['fixed acidity'].isnull(), 'fixed acidity'] = data.groupby('density')['fixed acidity'].transform('mean')
data.loc[data['fixed acidity'].isnull(), 'fixed acidity'] = data.loc[(data['density']>0.9963)&(data['density']<0.9964)]['fixed acidity'].mean()
data[data['fixed acidity'].isnull()]
data.corrwith(data['volatile acidity']).sort_values(ascending=False)
((data.groupby('volatile acidity')['chlorides'].value_counts()).sort_values(ascending=False))
data.loc[data['volatile acidity'].isnull(), 'volatile acidity'] = data.groupby('chlorides')['volatile acidity'].transform('mean')
data[data['volatile acidity'].isnull()]
data.corrwith(data['citric acid']).sort_values(ascending=False)
data.loc[data['citric acid'].isnull(), 'citric acid'] = data.groupby('citric acid')['fixed acidity'].transform('mean')
data.loc[data['citric acid'].isnull(), 'citric acid'] = data.groupby('citric acid')['volatile acidity'].transform('mean')
data.loc[data['citric acid'].isnull(), 'citric acid'] = data.groupby('citric acid')['pH'].transform('mean')
data.loc[data['citric acid'].isnull(), 'citric acid'] = data.loc[(data['fixed acidity']>5.2)&(data['fixed acidity']<5.4)]['citric acid'].mean()
data[data['citric acid'].isnull()]
data.corrwith(data['pH']).sort_values(ascending=False)
((data.groupby('volatile acidity')['pH'].value_counts()).sort_values())#ascending=False))
data[(data['volatile acidity']>1)].groupby(['pH'])['citric acid'].sum().plot(grid=True, xticks=range(0,10))
# data_27_45[(data_27_45['Item_Identifier']=='FDA15')].groupby(['Outlet_Location_Type'])['Item_Outlet_Sales'].sum().plot(grid=True)#, xticks=range(0,10))
# data_27_45[(data_27_45['Item_Identifier']=='FDZ20')].groupby(['Outlet_Location_Type'])['Item_Outlet_Sales'].sum().plot(grid=True)
# data_27_45[(data_27_45['Item_Identifier']=='FDF05')].groupby(['Outlet_Location_Type'])['Item_Outlet_Sales'].sum().plot(grid=True)
# data_27_45[(data_27_45['Item_Identifier']=='FDA04')].groupby(['Outlet_Location_Type'])['Item_Outlet_Sales'].sum().plot(grid=True)

data.loc[(data['pH'].isnull())&(data['volatile acidity']>=0.2)&(data['volatile acidity']<=0.23), 'pH'] \
= data.groupby('volatile acidity')['pH'].transform('mean')
data.loc[(data['pH'].isnull())&(data['volatile acidity']>=0.28)&(data['volatile acidity']<=0.32), 'pH'] \
= data.groupby('volatile acidity')['pH'].transform('mean')
data.loc[(data['pH'].isnull())&(data['volatile acidity']>=0.43)&(data['volatile acidity']<=0.45), 'pH'] \
= data.groupby('volatile acidity')['pH'].transform('mean')
data.loc[(data['pH'].isnull())&(data['volatile acidity']>=0.695)&(data['volatile acidity']<=0.71), 'pH'] \
= data.groupby('volatile acidity')['pH'].transform('mean')
data[data['pH'].isnull()]
data.corrwith(data['sulphates']).sort_values(ascending=False)
data['sulphates'].value_counts()
((data.groupby('chlorides')['sulphates'].value_counts()).sort_values(ascending=False))
data.loc[data['sulphates'].isnull(), 'sulphates'] = data.groupby('chlorides')['sulphates'].transform('mean')
data[data['sulphates'].isnull()]
data.corrwith(data['residual sugar']).sort_values(ascending=False)
data.loc[data['residual sugar'].isnull(), 'residual sugar'] = data.groupby('density')['residual sugar'].transform('mean')
data[data['residual sugar'].isnull()]
data.corrwith(data['chlorides']).sort_values(ascending=False)
data.loc[data['chlorides'].isnull(), 'chlorides'] = data.groupby('sulphates')['chlorides'].transform('mean')
data[data['chlorides'].isnull()]
data.isnull().sum()
y = data['quality']
X = data.drop(['quality'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.shape, y_train.shape, X_test.shape, y_train.shape
plot_variable_importance(X_train, y_train)
model_cat = XGBClassifier(n_estimators=1000, learning_rate=0.2, max_depth=4, silent=True)

model_cat.fit( X_train , y_train )
#print (score(model_cat, X_train, X_test, y_train, y_test))
print (model_cat.score( X_train , y_train ) , model_cat.score( X_test , y_test ))
#Линейную регрессию
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_test = model_lr.predict(X_test)
y_pred_train = model_lr.predict(X_train)
print("R2: \t", r2_score(y_train, y_pred_train),r2_score(y_test, y_pred_test))
#Бустинг
model_cat = XGBRegressor(n_estimators=1000, learning_rate=0.2, max_depth=4, silent=True)
model_cat.fit(X_train, y_train)
y_pred_test = model_cat.predict(X_test)
y_pred_train = model_cat.predict(X_train)
print(mean_squared_error(y_train, y_pred_train), mean_squared_error(y_test, y_pred_test))
print(r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)  )   
skf = KFold(n_splits=10, random_state=None, shuffle=False)
train_metric, test_metric = [], []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    clf_tree = RandomForestRegressor(n_estimators=1000, max_features=5)
    clf_tree.fit(X_train, y_train)
    y_pred_train_rf, y_pred_test_rf = model_cat.predict(X_train), model_cat.predict(X_test)
    mean_squared_error(y_train, y_pred_train_rf), mean_squared_error(y_test, y_pred_test_rf)
    train_metric.append(r2_score(y_train, y_pred_train_rf))
    test_metric.append(r2_score(y_test, y_pred_test_rf))
    print(r2_score(y_train, y_pred_train_rf), r2_score(y_test, y_pred_test_rf))
print(sum(train_metric)/len(train_metric))
print(sum(test_metric)/len(test_metric))