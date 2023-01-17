# Handle table-like data and matrices

import numpy as np

import pandas as pd



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
#Setup Helper Functions

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



def plot_correlation_map( df ):

    corr = df.corr()

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
data = pd.read_csv("../input/pollution_us_2000_2016.csv")
data.shape
data.columns
data.head(3)
NY_mask = data['City'].str.contains('New York')

ny = data[NY_mask]
ny.head(2)
del ny['State']

del ny['Unnamed: 0']

del ny['State Code']

del ny['City']

del ny['Address']
ny.head()
ny.shape
ny.info()
del ny['NO2 Units']

del ny['NO2 1st Max Hour']

del ny['O3 Units']

del ny['O3 1st Max Hour']

del ny['SO2 Units']

del ny['SO2 1st Max Hour']

del ny['CO Units']

del ny['CO 1st Max Hour']
ny.head(2)
ny.shape
ny.describe()
plot_correlation_map(ny)
newdata = pd.DataFrame(ny, columns = ['NO2 Mean', 'NO2 1st Max Value', 'NO2 AQI', 'O3 Mean', 'O3 1st Max Value', 'O3 AQI', 'SO2 Mean', 'SO2 1st Max Value', 'SO2 AQI', 'CO Mean', 'CO 1st Max Value', 'CO AQI']) 
newdata.isnull().any()
newdata.shape
X = newdata.interpolate()

X.shape
X.isnull().any()
X = X.dropna()

X.shape
x_before_pca = pd.DataFrame(X)

x_before_pca.describe()
x_before_pca.shape
plot_correlation_map(x_before_pca)
from sklearn.decomposition import PCA, RandomizedPCA
pca = PCA(n_components=12)
pca.fit(x_before_pca)
#The amount of variance that each PC explains

var = pca.explained_variance_ratio_
#Cumulative Variance explains

var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
var1
plt.plot(var1)
x_pca = PCA(n_components=3)
x_pca.fit(x_before_pca)
x = x_pca.fit_transform(x_before_pca)
type(x[:,0])
x.shape
d = {'pc1': x[:,0], 'pc2': x[:, 1], 'pc3': x[:,2]}

x_df = pd.DataFrame(d)
x_df.shape
x_df.head(3)
x_df.describe()
x_new_ndarray = x_pca.inverse_transform(x_df)

x_new = pd.DataFrame(x_new_ndarray)

x_new.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']
x_new.shape
x_new.head(3)
x_before_pca.head(3)
plt.scatter(x_df['pc1'], x_df['pc2'], color = 'blue')
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs = x_df['pc1'], ys = x_df['pc2'], zs= x_df['pc3'], zdir='z')