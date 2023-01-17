import warnings

warnings.filterwarnings('ignore')



# Handle table-like data and matrices

import numpy as np

import pandas as pd



import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns
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

    

def plot_correlation_map( df ):

    corr = titanic.corr()

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

    

def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()
# get titanic & test csv files as a DataFrame

from sklearn.preprocessing import OneHotEncoder

from sklearn import preprocessing



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print(train.head(5))

# print(train.to_string())





label_enc = preprocessing.LabelEncoder()

label_enc.fit(['male', 'female'])





sex_enc = label_enc.transform(train['Sex'])

print(sex_enc)

# LabelEncoder()



one_hot_enc = OneHotEncoder(categorical_features='all', dtype='numpy.string',

       handle_unknown='error', n_values='auto', sparse=True)







full = train.append( test , ignore_index = True )

titanic = full[ :891 ]

print('test:', test.shape)







print ('Datasets:' , 'full:' , full.shape , 'titanic:', titanic.shape)





plot_correlation_map(titanic)
plot_distribution(titanic, var = 'Fare', target = "Survived")

plot_distribution(titanic, var = 'Age', target = "Survived")



print ("!!!")