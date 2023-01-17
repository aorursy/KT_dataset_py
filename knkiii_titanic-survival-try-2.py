%matplotlib inline

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
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

    

def plot_categories( df , cat , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , row = row , col = col )

    facet.map( sns.barplot , cat , target )

    facet.add_legend()
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
full = train.append(test, ignore_index='True')
titanic = full[:len(train.index)]

titanic.head()

del train, test
titanic.describe()
plot_correlation_map( titanic )
# Plot distributions of Age of passangers who survived or did not survive

plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
# Plot survival rate by Embarked

plot_categories( titanic , cat = 'Embarked' , target = 'Survived' )

plot_categories( titanic , cat = 'Sex' , target = 'Survived' )

plot_categories( titanic , cat = 'Pclass' , target = 'Survived' )
sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

sex.tail()
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

pclass.head()
full_X = pd.concat( [ pclass , sex ] , axis=1 )

full_X.head()
# Create all datasets that are necessary to train, validate and test models

train_valid_X = full_X[ 0:891 ]

train_valid_y = titanic.Survived

test_X = full_X[ 891: ]

train_X , valid_X , train_y , valid_y = train_test_split(train_valid_X , train_valid_y , train_size = .7 )



print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape , valid_y.shape , test_X.shape)
model = KNeighborsClassifier(n_neighbors = 3)
model.fit( train_X , train_y )
print (model.score( train_X , train_y ) , model.score( valid_X , valid_y ))
test_Y = model.predict( test_X ).astype(int)

print(test_Y)

passenger_id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y} )

test.shape

test.to_csv( 'titanic_pred.csv' , index = False )
