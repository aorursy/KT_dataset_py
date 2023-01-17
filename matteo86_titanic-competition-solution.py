# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
#%matplotlib inline
#mpl.style.use( 'ggplot' )
#sns.set_style( 'white' )
#pylab.rcParams[ 'figure.figsize' ] = 8 , 6



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
    
    
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
edge = range(0,90,10)
lbs = ["%d"%i for i in range(len(edge)-1)]
train['Age_cat']=pd.cut(train['Age'],bins=8,labels=lbs,include_lowest=True)
train.info()
train.head()
plot_categories(train,cat='Pclass', target = 'Survived')
train.groupby(['Pclass'])['Fare'].mean()

train.groupby(['Pclass'])['Fare'].max()

_=train.loc[train['Fare']>0]
_.groupby(['Pclass'])['Fare'].min()
edge = range(0,500,50)
lbs = ["%d"%i for i in range(len(edge)-1)]
train['Fare_cat']=pd.cut(train['Fare'],bins=9,labels=lbs,include_lowest=True)
plot_categories(train,cat='Fare_cat',target='Survived')
plot_categories(train,cat='Fare_cat',target='Survived',col='Pclass',row='Sex')
_=train.loc[train['Sex']=='male']
plot_categories(_,cat='Age_cat',target='Survived',row='Pclass')
embarked = pd.DataFrame()
embarked = pd.get_dummies(train['Embarked'])
embarked['class']=train['Pclass']
mtx = np.zeros((3,3))
mtx = np.zeros((3,3))
for i in [1,2,3]:
    j_aux=0
    for j in ['C','Q','S']:
        mtx[i-1][j_aux]=embarked[(embarked['class']==i) & (embarked[j]==1)].sum()[j_aux]
        j_aux=j_aux+1
        
label=['1st class','2nd class','3rd class']        
class_embark= pd.DataFrame(columns=['C','Q','S'],index=label)
class_embark['C']=mtx[:,0]
class_embark['Q']=mtx[:,1]
class_embark['S']=mtx[:,2]
class_embark.plot.bar()
class_embark_trnsp=class_embark.transpose()
class_embark_trnsp.plot.bar()

plot_categories(train,cat='Embarked',target='Survived')

plot_categories(train,cat='Sex',target='Survived')
#edge = range(0,90,10)
#lbs = ["%d"%i for i in range(len(edge)-1)]
#train['Age_cat']=pd.cut(train['Age'],bins=8,labels=lbs,include_lowest=True)
plot_categories(train,cat='Age_cat', target = 'Survived')
plot_categories( train , cat = 'Age_cat' , target = 'Survived' , col = 'Sex' )
plot_categories( train , cat = 'Age_cat' , target = 'Survived' , col = 'Pclass' )
#get a second name list for all passenger
numb_passg = int(train.describe().loc['count','PassengerId'])
list_surname=[]
list_surname_full=[]
for i in range(0,(numb_passg)):
    tmp=(train.loc[i,'Name'].split( ',' )[0])
    list_surname_full.append(tmp)
    if list_surname.count(tmp) == 0:
        list_surname.append(tmp)
family_size = pd.DataFrame()
family_size['LastName']=list_surname
family_size['counts']=list(map(lambda x: list_surname_full.count(x),family_size['LastName']))
train['LastName']=list_surname_full
train['NotAlone']=list(map(lambda x: list_surname_full.count(x),train['LastName']))
train['NotAlone_v2']=np.where(((train['NotAlone']>1) & (train['SibSp']>0)) | ((train['NotAlone']>1) & (train['Parch']>0)),train["NotAlone"],train['SibSp']+train['Parch']+1)
title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = train[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )

# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"

                    }

# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )
#title = pd.concat( [ title , titles_dummies ] , axis = 1 )
title['TitleCat']=np.zeros(891)
title['Sex']=train['Sex']
title.head()
#title['TitleCat']=np.where(title['Royalty']==1,'Roy',title['TitleCat'])
title['TitleCat']=np.where(title['Officer']==1,'Off',title['TitleCat'])
title['TitleCat']=np.where(title['Mrs']==1,'Mrs',title['TitleCat'])
title['TitleCat']=np.where(title['Mr']==1,'Mr',title['TitleCat'])
title['TitleCat']=np.where(title['Miss']==1,'Miss',title['TitleCat'])
title['TitleCat']=np.where(title['Master']==1,'Master',title['TitleCat'])
title['TitleCat']=np.where((title['Royalty']==1)&(title['Sex']=='male'),'Roy_m',title['TitleCat'])
title['TitleCat']=np.where((title['Royalty']==1)&(title['Sex']=='female'),'Roy_f',title['TitleCat'])
title['Survived']=train['Survived']
plot_categories(title,cat='TitleCat',target='Survived')