# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import matplotlib as mpl

% matplotlib inline



import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier



from sklearn.cross_validation import train_test_split,StratifiedKFold

from sklearn.preprocessing import Imputer,Normalizer,scale

from sklearn.feature_selection import RFECV



mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 8,6
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



full = train.append(test,ignore_index=True)

titanic = full[:891]



full.shape,titanic.shape
del train,test
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
titanic.head()
titanic.describe()
plot_correlation_map(titanic)
plot_distribution( titanic, var = 'Age', target= 'Survived', row = 'Sex')
plot_distribution(titanic, var = 'Fare', target = 'Survived')
plot_categories(df = titanic, cat = 'Embarked', target= 'Survived')
plot_categories(df = titanic, cat = 'Sex', target= 'Survived')

plot_categories(df = titanic, cat = 'SibSp', target= 'Survived')

plot_categories(df = titanic, cat = 'Parch', target= 'Survived')

plot_categories(df = titanic, cat = 'Pclass', target= 'Survived')
sex = pd.Series(np.where(full.Sex == 'male',1,0),name='Sex')
embarked = pd.get_dummies(full.Embarked,prefix='Embarked')

embarked.head()
pclass = pd.get_dummies(full.Pclass,prefix='Pclass')

pclass.head()
print(full.Embarked)
imputed = pd.DataFrame()

imputed['Age'] = full.Age.fillna(full.Age.mean())

imputed['Fare'] = full.Fare.fillna(full.Fare.mean())



imputed.head()
title = pd.DataFrame()

title['Title'] = full['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())



title.describe()
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



title.Title = title.Title.map(Title_Dictionary)

title = pd.get_dummies(title.Title)



title.head()
cabin = pd.DataFrame()

cabin['Cabin'] = full.Cabin.fillna("U")

cabin.Cabin = cabin.Cabin.str.extract('([A-Z])',expand=False)

cabin = pd.get_dummies(cabin.Cabin,prefix='Cabin')

cabin.head()
def cleanTicket(ticket):

    ticket = ticket.replace('.','')

    ticket = ticket.replace('/','')

    ticket = ticket.split()

    ticket = map(lambda c:c.strip(),ticket)

    ticket = list(filter(lambda c:not c.isdigit(),ticket))

    if len(ticket) > 0:

        return ticket[0]

    else:

        return 'XXX'

    

ticket = pd.DataFrame()

ticket['Ticket'] = full.Ticket.map(cleanTicket)

ticket = pd.get_dummies(ticket,prefix='Ticket')



ticket.head()

                            
familysize = pd.DataFrame()

familysize['FamilySize'] = full.Parch + full.SibSp + 1



familysize['FamilySingle'] = familysize.FamilySize.map(lambda s:1 if s==1 else 0)

familysize['FamilySmall'] = familysize.FamilySize.map(lambda s:1 if 2<=s<=4 else 0)

familysize['FamilyLarge'] = familysize.FamilySize.map(lambda s:1 if s>=5 else 0)



familysize.head()
full_x = pd.concat([imputed,familysize,cabin,embarked,sex],axis = 1)

full_x.head()
train_valid_x = full_x[:891]

test_x = full_x[891:]

train_valid_y = titanic.Survived

train_x,valid_x,train_y,valid_y = train_test_split(train_valid_x,train_valid_y,train_size = 0.7)



train_x.shape,valid_x.shape,train_y.shape,valid_y.shape,test_x.shape
plot_variable_importance(train_x,train_y)
model = SVC()

model.fit(train_x,train_y)

print(model.score(train_x,train_y),model.score(valid_x,valid_y))
model = LinearSVC()

model.fit(train_x,train_y)

print(model.score(train_x,train_y),model.score(valid_x,valid_y))

model = LogisticRegression()

model.fit(train_x,train_y)

print(model.score(train_x,train_y),model.score(valid_x,valid_y))

model = GaussianNB()

model.fit(train_x,train_y)

print(model.score(train_x,train_y),model.score(valid_x,valid_y))

model = RandomForestClassifier(n_estimators=100)

model.fit(train_x,train_y)

print(model.score(train_x,train_y),model.score(valid_x,valid_y))

model = KNeighborsClassifier(n_neighbors=5)

model.fit(train_x,train_y)

print(model.score(train_x,train_y),model.score(valid_x,valid_y))

model = GradientBoostingClassifier()

model.fit(train_x,train_y)

print(model.score(train_x,train_y),model.score(valid_x,valid_y))
test_y = model.predict(test_x)

test_y = test_y.astype(int)

predict_y = pd.DataFrame({

    'PassengerId':full[891:].PassengerId,

    'Survived':test_y

})

predict_y.to_csv('predict_y.csv',index = False)
model.score(train_x,train_y)