# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale , LabelEncoder

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn import metrics



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



# Neural networks

import tensorflow as  tf

import keras



from keras.models import Sequential

from keras.layers import Dense, Activation

from keras.utils import np_utils



# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
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
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head()
# extract titles from names

df_train['NameTitle'] = df_train['Name'].str.split('(.*, )|(\\..*)').str[3]

df_test['NameTitle'] = df_test['Name'].str.split('(.*, )|(\\..*)').str[3]



df_train['NameTitle'].str.replace('Ms | Mlle','Miss')

df_train['NameTitle'].str.replace('Mme','Mr')

df_train['NameTitle'].str.replace('Mme','Mr')

df_train['NameTitle'][~df_train['NameTitle'].str.contains('Mr|Mrs|Miss|Master')] = 'Other'



df_test['NameTitle'].str.replace('Ms | Mlle','Miss')

df_test['NameTitle'].str.replace('Mme','Mr')

df_test['NameTitle'].str.replace('Mme','Mr')

df_test['NameTitle'][~df_test['NameTitle'].str.contains('Mr|Mrs|Miss|Master')] = 'Other'
# encode sex into ints

le = LabelEncoder()

df_train['Sex'] = le.fit_transform(df_train['Sex'])

df_test['Sex'] = le.fit_transform(df_test['Sex'])



df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())

df_test['Fare'] = df_train['Fare'].fillna(df_train['Fare'].mean())





df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

df_test['Age'] = df_train['Age'].fillna(df_train['Age'].mean())



df_train['Embarked'] = df_train['Embarked'].fillna("S")

df_test['Embarked'] = df_test['Embarked'].fillna("S")

df_train['Embarked'] = le.fit_transform(df_train['Embarked'])

df_test['Embarked'] = le.fit_transform(df_test['Embarked'])



df_train['Cabin'] = df_train['Cabin'].fillna("None")

df_test['Cabin'] = df_test['Cabin'].fillna("None")

df_train['Cabin'] = le.fit_transform(df_train['Cabin'])

df_test['Cabin'] = le.fit_transform(df_test['Cabin'])



df_train['Ticket'] = le.fit_transform(df_train['Ticket'])

df_test['Ticket'] = le.fit_transform(df_test['Ticket'])



df_train['NameTitle'] = le.fit_transform(df_train['NameTitle'])

df_test['NameTitle'] = le.fit_transform(df_test['NameTitle'])
plot_correlation_map(df_train)
features = list(df_train.columns.values)

# Remove unwanted features

features.remove('Name')

features.remove('PassengerId')

features.remove('Survived')

#features.remove('Ticket')

#features.remove('SibSp')

#features.remove('Parch')

#features.remove('Fare')

#features.remove('Cabin')

#features.remove('Embarked')

print(features)



y = df_train['Survived']

x = df_train[features]

x_t = df_test[features]

x.head()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=32)
x.shape[1]
model = Sequential()

model.add(Dense(output_dim=256, input_dim=x.shape[1]))

model.add(Activation("relu"))

model.add(keras.layers.core.Dropout(0.3))

model.add(Dense(output_dim=128))

model.add(Activation("relu"))

model.add(keras.layers.core.Dropout(0.2))

model.add(Dense(output_dim=64))

model.add(Activation("relu"))

model.add(Dense(output_dim=2))

model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

#a = model.fit(X_train.values, y_train, nb_epoch=500)
#loss_and_metrics = model.evaluate(X_test.values, y_test)

#print(loss_and_metrics)

#print("%s: %.2f%%" % (model.metrics_names[1], loss_and_metrics[1]*100))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

print("acc: %.2f%%" % (rf.score(X_test.values, y_test)*100))
classes = model.predict_classes(x_t.values, batch_size=32)
print(classes)
submission = pd.DataFrame({

    "PassengerId": df_test["PassengerId"],

    "Survived": classes})

print(submission)



submission.to_csv('titanic_lin.csv', index=False)