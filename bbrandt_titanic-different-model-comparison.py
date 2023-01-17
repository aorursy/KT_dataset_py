# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



# Modelling Algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier



# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV



# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")





full = train.append( test , ignore_index = True )



print ('train:' , train.shape)

train.head()
train.describe()
pd.isnull(full).sum() > 0
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

    

plot_correlation_map( train )
#not working right now due to a bug in seaborn. Is expected to work again after library upgrade of kaggle

def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()

    

#plot_distribution( titanic , var = 'Age' , target = 'Survived' , row = 'Sex' )
def plot_categories( df , cat , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , row = row , col = col )

    facet.map( sns.barplot , cat , target )

    facet.add_legend()



plot_categories( train , cat = 'Embarked' , target = 'Survived' )

plot_categories( train , cat = 'Sex' , target = 'Survived' )

plot_categories( train , cat = 'Pclass' , target = 'Survived' )

plot_categories( train , cat = 'SibSp' , target = 'Survived' )

plot_categories( train , cat = 'Parch' , target = 'Survived' )
# Transform Sex into binary values 0 and 1

sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )

sex.head()
# Create a new variable for every unique value of Embarked

embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )

embarked.head()
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )

pclass.head()
def splitXY(dataset):

    X=dataset[dataset.columns.difference(['Survived'])]

    Y=dataset.Survived

    return [X,Y]



def trainValidationAndTestData(dataset,lastTrainIndex): 

    datasetX, datasetY = splitXY(dataset)

    train_X , valid_X , train_Y , valid_Y = train_test_split( datasetX[ 0:lastTrainIndex ] , datasetY[ 0:lastTrainIndex ] , train_size = .7 )

    test_X = datasetX[ lastTrainIndex: ] 

    test_Y = datasetY[ lastTrainIndex: ]

    trainValidationAndTest=[ train_X, train_Y, valid_X, valid_Y, test_X, test_Y ]

    return trainValidationAndTest



# Create dataset

dataset1 = pd.DataFrame()

# Fill missing values of Age with the median of Age (median)

dataset1[ 'Age' ] = full.Age.fillna( full.Age.median() )

# Fill missing values of Fare with the median of Age (median)

dataset1[ 'Fare' ] = full.Fare.fillna( full.Fare.median() )

# Additional features to include  

dataset1 =  pd.concat( [ dataset1 , embarked , pclass, sex, full.Survived ] , axis=1 ); 

trainValidationAndTest1 = trainValidationAndTestData(dataset1,891)





# Create dataset

dataset2 = pd.DataFrame()

dataset2[ 'SibSp' ] = full.SibSp

dataset2[ 'Parch' ] =  full.SibSp

# Additional features to include  

dataset2 =  pd.concat( [ dataset1 , dataset2 ] , axis=1 ); 

trainValidationAndTest2 = trainValidationAndTestData(dataset2,891)





 # Create dataset

dataset3 = pd.DataFrame()

# Additional features to include  

dataset3 = pd.concat( [ full.Age , full.Fare, embarked , pclass, sex, full.Survived, full.SibSp,

                      full.Parch ] , axis=1 );

trainValidationAndTest3 = trainValidationAndTestData(dataset3,891)

train3 = pd.DataFrame()

train3 = pd.concat([trainValidationAndTest3[0], trainValidationAndTest3[1]],axis=1)

valid3 = pd.DataFrame()

valid3 = pd.concat([trainValidationAndTest3[2], trainValidationAndTest3[3]],axis=1)

test3 = pd.DataFrame()

test3 = trainValidationAndTest3[4]

 

train3=train3.dropna(axis=0, how='any')

valid3=valid3.dropna(axis=0, how='any')

test3.Age.fillna( full.Age.median(),inplace=True )

test3.Fare.fillna( full.Fare.median(),inplace=True )  



train3X, train3Y = splitXY(train3)

valid3X, valid3Y = splitXY(valid3)

trainValidationAndTest3=[train3X,train3Y,valid3X,valid3Y,test3]

test3.info()
def runallmodels(trainValidationAndTest):

    train_X=trainValidationAndTest[0]

    train_Y=trainValidationAndTest[1]

    

    modellr = LogisticRegression()

    modellr.fit( train_X , train_Y )

    

    modelrfc = RandomForestClassifier(n_estimators=100)

    modelrfc.fit( train_X , train_Y )

    

    modelrfc2 = ExtraTreesClassifier(n_estimators=100)

    modelrfc2.fit( train_X , train_Y )

    

    modelsvc = SVC()

    modelsvc.fit( train_X , train_Y )



    modelgbc = GradientBoostingClassifier()

    modelgbc.fit( train_X , train_Y )

    

    modelknc = KNeighborsClassifier(n_neighbors = 3)

    modelknc.fit( train_X , train_Y )

    

    modelgnb = GaussianNB()

    modelgnb.fit( train_X , train_Y )



    models = [modellr,modelrfc, modelrfc2, modelsvc, modelgbc, modelknc, modelgnb]

    

    return models



def scoreallmodels(models,trainValidationAndTest):

    train_X=trainValidationAndTest[0]

    train_Y=trainValidationAndTest[1]

    valid_X=trainValidationAndTest[2]

    valid_Y=trainValidationAndTest[3]

    

    modellr = models[0]

    modelrfc = models[1]

    modelrfc2 = models[2]

    modelsvc = models[3]

    modelgbc = models[4]

    modelknc = models[5]

    modelgnb = models[6]

    

    print ("Logistic Regression (training,validation)          :" , 

           modellr.score( train_X , train_Y ) , modellr.score( valid_X , valid_Y ))

    print ("Random Forest Model (training,validation)          :" , 

           modelrfc.score( train_X , train_Y ) , modelrfc.score( valid_X , valid_Y ))

    print ("ExtraTreesClassifier (training,validation)         :" , 

           modelrfc2.score( train_X , train_Y ) , modelrfc2.score( valid_X , valid_Y ))

    print ("Support Vector Machines (training,validation)      :" , 

           modelsvc.score( train_X , train_Y ) , modelsvc.score( valid_X , valid_Y ))

    print ("Gradient Boosting Classifier (training,validation) :" , 

           modelgbc.score( train_X , train_Y ) , modelgbc.score( valid_X , valid_Y ))

    print ("k-nearest neighbors (training,validation)          :" , 

           modelknc.score( train_X , train_Y ) , modelknc.score( valid_X , valid_Y ))

    print ("Gaussian Naive Bayes (training,validation)         :" , 

           modelgnb.score( train_X , train_Y ) , modelgnb.score( valid_X , valid_Y ))

    



    

models1=runallmodels(trainValidationAndTest1);

scoreallmodels(models1,trainValidationAndTest1)

print("")

models2=runallmodels(trainValidationAndTest2);

scoreallmodels(models2,trainValidationAndTest2)

print("")

models3=runallmodels(trainValidationAndTest3);

scoreallmodels(models3,trainValidationAndTest3)

selectedmodel=models4[1]



test_Y = selectedmodel.predict( trainValidationAndTest2[4] ).astype(int)

passenger_id = full[891:].PassengerId

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test.shape

test.head()

test.to_csv( 'titanic_pred.csv' , index = False )