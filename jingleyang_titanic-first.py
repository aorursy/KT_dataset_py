import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image

%matplotlib inline



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

%matplotlib inline

mpl.style.use( 'ggplot' )

sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6







data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

data_train.sample(5)



sns.barplot(x="Pclass",y="Survived", hue="Sex", data=data_train)
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"]);
def simplify_ages(df):

    mean = df.Age.mean()

    df.Age = df.Age.fillna(mean)

    bins=(-1,0,5,12,18,25,30,60,120)

    group_names=['Error','Baby','Child','Teen','Student','Young','Adult','Senior']

    categories = pd.cut(df.Age,bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('X')

    df.Cabin = df.Cabin.apply(lambda x:x[0].upper())

    return df



def simplify_fares(df):

    mean = df.Fare.mean()

    df.Fare = df.Fare.fillna(mean)

    bins = (-1,0,8,15,32,1000)

    group_names = ['Error','low','normal','high','VIP']

    categories = pd.cut(df.Fare,bins, labels = group_names)

    df.Fare = categories

    return df



def title_map_func(val):

    if val in ['Mr','Mrs','Miss','Dr','Capt','Col','Major','Ms','Master']:

        return val

    else:

        return 'Empty'

def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split()[0][:-1])

    df['Title'] = df.Name.apply(lambda x: x.split()[1][:-1])

    df['Title'] = df.Title.apply(title_map_func)

    return df

def drop_features(df):

    df = df.drop(['Ticket','PassengerId','Name','SibSp','Parch','Lname','Family'],axis=1)

    return df

def family_size(df):

    df['Family'] = df['SibSp']+df['Parch']

    df['IsAlone'] = df.Family.apply(lambda x: 'alone' if x==0 else 'not')

    return df



def fillna_embarked(df):

    df.Embarked = df.Embarked.fillna('X')

    return df

def lable_pclass(df):

    level_map={1:'First',2:'Middle',3:'Low'}

    df.Pclass = df.Pclass.map(level_map)

    return df



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = family_size(df)

    df = fillna_embarked(df)

    df = lable_pclass(df)

    df = drop_features(df)

    return df



data_train = transform_features(data_train)

data_test = transform_features(data_test)

data_train.head(10)
from sklearn import preprocessing



def encode_features(df_train, df_test):

    features = list(data_train.columns.values)

    features.remove('Survived')

    #print(features)

    df_combined = pd.concat([df_train[features],df_test[features]])

    #print(df_combined.sample(5))

    for f in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[f])

        df_train[f] = le.transform(df_train[f])

        df_test[f] = le.transform(df_test[f])

        

        ohe = preprocessing.OneHotEncoder()

        ohe = ohe.fit(df_combined[f])

        df_train[f] = ohe.transform(df_train[f])

        df_test[f] = ohe.transform(df_test[f])

        

    return df_train, df_test



#encode_features(data_train, data_test)

features = list(data_train.columns.values)

features.remove('Survived')

df_combined = pd.concat([data_train[features],data_test[features]])

print(df_combined.sample(10))

print(data_train.shape,data_test.shape, df_combined.shape)

df_combined = pd.get_dummies(df_combined)
from sklearn.model_selection import train_test_split



X_all = df_combined.head(data_train.shape[0])

y_all = data_train['Survived']

num_test = 0.2



X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

X_test.sample(10)
from sklearn.ensemble import RandomForestClassifier

from mlxtend.plotting import plot_decision_regions



forest = RandomForestClassifier(criterion='entropy',

                               n_estimators=10,

                               random_state=0,

                               n_jobs=2)



forest.fit(X_train, y_train)

print(forest)

print(forest.score(X_test, y_test))

print(forest.feature_importances_)

colums_size = X_train.shape[1]

plt.plot(range(0,colums_size), forest.feature_importances_)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



#clf = RandomForestClassifier()



parameters = {'n_estimators': [4, 10], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2,6], 

              'min_samples_split': [2, 5],

              'min_samples_leaf': [1,4]

             }



#acc_scorer = make_scorer(accuracy_score)

#grid_obj = GridSearchCV(clf, parameters, scoring = acc_scorer)



#grid_obj = grid_obj.fit(X_train,y_train)

#clf = grid_obj.best_estimator_

clf = RandomForestClassifier(criterion='entropy',

                               n_estimators=10,

                               random_state=0,

                               n_jobs=2)

clf.fit(X_train,y_train)
print(clf)

print(clf.score(X_test, y_test))

print(clf.feature_importances_)
clf = SVC()

print(clf.fit(X_train,y_train))

print(clf.score(X_test, y_test))
clf = GradientBoostingClassifier()

print(clf.fit(X_train,y_train))

print(clf.score(X_test, y_test))
clf = KNeighborsClassifier(n_neighbors = 3)

print(clf.fit(X_train,y_train))

print(clf.score(X_test, y_test))
clf = GaussianNB()

print(clf.fit(X_train,y_train))

print(clf.score(X_test, y_test))
clf = LogisticRegression()

print(clf.fit(X_train,y_train))

print(clf.score(X_test, y_test))
clf = SVC(kernel='rbf', random_state=0,gamma=0.1, C=10)

print(clf.fit(X_train,y_train))

print(clf.score(X_test, y_test))

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from mlxtend.plotting import plot_decision_regions

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)



lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train_std, y_train)

X_test_lda = lda.transform(X_test_std)



clf = LogisticRegression()

print(clf.fit(X_train_lda,y_train))

print(clf.score(X_test_lda, y_test))





plot_decision_regions(X_test_lda, y_test.values, clf)

plt.show()
pca = PCA()

X_train_pca = pca.fit_transform(X_train_std)

var_len = len(pca.explained_variance_ratio_)

plt.bar(range(1,var_len+1), pca.explained_variance_ratio_)

plt.step(range(1,var_len+1), np.cumsum(pca.explained_variance_ratio_))

plt.show()



pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_std)

X_test_pca = pca.transform(X_test_std)



plt.scatter(X_train_pca[:,0],X_train_pca[:,1])

plt.show()



clf = SVC()

clf = clf.fit(X_train_pca, y_train)



plot_decision_regions(X_test_pca, y_test.values, clf)

plt.show()
import xgboost as xgb

clf= xgb.XGBClassifier()
from sklearn.cross_validation import KFold

def run_kfold(clf):

    tot_size = X_all.shape[0]

    kf = KFold(tot_size,  n_folds = 5)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold +=1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        sc = StandardScaler()

        X_train_std = sc.fit_transform(X_train)

        X_test_std = sc.transform(X_test)

        clf.fit(X_train_std, y_train)

        outcomes.append(clf.score(X_test_std, y_test))

    return outcomes

print(clf)

out = run_kfold(clf)

mean_val = np.mean(out)

print(out)

print(mean_val)
# train them all

clf.fit(X_all,y_all)
X_all.sample(10)

data_test = pd.read_csv('../input/test.csv')

data_test.sample(10)

ids = data_test['PassengerId']



predict_input = df_combined.tail(data_test.shape[0])



predict_output = clf.predict(predict_input)



output = pd.DataFrame({'PassengerId':ids, 'Survived':predict_output })

print(output.head())

print(output.tail())

print(clf)

output.to_csv("titanic-output-0113-v2.csv",index=False)


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
plot_variable_importance(X_train, y_train)