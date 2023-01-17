import pandas as pd

import numpy as np

import scipy 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.options.display.max_rows = 999

verbose = False # param for debugging
# Load data

train = pd.read_csv('../input/train.csv', header=0,sep=',')

test = pd.read_csv('../input/test.csv', header=0)
# Let's look at the general information about the training set

train.info()
# Create a function for processing NaN values



def fillNaN(df=train,drop=True):

    if drop:

        df.dropna(subset=['Embarked','Fare'],inplace=True,axis=0)  # Drop  NaN for Embarked and Fare

    else:

        df['Fare']=df['Fare'].fillna(0.)                           # Fill NaN as 0 for Fare

        df['Embarked']=df['Embarked'].fillna('S')                  # Fill NaN as S for Embarked

    df['Age']=df['Age'].fillna(95.)                                # Fill NaN as 95 for Age. I chose 95 because max Age is 80

    df['Cabin']=df['Cabin'].fillna('-1')                           # Fill NaN as -1 for Cabin 

fillNaN()

fillNaN(test,False)



# Check empty values

train.info()
train.groupby('Cabin').count()

# We will see 'T' as noise
# I want to get code for Cabine, 

# I suppose that the first letter is ship's deck ( depends on location and class of passenger)



data_clear = train.copy()  



# T is noise in the field

data_clear = data_clear[train.Cabin!= 'T']    





cabin,cab_map = [],[]



def modCabin(df):

    global cabin,cab_map

    df['cab'] = df['Cabin']                  # Create a new column

    df['cab'] = [ x[0] for x in df.Cabin]         # get the first letter from Cabin

    if len(cabin) == 0:

        cabin = df['cab'].unique()                         # get unique values

        cabin.sort(axis=0)                                         # sort values

        print(cabin)                                                # note NA = '-'

        cab_map =np.arange(len(cabin))                             # List of numbers for mapping 

    df['cab'].replace(cabin,cab_map,inplace=True)      # Map letters to numbers 

    df['cab'].astype(dtype='int64')                  # Convert data type to  int64
modCabin(data_clear)

modCabin(test)
# Create a function to mapping string to int 

def modSexEmb(df):

    df['Sex'].replace(['male','female'],[0,1],inplace=True) # Map male = 0, female = 1

    df['Sex'].astype(dtype='int64')                         # Convert data type to  int64



    df['Embarked'].replace(['C','Q','S'],[0,1,2],inplace=True) # Map C = 0, Q = 1, S = 2, N = 3

    df['Embarked'].astype(dtype='int64')                    # Convert data type to  int64

modSexEmb(data_clear)

modSexEmb(test)
# Create a function for dropping useless fields 

def drop(df, delPas=True):

    df.drop('Cabin',axis=1,inplace=True)             # Drop 'Cabin' field

    if delPas == True:

        df.drop('PassengerId',axis=1,inplace=True)       # Drop 'PassengerId' field

    df.drop('Name',axis=1,inplace=True)              # Drop 'Name' field

    df.drop('Ticket',axis=1,inplace=True)            # Drop 'Ticket' field

    df.info()                                        # Check  

drop(data_clear)

drop(test,False)
# Create a function for groupping

def split(col_range, df,col):

    n=0

    for i in col_range:

        

        if i == col_range[len(col_range)-1]:

            df[col].replace(df[(df[col]>=i)][col],i,inplace=True)

        else:

            alpha = col_range[n+1]-col_range[n]

            df[col].replace( df[(df[col]>=i) & (df[col]<i+alpha)][col],i,inplace=True)

        n+=1

    return df[col]



"""Age"""

ages_range = [0, 9.5, 15, 20, 35, 40, 50, 60, 90]

data_clear['Age'] = split(ages_range, data_clear.copy(), 'Age')

test['Age'] = split(ages_range, test.copy(), 'Age')

"""Fare"""

fare_range = [0,20,40,60,80,100,150,200,250,500]

data_clear['Fare'] = split(fare_range, data_clear.copy(), 'Fare')

test['Fare'] = split(fare_range, test.copy(), 'Fare')
params = {'Sex':[0,1],       # Who is Survived (Male vs Female)

         'Pclass':[1,2,3],               # What class

         'Embarked':[0,1,2,3],  # Embarked

         'SibSp':[0,1,2,3,4,5,6,7,8],    # № of siblings

         'Parch':[0,1,2,3,4,5,6],        # № parents / children

         'cab':np.arange(len(cabin)),

         'Age':ages_range,

         'Fare':fare_range

         }             

plt.figure(figsize=(20, 40))

plot_number=0

for key, value in params.items():

    vals =[]

    if verbose:

        print( key)

    for i in value:

        c = data_clear[data_clear[key]==i][key].count()

        v = data_clear[data_clear[key]==i]['Survived']

        if verbose:

            print  (i , c)

            print ('Survived', data_clear[(data_clear[key]==i)&(data_clear['Survived']==1)]['Survived'].count())

            print ('Died', data_clear[(data_clear[key]==i)&(data_clear['Survived']==0)]['Survived'].count())

        vals.append(v)

    plot_number+=1

    ax = plt.subplot(5, 2, plot_number)

    

    plt.title(key)

    plt.xlabel(key)

    plt.ylabel('Numbers of people')

    plt.hist((vals), histtype='bar', bins=5,label=value,cumulative =False, normed=True)

    ax.legend(prop={'size': 10})

plt.show()
def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()

plot_distribution(train , var = 'Age' , target = 'Survived' , row = 'Sex' )   
# Separarte numeric, categorial and target(y) parameters



numeric_cols=['SibSp','Parch','Fare']



y = data_clear['Survived'].copy()

data_clear.drop(['Survived'],axis=1,inplace=True) 



categorical_cols = list(set(data_clear.columns.values.tolist()) - set(numeric_cols))
from sklearn.linear_model import LogisticRegression as LR

from sklearn.feature_extraction import DictVectorizer as DV

from sklearn.cross_validation import train_test_split

from sklearn.svm import SVC, LinearSVC

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn import cross_validation, datasets, metrics, tree
# Create copies of data for numeric and categorial columns

X_cat = data_clear[categorical_cols].copy()

for i in X_cat.columns.values:

    X_cat[i] = X_cat[i].astype(str)



X_num = data_clear[numeric_cols].copy()
# Create function for transform categorial data to matrix

def catTransform(X_cat):

    encoder = DV(sparse = False)

    X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())

    np.set_printoptions(threshold=np.nan)

    print (X_cat_oh.shape)

    return X_cat_oh

X_cat_oh = catTransform(X_cat)
# Devide data  to train and test as 70/30



(X_train, 

 X_test, 

 y_train, y_test) = train_test_split(X_num, y, 

                                     test_size=0.3, 

                                     random_state=0,

                                    stratify=y)

(X_train_cat_oh,

 X_test_cat_oh) = train_test_split(X_cat_oh, 

                                   test_size=0.3, 

                                   random_state=0,

                                  stratify=y)

(X_train_cat,

 X_test_cat) = train_test_split(X_cat, 

                                   test_size=0.3, 

                                   random_state=0,

                                  stratify=y)
# LogisticRegression

param_grid = {'C': np.linspace(1,2,50),'class_weight':['balanced']}

cv = 3



estimator = LR('l1')

grid = GridSearchCV(estimator, param_grid,cv=cv)

grid.fit(np.hstack([X_train,X_train_cat_oh]), y_train)



lr_pred = grid.predict(np.hstack([X_test,X_test_cat_oh]))

cv_score_lr = cross_validation.cross_val_score(grid, np.hstack([X_test,X_test_cat_oh]), y_test, cv = 10).mean()

print (cv_score_lr)
# SVC



svc = SVC()

svc.fit(np.hstack([X_train,X_train_cat_oh]), y_train)

svc_pred = svc.predict(np.hstack([X_test,X_test_cat_oh]))

cv_score_svc = cross_validation.cross_val_score(svc, np.hstack([X_test,X_test_cat_oh]), y_test, cv = 10).mean()

print (cv_score_svc)
# Decision Tree

clf = tree.DecisionTreeClassifier(random_state = 1, min_samples_leaf = 5, max_depth = 6)

clf.fit(np.hstack([X_num,X_cat]), y)

clf_pred = clf.predict(np.hstack([X_num,X_cat]))



cv_score_clf = cross_validation.cross_val_score(clf, np.hstack([X_num,X_cat]), y, cv = 10).mean()  

print (cv_score_clf)
from sklearn import ensemble, learning_curve 



# Random Forest





rf_classifier = ensemble.RandomForestClassifier(n_estimators = 200, random_state = 1, min_samples_leaf = 5)

train_sizes, train_scores, test_scores = learning_curve.learning_curve(rf_classifier, np.hstack([X_num,X_cat]), y, 

                                                                       train_sizes=np.arange(0.1,1., 0.2), 

                                                                       cv=10, scoring='accuracy')

plt.grid(True)

plt.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')

plt.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')

plt.ylim((0.0, 1.05))

plt.legend(loc='lower right')

rf_classifier.fit(np.hstack([X_num,X_cat]), y)

rf_pred = rf_classifier.predict(np.hstack([X_num,X_cat]))



cv_score_rt = cross_validation.cross_val_score(rf_classifier, np.hstack([X_num,X_cat]), y, cv = 15).mean()

print (cv_score_rt)
# Bagging

clf2 = tree.DecisionTreeClassifier(random_state = 1, min_samples_leaf = 5)

bagging = ensemble.BaggingClassifier(clf2,n_estimators =100)

bagging.fit(np.hstack([X_num,X_cat]), y)

bag_pred = bagging.predict(np.hstack([X_num,X_cat]))

cv_score_bag = cross_validation.cross_val_score(bagging, np.hstack([X_num,X_cat]), y, cv = 10).mean()

print (cv_score_bag)
# Bagging with features



d =  np.hstack([X_train,X_train_cat]).shape[1]

bagging = ensemble.BaggingClassifier(clf2,n_estimators =100,max_features=d)

bagging.fit(np.hstack([X_num,X_cat]), y)

bagf_pred = bagging.predict(np.hstack([X_num,X_cat]))

cv_score_bagf = cross_validation.cross_val_score(bagging, np.hstack([X_num,X_cat]), y, cv = 10).mean()

print (cv_score_bagf)
abc = ensemble.AdaBoostClassifier(n_estimators = 50)

abc.fit(np.hstack([X_num,X_cat]), y)

abc_pred = abc.predict(np.hstack([X_num,X_cat]))

cv_score_abc = cross_validation.cross_val_score(abc, np.hstack([X_num,X_cat]), y, cv = 10).mean()

print (cv_score_abc)
bc = ensemble.GradientBoostingClassifier(n_estimators = 100,learning_rate=1,max_depth=1, random_state=0)

bc.fit(np.hstack([X_num,X_cat]), y)

bc_pred = bc.predict(np.hstack([X_num,X_cat]))

cv_score_bc = cross_validation.cross_val_score(bc, np.hstack([X_num,X_cat]), y, cv = 10).mean()

print (cv_score_bc)
res = {'algorithm':['Logistic Regression','SVC','Decision Tree', 'Random Forest','Bagging','Bagging with features','AdaBoost','Gradient Boosting'],

      'accuracy':[cv_score_lr,cv_score_svc,cv_score_clf,cv_score_rt,cv_score_bag,cv_score_bagf,cv_score_abc,cv_score_bc]}
import pandas as pd

df = pd.DataFrame(res)

df.sort_values('accuracy').head(10)