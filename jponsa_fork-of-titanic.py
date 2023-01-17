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
train = pd.read_csv("../input/train.csv", index_col= 0)

train.to_csv('Kaggle_titanic_imp_train.csv')

test= pd.read_csv("../input/test.csv", index_col= 0)

test.to_csv('Kaggle_titanic_imp_test.csv')
# Joan Ponsa - May 2017



# This is a tutorial from Kaggle, called Titanic, to learn more 

# about machine learning using scikit-learn (a.k.a. sklean). In the

# tutorial thei seaborn for plotting but I prefer maplotlib (also used

# in the tutorial)



# The "goal" of the analysis is to predict the number of survivors from

# the Titanic



# Source: https://www.kaggle.com/c/titanic#tutorials



%matplotlib inline

import os

import time

import numpy as np

import scipy

import pandas as pd

import matplotlib.pyplot as plt

import itertools



import subprocess

from subprocess import call



import xgboost as xgb



import sklearn



import sklearn.metrics

import sklearn.preprocessing



from sklearn.model_selection import train_test_split, RandomizedSearchCV, ShuffleSplit

from sklearn.feature_selection import SelectFromModel



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier

from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, SGDClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.svm import SVC, NuSVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



######### Functions #############################################



# Correlation among feature



def f_hm_pd_df_coor(df, method= "pearson"):

    hm=  plt.imshow(df.corr(), cmap='bwr', interpolation='nearest', vmin=-1, vmax= 1)



    thresh = df.corr().max().max() / 2.

    

    for i, j in itertools.product(range(df.corr().shape[0]), range(titanic.corr().shape[1])):

        if  df.corr().ix[i, j] > 0:

            plt.text(j, i+0.15, round( df.corr().ix[i, j], 2,), fontsize= 12,

                     horizontalalignment="center",

                     color="white" if  df.corr().ix[i, j] > thresh else "black")

        else:

            plt.text(j, i+.15, round( df.corr().ix[i, j], 2,), fontsize= 12,

                     horizontalalignment="center", color="black")



    cb = plt.colorbar(hm)

    cb.ax.get_yaxis().labelpad = 15

    cb.ax.set_ylabel(method+"'s corelation", rotation= 270, fontsize= 12)



    plt.yticks(range(df.corr().shape[1]), list(df.corr().columns))

    plt.xticks(range(df.corr().shape[1]), list(df.corr().columns), 

               rotation= 45, ha= "right", )

    

# a function that extracts each prefix of the ticket, 

# returns 'XXX' if no prefix (i.e the ticket is a digit)

def f_cleanTicket( ticket ):

    ticket = ticket.replace( '.' , '' )

    ticket = ticket.replace( '/' , '' )

    ticket = ticket.split()

    ticket = map( lambda t : t.strip() , ticket )

    ticket = list(filter( lambda t : not t.isdigit() , ticket ))

    if len( ticket ) > 0:

        return ticket[0]

    else: 

        return 'XXX'

    

    

# Run bash command



def RunBashCmd(cmd):

    call(cmd, shell=True)





# Function that call different types of

# Machine Learning methods



def def_classifier(method_name, n_cpus):



    check = False

    

    if method_name == "XBG":

        classifier = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

        param_distribution= {}

        check= True

    

    if method_name == "Logistic_Regression":

        classifier= LogisticRegression(n_jobs= n_cpus, random_state= 1,max_iter= 250)

        

        param_distribution= {'C': scipy.stats.expon(scale=100), 'class_weight':['balanced', None],

                            'solver': ['newton-cg', 'lbfgs', 'sag'],

                             'multi_class': ['ovr', 'multinomial']}

        check= True



    if method_name == "LR_CV":

        classifier= LogisticRegressionCV(n_jobs= n_cpus, random_state= 1, max_iter= 10,

                                        scoring= sklearn.metrics.make_scorer(

                                             sklearn.metrics.accuracy_score))

        

        param_distribution= {'Cs': [range(1,11),]*25,

                             'class_weight':['balanced', None],

                             'solver': ['newton-cg', 'lbfgs', 'sag'],

                             'multi_class': ['ovr', 'multinomial']}

        check= True





    if method_name == "PAC":

        classifier= PassiveAggressiveClassifier(n_jobs= n_cpus, random_state= 1)

        

        param_distribution= {'C': scipy.stats.expon(scale=100), 'class_weight':['balanced', None]}

        check= True



    if method_name == "Ridge_Classifier":

        classifier= RidgeClassifier(random_state= 1)

        

        param_distribution= {'alpha': scipy.stats.expon(scale=100), 'class_weight':['balanced', None],

                             'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']}

        check= True



    if method_name == "RC_CV":

        classifier= RidgeClassifierCV()

    

        param_distribution= {'class_weight':['balanced', None],

                            'alphas': [(0.1, 1.0, 10.0),]*50}

        check= True



    if method_name == "SGDC":

        classifier= SGDClassifier(n_jobs= n_cpus, random_state= 1, eta0= 0.0)

        

        param_distribution= {'class_weight':['balanced', None],

                             'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge',

                                     'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive',

                                     'squared_epsilon_insensitive'],

                             'learning_rate': ['constant', 'optimal', 'invscaling'],

                             'penalty' :['none', 'l2', 'l1','elasticnet'],

                             'alpha': [1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3]}

        

        check= True



    if method_name == "Random_Forest":

        classifier= RandomForestClassifier(n_estimators= 250, n_jobs= n_cpus, random_state= 1)

        param_distribution= {'class_weight':['balanced', None], 'criterion': ['gini', 'entropy'],

                            'max_features': ['auto', 'sqrt', 'log2', None], 

                            "max_depth": [3, None], "max_features": scipy.stats.randint(1, 11),

                             "min_samples_split": scipy.stats.randint(1, 11),

                             "min_samples_leaf": scipy.stats.randint(1, 11),

                             "bootstrap": [True, False]}

        check= True



    if method_name == "ABC":

        classifier= AdaBoostClassifier(n_estimators= 250,random_state= 1)

        param_distribution= {'algorithm': ['SAMME', 'SAMME.R'],

                             'n_estimators': range(1,50),

                             'learning_rate': np.arange(0.25, 1.1, 0.25)}

        check= True



    if method_name == "MLPC": # MLPClassifer

        classifier= MLPClassifier(random_state= 1)

        param_distribution= {'activation': ['identity', 'logistic', 'tanh', 'relu'],

                             'solver': ['lbfgs', 'sgd', 'adam'],

                             'alpha': scipy.stats.expon(scale=100),

                             'learning_rate' : ['constant', 'invscaling', 'adaptive']}

        check= True



    if method_name == "KNC": # KNeighboursClassifer

        classifier= KNeighborsClassifier(n_jobs= n_cpus)

        param_distribution= {'n_neighbors': range(1,11),

                             'weights': ['uniform', 'distance'],

                             'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],

                             'leaf_size': range(5, 55, 5)}

        check= True



    if method_name == "GPC": # GaussicanProcessClassifier

        classifier= GaussianProcessClassifier(n_jobs= n_cpus, random_state= 1, max_iter_predict= 250)

        param_distribution= {'multi_class': ['one_vs_one', 'one_vs_rest']}

        check= True



    if method_name == "SVC": 

        classifier= SVC(random_state= 1, max_iter= 10)

        param_distribution= {'C': scipy.stats.expon(scale=100), 'class_weight':['balanced', None],

                            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],

                            'gamma': scipy.stats.expon(scale=.1),

                            'decision_function_shape': ['ovo', 'ovr',None]}

        check= True



    if method_name == "LinearSVC": 

        classifier= LinearSVC(random_state= 1)

        param_distribution= {'class_weight':['balanced', None],

                             'C': scipy.stats.expon(scale=100),

                             'loss': ['hinge', 'squared_hinge'],

                             'multi_class': ['ovr', 'crammer_singer']}

        check= True



    if method_name == "NuSVC":

        classifier= NuSVC(random_state= 1, max_iter= 10)

        param_distribution= {'class_weight':['balanced', None],

                             'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],

                             'gamma': scipy.stats.expon(scale=.1),

                             'decision_function_shape': ['ovo', 'ovr', None]}

        check= True



    if method_name == "GNB":

        classifier= GaussianNB()

        param_distribution= {}

        check= True



    if method_name == "QDA":

        classifier= QuadraticDiscriminantAnalysis()

        param_distribution= {}

        check= True



    if method_name == False:

        print("ERROR: "+method_name+" not available")

    else:

        return classifier, param_distribution



######### Variables ############################################

pwd= "" # Path work directory

ptd= "" # Path temp directory

pod= "" # Path output directory



######### Main ################################################



#Load data 

## read csv input files as  pd.Dataframes

train = pd.read_csv("../input/train.csv", index_col= 0)

test= pd.read_csv("../input/test.csv", index_col= 0)



## save it as uniquue dataframe

full= train.append(test)

titanic= full[ :891 ] # create a dataset with the top 891 entries/rows (?)



# Full / Titanic - Variable description

# Survived: Survived (1) or died (0)

# Pclass: Passenger's class

# Name: Passenger's name

# Sex: Passenger's sex

# Age: Passenger's age

# SibSp: Number of siblings/spouses aboard

# Parch: Number of parents/children aboard

# Ticket: Ticket number

# Fare: Fare

# Cabin: Cabin

#Embarked: Port of embarkation



## delete useless var

del train, test





#Load data 

## read csv input files as  pd.Dataframes

train = pd.read_csv("../input/train.csv", index_col= 0)

test= pd.read_csv("../input/test.csv", index_col= 0)



## save it as uniquue dataframe

full= train.append(test)

titanic= full[ :891 ] # create a dataset with the top 891 entries/rows (?)



# Full / Titanic - Variable description

# Survived: Survived (1) or died (0)

# Pclass: Passenger's class

# Name: Passenger's name

# Sex: Passenger's sex

# Age: Passenger's age

# SibSp: Number of siblings/spouses aboard

# Parch: Number of parents/children aboard

# Ticket: Ticket number

# Fare: Fare

# Cabin: Cabin

#Embarked: Port of embarkation



## delete useless var

del train, test





# # Correlation among feature

# f_hm_pd_df_coor(titanic, 'spearman')



# # distributions of Age of passangers who survived or did not survive

# f, (ax1, ax2)= plt.subplots(2,1, sharex= True)

# plt.subplots_adjust(hspace= 0.5)



# for axn, sex in zip([ax1, ax2], np.unique(titanic["Sex"])):

#     for survived in np.unique(titanic["Survived"]):

#         titanic[((titanic["Survived"]== survived) & 

#                  (titanic["Sex"]== sex))]["Age"].plot.density(ax= axn, label= survived)

        

#     axn.set_title("Sex= "+sex)



# plt.legend(title= "Survived")

# plt.xlim(0, titanic["Age"].max())

# plt.show()



# # Survival rate by Embarked

# ((titanic.groupby("Embarked")["Survived"].sum())/  

#  (titanic["Embarked"].value_counts())).plot(kind="barh", color= "black")



# plt.xlabel("Survived (fraction)")

# plt.ylabel("Embarked")

# plt.show()



# Prepare data for ML

## Transform Sex into binary (male= 1)

full["Sex"]= pd.Series(np.where(full.Sex == 'male' , 1 , 0 ), name= 'Sex')

## Create a new variable for every unique value of Embarked

full= full.join(pd.get_dummies(full.Embarked , prefix='Embarked'))

## Create a new variable for every unique value of Embarked

full= full.join(pd.get_dummies(full.Pclass , prefix='Pclass'))

## Fill missing values of Age with the average of Age (mean)

full["Age"]= full.Age.fillna(full.Age.mean())

## Fill missing values of Fare with the average of Fare (mean)

full["Fare"]= full.Fare.fillna(full.Fare.mean())



# Get the title of the person/ passanger

title = pd.DataFrame()

## Extract the title from each name

title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )



## Dictionary of more aggregated titles

dTitle = {"Capt":       "Officer",

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

          "Lady" :      "Royalty"}



full['Title'] = title.Title.map(dTitle)



# Cabin class

## replacing missing cabins with U (for Uknown)

full[ 'Cabin' ] = full.Cabin.fillna( 'U' )

## mapping each Cabin value with the cabin letter

full[ 'Cabin' ] = full[ 'Cabin' ].map( lambda c : c[0] )



## dummy encoding ...

full= full.join(pd.get_dummies(full['Cabin'] , prefix= 'Cabin'))



# Ticket class

## Extracting dummy variables from tickets:

full['Ticket'] = full['Ticket'].map(f_cleanTicket )

full= full.join(pd.get_dummies(full['Ticket'], prefix= 'Ticket'))



# Family size

full['FamilySize'] = full[ 'Parch' ] + full['SibSp'] + 1

full['Family_Single'] = full['FamilySize'].map( lambda s : 1 if s == 1 else 0 )

full['Family_Small']  = full['FamilySize'].map( lambda s : 1 if 2 <= s <= 4 else 0 )

full['Family_Large']  = full['FamilySize'].map( lambda s : 1 if 5 <= s else 0 )



## check highly-correlated features (abs(spearman) > 0.65)

highly_correlated= []

for x, y in list(itertools.combinations(full.corr().columns, 2)):

    if abs(full.corr().ix[x, y])>= 0.65:

        highly_correlated.append(x+"-"+y)



#highly_correlated

        

features= ['Age', 'Fare', 'Sex', 'SibSp',

           'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1', 'Pclass_2',

           'Pclass_3', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E',

           'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_U', 'Ticket_A', 'Ticket_A4',

           'Ticket_A5', 'Ticket_AQ3', 'Ticket_AQ4', 'Ticket_AS', 'Ticket_C',

           'Ticket_CA', 'Ticket_CASOTON', 'Ticket_FC', 'Ticket_FCC', 'Ticket_Fa',

           'Ticket_LINE', 'Ticket_LP', 'Ticket_PC', 'Ticket_PP', 'Ticket_PPP',

           'Ticket_SC', 'Ticket_SCA3', 'Ticket_SCA4', 'Ticket_SCAH', 'Ticket_SCOW',

           'Ticket_SCPARIS', 'Ticket_SCParis', 'Ticket_SOC', 'Ticket_SOP',

           'Ticket_SOPP', 'Ticket_SOTONO2', 'Ticket_SOTONOQ', 'Ticket_SP',

           'Ticket_STONO', 'Ticket_STONO2', 'Ticket_STONOQ', 'Ticket_SWPP',

           'Ticket_WC', 'Ticket_WEP', 'Ticket_XXX', 'Family_Single',

           'Family_Small', 'Family_Large']



label= "Survived"





######### ML - Trainning ################################################



methods= ["RC_CV", "SGDC", "Random_Forest", "ABC", "MLPC", "GPC",

          "SVC", "LinearSVC", "NuSVC", "Logistic_Regression",

          "LR_CV", "PAC", "Ridge_Classifier",]



data_labeled= full[0:891]

data_unlabeled= full[891:]



# DataSet ML



# Full dataset

F_predictors= data_labeled[features]

F_predictors= F_predictors.astype('float64')

F_predictors= pd.DataFrame(sklearn.preprocessing.scale(F_predictors, with_mean= True),

                           index= F_predictors.index, columns= F_predictors.columns)



F_target= np.array(data_labeled[label], dtype= float)





# Split data in training set and test set



ncpus= 1

n_permutations= 1

test_size= 0.3 



(predictors_train, predictors_test, target_train,

target_test) = train_test_split(F_predictors, F_target,

        test_size= test_size, random_state= 1)



# DataFrame metric scores

metric_scores_df= pd.DataFrame([], index= methods, 

                            columns= ["F1 score", "Accuracy", "Precision", "Recall"])



#Dictionary confusion matrices

dCMs= {}

dBestModel= {}

for method_name in methods:

    dCMs[method_name]= ""

    dBestModel[method_name]= ""



for method_name in (methods + ["Voting"]):



    # Define classifier

    

    if method_name != "Voting":

    

        clf, param_distribution= def_classifier(method_name, ncpus)

        

        Model_search= RandomizedSearchCV(clf, param_distribution, n_iter= 100,

                                         scoring= sklearn.metrics.make_scorer(

                                             sklearn.metrics.accuracy_score),

                                         n_jobs= ncpus, 

                                         cv= ShuffleSplit(n_splits= 5, test_size= test_size, random_state= 1),

                                         random_state= 1)

        

        Model_search.fit(F_predictors, F_target)

        dBestModel[method_name]= Model_search.best_estimator_

        clf= Model_search.best_estimator_

                                         

    else:

        estimators= []



        clf1= ""

        clf2= ""

        clf3= ""

        clf4= ""

        clf5= ""



        for method_name2, clfn in zip(metric_scores_df.sort_values(by="Accuracy", ascending= False).head(5).index,

                                   [clf1, clf2, clf3, clf4, clf5]):



            clfn= def_classifier(method_name2, ncpus)

            estimators.append((method_name2, clfn))

          

        clf = VotingClassifier(estimators= estimators, voting= 'hard')

        

    clf.fit(predictors_train, target_train)

    

    predicted_targets= clf.predict(predictors_test)

        

    metric_scores_df.ix[method_name, "F1 score"]= sklearn.metrics.f1_score(

        target_test, predicted_targets, average= "weighted")

    

    metric_scores_df.ix[method_name, "Accuracy"]= sklearn.metrics.accuracy_score(

        target_test, predicted_targets)



    

    metric_scores_df.ix[method_name, "Precision"]= sklearn.metrics.precision_score(

        target_test, predicted_targets, average= "weighted")

    

    metric_scores_df.ix[method_name, "Recall"]= sklearn.metrics.recall_score(

        target_test, predicted_targets, average= "weighted")



    tmp_cm= np.array(sklearn.metrics.confusion_matrix(target_test, predicted_targets), dtype= float)



    # Normalize CM

    for i in range(len(np.unique(F_target))):

        tmp_cm[i]= tmp_cm[i]/ np.sum(tmp_cm[i])



    dCMs[method_name]= tmp_cm



    print ("Finished: "+method_name)

    #print (time.strftime("%H:%M:%S")+" - "+time.strftime("%d/%m/%Y")
metric_scores_df.sort_values(by="Accuracy", ascending= False).head()
for method_name in metric_scores_df.sort_values(

    by="Accuracy", ascending= False).head(5).index:

    

    cm= dCMs[method_name]

    plt.imshow(cm, interpolation='nearest',vmin=0, vmax= 1, cmap= plt.cm.Greys)



    cb= plt.colorbar(pad= 0.035, label= "Fraction",orientation= "vertical",

                     ticks= [0, 0.25, 0.5, 0.75, 1], aspect= 6.8, fraction= 0.125)

    

    cb.ax.get_yaxis().labelpad = 15

    cb.ax.set_ylabel("Fraction", rotation= 270, fontsize= 14)

    

    tick_marks = np.arange(len(cm))

    plt.xticks(tick_marks, range((2-len(cm)), 4), rotation=0, fontsize= 14)

    plt.yticks(tick_marks, range((2-len(cm)), 4), fontsize= 14)

    

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if cm[i, j] > 0:

            plt.text(j, i+0.15, round(cm[i, j], 2,), fontsize= 14,

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, round(cm[i, j], 2,), fontsize= 14,

                     horizontalalignment="center",

                     color="black")

        

    plt.title(method_name)

    plt.tight_layout()

    plt.ylabel('True label', fontsize= 14)

    plt.xlabel('Predicted label', fontsize= 14)

    plt.show()
# Multi-voting



estimators= []



clf1= ""

clf2= ""

clf3= ""

clf4= ""

clf5= ""

clf6= ""

clf7= ""

clf8= ""

clf9= ""

clf10= ""

clf11= ""

clf12= ""

clf13= ""

clf14= ""

clf15= ""

clf16= ""

clf17= ""

clf18= ""



for method_name, clfn in zip(metric_scores_df.sort_values(by="F1 score", ascending= False).head(10).index,

                           [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10,

                           clf11, clf12, clf13, clf14, clf15, clf16, clf17, clf18][0:10]):

    

    clfn= def_classifier(method_name, ncpus)

    estimators.append((method_name, clfn))
eclf1 = VotingClassifier(estimators= estimators, voting= 'hard')

eclf1 = eclf1.fit(predictors_train, target_train)

predicted_targets= eclf1.predict(predictors_test)

        

print(sklearn.metrics.f1_score(

    target_test, predicted_targets, average= "weighted"))



print(sklearn.metrics.accuracy_score(

    target_test, predicted_targets))



# DataSet ML



# Full dataset

F_predictors= data_labeled[features]

F_predictors= F_predictors.astype('float64')

F_predictors= pd.DataFrame(sklearn.preprocessing.scale(F_predictors, with_mean= True),

                            index= F_predictors.index, columns= F_predictors.columns)



F_target= np.array(data_labeled[label], dtype= float)



test_X= data_unlabeled[features]

test_X.ix[1309, "Sex"]= 1



test_X= test_X.astype('float64')

test_X= pd.DataFrame(sklearn.preprocessing.scale(test_X, with_mean= True),

                           index= test_X.index, columns= test_X.columns)



test_Y = eclf1.predict(test_X)

passenger_id = list(data_unlabeled.index)

test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )

test.shape

test.head()

test.to_csv('JPonsa_titanic_pred_V3.csv' , index = False )
len(target_test)
len(predicted_targets)
[clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10,

 clf11, clf12, clf13, clf14, clf15, clf16, clf17, clf18]
len(estimators)
import scipy.stats

test= scipy.stats.expon(scale=100)
test.stats()
import numpy as np

test= (np.concatenate([[0.1], np.arange(0.25, 10.1, 0.25)]),)*50
[np.linspace(0.1, 10, 20),]*5
test= [(0.1, 1.0, 10.0),]*25
test[0]
method_name
[1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3]