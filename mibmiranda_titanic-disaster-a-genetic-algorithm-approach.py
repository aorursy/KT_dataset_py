#Here some imports...

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



#Genetic Algorithm - https://github.com/deap/deap

from deap import base

from deap import creator

from deap import tools



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning models

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier

# machine learning auxiliaries

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_regression

from sklearn.pipeline import make_pipeline



#loading data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

dfs = [train_df, test_df]

full_origin = pd.concat([train_df.drop('Survived',1),test_df])



print('Data loaded...')



# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

#From https://www.kaggle.com/sachinkulkarni/an-interactive-data-science-tutorial

def cleanTicket(ticket):

    ori_ticket = ticket

    ticket = ticket.replace( '.' , '' )

    ticket = ticket.replace( '/' , '' )

    ticket = ticket.split()

    ticket = map( lambda t : t.strip() , ticket )

    ticket = list(filter( lambda t : not t.isdigit() , ticket ))

    if len(ticket) > 0:

        return ticket[0]

    else: 

        return 'XXX'



#from https://www.kaggle.com/rajatshah/scikit-learn-ml-from-start-to-finish?scriptVersionId=1260742

def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df['AgeBins'] = categories

    return df



#From https://www.kaggle.com/rajatshah/scikit-learn-ml-from-start-to-finish?scriptVersionId=1260742

def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df['FareBins'] = categories

    return df



#From https://www.kaggle.com/rajatshah/scikit-learn-ml-from-start-to-finish?scriptVersionId=1260742

#Map features to Numeric values

def encode_features(df_train, df_test, features):

    df_combined = pd.concat([df_train[features], df_test[features]])    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test



#Creating Title feature and mapping some synonyms

for dataset in dfs:

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



#Converting Sex categorical feature to int value

for dataset in dfs:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



#Checking Age missing values

for dataset in dfs:

    dataset['Age_known'] = dataset['Age'].isnull() == False



#Filling in missing values: Embarked (missing only in train dataset)

train_df['Embarked'].iloc[61] = "C"

train_df['Embarked'].iloc[829] = "C"



#Filling in missing values: Fare (missing only in test dataset)

#using the median value for the 3st class 

all_df = pd.concat([train_df.drop('Survived',1),test_df])

test_df['Fare'].iloc[152] = all_df['Fare'][all_df['Pclass'] == 3].dropna().median()



#Filling in missing values: Age

#Guessing value from similares sex-title and sex-class median values

all_df = pd.concat([train_df.drop('Survived',1),test_df])

titleList = all_df['Title'].unique().tolist()

guess_ages_sex_title = np.zeros((2,len(titleList)))

guess_ages_sex_pclass = np.zeros((2,3))

for dataset in dfs:

    for i in range(0, 2):#Sex

        for j in range(0, 3):#Pclass

            guess_df = all_df[(all_df['Sex'] == i) & (all_df['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            guess_ages_sex_pclass[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            for title in titleList:#Title

                k = titleList.index(title)

                guess_df = all_df[(all_df['Title'] == title) & (all_df['Sex'] == i)]['Age'].dropna()

                                

                # age_mean = guess_df.mean()

                # age_std = guess_df.std()

                # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



                age_guess2 = guess_df.median()                

                if (age_guess2!=age_guess2):#if Nan

                    age_guess2 = age_guess

                

                guess_ages_sex_title[i,k] = int( age_guess2/0.5 + 0.5 ) * 0.5

    

    for i in range(0, 2):

        for j in range(0, 3):

            for k in range(0, len(titleList)):

                dataset.loc[ (dataset.Age.isnull()) & (dataset.Title == titleList[k]) & (dataset.Sex == i) ,\

                    'Age'] = guess_ages_sex_title[i,k]

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages_sex_pclass[i,j]



    dataset['Age'] = dataset['Age'].astype(int)

    

#creating Deck and FamilyName features

for dataset in dfs:    

    dataset['Deck'] = dataset['Cabin'].str[0]

    dataset['FamilyName'] = dataset['Name'].str.split(", ", expand=True)[0]

    

all_df = pd.concat([train_df.drop('Survived',1),test_df])

print("Initial missing Deck values: ",len(all_df.loc[all_df['Deck'].isnull()]))
#Guessing Deck missing values from the Ticket value. 

TicketList = all_df['Ticket'].unique().tolist()

for dataset in dfs:    

    for ticket in TicketList:

        guess_deck = all_df[(all_df['Ticket'] == ticket)]['Deck'].dropna()

        if(len(guess_deck.index)>0):

            guess_deck = guess_deck.iloc[0][0]

            dataset.loc[(dataset.Deck.isnull()) & (dataset.Ticket == ticket),'Deck'] = guess_deck

all_df = pd.concat([train_df.drop('Survived',1),test_df])

print("Missing Deck values after apply shared ticket heuristic: ",len(all_df.loc[all_df['Deck'].isnull()]))
#Trying to guess Deck missing values using a prediction Model.

all_df = pd.concat([train_df,test_df])

df = all_df[['Pclass','Fare','Embarked','Deck']]

df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)   

df1 = df[df['Deck'].notnull()]

XDeck_train = df1[['Pclass','Fare','Embarked']]

YDeck_train = df1.Deck

random_forest = RandomForestClassifier(n_estimators=100)



scores = cross_val_score(random_forest, XDeck_train, YDeck_train, cv=5, n_jobs=-1)

print("CV score: ",scores.mean())



random_forest.fit(XDeck_train, YDeck_train)

XDeck_test = df[['Pclass','Fare','Embarked']]

YDeck_pred = random_forest.predict(XDeck_test)

all_df['DeckPred'] = YDeck_pred

all_df.loc[(all_df.Deck.isnull()),'Deck'] = all_df.loc[(all_df.Deck.isnull()),'DeckPred']

train_df['Deck'] = all_df[ 0:891 ]['Deck']

test_df['Deck'] = all_df[ 891: ]['Deck']



'''

#Guessing Deck missing values from the Embarked distribution

#print(pd.crosstab(all_df['Embarked'], all_df['Deck']))

all_df = pd.concat([train_df.drop('Survived',1),test_df])

print("Missing Deck values: ",len(all_df.loc[all_df['Deck'].isnull()]))

for dataset in dfs:    

    for index, row in dataset.iterrows():

        if(row["Deck"]==row["Deck"]): continue#not Null

        if row['Embarked'] == 'C':

            Cchoice = np.random.choice(['A', 'B', 'C', 'D','E', 'F', 'G', 'T'], p=[11/129, 36/129, 46/129, 20/129,11/129, 5/129, 0/129, 0/129])

            dataset.loc[index, 'Deck'] = Cchoice

        elif row['Embarked'] == 'S':

            Schoice = np.random.choice(['A', 'B', 'C', 'D','E', 'F', 'G', 'T'], p=[11/177, 32/177, 55/177, 26/177,30/177, 17/177, 5/177, 1/177])

            dataset.loc[index, 'Deck'] = Schoice

all_df = pd.concat([train_df.drop('Survived',1),test_df])

print("Missing Deck values: ",len(all_df.loc[all_df['Deck'].isnull()]))



for dataset in dfs:    

    dataset['Deck'] = dataset['Deck'].fillna(value='U')

'''

                

all_df = pd.concat([train_df.drop('Survived',1),test_df])

print("Missing Deck values: ",len(all_df.loc[all_df['Deck'].isnull()]))

#Creating feature NameContainsP

for dataset in dfs:    

    dataset["NameContainsP"] = dataset["Name"].apply(lambda x: "(" in x) #If the name contain "("



g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'NameContainsP')



pd.crosstab(train_df['NameContainsP'], train_df['Survived'])
#creating some new features

all_df = pd.concat([train_df.drop('Survived',1),test_df])

for dataset in dfs:    

    simplify_fares(dataset)#ranges of fares

    simplify_ages(dataset)#ranges of ages

    dataset['Child'] = dataset['Age']<=10

    dataset['MedianAge'] = (dataset['Age']>=18) & (dataset['Age']<=40)

    dataset['Young_m'] = (dataset['Age']>=18) & (dataset['Age']<=40) & (dataset['Sex']==0)

    dataset['Young_f'] = (dataset['Age']>=18) & (dataset['Age']<=40) & (dataset['Sex']==1)

    dataset['Family'] = dataset['SibSp'] + dataset['Parch']

    dataset['Alone']  = (dataset['SibSp'] + dataset['Parch']) == 0

    dataset['Cabin_known'] = dataset['Cabin'].isnull() == False

    dataset["Cabin_known"] = dataset["Cabin_known"].astype("int")    

    dataset['Ttype'] = dataset['Ticket'].str[0]

    dataset['Ttype2'] = dataset['Ticket'].map(cleanTicket)    

    dataset['Bad_ticket'] = dataset['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])

    dataset["NameLength"] = dataset["Name"].apply(lambda x: len(x))  #Create feture for name length         

    dataset['Ticket_group'] = dataset.groupby('Ticket')['Name'].transform('count')

    dataset['Fare_eff'] = dataset['Fare']/dataset['Ticket_group']

    dataset['Shared_ticket'] = 3

    for i in range(len(dataset)):

        if dataset['Shared_ticket'].iloc[i]==3:            

            if ((len(all_df.groupby('Ticket').get_group(dataset['Ticket'].iloc[i]))) > 1 ):

                dataset.loc[dataset['Ticket'] == dataset['Ticket'].iloc[i], 'Shared_ticket'] = 1

            else:

                dataset.loc[dataset['Ticket'] == dataset['Ticket'].iloc[i], 'Shared_ticket'] = 0

    

    dataset['Young'] = (dataset['Age']<=20) | (dataset['Title'].isin(['Master','Miss','Mlle','Mme']))

     #FareBand

    dataset['FareBand'] = 0

    dataset.loc[ dataset['Fare'] <= 7.91, 'FareBand'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'FareBand'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'FareBand']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'FareBand'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    #AgeBand

    dataset['AgeBand'] = 0

    dataset.loc[ dataset['Age'] <= 16, 'AgeBand'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'AgeBand'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'AgeBand'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'AgeBand'] = 3

    dataset.loc[ dataset['Age'] > 64, 'AgeBand'] = 4    

    

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



#Converting categorical features to integer values

for dataset in dfs:      

    #Title

    # we map each title

    dataset[ 'Title' ] = dataset.Title.map( Title_Dictionary )

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Officer": 5, "Royalty": 6}    

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Title'] = dataset['Title'].astype(int)    

    dataset['Ttype'] = dataset['Ttype'].map( {'1': 1, '2': 2, '3': 3, '4': 4,'5': 5,'6': 6, '7': 7, '8': 8,'9': 9,'A': 10, 'C': 11, 'F': 12,'L': 13, 'P': 14, 'S': 15,'W': 16} ).astype(int)    

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)    

    dataset['Deck'] = dataset['Deck'].map( {'U': 0, 'C': 1, 'E': 2,'G': 3, 'D': 4, 'A': 5,'B': 6, 'F': 7, 'T': 8} ).astype(int)    

    for col in dataset.columns:

        if(dataset[col].dtype == 'bool'):

            dataset[col] = dataset[col].astype(int)



print("Ok. Data are now almost preprocessed...")
# preview the final data

all_df = pd.concat([train_df.drop('Survived',1),test_df])

print(all_df.info())

all_df.head()

#Encoding to Numeric values

train_df, test_df = encode_features(train_df, test_df, ['FamilyName','AgeBins','FareBins','Ttype2'])

#preparing dataset

selCols = []

#filtering only numeric attributes

for col in test_df.columns:

    if(test_df[col].dtype == 'int64' or test_df[col].dtype == 'float64' or test_df[col].dtype == 'uint8'):

        selCols.append(col)        



#removing some supposed helpless attributes

#if 'SibSp' in selCols: selCols.remove('SibSp')

#if 'Parch' in selCols: selCols.remove('Parch')

#if 'AgeBand' in selCols: selCols.remove('AgeBand')

#if 'FareBand' in selCols: selCols.remove('FareBand')

if 'PassengerId' in selCols: selCols.remove('PassengerId')

if 'Survived' in selCols: selCols.remove('Survived')



train_df = train_df.loc[:,selCols+['Survived']]

test_df = test_df.loc[:,selCols+['PassengerId']]

train_df.head()



print("Number of selected cols ",len(selCols)," :",selCols)

print()

all_df = pd.concat([train_df,test_df])

print(all_df.describe())

colsRF =  ['Pclass', 'Sex', 'Embarked', 'Title', 'Age_known', 'Deck', 'FareBins', 'AgeBins', 'Alone', 'Ttype', 'Ttype2', 'NameLength', 'NameContainsP', 'Young']

#colsRF =  ['Pclass', 'Sex', 'Embarked', 'Deck', 'FareBins', 'AgeBins', 'Alone', 'Ttype', 'Ttype2', 'NameContainsP']

tcols = np.append(['Survived'],colsRF)

df = train_df.loc[:,tcols].dropna()

X_train = df.loc[:,colsRF]

Y_train = np.ravel(df.loc[:,['Survived']])



model = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(model, X_train, Y_train, cv=5, n_jobs=-1)

cv_rf_score = scores.mean()

print("RF CV score: ",scores.mean())

model.fit( X_train , Y_train )

print("Training score: ",model.score(X_train, Y_train))



X_submit = test_df.loc[:,colsRF]

Y_pred = model.predict(X_submit)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv( 'titanic_pred_RF.csv' , index = False )

#colsRF =  ['Pclass', 'Sex', 'Embarked', 'Title', 'Age_known', 'Deck', 'FareBins', 'AgeBins', 'Alone', 'Ttype', 'Ttype2', 'NameLength', 'NameContainsP', 'Young']

colsRF =  ['Pclass', 'Sex', 'Embarked', 'Deck', 'FareBins', 'AgeBins', 'Alone', 'Ttype', 'Ttype2', 'NameContainsP']

tcols = np.append(['Survived'],colsRF)

df = train_df.loc[:,tcols].dropna()

X_train = df.loc[:,colsRF]

Y_train = np.ravel(df.loc[:,['Survived']])



model = KNeighborsClassifier(n_neighbors = 3)

scores = cross_val_score(model, X_train, Y_train, cv=5, n_jobs=-1)

cv_knn_score = scores.mean()

print("KNN CV score: ",scores.mean())

model.fit( X_train , Y_train )

print("Training score: ",model.score(X_train, Y_train))



X_submit = test_df.loc[:,colsRF]

Y_pred = model.predict(X_submit)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv( 'titanic_pred_KNN.csv' , index = False )

#colsRF =  ['Pclass', 'Sex', 'Embarked', 'Title', 'Age_known', 'Deck', 'FareBins', 'AgeBins', 'Alone', 'Ttype', 'Ttype2', 'NameLength', 'NameContainsP', 'Young']

colsRF =  selCols

tcols = np.append(['Survived'],colsRF)

df = train_df.loc[:,tcols].dropna()

X_train = df.loc[:,colsRF]

Y_train = np.ravel(df.loc[:,['Survived']])

scaler = preprocessing.StandardScaler().fit(X_train)    

X_train = scaler.transform(X_train)



model = SVC(kernel='rbf')

scores = cross_val_score(model, X_train, Y_train, cv=5, n_jobs=-1)

cv_svm_score = scores.mean()

print("SVM CV score: ",scores.mean())

model.fit( X_train , Y_train )

print("Training score: ",model.score(X_train, Y_train))



X_submit = test_df.loc[:,colsRF]

X_submit = scaler.transform(X_submit)

Y_pred = model.predict(X_submit)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv( 'titanic_pred_SVM.csv' , index = False )

#GA applied to select optimal features and parameters of a SVM model

#Extra examples of this GA library could be find here: http://deap.readthedocs.io/en/master/examples/



cols = selCols#initial set of features, the space search...

training = pd.concat([train_df]) #A working copy of the training dataset



#Random generator of SVM parameter C

def getC():

    #[0.01 - 3[

    r = rnd.random()

    r2 = rnd.randint(0,2)

    r3 = r+r2+0.000000001    

    #print("New C: ",r3)

    return r3

#Random generator of SVM parameter Gamma

def getGamma():

    #[0.01 - 1[

    r = rnd.random()#[0-1]

    r2 = rnd.randint(0,3)

    r3 = 0.000000001+(r/(10**r2))

    #r3=0.01

    #print("New Gamma: ",r3)

    return r3

#Random generator of SVM Kernel

def getKernel():

    kernels = ['rbf','linear','svcLinear']

    ind = rnd.randint(0,len(kernels)-1)   

    r = kernels[ind]

    #print("New Kernel: ", r)

    return r

    

creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)



toolbox = base.Toolbox()



toolbox.register("attr_bool", rnd.randint, 0, 1)

toolbox.register("attr_C", getC)

toolbox.register("attr_Gamma", getGamma)

toolbox.register("attr_kernel", getKernel)



#features

func_seq = [toolbox.attr_C , toolbox.attr_Gamma,toolbox.attr_kernel]#[C,Gamma,kernel]

for c in cols:

    func_seq.append(toolbox.attr_bool)



print("individuals size: ",len(func_seq))



toolbox.register("individual", tools.initCycle, creator.Individual, func_seq, 1)



# define the population to be a list of individuals

toolbox.register("population", tools.initRepeat, list, toolbox.individual)



def getModel(individual):

    k = individual[2]

    if k == 'svcLinear':

        clf = LinearSVC(C=individual[0])

    elif k == 'rbf':

        clf = SVC(kernel=k, C=individual[0],gamma=individual[1])

    else:

        #linear

        clf = SVC(kernel=k, C=individual[0])

    return clf



def getXy(individual):

    scols = list(cols)

    for i in range(len(individual[3:])):

        if individual[3+i]<1: scols.remove(cols[i])

    #print("Selected cols: ",scols)

    tcols = np.append(['Survived'],scols)

    df = training.loc[:,tcols].dropna()

    X = df.loc[:,scols]

    scaler = preprocessing.StandardScaler().fit(X)    

    #scaler= preprocessing.MinMaxScaler().fit(X)

    X = scaler.transform(X)

    y = np.ravel(df.loc[:,['Survived']])

    return [X,y,scols,scaler]



# the goal ('fitness') function to be maximized

def evalOneMax(individual):

    clf = getModel(individual)

    Xy = getXy(individual)

    scores = cross_val_score(clf, Xy[0], Xy[1], cv=5, n_jobs=-1)

    res1 = scores.mean(),

    return res1



def myMutate(individual,indpb=0.05):

    #print(individual)

    #C

    if rnd.random() < indpb:

        individual[0] = toolbox.attr_C()

    #Gamma

    if rnd.random() < indpb:

        individual[1] = toolbox.attr_Gamma()

    #Kernel

    if rnd.random() < indpb:

        individual[2] = toolbox.attr_kernel()

    #features

    for i in range(len(individual[3:])):

        if rnd.random() < indpb:

            individual[3+i] = toolbox.attr_bool()

    #print(individual)

                  

#----------

# Operator registration

#----------

# register the goal / fitness function

toolbox.register("evaluate", evalOneMax)



# register the crossover operator

toolbox.register("mate", tools.cxTwoPoint)



# register a mutation operator with a probability to

# flip each attribute/gene of 0.05

toolbox.register("mutate", myMutate, indpb=0.15)



# operator for selecting individuals for breeding the next

# generation: each individual of the current generation

# is replaced by the 'fittest' (best) of three individuals

# drawn randomly from the current generation.

toolbox.register("select", tools.selTournament, tournsize=3)



rnd.seed(66)

                

# CXPB  is the probability with which two individuals

#       are crossed

#

# MUTPB is the probability for mutating an individual

#

# NGEN  is the number of generations for which the

#       evolution runs

CXPB, MUTPB, NGEN, POPSIZE = 0.5, 0.2, 40, 100



# create an initial population of 300 individuals (where

# each individual is a list of integers)

pop = toolbox.population(n=POPSIZE)    

#print(pop)



print("Start of evolution SVM")



# Evaluate the entire population

fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):

    ind.fitness.values = fit



print("  Evaluated %i individuals" % len(pop))



# Begin the evolution

for g in range(NGEN):

    print("-- Generation %i --" % g)

    

    # Select the next generation individuals

    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals

    offspring = list(map(toolbox.clone, offspring))



    # Apply crossover and mutation on the offspring

    for child1, child2 in zip(offspring[::2], offspring[1::2]):



        # cross two individuals with probability CXPB

        if rnd.random() < CXPB:

            #print("CX")

            #print(child1,child2)

            c1 = toolbox.clone(child1)

            c2 = toolbox.clone(child2)

            toolbox.mate(child1, child2)

            #print(child1,child2)

            # fitness values of the children

            # must be recalculated later

            if c1!=child1: del child1.fitness.values

            if c2!=child2: del child2.fitness.values



    for mutant in offspring:



        # mutate an individual with probability MUTPB

        if rnd.random() < MUTPB:

            #print("mut")

            #print(mutant)

            m1 = toolbox.clone(mutant)

            toolbox.mutate(mutant)

            if m1!=mutant: del mutant.fitness.values

            #print(mutant)



    # Evaluate the individuals with an invalid fitness

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    #print(invalid_ind)

    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):

        ind.fitness.values = fit

    

    print("  Evaluated %i individuals" % len(invalid_ind))

    #print(invalid_ind)

    

    # The population is entirely replaced by the offspring

    pop[:] = offspring

    

    # Gather all the fitnesses in one list and print the stats

    fits = [ind.fitness.values[0] for ind in pop]

    

    length = len(pop)

    mean = sum(fits) / length

    sum2 = sum(x*x for x in fits)

    std = abs(sum2 / length - mean**2)**0.5

    best_ind = tools.selBest(pop, POPSIZE)[0]

    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))        

    print("  Min %s" % min(fits))

    print("  Max %s" % max(fits))

    print("  Avg %s" % mean)

    print("  Std %s" % std)



print("-- End of (successful) evolution --")



best_ind = tools.selBest(pop, POPSIZE)[0]

print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))





model = getModel(best_ind)

Xy = getXy(best_ind)

colsSVM = Xy[2]

scaler = Xy[3]

print("Selected Features: ",colsSVM)



X_train = Xy[0]

Y_train = Xy[1]



scores = cross_val_score(model, X_train, Y_train, cv=5).mean()

cv_SVMGA_score = scores.mean()

print("SVMGA CV score: ",scores.mean())

model.fit( X_train , Y_train )

print("Training score: ",model.score(X_train, Y_train))



#Train using all data from training dataset

tcols = np.append(['Survived'],colsSVM)

df = train_df.loc[:,tcols].dropna()

X = df.loc[:,colsSVM]

scaler = preprocessing.StandardScaler().fit(X)    

X = scaler.transform(X)

y = np.ravel(df.loc[:,['Survived']])

model.fit(X, y)



X_submit = test_df.loc[:,colsSVM]

X_submit = scaler.transform(X_submit)

Y_pred = model.predict(X_submit)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv( 'titanic_pred_SVMGA.csv' , index = False )

d = {'cv_rf_score': cv_rf_score, 'cv_knn_score': cv_knn_score, 'cv_svm_score': cv_svm_score, 'cv_SVMGA_score': cv_SVMGA_score}

df = pd.DataFrame(data=d,index=[0])

df.head()