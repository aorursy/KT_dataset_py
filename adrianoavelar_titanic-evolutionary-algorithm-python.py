# modules to handle data

import pandas as pd





train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
train_df.describe()
test_df.describe()
import seaborn as sns

import matplotlib.pyplot as plt



#Relation between PassengerId and Survived

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'PassengerId', bins=20)
#Relation between Pclass e Survived

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Pclass', bins=5)
#Relation between Age e Survived

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=5)
#Relation between SibSp e Survived

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'SibSp', bins=5)
#Relation between Parch e Survived

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Parch', bins=5)
#Relation between Fare e Survived

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Fare', bins=5)
#Stores the PassengerId Column of Test to be used in the submission

passengerId = test_df.PassengerId



train_df = train_df.drop('PassengerId', axis = 1)

test_df = test_df.drop('PassengerId', axis = 1)
train_df['Name'].describe()
train_df['Title'] = train_df.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

train_df['Title'].value_counts()
norm_titles = {

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







train_df.Title = train_df.Title.map(norm_titles)



train_df.Title.value_counts()
#Checking  Age mean Grouping Sex and Title.

train_grouped = train_df.groupby(['Sex','Title','Pclass'])

train_grouped.Age.mean()
#Applying the mean values of the Sex/Title/Pclass group to the empty values of Age.

train_df.Age = train_grouped.Age.apply(lambda x: x.fillna(x.mean()))
#Checking how many null values ​​there are in Age (Expected Value = 0)

train_df.Age.isnull().sum()
#Let's do the same with the test set



test_df['Title'] = test_df.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

test_df.Title = test_df.Title.map(norm_titles)

test_grouped = test_df.groupby(['Sex','Title','Pclass'])

test_df.Age = test_grouped.Age.apply(lambda x: x.fillna(x.mean()))

test_df.Age.isnull().sum()



test_df.Title.value_counts()


# find most frequent Embarked value and store in variable

most_embarked = train_df.Embarked.value_counts().index[0]



# fill NaN with most_embarked value

train_df.Embarked = train_df.Embarked.fillna(most_embarked)

# fill NaN with median fare

train_df.Fare = train_df.Fare.fillna(train_df.Fare.median())



train_df.Cabin = train_df.Cabin.fillna('U')

train_df.Cabin = train_df.Cabin.map(lambda x: x[0])

train_df['Cabin'] = train_df.Cabin.replace({'T': 'G'})

train_df.Cabin.value_counts()

#Doing the same with Test data



# fill Cabin NaN with U for unknown

test_df.Cabin = test_df.Cabin.fillna('U')

# find most frequent Embarked value and store in variable

most_embarked = test_df.Embarked.value_counts().index[0]



# fill NaN with most_embarked value

test_df.Embarked = test_df.Embarked.fillna(most_embarked)

# fill NaN with median fare

test_df.Fare = test_df.Fare.fillna(train_df.Fare.median())



#O Máximo que dá para fazer com a coluna Cabin por hora é isolar a primeira letra e agrupá-la

test_df['Cabin'] = test_df.Cabin.apply(lambda name: name[0])

test_df.Cabin.value_counts()
train_df['FamilySize'] = train_df.Parch + train_df.SibSp + 1

train_df['FamilySize'].describe()
#Relation between Family and Survived

g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'FamilySize', bins=20)
#Now for Test

test_df['FamilySize'] = test_df.Parch + test_df.SibSp + 1

test_df['FamilySize'].describe()
# TRAIN

train_df.Sex = train_df.Sex.map({"male": 0, "female":1})

# create dummy variables for categorical features

pclass_dummies = pd.get_dummies(train_df.Pclass, prefix="Pclass")

title_dummies = pd.get_dummies(train_df.Title, prefix="Title")

cabin_dummies = pd.get_dummies(train_df.Cabin, prefix="Cabin")

embarked_dummies = pd.get_dummies(train_df.Embarked, prefix="Embarked")

# concatenate dummy columns with main dataset

train_dummies = pd.concat([train_df, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)



# drop categorical fields

train_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



train_dummies.head()
train_dummies.columns
len(train_dummies.columns)
# TEST

test_df.Sex = test_df.Sex.map({"male": 0, "female":1})

# create dummy variables for categorical features

pclass_dummies = pd.get_dummies(test_df.Pclass, prefix="Pclass")

title_dummies = pd.get_dummies(test_df.Title, prefix="Title")

cabin_dummies = pd.get_dummies(test_df.Cabin, prefix="Cabin")

embarked_dummies = pd.get_dummies(test_df.Embarked, prefix="Embarked")

# concatenate dummy columns with main dataset

test_dummies = pd.concat([test_df, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)



# drop categorical fields

test_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



test_dummies.head()

test_dummies.columns
len(test_dummies.columns)
test_dummies.columns
train_dummies.columns
import operator

import math

import random



import numpy as np



from deap import algorithms

from deap import base

from deap import creator

from deap import tools

from deap import gp
def mydeap(mungedtrain, epochs): 

    

    inputs = mungedtrain.drop('Survived', axis = 1).values.tolist()

    outputs = mungedtrain['Survived'].values.tolist()

    

    # Define new functions

    def protectedDiv(left, right):

        try:

            return left / right

        except ZeroDivisionError:

            return 1

    

    pset = gp.PrimitiveSet("MAIN", 26) # eight input

    pset.addPrimitive(operator.add, 2)

    pset.addPrimitive(operator.sub, 2)

    pset.addPrimitive(operator.mul, 2)

    pset.addPrimitive(protectedDiv, 2)

    pset.addPrimitive(operator.neg, 1)

    pset.addPrimitive(math.cos, 1)

    pset.addPrimitive(math.sin, 1)

    pset.addPrimitive(max, 2)

    pset.addPrimitive(min, 2) # add more?

    pset.renameArguments(ARG0='x1')

    pset.renameArguments(ARG1='x2')

    pset.renameArguments(ARG2='x3')

    pset.renameArguments(ARG3='x4')

    pset.renameArguments(ARG4='x5')

    pset.renameArguments(ARG5='x6')

    pset.renameArguments(ARG6='x7')

    pset.renameArguments(ARG7='x8')

    pset.renameArguments(ARG8='x9')

    pset.renameArguments(ARG9='x10')

    pset.renameArguments(ARG10='x11')

    pset.renameArguments(ARG11='x12')

    pset.renameArguments(ARG12='x13')

    pset.renameArguments(ARG13='x14')

    pset.renameArguments(ARG14='x15')

    pset.renameArguments(ARG15='x16')

    pset.renameArguments(ARG16='x17')

    pset.renameArguments(ARG17='x18')

    pset.renameArguments(ARG18='x19')

    pset.renameArguments(ARG19='x20')

    pset.renameArguments(ARG20='x21')

    pset.renameArguments(ARG21='x22')

    pset.renameArguments(ARG22='x23')

    pset.renameArguments(ARG23='x24')

    pset.renameArguments(ARG24='x25')

    pset.renameArguments(ARG25='x26')

    

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))

    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) #

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)

    

    def evalSymbReg(individual):

        # Transform the tree expression in a callable function

        func = toolbox.compile(expr=individual)

        # Evaluate the accuracy

        return sum(round(1.-(1./(1.+np.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs))/len(mungedtrain),

    

    toolbox.register("evaluate", evalSymbReg)

    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("mate", gp.cxOnePoint)

    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    

    random.seed(318)

    

    pop = toolbox.population(n=300) #

    hof = tools.HallOfFame(1)

    

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    mstats.register("avg", np.mean)

    mstats.register("std", np.std)

    mstats.register("min", np.min)

    mstats.register("max", np.max)

    

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, epochs, stats=mstats,

                                   halloffame=hof, verbose=True) #

    

    print(hof[0])

    func2 =toolbox.compile(expr=hof[0])

    return func2
def Outputs(data):

    return np.round(1.-(1./(1.+np.exp(-data))))


#Main Function

if __name__ == "__main__":

    train = train_dummies

    test = test_dummies.columns

    

    #passengerId = test.PassengerId.astype(int)

    #mungedtrain = MungeData(train)

    mungedtrain = train_dummies.astype(float) #PreProcessing(train)

    

    #Genetic Programing 

    GeneticFunction = mydeap(mungedtrain, epochs = 100)

    

    #test

    #mytrain = mungedtrain.iloc[:,2:10].values.tolist()

    mytrain = mungedtrain.drop('Survived', axis = 1).values.tolist()

    

    trainPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytrain]))



    from sklearn.metrics import accuracy_score

    print(accuracy_score(mungedtrain.Survived.astype(int),trainPredictions.astype(int)))

    

    #mungedtest = MungeData(test)

    mungedtest = test_dummies.astype(float) #PreProcessing(test)

    #mytest = mungedtest.iloc[:,1:9].values.tolist()

    mytest = mungedtest.values.tolist()

     

    testPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytest]))



    pdtest = pd.DataFrame({'PassengerId': passengerId,

                            'Survived': testPredictions.astype(int)})

    pdtest.to_csv('gptest.csv', index=False)