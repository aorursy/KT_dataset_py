import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)



import matplotlib.pyplot as plt

import seaborn as sns



df_train = pd.read_csv('../input/titanic/train.csv', delimiter = ",")

df_test = pd.read_csv('../input/titanic/test.csv', delimiter = ",")

df_sub = pd.read_csv('../input/titanic/gender_submission.csv', delimiter = ",")
df = pd.concat([df_train, df_test],sort=False)

df.reset_index(drop=True,inplace=True)
age60=df[(df['Age']>60) & (df['Age']<70)]

age60.groupby(['Pclass','Embarked','SibSp','Parch']).median()

df.loc[df['PassengerId'] == 1044, 'Fare'] = 7.9
df['Fare']=np.round(df['Fare'])

p01 = df['Fare'].quantile(0.01)

p99 = df['Fare'].quantile(0.99)

df['Fare'] = df['Fare'].clip(p01, p99)
df['Farecut']=pd.cut(df['Fare'], 5, labels=False)
column = 'mrms'

df[column] = df['Name'].str.extract('([A-Za-z]+)\.', expand = False)

df[column] = df[column].replace(['Capt', 'Countess','Lady','Sir','Jonkheer', 'Don','Dona', 'Mlle','Mme','Major'], 'test_non')

df[column] = df[column].replace(['Ms'], 'Miss')

df[column] = df[column].replace(['Col','Dr', 'Rev'], 'Other')
column='Embarked'

df['Embarked'].fillna('S',inplace=True)
df.loc[df['Parch'] >= 3, 'Parch'] = 3

df.loc[df['SibSp'] >= 3, 'SibSp'] = 3
df['Parch+SibSp']=df['Parch']+df['SibSp']
df['Parch-SibSp']=df['Parch']-df['SibSp']
l=list(['Pclass','SibSp','Fare','Parch'])

df['Agemedian'] = df['Age'].fillna(df.groupby(l)['Age'].transform('median'))

df['Agemedian'] = df['Agemedian'].fillna(df['Agemedian'].median())

df['Agemedian'] = np.round(df['Agemedian'])
column = 'Cabin'

df[column+'head'] = df[column].str.extract('([A-Za-z]+)', expand = False)

labels, uniques = pd.factorize(df['Cabinhead'])

df['Cabinhead']=labels



df['Cabinhead'].fillna(0,inplace=True)

df['Cabinhead'].replace(-1,np.NaN,inplace=True)

Ticket = df['Ticket'].str.extract('(.*)\s(.*)')

df['Ticket_head']=Ticket[0]

df['Ticket_num']=Ticket[1]



l = list(['Pclass','Farecut'])

df['Cabinheadmedian'] = df['Cabinhead'].fillna(df.groupby(l)['Cabinhead'].transform('median'))



df['Cabinheadmedian'] = np.round(df['Cabinheadmedian'])
l = list(['Farecut'])

df['Cabinheadmedian'] = df['Cabinhead'].fillna(df.groupby(l)['Cabinhead'].transform('median'))
for column in ['mrms','Sex','Embarked','Ticket_head']:

    labels, uniques = pd.factorize(df[column])

    df[column]=labels
df = df.drop(['Name','Ticket','Cabin','Cabinhead','Age','Fare','Ticket_num'], axis=1)
import operator

import math

import random

from deap import gp

from deap import algorithms

from deap import base

from deap import creator

from deap import tools
train=df
target = train['Survived']
def Outputs(data):

    return np.round(1.-(1./(1.+np.exp(-data))))
def mydeap(mungedtrain):

    

    import operator

    import math

    import random

    

    import numpy

    

    from deap import algorithms

    from deap import base

    from deap import creator

    from deap import tools

    from deap import gp

    

    inputs = mungedtrain.iloc[:,2:14].values.tolist()

    outputs = mungedtrain['Survived'].values.tolist()

    

    # Define new functions

    def protectedDiv(left, right):

        try:

            return left / right

        except ZeroDivisionError:

            return 1

    

    pset = gp.PrimitiveSet("MAIN", 12) # eight input

    pset.addPrimitive(operator.add, 2)

    pset.addPrimitive(operator.sub, 2)

    pset.addPrimitive(operator.mul, 2)

    pset.addPrimitive(protectedDiv, 2)

    pset.addPrimitive(operator.neg, 1)

    pset.addPrimitive(math.cos, 1)

    pset.addPrimitive(math.sin, 1)

    pset.addPrimitive(max, 2)

    pset.addPrimitive(min, 2) # add more?

    pset.addEphemeralConstant("rand101", lambda: random.uniform(-10,10)) # adjust?

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

    

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))

    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) #深さ

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)#個体

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)#世代

    toolbox.register("compile", gp.compile, pset=pset)

    

    def evalSymbReg(individual):

        # Transform the tree expression in a callable function

        func = toolbox.compile(expr=individual)

        # Evaluate the accuracy

        return sum(round(1.-(1./(1.+numpy.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs))/len(mungedtrain),

    

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

    mstats.register("avg", numpy.mean)

    mstats.register("std", numpy.std)

    mstats.register("min", numpy.min)

    mstats.register("max", numpy.max)

    

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 1000, stats=mstats,

                                   halloffame=hof, verbose=True) #

    

    print(hof[0])

    func2 =toolbox.compile(expr=hof[0])

    return func2
train
if __name__ == "__main__":



    mungedtrain = train[:df_train.shape[0]]



    #GP

    GeneticFunction = mydeap(mungedtrain)

    

    #test

    mytrain = mungedtrain.iloc[:,2:14].values.tolist()

    trainPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytrain]))

    

    pdtrain = pd.DataFrame({'PassengerId': mungedtrain.PassengerId.astype(int),

                            'Predicted': trainPredictions.astype(int),

                            'Survived': mungedtrain.Survived.astype(int)})

    

    pdtrain.to_csv('MYgptrain.csv', index=False)
    from sklearn.metrics import accuracy_score

    print(accuracy_score(df_train.Survived.astype(int),trainPredictions.astype(int)))

    

    mungedtest = train[df_train.shape[0]:]

    mytest = mungedtest.iloc[:,2:14].values.tolist()

    testPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytest]))



    pdtest = pd.DataFrame({'PassengerId': mungedtest.PassengerId.astype(int),

                            'Survived': testPredictions.astype(int)})

    pdtest.to_csv('gptest.csv', index=False)
print('end')