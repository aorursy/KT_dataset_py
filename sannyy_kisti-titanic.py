# [Using DEAP]

# [데이터 정제]





# [dummies]

# Pclass_dum = pd.get_dummies(X_train.Pclass, prefix = "Pclass")

# Deck_dum = pd.get_dummies(X_train.Deck, prefix = "Deck")

# Embarked_dum = pd.get_dummies(X_train.Embarked, prefix = "Embarked")

# Fare_dum = pd.get_dummies(X_train.Fare, prefix = "Fare")

# title_dum = pd.get_dummies(X_train.Title, prefix = "Title")

# train_dum = pd.concat([X_train, Deck_dum, Embarked_dum, Fare_dum, title_dum], axis = 1)

# train_dum.drop(["Pclass", "Deck", "Embarked", "Fare","Title"], axis=1, inplace=True)

# train_dum.drop(["Deck_7"], axis=1, inplace=True)

# train_dum["Survived"] = train["Survived"]  

# train_dum.head()



# Pclass_dum = pd.get_dummies(X_test.Pclass, prefix = "Pclass")

# Deck_dum = pd.get_dummies(X_test.Deck, prefix = "Deck")

# Embarked_dum = pd.get_dummies(X_test.Embarked, prefix = "Embarked")

# Fare_dum = pd.get_dummies(X_test.Fare, prefix = "Fare")

# title_dum = pd.get_dummies(X_test.Title, prefix = "Title")

# test_dum = pd.concat([X_test, Deck_dum, Embarked_dum, Fare_dum, title_dum], axis = 1)

# test_dum.drop(["Pclass", "Deck", "Embarked", "Fare", "Title"], axis=1, inplace=True)

# test_dum.head()





# [GP: Genetic Programming] - DEAP이용(PYTHON)

# import operator

# import math

# import random



# import numpy as np



# from deap import algorithms

# from deap import base

# from deap import creator

# from deap import tools

# from deap import gp



# def mydeap(mungedtrain, epochs): 

    

#     inputs = mungedtrain.drop('Survived', axis = 1).values.tolist()

#     outputs = mungedtrain['Survived'].values.tolist()

    

#     # Define new functions

#     def protectedDiv(left, right):

#         try:

#             return left / right

#         except ZeroDivisionError:

#             return 1

    

#     pset = gp.PrimitiveSet("MAIN", 25) # eight input

#     pset.addPrimitive(operator.add, 2)

#     pset.addPrimitive(operator.sub, 2)

#     pset.addPrimitive(operator.mul, 2)

#     pset.addPrimitive(protectedDiv, 2)

#     pset.addPrimitive(operator.neg, 1)

#     pset.addPrimitive(operator.pos, 1)

#     pset.addPrimitive(math.cos, 1)

#     pset.addPrimitive(math.sin, 1)

#     pset.addPrimitive(math.tanh, 1)

#     pset.addPrimitive(max, 2)

#     pset.addPrimitive(min, 2) # add more?

#     pset.addEphemeralConstant("rand101", lambda: random.uniform(-10,10))

#     pset.renameArguments(ARG0='x1')

#     pset.renameArguments(ARG1='x2')

#     pset.renameArguments(ARG2='x3')

#     pset.renameArguments(ARG3='x4')

#     pset.renameArguments(ARG4='x5')

#     pset.renameArguments(ARG5='x6')

#     pset.renameArguments(ARG6='x7')

#     pset.renameArguments(ARG7='x8')

#     pset.renameArguments(ARG8='x9')

#     pset.renameArguments(ARG9='x10')

#     pset.renameArguments(ARG10='x11')

#     pset.renameArguments(ARG11='x12')

#     pset.renameArguments(ARG12='x13')

#     pset.renameArguments(ARG13='x14')

#     pset.renameArguments(ARG14='x15')

#     pset.renameArguments(ARG15='x16')

#     pset.renameArguments(ARG16='x17')

#     pset.renameArguments(ARG17='x18')

#     pset.renameArguments(ARG18='x19')

#     pset.renameArguments(ARG19='x20')

#     pset.renameArguments(ARG20='x21')

#     pset.renameArguments(ARG21='x22')

#     pset.renameArguments(ARG22='x23')

#     pset.renameArguments(ARG23='x24')

#     pset.renameArguments(ARG24='x25')

    

#     creator.create("FitnessMin", base.Fitness, weights=(1.0,))

#     creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    

#     toolbox = base.Toolbox()

#     toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) #

#     toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#     toolbox.register("compile", gp.compile, pset=pset)

    

#     def evalSymbReg(individual):

#         # Transform the tree expression in a callable function

#         func = toolbox.compile(expr=individual)

#         # Evaluate the accuracy

#         return sum(round(1.-(1./(1.+np.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs))/len(mungedtrain),

    

#     toolbox.register("evaluate", evalSymbReg)

#     toolbox.register("select", tools.selTournament, tournsize=3)

#     toolbox.register("mate", gp.cxOnePoint)

#     toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

#     toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    

#     toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

#     toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    

#     random.seed(318)

    

#     pop = toolbox.population(n=300) #

#     hof = tools.HallOfFame(1)

    

#     stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

#     stats_size = tools.Statistics(len)

#     mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

#     mstats.register("avg", np.mean)

#     mstats.register("std", np.std)

#     mstats.register("min", np.min)

#     mstats.register("max", np.max)

    

#     pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, epochs, stats=mstats,

#                                    halloffame=hof, verbose=True) #

    

#     print(hof[0])

#     func2 =toolbox.compile(expr=hof[0])

#     return func2

    

# def Outputs(data):

#     return np.round(1.-(1./(1.+np.exp(-data))))

   



# #Main Function

# if __name__ == "__main__":

#     train = train_dum

#     test = test_dum.columns

     

#     mungedtrain = train_dum.astype(float) 

    

#     #Genetic Programing 

#     GeneticFunction = mydeap(mungedtrain, epochs = 100)

    

#     #test 

#     mytrain = mungedtrain.drop('Survived', axis = 1).values.tolist()

    

#     trainPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytrain]))



#     from sklearn.metrics import accuracy_score

#     print(accuracy_score(mungedtrain.Survived.astype(int),trainPredictions.astype(int)))

    

#     mungedtest = test_dum.astype(float) 

#     mytest = mungedtest.values.tolist()

     

#     testPredictions = Outputs(np.array([GeneticFunction(*x) for x in mytest]))



#     pdtest = pd.DataFrame({'PassengerId': passenger_id,

#                             'Survived': testPredictions.astype(int)})

#     pdtest.to_csv('gptest.csv', index=False)

    

import numpy as np

import pandas as pd



def Outputs(data):

    return np.round(1.-(1./(1.+np.exp(-data))))



# [GP] - C++ 이용



def GeneticFunction(data):

    return ((np.minimum( ((((0.058823499828577 + data["Sex"]) - np.cos((data["Pclass"] / 2.0))) * 2.0)),  ((0.885868))) * 2.0) +

            np.maximum( ((data["SibSp"] - 2.5)),  ( -(np.minimum( (data["Sex"]),  (np.sin(data["Parch"]))) * data["Pclass"]))) +

            (0.1 * ((np.minimum( (data["Sex"]),  (((data["Parch"] / 2.0) / 2.0))) * data["Age"]) - data["Cabin"])) +

            np.minimum( ((np.sin((data["Parch"] * ((data["Fare"] - 0.7) * 2.0))) * 2.0)),  ((data["SibSp"] / 2.0))) +

            np.maximum( (np.minimum( ( -np.cos(data["Embarked"])),  (0.138462007045746))),  (np.sin(((data["Cabin"] - data["Fare"]) * 2.0)))) +

            -np.minimum( ((((data["Age"] * data["Parch"]) * data["Embarked"]) + data["Parch"])),  (np.sin(data["Pclass"]))) +

            np.minimum( (data["Sex"]),  ((np.sin( -(data["Fare"] * np.cos((data["Fare"] * 1.5)))) / 2.0))) +

            np.minimum( ((0.230145)),  (np.sin(np.minimum( (((67.0 / 2.0) * np.sin(data["Fare"]))),  (0.31830988618379069))))) +

            np.sin((np.sin(data["Cabin"]) * (np.sin((12.6275)) * np.maximum( (data["Age"]),  (data["Fare"]))))) +

            np.sin(((np.minimum( (data["Fare"]),  ((data["Cabin"] * data["Embarked"]))) / 2.0) *  -data["Fare"])) +

            np.minimum( (((2.6 * data["SibSp"]) * np.sin(((96) * np.sin(data["Cabin"]))))),  (data["Parch"])) +

            np.sin(np.sin((np.maximum( (np.minimum( (data["Age"]),  (data["Cabin"]))),  ((data["Fare"] * 0.2))) * data["Cabin"]))) +

            np.maximum( (np.sin(((12.4148) * (data["Age"] / 2.0)))),  (np.sin((-3.0 * data["Cabin"])))) +

            (np.minimum( (np.sin((((np.sin(((data["Fare"] * 2.0) * 2.0)) * 2.0) * 2.0) * 2.0))),  (data["SibSp"])) / 2.0) +

            ((data["Sex"] - data["SibSp"]) * (np.cos(((data["Embarked"] - 0.7) + data["Age"])) / 2.0)) +

            ((np.sin(data["Cabin"]) / 2.0) - (np.cos(np.minimum( (data["Age"]),  (data["Embarked"]))) * np.sin(data["Embarked"]))) +

            np.minimum( (0.3),  ((data["Sex"] * (2.212120056152344 * (0.720430016517639 - np.sin((data["Age"] * 2.0))))))) +

            (np.minimum( (np.cos(data["Fare"])),  (np.maximum( (np.sin(data["Age"])),  (data["Parch"])))) * np.cos((data["Fare"] / 2.0))) +

            np.sin((data["Parch"] * np.minimum( ((data["Age"] - 1.5707963267948966)),  ((np.cos((data["Pclass"] * 2.0)) / 2.0))))) +

            (data["Parch"] * (np.sin(((data["Fare"] * (0.623655974864960 * data["Age"])) * 2.0)) / 2.0)) +

            (0.31830988618379069 * np.cos(np.maximum( ((0.5 * data["Fare"])),  ((np.sin(0.720430016517639) * data["Age"]))))) +

            (np.minimum( ((data["SibSp"] / 2.0)),  (np.sin(((data["Pclass"] - data["Fare"]) * data["SibSp"])))) * data["SibSp"]) +

            np.tanh((data["Sex"] * np.sin((5.3 * np.sin((data["Cabin"] * np.cos(data["Fare"]))))))) +

            (np.minimum( (data["Parch"]),  (data["Sex"])) * np.cos(np.maximum( ((np.cos(data["Parch"]) + data["Age"])),  (3.2)))) +

            (np.minimum( (np.tanh(((data["Cabin"] / 2.0) + data["Parch"]))),  ((data["Sex"] + np.cos(data["Age"])))) / 2.0) +

            (np.sin((np.sin(data["Sex"]) * (np.sin((data["Age"] * data["Pclass"])) * data["Pclass"]))) / 2.0) +

            (data["Sex"] * (np.cos(((data["Sex"] + data["Fare"]) * ((8.48635) * (63)))) / 2.0)) +

            np.minimum( (data["Sex"]),  ((np.cos((data["Age"] * np.tanh(np.sin(np.cos(data["Fare"]))))) / 2.0))) +

            (np.tanh(np.tanh( -np.cos((np.maximum( (np.cos(data["Fare"])),  (0.094339601695538)) * data["Age"])))) / 2.0) +

            (np.tanh(np.cos((np.cos(data["Age"]) + (data["Age"] + np.minimum( (data["Fare"]),  (data["Age"])))))) / 2.0) +

            (np.tanh(np.cos((data["Age"] * ((-2.0 + np.sin(data["SibSp"])) + data["Fare"])))) / 2.0) +

            (np.minimum( (((281) - data["Fare"])),  (np.sin((np.maximum( ((176)),  (data["Fare"])) * data["SibSp"])))) * 2.0) +

            np.sin(((np.maximum( (data["Embarked"]),  (data["Age"])) * 2.0) * (((785) * 3.1415926535897931) * data["Age"]))) +

            np.minimum( (data["Sex"]),  (np.sin( -(np.minimum( ((data["Cabin"] / 2.0)),  (data["SibSp"])) * (data["Fare"] / 2.0))))) +

            np.sin(np.sin((data["Cabin"] * (data["Embarked"] + (np.tanh( -data["Age"]) + data["Fare"]))))) +

            (np.cos(np.cos(data["Fare"])) * (np.sin((data["Embarked"] - ((734) * data["Fare"]))) / 2.0)) +

            ((np.minimum( (data["SibSp"]),  (np.cos(data["Fare"]))) * np.cos(data["SibSp"])) * np.sin((data["Age"] / 2.0))) +

            (np.sin((np.sin((data["SibSp"] * np.cos((data["Fare"] * 2.0)))) + (data["Cabin"] * 2.0))) / 2.0) +

            (((data["Sex"] * data["SibSp"]) * np.sin(np.sin( -(data["Fare"] * data["Cabin"])))) * 2.0) +

            (np.sin((data["SibSp"] * ((((5.428569793701172 + 67.0) * 2.0) / 2.0) * data["Age"]))) / 2.0) +

            (data["Pclass"] * (np.sin(((data["Embarked"] * data["Cabin"]) * (data["Age"] - (1.07241)))) / 2.0)) +

            (np.cos((((( -data["SibSp"] + data["Age"]) + data["Parch"]) * data["Embarked"]) / 2.0)) / 2.0) +

            (0.31830988618379069 * np.sin(((data["Age"] * ((data["Embarked"] * np.sin(data["Fare"])) * 2.0)) * 2.0))) +

            ((np.minimum( ((data["Age"] * 0.058823499828577)),  (data["Sex"])) - 0.63661977236758138) * np.tanh(np.sin(data["Pclass"]))) +

            -np.minimum( ((np.cos(((727) * ((data["Fare"] + data["Parch"]) * 2.0))) / 2.0)),  (data["Fare"])) +

            (np.minimum( (np.cos(data["Fare"])),  (data["SibSp"])) * np.minimum( (np.sin(data["Parch"])),  (np.cos((data["Embarked"] * 2.0))))) +

            (np.minimum( (((data["Fare"] / 2.0) - 2.675679922103882)),  (0.138462007045746)) * np.sin((1.5707963267948966 * data["Age"]))) +

            np.minimum( ((0.0821533)),  (((np.sin(data["Fare"]) + data["Embarked"]) - np.cos((data["Age"] * (9.89287)))))))
# [데이터 정제]



def MungeData(data):

    # Sex

    data.drop(['Ticket', 'Name'], inplace=True, axis=1)

    data.Sex.fillna('0', inplace=True)

    data.loc[data.Sex != 'male', 'Sex'] = 0

    data.loc[data.Sex == 'male', 'Sex'] = 1

    # Cabin

    data.Cabin.fillna('0', inplace=True)

    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1

    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2

    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3

    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4

    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5

    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6

    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7

    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8

    # Embarked

    data.loc[data.Embarked == 'C', 'Embarked'] = 1

    data.loc[data.Embarked == 'Q', 'Embarked'] = 2

    data.loc[data.Embarked == 'S', 'Embarked'] = 3

    data.Embarked.fillna(0, inplace=True)

    data.fillna(-1, inplace=True)

    return data.astype(float)





if __name__ == "__main__":

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    

    mungedtrain = MungeData(train)

    print(mungedtrain)

    trainPredictions = Outputs(GeneticFunction(mungedtrain)) 

    pdtrain = pd.DataFrame({'PassengerId': mungedtrain.PassengerId.astype(int),

                            'Predicted': trainPredictions.astype(int),

                            'Survived': mungedtrain.Survived.astype(int)})

    pdtrain.to_csv('gptrain.csv', index=False)

    

    mungedtest = MungeData(test) 

    testPredictions = Outputs(GeneticFunction(mungedtest))



    pdtest = pd.DataFrame({'PassengerId': mungedtest.PassengerId.astype(int),

                            'Survived': testPredictions.astype(int)})

    pdtest.to_csv('gptest.csv', index=False)

 

 
# **REF**



# https://docs.python.org/3/library/operator.html

# https://www.kaggle.com/scirpus/genetic-programming-lb-0-88

# https://github.com/innjoshka/Genetic-Programming-Titanic-Kaggle

# http://geneticprogramming.com/software/

# https://github.com/DEAP/deap

# https://deap.readthedocs.io/en/master/index.html




