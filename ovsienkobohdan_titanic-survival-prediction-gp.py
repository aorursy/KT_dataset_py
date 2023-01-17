import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD



import re

import operator

import math

import random

random.seed(42)



import warnings



from deap import gp

from deap import algorithms

from deap import base

from deap import creator

from deap import tools
warnings.filterwarnings("ignore")
def Pset(names):

    # Define new functions

    def ifelse(input, output1, output2):

        return output1 if input else output2

 

    def or_(left, right):

        return int(left) | int(right)

        

    def and_(left, right):

        return int(left) & int(right)

    

    def xor_(left, right):

        return int(left) ^ int(right)



    def abs_(input):

        return abs(input)

    

    def pDiv(left, right):

        try:

            return left / right

        except ZeroDivisionError:

            return 1



    def pPow_(left, right):

        try:

            return abs(left) ** min(float(right),8)

        except ZeroDivisionError:

            return 1

        except OverflowError:

            return 1

    

    pset = gp.PrimitiveSet("MAIN", len(names))

    pset.addPrimitive(pDiv, 2)

    #pset.addPrimitive(pPow_, 2)

    pset.addPrimitive(abs_, 1)

    pset.addPrimitive(operator.add, 2)

    pset.addPrimitive(operator.sub, 2)

    pset.addPrimitive(operator.mul, 2)

    pset.addPrimitive(operator.neg, 1)

    #pset.addPrimitive(ifelse, 3)

    #pset.addPrimitive(operator.lt, 2)

    #pset.addPrimitive(operator.le, 2)

    #pset.addPrimitive(operator.eq, 2)

    #pset.addPrimitive(operator.ne, 2)

    #pset.addPrimitive(operator.gt, 2)

    #pset.addPrimitive(operator.ge, 2)

    #pset.addPrimitive(operator.not_, 1) 

    #pset.addPrimitive(and_, 2)

    #pset.addPrimitive(or_, 2)

    #pset.addPrimitive(xor_, 2)

    pset.addPrimitive(math.floor, 1)

    pset.addPrimitive(math.tanh, 1)

    pset.addPrimitive(math.sin, 1)

    pset.addPrimitive(math.cos, 1)

    pset.addPrimitive(max, 2)

    pset.addPrimitive(min, 2)



    #for i in range(25):

    #    pset.addEphemeralConstant(f"c{i}", lambda: round(random.random()+0.001,2))

        

    #pset.addTerminal(False)

    #pset.addTerminal(True)

    pset.addTerminal(2.0)

    pset.addTerminal(1.0)

    pset.addTerminal(0.5)

    pset.addTerminal(0.4)

    pset.addTerminal(0.3)

    pset.addTerminal(0.2)

    pset.addTerminal(0.1)

    pset.addTerminal(0.05)

    pset.addTerminal(0.02)

    pset.addTerminal(0.01)    



    # Rename arguments with columns names

    for i, a in enumerate(pset.arguments):

        new_name = names[i]

        pset.arguments[i] = new_name

        pset.mapping[new_name] = pset.mapping[a]

        pset.mapping[new_name].value = new_name

        del pset.mapping[a]



    return pset
def mydeap(mungedtrain, target):



    inputs = mungedtrain.values.tolist()

    outputs = target.values.tolist()



    pset = Pset(list(mungedtrain.columns))

    

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))

    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)

    

    def evalSymbReg(individual):

        # Transform the tree expression in a callable function

        func = toolbox.compile(expr=individual)

        # Evaluate the accuracy

        return sum(round(1.-(1./(1.+np.exp(-func(*in_))))) == out for in_,

                   out in zip(inputs, outputs))/len(mungedtrain),

    

    toolbox.register("evaluate", evalSymbReg)

    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("mate", gp.cxOnePoint)

    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    #toolbox.register("map", dtm.map)

    

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=37))

    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=37))

   

    random.seed(42)



    pop = toolbox.population(n=600)

    hof = tools.HallOfFame(1)

    

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    mstats.register("avg", np.mean)

    #mstats.register("std", np.std)

    #mstats.register("min", np.min)

    mstats.register("max", np.max)

    

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=500,

                                   stats=mstats, halloffame=hof, verbose=True)



    print(hof[0])

    print(hof[0].fitness.values)

    return hof[0], toolbox
def Outputs(data):

    return np.round(1.-(1./(1.+np.exp(-data))))
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

            return title_search.group(1)

    return ""



def PrepData(data):

    data['IsNull'] = data.isnull().sum(axis=1)

    data['Ticket'] = data['Ticket'].str.lower().replace('\W', '')

    # Sex

    data.Sex.fillna(0, inplace=True)

    data.loc[data.Sex != 'male', 'Sex'] = 1

    data.loc[data.Sex == 'male', 'Sex'] = 0

    data['NameLen'] = data['Name'].apply(len)

    bin_num = 4

    data['NameLen'] = pd.qcut(data['NameLen'], bin_num,labels=list(range(bin_num))).astype(float)   

    # Feature that tells whether a passenger had a cabin on the Titanic

    data['Has_Cabin'] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

    # Create new feature FamilySize as a combination of SibSp and Parch

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # Create new feature IsAlone from FamilySize

    data['isFamily'] = 1

    data.loc[data['isFamily'] == 1, 'notAlone'] = 0

    # Create a new feature Title, containing the titles of passenger names

    data['Title'] = data['Name'].apply(get_title)

    # Group all non-common titles into one single grouping "Rare"

    mapping = {'Mlle': 'Rare', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Rare', 'Rev': 'Mr',

               'Don': 'Mr', 'Mme': 'Rare', 'Jonkheer': 'Mr', 'Lady': 'Mrs',

               'Capt': 'Mr', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare'}

    data.replace({'Title': mapping}, inplace=True)

    # Mapping titles

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}

    data['Title'] = data['Title'].map(title_mapping)

    data['Title'] = data['Title'].fillna(0)

    # Remove all NULLS in the Embarked column

    data['Embarked'].fillna(method='backfill', inplace=True)

    # Mapping Embarked

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Remove all NULLS in the Fare column and create a new feature

    data['Fare'] = data['Fare'].fillna(train['Fare'].median())

    # Mapping Fare

    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2

    data.loc[ data['Fare'] > 31, 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)

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

    data['Cabin'] = data['Cabin'].astype(int)

    # Fillna Age

    grouped = data.groupby(['Sex','Pclass', 'Title'])

    data['Age'] = grouped['Age'].apply(lambda x: x.fillna(x.median()))

    data['Age'] = data['Age'].astype(int)

    # select females and masters (boys)

    boy = (data['Name'].str.contains('Master')) | ((data['Sex']==0) & (data['Age']<13))

    female = data['Sex']==1

    boy_or_female = boy | female   

    # no. females + boys on ticket

    n_ticket = data[boy_or_female].groupby('Ticket').Survived.count()

    # survival rate amongst females + boys on ticket

    tick_surv = data[boy_or_female].groupby('Ticket').Survived.mean()

    data['Boy'] = (data['Name'].str.contains('Master')) | ((data['Sex']==0) & (data['Age']<13))   

    # if ticket exists in training data, fill NTicket with no. women+boys

    # on that ticket in the training data.

    data['NTicket'] = data['Ticket'].replace(n_ticket)

    # otherwise NTicket=0

    data.loc[~data.Ticket.isin(n_ticket.index),'NTicket']=0

    # if ticket exists in training data, fill TicketSurv with

    # women+boys survival rate in training data  

    data['TicketSurv'] = data['Ticket'].replace(tick_surv)

    # otherwise TicketSurv=0

    data.loc[~data.Ticket.isin(tick_surv.index),'TicketSurv']=0

    data['TicketSurv'].fillna(0, inplace=True)

    # Mapping Age

    data.loc[ data['Age'] <= 16, 'Age'] = 5

    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1

    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2

    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3

    data.loc[ data['Age'] > 64, 'Age'] = 4

    tfidf_vec = TfidfVectorizer(max_features=15, token_pattern="\w+")

    svd = TruncatedSVD(n_components=10)

    tfidf_array = svd.fit_transform(tfidf_vec.fit_transform(data["Name"]))

    for i in range(tfidf_array.shape[1]):

        data.insert(len(data.columns), column = 'Name_' + str(i), value = tfidf_array [:,i])

    tfidf_vec = TfidfVectorizer(max_features=5, analyzer="char")

    svd = TruncatedSVD(n_components=3)

    tfidf_array = svd.fit_transform(tfidf_vec.fit_transform(data["Ticket"]))

    for i in range(tfidf_array.shape[1]):

        data.insert(len(data.columns), column = 'Ticket_' + str(i), value = tfidf_array [:,i])

    data['Ticket'] = data['Ticket'].str.extract('(\d+)', expand=False).fillna(0).astype(float)

    data['Ticket'] = np.round(np.log1p(data['Ticket'])*10)

    data.drop(['Name'],1,inplace=True)

    return data.astype(float)
train = pd.read_csv("../input/titanic/train.csv", dtype={"Age": np.float64}, index_col='PassengerId' )

test = pd.read_csv("../input/titanic/test.csv", dtype={"Age": np.float64}, index_col='PassengerId')
# test_data_with_labels = pd.read_csv('../input/titanic-test-data/titanic.csv')

# test_data = pd.read_csv('../input/titanic/test.csv')

# for i, name in enumerate(test_data_with_labels['name']):

#     if '"' in name:

#         test_data_with_labels['name'][i] = re.sub('"', '', name)

        

# for i, name in enumerate(test_data['Name']):

#     if '"' in name:

#         test_data['Name'][i] = re.sub('"', '', name)

# survived = []



# for name in test_data['Name']:

#     survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))



# test['Survived'] = survived 
df = pd.concat((train,test),0)

target = df['Survived'].astype(float)

df = PrepData(df)



df['Ticket'] = df['Ticket'].astype(int).astype('category')



col_to_use = ['Embarked', 'Fare', 'Parch', 'Pclass', 'Sex', 'Age', 

              'NameLen', 'Has_Cabin', 'Cabin', 'FamilySize', 'isFamily', 

              'Title', 'TicketSurv', 'NTicket', 'Boy',

              'Ticket_0', 'Ticket_1', 'Ticket_2',

              'Name_0', 'Name_1', 'Name_2', 'Name_3', 'Name_4',

              'Name_5', 'Name_6', 'Name_7', 'Name_8', 'Name_9']



df = pd.get_dummies(df[col_to_use])

df[col_to_use] += 0.0



mungedtrain = df.copy()

mungedtest = df[train.shape[0]:].copy()

mytrain = mungedtrain.values.tolist()

mytest = mungedtest.values.tolist()
#GP Train

GPhof, Tbox = mydeap(mungedtrain, target)
test = test.reset_index()

GPfunc = Tbox.compile(expr=GPhof)

testPredictions = Outputs(np.array([GPfunc(*x) for x in mytest]))

test['Survived'] = np.round(testPredictions).astype(int)

test[['PassengerId','Survived']].to_csv('gp_submit.csv', index=False)
# test_data_with_labels = pd.read_csv('../input/titanic-test-data/titanic.csv')

# test_data = pd.read_csv('../input/titanic/test.csv')

# for i, name in enumerate(test_data_with_labels['name']):

#     if '"' in name:

#         test_data_with_labels['name'][i] = re.sub('"', '', name)

        

# for i, name in enumerate(test_data['Name']):

#     if '"' in name:

#         test_data['Name'][i] = re.sub('"', '', name)

# survived = []



# for name in test_data['Name']:

#     survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))



# test['Survived'] = survived 
# GPfunc = Tbox.compile(expr=GPhof)

# testPredictions = Outputs(np.array([GPfunc(*x) for x in mytest]))

# test['Survived'] = np.round(testPredictions).astype(int)

# test[['PassengerId','Survived']].to_csv('gp_submit.csv', index=False)
# testPrediction = np.round(testPredictions).astype(int)

# print("Score :",accuracy_score(survived,testPredictions))

# # test['Survived'] = testPrediction

# # test[['PassengerId','Survived']].to_csv('gp_submit.csv', index=False)