# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import operator

import math

import random



import numpy



from deap import algorithms

from deap import base

from deap import creator

from deap import tools

from deap import gp



import warnings

warnings.filterwarnings("ignore")

from colorama import Fore, Back, Style 

def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
class Reader(object):

    

    def __init__(self,root = '/kaggle/input/titanic/',FE=False):

        

        self.train = pd.read_csv(root+'train.csv')

        self.test = pd.read_csv(root+'test.csv')

        self.simplified = FE

        

    def get(self):

        train = self.train

        test = self.test

        train['is_train'] = 1

        test['is_train'] = 0

        total = pd.concat([train,test],ignore_index=True,sort=False)

        

        if self.simplified==False:

            

            print('FE == FALSE')

            

            prRed('##Title##')

            total['Title'] = total.Name.str.split(".").str.get(0).str.split(',').str.get(1)

            #title_dict = {i:idx for idx, i in enumerate(total.Title.unique())}

            #total['Title'] = total.Title.map(title_dict)

            prRed('##Cabin##')

            total['Cabin'] = total.Cabin.str.get(0)

            cabin_dict = {i:idx for idx, i in enumerate(total.Cabin.unique())}

            total['Cabin'] = total.Cabin.map(cabin_dict)

            prRed('##Sex##')

            sex_dict = {i: idx for idx , i in enumerate(total.Sex.unique())}

            total['Sex'] = total.Sex.map(sex_dict)

            prRed('##Embarked##')

            embarked_dict = {i: idx for idx, i in enumerate(total.Embarked.unique())}

            total['Embarked'] = total.Embarked.map(embarked_dict)

            prRed('##Age##')

            age_dict = total.groupby('Title').Age.mean().to_dict()

            total.loc[total.Age.isnull(),'Age'] = total.loc[total.Age.isnull()].Title.map(age_dict)

            total.drop(columns=['Name','Ticket','PassengerId','PassengerId','Cabin', 'Ticket'],inplace=True)

        else:

            

            print('FE == TRUE')

            stat_min = 10

            prRed('##Title##')

            total['Title'] = total.Name.str.split(".").str.get(0).str.split(',').str.get(1)

            title_names = (total['Title'].value_counts() < stat_min)

            total['Title'] = total['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

            

            prRed('##Age##')

            age_dict = total.groupby('Title').Age.mean().to_dict()

            total.loc[total.Age.isnull(),'Age'] = total.loc[total.Age.isnull()].Title.map(age_dict)

            total['AgeBin'] = pd.cut(total['Age'].astype(int), 5)

            agebin_dict = {i:idx for idx, i in enumerate(total.AgeBin.unique())}

            total['AgeBin'] = total.AgeBin.map(agebin_dict)

            

            prRed('##Embarked##')

            total['Embarked'].fillna(total['Embarked'].mode()[0], inplace = True)

            prRed('##Fare##')

            total['Fare'].fillna(total['Fare'].median(), inplace = True)

            total['FareBin'] = pd.qcut(total['Fare'], 4)

            agebin_dict = {i:idx for idx, i in enumerate(total.FareBin.unique())}

            total['FareBin'] = total.FareBin.map(agebin_dict)

            

            prRed('##FamilySize##')

            total['FamilySize'] = total.SibSp + total.Parch + 1

            

            prRed('##Alone##')

            total['Alone'] = 1

            total['Alone'].loc[total['FamilySize'] > 1] = 0

            

            total.loc[(total.Age<=13)|(total.Title=='Master'),'Kid'] = 1

            

            prRed('##Kid##')

            total['Kid'].fillna(0,inplace=True)



            

            prRed('##Drop and OHE##')

            total.drop(columns=['Age', 'SibSp', 'Parch', 'Fare','Name','PassengerId','Cabin', 'Ticket'], axis=1, inplace = True)

            total = pd.get_dummies(total,columns=['Sex','Pclass', 'Embarked', 'Title','FareBin', 'AgeBin', 'Alone','Kid'])

            

        train = total.loc[total.is_train==1]

        test = total.loc[total.is_train==0]

        train.drop(columns=['is_train'],inplace=True)

        test.drop(columns=['is_train','Survived'],inplace=True)

        test.fillna(0,inplace=True)

        print('Done!')

        print(f'Train : {train.shape} , Test : {test.shape}')

        return train,test
reader = Reader(FE=True)

train,test = reader.get()
class GP(object):

    def __init__(self,features,target):

        self.features = features.values.tolist()

        self.target = target.values.tolist()

        

    def fit(self):

        

        pset = gp.PrimitiveSet("MAIN", 27)

        pset.addPrimitive(operator.add,2)

        pset.addPrimitive(operator.sub,2)

        pset.addPrimitive(operator.mul,2)

        pset.addPrimitive(self.div,2)

        pset.addPrimitive(math.cos,1)

        pset.addPrimitive(math.sin,1)

        pset.addPrimitive(math.tanh,1)

        pset.addPrimitive(math.floor,1)

        pset.addPrimitive(math.ceil,1)

        pset.addPrimitive(self.sqrt,1)

        pset.addPrimitive(self.abs,1)

        

        

        def evaluation(individual):

        

            func = toolbox.compile(expr=individual)

            result =sum(round(1.-(1./(1.+np.exp(-func(*in_))))) == out for in_, out in zip(self.features,self.target))/len(self.features)

            return result,

        

        

        creator.create('FitnessMin', base.Fitness, weights=(1.0,))

        creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin)

        

        toolbox = base.Toolbox()

        toolbox.register('expr', gp.genHalfAndHalf,pset=pset, min_=1,max_=3)

        toolbox.register('individual', tools.initIterate, creator.Individual,toolbox.expr)

        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('compile', gp.compile, pset=pset)

        

        toolbox.register('evaluate', evaluation)

        toolbox.register('select', tools.selTournament, tournsize=3)

        toolbox.register('mate', gp.cxOnePoint)

        toolbox.register('expr_mut', gp.genFull, min_=1,max_=3)

        toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut,pset=pset)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=80))

        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=80))

        

        pop = toolbox.population(n=200)

        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)

        stats_size = tools.Statistics(len)

        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        mstats.register("max", numpy.max)

        pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.5, 30, stats=mstats,

                                   halloffame=hof, verbose=True)

        

        print('Expression')

        print(hof[0])

        

        return toolbox,hof[0]

    

    def div(self,left,right):

        try : 

            return left/right

            

        except ZeroDivisionError:

            return left/1e-4

        

    def sqrt(self,inp):

        

        return math.sqrt(abs(inp))

    

    def log1p(self,inp):

        

        return np.log1p(abs(inp))

    

    def abs(self,inp):

        

        return abs(inp)
GPro = GP(train.iloc[:,1:],train.iloc[:,0])
prediction  = []

for i in range(5):

    toolbox,hof = GPro.fit()

    func = toolbox.compile(expr=hof)

    prediction.append([np.round(1.-(1./(1.+np.exp(-func(*x))))) for x in test.values.tolist()])
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission['Survived'] = np.round(np.mean(np.vstack(prediction),axis=0))

submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('submission.csv', index=False)