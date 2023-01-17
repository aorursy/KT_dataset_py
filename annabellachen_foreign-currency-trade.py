# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras import optimizers

import os

import matplotlib.pyplot as plt

import copy

import keras

from keras.models import Sequential

from keras.layers import Dense,Activation

from keras.wrappers.scikit_learn import KerasRegressor

import itertools

from keras.models import Sequential

from keras.layers.recurrent import LSTM

from keras.layers.core import Dense, Activation, Dropout

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.utils import shuffle

import numpy.lib.recfunctions as rf

from math import sqrt

from numpy import concatenate

from matplotlib import pyplot

from pandas import read_csv

from pandas import DataFrame

from pandas import concat

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.impute import SimpleImputer

# Any results you write to the current directory are saved as output.
# get data

#train  = pd.read_csv('../input/financialdata-mytraindata/train_data.csv')

#test = pd.read_csv('../input/financialdata-mytestdata/test_data.csv')

#data = pd.read_csv('../input/latestfin/FinancialData.csv')

#data = pd.read_csv('../input/sourcedata/FinancialData.csv')

data = pd.read_csv('../input/finaldata/FinancialData_fullcsv.csv')
data.info()

# missing_val_count_by_column = (data.isnull().sum())

# print(missing_val_count_by_column[missing_val_count_by_column > 0])
data.drop(data.columns[[0,2,3,4,5,7]], axis=1, inplace=True)

data.head()
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

#We use the train dataframe from Titanic dataset

#fancy impute removes column names.



data = KNN(k=4).fit_transform(data)
#data = data.values.astype("float32")

# We apply the MinMax scaler from sklearn

# to normalize data in the (0, 1) interval.

scaler = MinMaxScaler(feature_range = (0, 1)).fit(data)

dataset = scaler.fit_transform(data)
#FUNCTION TO CREATE FORMAT DATASET WITH LAG DATA

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if type(data) is list else data.shape[1]

    df = DataFrame(data)

    cols, names = list(), list()

    # input sequence (t-n, ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together

    agg = concat(cols, axis=1)

    agg.columns = names

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
# n_hours=4#prediction based on previous 4 time period

# n_features=10

# reframed = series_to_supervised(dataset,n_hours,1)

# reframed.head()
n_hours=208#prediction based on previous 4 time period

n_features=10

reframed = series_to_supervised(dataset,n_hours,104)

reframed.head()
#if not all features are to be predicted

#removed_columns=[i for i in range(n_hours*n_features,n_hours*n_features+5)]

# removed_columns.extend([i for i in range(n_hours*n_features+8,n_hours*n_features+10)])

#removed_columns
#drop columns doesnt required to be predicted

#reframed.drop(reframed.columns[removed_columns], axis=1, inplace=True)

reframed.shape
# split into train and test sets

values = reframed.values

#n_train_hours = 498

n_train_hours = 250

train = values[:n_train_hours, :]

test = values[n_train_hours:, :]

# split into input and outputs

n_obs = n_hours * n_features

train_X, train_y = train[:, :n_obs], train[:, n_obs:]

test_X, test_y = test[:, :n_obs], test[:, n_obs:]

print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]

train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))

test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# design network

model = Sequential()

model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]),dropout=0.15))

model.add(Dense(1040))

#model.compile(loss='mae', optimizer='adam')

optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

model.compile(loss='mae', optimizer='adam')

# fit network

history = model.fit(train_X, train_y, epochs=150, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history

pyplot.plot(history.history['loss'], label='training')

pyplot.plot(history.history['val_loss'], label='validation')

pyplot.legend()

pyplot.show()
# #DJI	STI	SGPRIME	UKPRIME	USPRIME	USINF	UKINF	SGINF	SG/D	P/D

# # make a prediction

# yhat = model.predict(test_X)

# test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

# # invert scaling for forecast

# #inv_yhat = concatenate((test_X[:, 0:n_features-5],yhat), axis=1)

# #inv_yhat = scaler.inverse_transform(inv_yhat)

# inv_yhat = scaler.inverse_transform(yhat)

# inv_yhat_uk_exchange = inv_yhat[:,n_features-1]

# inv_yhat_sg_exchange = inv_yhat[:,n_features-2]

# inv_yhat_USPRIME = inv_yhat[:,n_features-3]

# inv_yhat_UKPRIME = inv_yhat[:,n_features-4]

# inv_yhat_SGPRIME= inv_yhat[:,n_features-5]

# # invert scaling for actual

# test_y = test_y.reshape((len(test_y), 10))

# #inv_y = concatenate((test_X[:, 0:n_features-5],test_y), axis=1)

# #inv_y = scaler.inverse_transform(inv_y)

# inv_y = scaler.inverse_transform(test_y)

# inv_y_uk_exchange = inv_y[:,n_features-1]

# inv_y_sg_exchange = inv_y[:,n_features-2]

# inv_y_USPRIME = inv_y[:,n_features-3]

# inv_y_UKPRIME = inv_y[:,n_features-4]

# inv_y_SGPRIME = inv_y[:,n_features-5]

# # calculate RMSE

# rmse = sqrt(mean_squared_error(inv_y_uk_exchange, inv_yhat_uk_exchange))

# print('Test RMSE(uk_exchange): %.3f' % rmse)

# rmse = sqrt(mean_squared_error(inv_y_sg_exchange, inv_yhat_sg_exchange))

# print('Test RMSE(sg_exchange): %.3f' % rmse)

# rmse = sqrt(mean_squared_error(inv_y_USPRIME, inv_yhat_USPRIME))

# print('Test RMSE(USPRIME): %.3f' % rmse)

# rmse = sqrt(mean_squared_error(inv_y_UKPRIME, inv_yhat_UKPRIME))

# print('Test RMSE(UKPRIME): %.3f' % rmse)

# rmse = sqrt(mean_squared_error(inv_y_SGPRIME, inv_yhat_SGPRIME))

# print('Test RMSE(SGPRIME): %.3f' % rmse)
yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))

yhat=yhat.reshape((yhat.shape[0], 104,n_features))

inv_yhat =[scaler.inverse_transform(i) for i in yhat] 

inv_yhat = np.array(inv_yhat)

inv_yhat_uk_exchange = inv_yhat[:,:,n_features-1]

inv_yhat_sg_exchange = inv_yhat[:,:,n_features-2]

inv_yhat_USPRIME = inv_yhat[:,:,n_features-3]

inv_yhat_UKPRIME = inv_yhat[:,:,n_features-4]

inv_yhat_SGPRIME= inv_yhat[:,:,n_features-5]

# invert scaling for actual

test_y = test_y.reshape(test_y.shape[0], 104,n_features)

#inv_y = concatenate((test_X[:, 0:n_features-5],test_y), axis=1)

#inv_y = scaler.inverse_transform(inv_y)

inv_y =[scaler.inverse_transform(i) for i in test_y] 

inv_y = np.array(inv_y)

inv_y_uk_exchange = inv_y[:,:,n_features-1]

inv_y_sg_exchange = inv_y[:,:,n_features-2]

inv_y_USPRIME = inv_y[:,:,n_features-3]

inv_y_UKPRIME = inv_y[:,:,n_features-4]

inv_y_SGPRIME = inv_y[:,:,n_features-5]

# calculate RMSE

rmse = sqrt(mean_squared_error(inv_y_uk_exchange, inv_yhat_uk_exchange))

print('Test RMSE(uk_exchange): %.3f' % rmse)

rmse = sqrt(mean_squared_error(inv_y_sg_exchange, inv_yhat_sg_exchange))

print('Test RMSE(sg_exchange): %.3f' % rmse)

rmse = sqrt(mean_squared_error(inv_y_USPRIME, inv_yhat_USPRIME))

print('Test RMSE(USPRIME): %.3f' % rmse)

rmse = sqrt(mean_squared_error(inv_y_UKPRIME, inv_yhat_UKPRIME))

print('Test RMSE(UKPRIME): %.3f' % rmse)

rmse = sqrt(mean_squared_error(inv_y_SGPRIME, inv_yhat_SGPRIME))

print('Test RMSE(SGPRIME): %.3f' % rmse)
inv_yhat_uk_exchange_result = inv_yhat_uk_exchange[-1]

inv_yhat_sg_exchange_result = inv_yhat_sg_exchange[-1]

inv_yhat_USPRIME_result = inv_yhat_USPRIME[-1]

inv_yhat_UKPRIME_result = inv_yhat_UKPRIME[-1]

inv_yhat_SGPRIME_result= inv_yhat_SGPRIME[-1]

inv_y_uk_exchange_result = inv_y_uk_exchange[-1]

inv_y_sg_exchange_result = inv_y_sg_exchange[-1]

inv_y_USPRIME_result = inv_y_USPRIME[-1]

inv_y_UKPRIME_result = inv_y_UKPRIME[-1]

inv_y_SGPRIME_result = inv_y_SGPRIME[-1]
output = pd.DataFrame({'uk_exchange': inv_y_uk_exchange_result, 'uk_exchangep': inv_yhat_uk_exchange_result,

                       'sg_exchange': inv_y_sg_exchange_result, 'sg_exchangep': inv_yhat_sg_exchange_result,

                       'USPRIME': inv_y_USPRIME_result, 'USPRIMEp': inv_yhat_USPRIME_result,'UKPRIME': inv_y_UKPRIME_result, 

                       'UKPRIMEp': inv_yhat_UKPRIME_result,'SGPRIME': inv_y_SGPRIME_result, 'SGPRIMEp': inv_yhat_SGPRIME_result})
output.to_csv('output.csv')
import random

import copy

class Chromosome:

    def __init__(self, timeLength,initAmount,exchangeRate,primeRate):

        self.timeLength = timeLength

        self.initAmount = initAmount

        self.exchangeRate = exchangeRate

        self.primeRate = primeRate

        

    def generateChromosome(self):

        patternList=[np.random.choice(np.arange(0, 10), p=[0.1,0.1,0.1,0.1,0.1,0.1,0.25,0.05,0.05,0.05]) for i in range(0,self.timeLength)]

        #patternList=[random.randint(0, 9) for i in range(0,self.timeLength)]#generate patterns for each week

        amountList=[]

        for i in range(0,self.timeLength):

            amount=self.generateAmountByPattern(patternList[i])

            amountList.append(amount)

#             if(patternList[i]==0 or patternList[i]==2):

#                 percentage=round(random.uniform(0.001, 1.0), 3)

#                 amount=[0,0,percentage]

#             elif(patternList[i]==1 or patternList[i]==4):

#                 percentage=round(random.uniform(0.001, 1.0), 3)

#                 amount=[0,percentage,0]

#             elif(patternList[i]==3 or patternList[i]==5):

#                 percentage=round(random.uniform(0.001, 1.0), 3)

#                 amount=[percentage,0,0]

#             elif(patternList[i]==6):

#                 percentage=round(random.uniform(0.001, 1.0), 3)

#                 amount=[0,0,0]

#             elif(patternList[i]==7):

#                 percentage_1=round(random.uniform(0.001, 0.99), 3)

#                 percentage_2=round(random.uniform(0.001, 0.99), 3)

#                 amount=[0,percentage_1,percentage_2]

#             elif(patternList[i]==8):

#                 percentage_1=round(random.uniform(0.001, 0.99), 3)

#                 percentage_2=round(random.uniform(0.001, 0.99), 3)

#                 amount=[percentage_1,0,percentage_2]

#             elif(patternList[i]==9):

#                 percentage_1=round(random.uniform(0.001, 0.99), 3)

#                 percentage_2=round(random.uniform(0.001, 0.99), 3)

#                 amount=[percentage_1,percentage_2,0]

            

        self.chromosomeList=pd.DataFrame({'pattern': patternList, 'amount': amountList})

        return self.chromosomeList

    

    def generateAmountByPattern(self,pattern):

        amount=[0,0,0]

        if(pattern==0 or pattern==2):

            percentage=round(random.uniform(0.001, 0.990), 3)

            amount=[0,0,percentage]

        elif(pattern==1 or pattern==4):

            percentage=round(random.uniform(0.001, 0.990), 3)

            amount=[0,percentage,0]

        elif(pattern==3 or pattern==5):

            percentage=round(random.uniform(0.001, 0.990), 3)

            amount=[percentage,0,0]

        elif(pattern==6):

            percentage=round(random.uniform(0.001, 0.990), 3)

            amount=[0,0,0]

        elif(pattern==7):

            percentage_1=round(random.uniform(0.001, 0.990), 3)

            percentage_2=round(random.uniform(0.001, 0.990), 3)

            amount=[0,percentage_1,percentage_2]

        elif(pattern==8):

            percentage_1=round(random.uniform(0.001, 0.990), 3)

            percentage_2=round(random.uniform(0.001, 0.990), 3)

            amount=[percentage_1,0,percentage_2]

        elif(pattern==9):

            percentage_1=round(random.uniform(0.001, 0.990), 3)

            percentage_2=round(random.uniform(0.001, 0.990), 3)

            amount=[percentage_1,percentage_2,0]

        return amount

            

    

    def calculateUSAmount(self,initAmount):       

        amount_sg=initAmount[0]

        amount_uk=initAmount[1]

        amount_us=initAmount[2]

        exchangeRate_sg_us=self.exchangeRate[0]

        exchangeRate_uk_us=self.exchangeRate[1]

        return amount_sg*exchangeRate_sg_us[0]+amount_uk*exchangeRate_uk_us[1]+amount_us

    

    def calculateInterest(self,amount,primeRate):

        return [amount[0]+amount[0]*primeRate[0]/100*7/365,amount[1]+amount[1]*primeRate[1]/100*7/365,amount[2]+amount[2]*primeRate[2]/100*7/365]

    

    def calculateAmountChange(self,pattern,amount_change,amount,exchangeRate):

        # SG|UK|US= 0,1,-1(0) ; 0,-1,1(1) ; 1,0,-1(2) ; -1,0,1(3) ; 1,-1,0(4) ; -1,1,0(5) ; 0,0,0(6) ; 1,-1,-1(7); 

        # -1, 1 ,-1(8); -1, -1, 1(9)

        #exchangeRate[0]:sg,exchangeRate[1]:pd

        #amount[0]:sg,amount[1]:uk,amount[2]:us

        if(pattern==0):

            amount_transfer=amount[2]*amount_change[2]#us->uk

            amount[2]-=amount_transfer*1.01

            amount[1]+=amount_transfer/exchangeRate[1]

        elif(pattern==1):

            amount_transfer=amount[1]*amount_change[1]#uk->us

            amount[1]-=amount_transfer*1.01

            amount[2]+=amount_transfer*exchangeRate[1]

        elif(pattern==2):

            amount_transfer=amount[2]*amount_change[2]#us->sg

            amount[2]-=amount_transfer*1.01

            amount[0]+=amount_transfer/exchangeRate[0]

        elif(pattern==3):

            amount_transfer=amount[0]*amount_change[0]#sg->us

            amount[0]-=amount_transfer*1.01

            amount[2]+=amount_transfer*exchangeRate[0]

        elif(pattern==4):

            amount_transfer=amount[1]*amount_change[1]#uk->sg

            amount[1]-=amount_transfer*1.01

            amount[0]+=amount_transfer*exchangeRate[1]/exchangeRate[0]

        elif(pattern==5):

            amount_transfer=amount[0]*amount_change[0]#sg->uk

            amount[0]-=amount_transfer*1.01

            amount[1]+=amount_transfer*exchangeRate[0]/exchangeRate[1]

#         elif(pattern==6):

#             print('no change')

        elif(pattern==7):

            amount_transfer_1=amount[1]*amount_change[1]#uk,us->sg

            amount_transfer_2=amount[2]*amount_change[2]#uk,us->sg

            amount[1]-=amount_transfer_1*1.01

            amount[2]-=amount_transfer_2*1.01

            amount[0]+=amount_transfer_1*exchangeRate[1]/exchangeRate[0]+amount_transfer_2/exchangeRate[0]

        elif(pattern==8):

            amount_transfer_1=amount[0]*amount_change[0]#sg,us->uk

            amount_transfer_2=amount[2]*amount_change[2]#sg,us->uk

            amount[0]-=amount_transfer_1*1.01

            amount[2]-=amount_transfer_2*1.01

            amount[1]+=amount_transfer_1*exchangeRate[0]/exchangeRate[1]+amount_transfer_2/exchangeRate[1]

        elif(pattern==9):

            amount_transfer_1=amount[0]*amount_change[0]#sg,uk->us

            amount_transfer_2=amount[1]*amount_change[1]#sg,uk->us

            amount[0]-=amount_transfer_1*1.01

            amount[1]-=amount_transfer_2*1.01

            amount[2]+=amount_transfer_1*exchangeRate[0]+amount_transfer_2*exchangeRate[1]

        return amount

    

    def calculateServiceFee(self,amount):

        return amount*0.01

    

    def calculateFinalAmount(self):       

        amount=copy.deepcopy(self.initAmount)

        for i in range(0,self.timeLength):

            amount=self.calculateInterest(amount,self.primeRate[i])

            amount=self.calculateAmountChange(self.chromosomeList['pattern'][i],self.chromosomeList['amount'][i],amount,self.exchangeRate[i])

        return amount

    def calculateFinalAmount_interestOnly(self):       

        amount=copy.deepcopy(self.initAmount)

        for i in range(0,self.timeLength):

            amount=self.calculateInterest(amount,self.primeRate[i])

            #amount=self.calculateAmountChange(self.chromosomeList['pattern'][i],self.chromosomeList['amount'][i],amount,exchangeRate[i])

        return amount

    

    def getFitnessValue(self):

        baseline=self.calculateUSAmount(self.calculateFinalAmount_interestOnly())

        final=self.calculateUSAmount(self.calculateFinalAmount())

        if(baseline>final):

            self.fitnessValue=10*(final-baseline)

            return 10*(final-baseline)#negative value

        else:

            self.fitnessValue=final-baseline

            return final-baseline
class GA:

    def __init__(self,initAmount,exchangeRate,primeRate,timeLength, num_population,num_iteration,crossoverRate,mutateRate):

        self.initAmount = initAmount

        self.exchangeRate = exchangeRate

        self.primeRate = primeRate

        self.timeLength = timeLength

        self.num_population = num_population

        self.num_iteration = num_iteration

        self.crossoverRate = crossoverRate

        self.mutateRate = mutateRate

        self.generatePopulation()

        

    def generatePopulation(self):

        population=[]

        for i in range(0,self.num_population):

            #initialize chromosome

            chromosome=Chromosome(self.timeLength,self.initAmount,self.exchangeRate,self.primeRate)

            chromosome.generateChromosome()

            chromosome.getFitnessValue()

            population.append(chromosome)

        self.populationList=population

    

    def selectPopulation(self): 

        '''

        Tournament

        '''

        newCouples=[]

        for i in range(0,int(self.num_population/2)):

            parents=[None]*2

            for j in [0,1]:

                selection=random.sample(set([x for x in range(self.num_population)]), 5)

                fitness=[self.populationList[i].fitnessValue for i in selection]

                index=fitness.index(max(fitness))

                parents[j]=copy.deepcopy(self.populationList[selection[index]])

            newCouples.append(parents) 

        return newCouples

           

    def crossover(self,coupleList):

        rand=random.random()

        population_crossover=[]

        if(rand<self.crossoverRate):

            split_index=random.randint(1,self.timeLength-1)

            for item in coupleList:

                couple=item

                pop_1=couple[0]

                pop_2=couple[1]

                pop_1_child=copy.deepcopy(pop_1)

                pop_2_child=copy.deepcopy(pop_2)

#                 print('original')

#                 print(pop_1_child.chromosomeList)

#                 print(pop_2_child.chromosomeList)

                pop_1_child.chromosomeList.iloc[0:split_index,:]=pop_2.chromosomeList.iloc[0:split_index,:]

                pop_2_child.chromosomeList.iloc[0:split_index,:]=pop_1.chromosomeList.iloc[0:split_index,:]

#                 print('after')

#                 print(pop_1_child.chromosomeList)

#                 print(pop_2_child.chromosomeList)

                pop_1_child.getFitnessValue()

                pop_2_child.getFitnessValue()

                population_crossover.append(pop_1_child)

                population_crossover.append(pop_2_child)

            print('crossover')

            return population_crossover

        else:

            print('no crossover')

            for item in coupleList:

                population_crossover.append(item[0])

                population_crossover.append(item[1])

            return population_crossover

        

    def mutate(self,population_crossover):

        rand=random.random()

        if(rand<self.mutateRate):

            selection=random.sample(set( [x for x in range(0, self.num_population)]), int(self.num_population/20))

            if(rand<self.mutateRate/3):

                print('mutate-1')

                #change pattern+amount both

                for index in selection:

                    index_mutate=random.randint(0, self.timeLength-1)

                    print('notify!!!')

                    print(index)

                    print(index_mutate)

                    print(len(population_crossover))

                    pop_pattern=population_crossover[index].chromosomeList.iloc[index_mutate,0]

                    pop_amount=population_crossover[index].chromosomeList.iloc[index_mutate,1]

                    pattern_mutate= random.choice([i for i in range(0,10) if i!=pop_pattern])   

                    amount_mutate=population_crossover[index].generateAmountByPattern(pattern_mutate)

                    population_crossover[index].chromosomeList.iat[index_mutate,0]=pattern_mutate

                    population_crossover[index].chromosomeList.iat[index_mutate,1]=amount_mutate

                    population_crossover[index].getFitnessValue()

                    #print('after')

                    #print(population_crossover[index].fitnessValue)

            else:

                print('mutate-2')

                #change amount only

                for index in selection:

                    index_mutate=random.randint(0, self.timeLength-1)

                    print('notify!!!')

                    print(index)

                    print(index_mutate)

                    print(len(population_crossover))

                    pop_amount=population_crossover[index].chromosomeList.iloc[index_mutate,1]

                    amount_mutate=[]

                    for amount in pop_amount:

                        if(amount!=0):

                            value=round(random.uniform(0.001, 0.990), 3)

                            amount_mutate.append(value)

                        else:

                            value=0

                            amount_mutate.append(value)

                    #print('before')

                    #print(population_crossover[index].fitnessValue)

                    population_crossover[index].chromosomeList.iat[index_mutate,1]=amount_mutate

                    population_crossover[index].getFitnessValue()

                    #print('after')

                    #print(population_crossover[index].fitnessValue)

        return population_crossover

    

    def finalSelection(self,population_mutate):

        parent_sorter=sorted(self.populationList,key=lambda x:x.fitnessValue,reverse=True)

        children_sorter=sorted(population_mutate,key=lambda x:x.fitnessValue,reverse=True)

        p1=parent_sorter[0:int(0.4*self.num_population)]

        print([i.fitnessValue for i in parent_sorter])

        c1=children_sorter[0:int(0.6*self.num_population)]

        print([i.fitnessValue for i in children_sorter])

        p1.extend(c1)

        result=sorted(p1,key=lambda x:x.fitnessValue,reverse=True)

        return result

    

    def process(self):

        for i in range(self.num_iteration):

            print('-----------------------'+str(i+1)+' round--------------------------\n\n')

            selectedPopulation=self.selectPopulation()

            print('-----------------------selected couples')

            print([i[0].fitnessValue for i in selectedPopulation])

            print([i[1].fitnessValue for i in selectedPopulation])

            population_crossover=self.crossover(selectedPopulation)

            print('-----------------------after crossover')

            print([i.fitnessValue for i in population_crossover])

            population_mutate=self.mutate(population_crossover)

            print('-----------------------after mutate')

            print([i.fitnessValue for i in population_mutate])

            population_final_selection=self.finalSelection(population_mutate)

            print('-----------------------after final selection')

            print([i.fitnessValue for i in population_final_selection])

            self.populationList=population_final_selection
# inv_yhat_uk_exchange

# inv_yhat_sg_exchange

# inv_yhat_USPRIME

# inv_yhat_UKPRIME

# inv_yhat_SGPRIME

initAmount=[10000,10000,10000]

exchangeRate=list(zip(inv_yhat_sg_exchange_result,inv_yhat_uk_exchange_result))

primeRate=list(zip(inv_yhat_SGPRIME_result,inv_yhat_UKPRIME_result,inv_yhat_USPRIME_result))
ga=GA(initAmount,exchangeRate,primeRate,104, 150,150,0.8,0.6)

ga.process()
# population_selected=ga.selectPopulation()

# result=ga.crossover(population_selected)

# result_mutate=ga.mutate(result)

# result_selection=ga.finalSelection(result_mutate)
ga.populationList[0].chromosomeList
output = ga.populationList[0].chromosomeList

output.to_csv('plan.csv')