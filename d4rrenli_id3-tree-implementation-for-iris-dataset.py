#ID3 Decsion Tree implmentation

import numpy as np

import pandas as pd

import random

random.seed(456641)

#We're using the Iris data set

import re

from sklearn.datasets import load_iris





iris = load_iris()





#load in data

data = pd.DataFrame(

    data=iris['data'],

    columns = iris['feature_names']

)

data.head()
#load in target column

target= iris['target']

target_names = iris['target_names']

mapper = dict( (i,target_names[i]) for i in range(3))

#mapp values in target to named classes

target_series = pd.Series([(lambda x:mapper[x])(i) for i in target])

data['species'] = target_series

data.head()
#check for nulls and nan

data.isnull().values.any()
#split so 80-20 train-test

TEST_AMOUNT= 0.2

DATA_LENGTH = len(data)

#sample random indices to split



test_indicies = random.sample(range(DATA_LENGTH),int(TEST_AMOUNT * DATA_LENGTH))

test_data = data.loc[test_indicies]

test_data.head()
#set training data

train_data = pd.concat([data,test_data]).drop_duplicates(keep=False)

train_data.head()
#calculate entropy of samples

def entropy(samples):

    if len(samples) < 2:

        return 0

    

    freq = np.array(samples.value_counts(normalize=True))

    return -(freq * np.log2(freq + 1e-6)).sum()



#calculate information gain

def information_gain (samples,target, attribute):

    values = samples[attribute].value_counts(normalize=True)

    split_entropy = 0

    for v,fr in values.iteritems():

        

        sub_ent = entropy(samples[samples[attribute] == v][target])

        split_entropy += fr * sub_ent

    ent = entropy(samples[target])



    return ent - split_entropy



#given a sorted array, return an array that contains the average of each adjacent number e.g. [1,2,3] -> [1.5,2.5]

def averaged_array(array):

    length = len(array) -1

    mean = lambda x,y: (x+y)/2

    return[mean(array[i],array[i+1]) for i in range(length)]
#ONLY WORKS FOR NUMERICAL DATA ONLY

class tree_node:



    def __init__(self,samples,target):

        self.decsion = None

        self.samples = samples

        self.target = target

        self.split_attribute = None

        self.children = None

        self.parent = None



    def make(self):

        target = self.target

        samples = self.samples

        #if no samples, something has gone very wrong

        if len(samples)  == 0:

            print("?")

            self.decsion = "ya done goffed"

        #if samples are of one kind, make decsion to be of teh sample kind

        elif len(samples[target].unique()) == 1:

            self.decsion = samples[target].unique()[0]

            print(self.decsion)

            return

        #if samples are mixed, then...

        else:



            best_info_gain = 0

            revised_samples = pd.DataFrame()



            #create a data frame that comapres teh values of each column of irsi in boolean form

            # this is teh part that makes things slow

            for attribute in samples.keys():

                if attribute == target:

                    continue  

                #find all unqiue values of teh attribute

                unique_values = samples[attribute].sort_values().unique()

                #avagere array

                unique_values = averaged_array(unique_values)

                for divider in unique_values:

                #check if value is greater than divider value and make nwew columnregarding this

                    name = attribute +" > " + str(divider)

                    revised_samples[name] = samples[attribute] >  divider



            revised_samples[target] = samples[target]



            # then from this revised df just pick out columns with bets information gain

            for attribute in revised_samples.keys():

                if attribute == target:

                    continue               

                

                info_gain = information_gain(revised_samples,target,attribute)

                #print("Information Gain at {}: {:.2f}".format(attribute,info_gain))

                if info_gain > best_info_gain:

                    best_info_gain = info_gain

                    self.split_attribute = attribute

            #print("Spliting by: {} Info Gain: {:.2f}".format(self.split_attribute,best_info_gain))



            self.children = {}

            #then set value and make the child nodes

            for value in revised_samples[self.split_attribute].unique():

                index = revised_samples[self.split_attribute] == value

                

                print((self.split_attribute,value))

                self.children[value] = tree_node(samples[index],target)

                self.children[value].make()

            

    #basic print desion logic to get to a result

    def print(self,prefix=''):

        if self.split_attribute is not None:

            for k,v in self.children.items():

                v.print("{} If {} is ({}) and".format(prefix,self.split_attribute, k))

        else:

            final = re.sub("and$","",prefix)

            print("{}, Then flower is: {}".format(final,self.decsion))





    def predict(self,sample):

        if self.decsion is not None:

            return self.decsion

        else:

            #split spit_attirbute to actual cloumn name and value

            split_attribute = self.split_attribute

            column,value = re.split(" > ",split_attribute)



            value = float(value)

            

            #check value comap[rd to split value , then set child to crrect one

            child = self.children[float(sample[column]) > value]

            return child.predict(sample)





class ID3_tree:

    def __init__(self):

        self.root = None

    def fit (self,samples,target):

        self.root = tree_node(samples, target)

        self.root.make()



t = ID3_tree()

t.fit(train_data,'species')

#basic decsion logic

t.root.print('')
#find number of correctly classfied samples

correct = int(TEST_AMOUNT * DATA_LENGTH)



for i,row in test_data.iterrows():

    pred = t.root.predict(row)

    actual = row['species']

    if actual != pred:

        correct += -1

        print(row[0:4])

        print("Predicted: " + pred + " | Actual: " + actual)



print( str(correct) + " / " + str(int(TEST_AMOUNT * DATA_LENGTH)) + " correct (" + str(100 * correct /float(TEST_AMOUNT * DATA_LENGTH)) + "%)")
