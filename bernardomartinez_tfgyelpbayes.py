import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import random,math

import json,gc,os,re,time

from scipy import sparse

from sklearn.preprocessing import LabelEncoder

from tqdm import tnrange, tqdm_notebook,tqdm

from pandas.io.json import json_normalize

from joblib import Parallel, delayed

from scipy.misc import logsumexp

from collections import defaultdict

gc.collect() 



print(os.listdir("../input/"))
class EM_NaiveBayes():

    def __init__(self,seed = 0):

        self.seed = seed

        random.seed(seed)

        

        self.pxy = {}

        self.py = {}

        

    def class_probability(self,y,random_prob=False):

        """

        Obtains the probability of the class variable by iterating through all its instances.

    

        Keyword arguments:

            y (dictionary): the variable class for each instance

            random_prob (boolean): indicates whether we want to compute real or random probabilities (default False)

            seed (int): Establish a seed to generate random numbers (default 0)

        Returns:

            A dictionary containing the probability for each state of the class variable

        """

        self.py.clear()

    

        #Generation of random probabilities

        if random_prob:

            rnd_num = random.random()

        

            self.py[0] = rnd_num

            self.py[1] = 1 - rnd_num

        

        #Iteration through the instances and counting the probability for each state

        else:

            

            self.py = {classes:0 for classes in range(2)}

            

            for instance in y:

                

                for k,v in instance.items():

                    self.py[k]+=v

                        

            #Divide the sum of probabilities between the number of instances

            constant = float(len(y))

            

            self.py = {k: v/constant for k,v in self.py.items()}



    

    def probability_table(self,X,y,text_dic,kind_col):

    

        dic = defaultdict(lambda: defaultdict(float))

        cons_norm = {classes:0 for classes in range(2)}

                

        nonzero_row_ind = X.nonzero()[0]

        

        #Iterates through all the instances

        for i in nonzero_row_ind:

            

            if kind_col == 0:

                values = [X.A[i][0]]

            else:

                values = text_dic[X.A[i][0]]

    

            #Computing the frequency for the probability table

            for term in values:

                    

                #Iterates through the variable class

                for classes in range(2):

                    dic[term][classes]+= y[i][classes]

                    cons_norm[classes]+=y[i][classes]

                

        #Apply normalization constant to probability table

        

        for classes in range(2):

            

            if cons_norm[classes] != 0:

        

                for key,val in dic.items():

                    dic[key][classes]/=cons_norm[classes]

            

            else:

                equalProb = 1/len(dic)

                for key,val in dic.items():

                    dic[key][classes] = equalProb

        

        return dic

    

    def random_probability_table(self,X,y,text_dic,kind_col = 0):

    

        dic = defaultdict(lambda: defaultdict(float))

        cons_norm = {classes:0 for classes in range(2)}

    

        nonzero_row_ind = X.nonzero()[0]

        #Iterates through all the instances

        for i in nonzero_row_ind:

        

            if kind_col == 0:

            

                for star in range(1,6):      

                    for classes in range(2):

                        dic[star][classes] = random.random()

                        cons_norm[classes]+=dic[star][classes]

            else:

                values = text_dic[X.A[i][0]]

                

                #Computing the frequency for the probability table

                for term in values:

                    

                    #Iterates through the variable class

                    for classes in range(2):

                        rnd = random.random()

                        dic[term][classes]+= rnd

                        cons_norm[classes]+=rnd

                     

        #Apply normalization constant to probability table

        

        for classes in range(2):

            

            if cons_norm[classes] != 0:

        

                for key,val in dic.items():

                    dic[key][classes]/=cons_norm[classes]

            

            else:

                equalProb = 1/len(dic)

                for key,val in dic.items():

                    dic[key][classes] = equalProb

        

        return dic

        

    

    def cond_probability_all(self,X,y,text_dic,random_prob=False):

        """

        Obtains the probability distribution tables p(x|y) for each predictor variable.

    

        Keyword arguments:

            X (list): A CSC numpy sparse matrix storing the values for all predictor variables

            y (dictionary): the variable class for each instance

        Returns:

            A dictionary containing all probability tables

        """

        size = X.shape[1]

        self.pxy.clear()

        

        if random_prob:

            self.pxy = {col: self.random_probability_table(X[:,col],y,text_dic,kind_col = col%2) for col in tnrange(size)}

       

        else:

            self.pxy = {col: self.probability_table(X[:,col],y,text_dic,kind_col = col%2) for col in tnrange(size)}

            

    def predict_proba_example(self,x,text_dic):  

        """

        Infers the probability of an instances.

    

        Keyword arguments:

            x (list): instance to predict

            py (dictionary): table with the probability of the class variable

            pxy (dictionary): table containing all the conditional probabilities

        Returns:

            A dictionary the prediction of the instance

        """

        inference = {}

        

        nonzero_col_ind = x.nonzero()[1]

        iterations = len(nonzero_col_ind)

        

        if iterations == 0:

            return {n: 0 for n in py}



        for k in self.py: #Computing P(ci|x) for i being each different state of the variable class

            #inference[k] = math.log(self.py[k])

            inference[k] = self.py[k]

            

            #Iterate through the table of conditional probabilities for that instance

            for i in nonzero_col_ind:

                

                #Checking if we are dealing with number of stars (even index) or text review (odd index)

                if i % 2 == 0:

                    try:

                        inference[k]+=math.log(self.pxy[i][x.A[0][i]][k])

                    except:

                        inference[k] = -float("inf")

                        break

            

                else:

                    values = text_dic[x.A[0][i]]

                    try:

                        for term in values:

                            inference[k]+=math.log(self.pxy[i][term][k])

                    except:

                        inference[k] = -float("inf")

                        break

           

        

        A = max(inference.values())

       

        #Computing divisor

        divisor = A

        

        aux = 0

        for i in self.py:

            aux+= math.exp(inference[i]-A)

        

        divisor+= math.log(aux)

        

        return {n: math.exp(inference[n] - divisor) for n in self.py}

    

     

    def predict_proba(self,X,text_dic):

        size = X.shape[0]

        return [self.predict_proba_example(X[row,:],text_dic) for row in tnrange(size)]
class EM_Algorithm():

    

    def __init__(self,X,text_dic,seed = 0):

        self.seed = seed

        self.nb = EM_NaiveBayes(self.seed)

        self.repeated_results = []

        self.text_dic = text_dic

        self.X = X

        self.y = None

    """

    Initialises the algorithm by creating a random probability class table and random conditional probabilities tables

    """

    def fit(self):

    

        self.nb.class_probability(None,random_prob=True)

        print("Computed p(y): " + str(self.nb.py))

        print("Computing random conditional probability tables")

        

        self.nb.cond_probability_all(self.X,self.y,self.text_dic,random_prob=True)

        print("End random conditional probability tables")

        

    def release_memory(self):

        self.nb.pxy.clear()

        

    def predict(self,iters=1):

        

        print("Computing iteration 1")

        print("Predicting probabilities")

        self.y = self.nb.predict_proba(self.X,self.text_dic)

        

        for i in range(1,iters):

            print("Reestimating p(y)") 

            self.nb.class_probability(self.y)

            print("p(y):" + str(self.nb.py))

            

            print("Reestimating p(xy)")

            self.nb.cond_probability_all(self.X,self.y,self.text_dic,False)

            

            print("Computing iteration " + str(i+1))

            print("Predicting probabilities")

            self.y = self.nb.predict_proba(self.X,self.text_dic)

        

        #self.release_memory()

        return self.y



    

    def repeatedEM(self,repeats=1,iters_per_repeat=1):

        

        for i in range(repeats):

            print("repeated EM " + str(i))

            self.seed+=5

            self.nb = EM_NaiveBayes(self.seed)

            

            self.fit()

            aux_result = self.predict(iters_per_repeat)

            

            self.repeated_results.append(aux_result)

            aux_result = None

        

        return self.repeated_results

review = pd.read_csv('../input/text-processing-with-spacy/clean_data.csv')

#review = pd.read_csv('../input/smallbusiness/small.csv')
review.shape
gc.collect()
review[review.text.isna()]
review.dropna(inplace=True)
def prepareData(review):

    le1 = LabelEncoder()

    le2 = LabelEncoder()



    #Convert values in range 0-n

    review.user_id = le1.fit_transform(review.user_id)

    review.business_id  = le2.fit_transform(review.business_id)



    #Index starting from 1

    review.text.index+=1

    

    #Converting reviews into a dict

    text_dic = review.text.to_dict()

    

    

    row = review.business_id.append(review.business_id)



    col = review.user_id*2

    col = col.append(review.user_id*2 + 1)



    #Create sparse array

    arr = sparse.csc_matrix(

        (

        np.append(review["stars"],list(text_dic.keys())), (row, col)

        )

    )

    

    

    inference_index = list(le2.inverse_transform([x for x in range(arr.shape[0])]))

    text_dic = {k:v.split() for k,v in text_dic.items()}

    



    return arr,inference_index,text_dic
arr, inference_index, text_dic = prepareData(review)
em = EM_Algorithm(arr,text_dic,seed=20)
%%time

y = em.repeatedEM(repeats=5,iters_per_repeat=3)
y
gc.collect()
dat1 = pd.concat([pd.DataFrame(data=partial_result).apply(round) for partial_result in y], axis=1)

dat1.index = inference_index
dat1.to_csv("results.csv")