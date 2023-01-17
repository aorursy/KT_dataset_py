import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer#nltk means natural language toolkit
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#importing dataset
columns=['category','text']
data=pd.read_csv("../input/smsspam/SMSSpamCollection.txt",names=columns,sep='[\t]',header=None,engine='python')
print(data)
print(data.shape)
data['bool_cat']=data['category'].map({'ham':0,'spam':1})
print(data)
data['Count']=0
#counting the length of each sms
for i in np.arange(0,len(data['text'])):
    data.loc[i,'Count']=len(data.loc[i,'text'])
data
#storing spam and sorting based on index
spam=data[data['bool_cat']==1]
print(spam,spam.shape)
spam_count=pd.DataFrame(pd.value_counts(spam['Count']).sort_index())
spam_count
#storing ham and sorting based on index
ham=data[data['bool_cat']==0]
print(ham,ham.shape)
ham_count=pd.DataFrame(pd.value_counts(ham['Count']).sort_index())
ham_count
#plotting the spam and ham on subplot to get idea about the word count of spam messages
fig, ax = plt.subplots(figsize=(60,5))
spam_count['Count'].sort_index().plot(ax=ax, kind='bar',facecolor='red');
ham_count['Count'].sort_index().plot(ax=ax, kind='bar',facecolor='green');
#lematizing the sms after removing stopwords from text and tokenizing
wordnet_lemmatizer=WordNetLemmatizer()#wordnet lemmatizer has been used
#function to lemmatize
def lemmatize_txt(text):
    text=text.replace('\d+(\.\d+)?', 'numbr')
    token_words=word_tokenize(text)#tokenizing the text
    lemmatized_text=[]
    for word in token_words:
        word=str.lower(word) #converting all text to lowercase
        if not word in stopwords.words():  #removingg stopwords
            lemmatized_text.append(wordnet_lemmatizer.lemmatize(word,pos='v'))
            lemmatized_text.append(" ")
    return "".join(lemmatized_text) #again join lemmatized tokens to form sentences
data=data.head(500)
for i in data.index:
    text=data.loc[i,'text']
    data.loc[i,'lemmatized_text']=lemmatize_txt(text) #passing the sms text for lemmatization
data
#vectorizing the words in sms text for feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#tf idf vectorizer function is used here
Vectorizer=TfidfVectorizer() 
X_sparse=Vectorizer.fit_transform(data.lemmatized_text)
#print(X)
#X=X.toarray();
Y=data.bool_cat
#print(Y)
X_sparse.shape
X_sparse
print(X_sparse.toarray())
"""
#------------------------------------------------------------------------------+
#
#   Samuel C P Oliveira
#   samuelcpoliveira@gmail.com
#   Artificial Bee Colony Optimization
#   This project is licensed under the MIT License.
#   v1.0 (April 2020)
#
#------------------------------------------------------------------------------+
#
# Bibliography
#
# [1] Karaboga, D. and Basturk, B., 2007
#     A powerful and efficient algorithm for numerical function optimization:
#     artificial bee colony (ABC) algorithm. Journal of global optimization, 39(3), pp.459-471.
#     doi: https://doi.org/10.1007/s10898-007-9149-x
# 
# [2] Liu, T., Zhang, L. and Zhang, J., 2013
#     Study of binary artificial bee colony algorithm based on particle swarm optimization.
#     Journal of Computational Information Systems, 9(16), pp.6459-6466.
#     link: https://api.semanticscholar.org/CorpusID:8789571
#
# [3] Anuar, S., Selamat, A. and Sallehuddin, R., 2016
#     A modified scout bee for artificial bee colony algorithm and its performance on optimization
#     problems. Journal of King Saud University-Computer and Information Sciences, 28(4), pp.395-406.
#     doi: https://doi.org/10.1016/j.jksuci.2016.03.001
#
#------------------------------------------------------------------------------+
"""

# %%
import numpy as np
import numpy.random as rng

class abc:
    """
    Class that applies Artificial Bee Colony (ABC) algorithm to find minimun
    or maximun of a function thats receive a vector of floats as input and
    returns a float as output.
    Parameters
    ----------
    function : Name
        A name of a function to minimize/maximize.
        Example: if the function is:
            def my_func(x): return x[0]**2 + x[1]**2 + 5*x[1]
            Put "my_func" as parameter.
    boundaries : List of Tuples
        A list of tuples containing the lower and upper boundaries of each
        dimension of function domain.
        Obs.: The number of boundaries determines the dimension of function.
        Example: [(-5,5), (-20,20)]
    [colony_size] : Int --optional-- (default: 40)
        A value that determines the number of bees in algorithm. Half of this
        amount determines the number of points analyzed (food sources).
        According articles, half of this number determines the amount of
        Employed bees and other half is Onlooker bees.
    [scouts] : Float --optional-- (default: 0)
        Determines the limit of tries for scout bee discard a food source and
        replace for a new one.
        If scouts = 0 : Scout_limit = colony_size * dimension
        If scouts = (0 to 1) : Scout_limit = colony_size * dimension * scouts
            Obs.: scout = 0.5 is used in [3] as benchmark.
        If scout = (1 to iterations) : Scout_limit = scout
        If scout >= iterations: Scout event never occurs
        Obs.: Scout_limit is rounded down in all cases.
    [iterations] : Int --optional-- (default: 50)
        The number of iterations executed by algorithm.
    [min_max] : String --optional-- (default: 'min')
        Determines if algorithm will minimize or maximize the function.
        If min_max = 'min' : Try to localize the minimum of function.
        If min_max = 'max' : Try to localize the maximum of function.
    Methods - Useful to user
    ----------
    fit()
        Execute the algorithm with defined parameters.
        Obs.: Returns a list with values found as minimum/maximum coordinate.
    get_solution()
        Returns the value obtained after fit() the method.
        Obs.: If fit() is not executed, return the position of
              best initial condition.
    get_status()
        Returns a tuple with:
            - Number of iterations executed
            - Number of scout events during iterations
    get_agents()
        Returns a list with the position of each food source during
        each iteration.
    
    Methods - Useful to developers
    ----------
    calculate_fit(evaluated_position = Position of evaluated point)
        Calculate the cost function (value of function in point) and
        returns the "fit" value associated (float) (according [2]).
    food_source_dance(index = index (int) of food source target)
        Randomizes a "neighbor" point to evaluate the <index> food source.
    """

    def __init__(self, function, boundaries, colony_size=40, scouts=0, iterations=50, min_max='min'):
        self.boundaries = boundaries
        self.min_max_selector = min_max
        self.cost_function = function
        self.max_iterations = int(iterations)
        
        self.employed_onlookers_count = int(colony_size/2)
        
        if (scouts == 0):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries))
        elif (scouts < 1):
            self.scout_limit = int(self.employed_onlookers_count * len(self.boundaries) * scouts)
        else:
            self.scout_limit = int(scouts)
        
        self.foods = []
        for i in range(self.employed_onlookers_count):
            self.foods.append(_FoodSource(self))
            while np.isnan(self.foods[i].fit):
                self.foods[i] = _FoodSource(self)
        
        self.best_food_source = self.foods[np.argmax([food.fit for food in self.foods])]
        self.agents = []
        self.agents.append([food.position for food in self.foods])
        
        self.scout_status = 0
        self.iteration_status = 0
        

    def calculate_fit(self,evaluated_position):
        #eq. (2) [2] (Convert "cost function" to "fit function")
        cost = self.cost_function(evaluated_position)
        if (self.min_max_selector == 'min'): #Minimize function
            fit_value = (1 + np.abs(cost)) if (cost < 0) else (1/(1 + cost))
        else: #Maximize function
            fit_value = (1 + cost) if (cost > 0) else (1/(1 + np.abs(cost)))
        return fit_value
    
    def food_source_dance(self,index):
        #Generate a partner food source to generate a neighbor point to evaluate
        while True: #Criterion from [1] geting another food source at random
            d = int(rng.randint(0,self.employed_onlookers_count))
            if (d != index):
                break
        self.foods[index].evaluate_neighbor(self.foods[d].position)
    
    def fit(self):
        for iteration in range(1,self.max_iterations+1):
            #--> Employer bee phase <--
            #Generate and evaluate a neighbor point to every food source
            [self.food_source_dance(i) for i in range(self.employed_onlookers_count)]
            max_fit = np.max([food.fit for food in self.foods])
            onlooker_probability = [0.9*(food.fit/max_fit)+0.1 for food in self.foods]
    
            #--> Onlooker bee phase <--
            #Based in probability, generate a neighbor point and evaluate again some food sources
            #Same food source can be evaluated multiple times
            p = 0 #Onlooker bee index
            i = 0 #Food source index
            while (p < self.employed_onlookers_count):
                if (rng.uniform(0,1) < onlooker_probability[i]):
                    p += 1
                    self.food_source_dance(i)
                    max_fit = np.max([food.fit for food in self.foods])
                    onlooker_probability = [0.9*(food.fit/max_fit)+0.1 for food in self.foods]
                i = (i+1) if (i < (self.employed_onlookers_count-1)) else 0

            #--> Memorize best solution <--
            iteration_best_food_index = np.argmax([food.fit for food in self.foods])
            self.best_food_source = self.best_food_source if (self.best_food_source.fit >= self.foods[iteration_best_food_index].fit) \
                else self.foods[iteration_best_food_index]
            
            #--> Scout bee phase <--
            #Generate up to one new food source that does not improve over scout_limit evaluation tries
            trial_counters = [food.trial_counter for food in self.foods]
            if (max(trial_counters)>self.scout_limit):
                #Take the index of replaced food source
                trial_counters = np.where(np.array(trial_counters) == max(trial_counters))[0].tolist()
                i = trial_counters[rng.randint(0,len(trial_counters))]
                
                self.foods[i] = _FoodSource(self) #Replace food source
                self.scout_status += 1
            
            self.agents.append([food.position for food in self.foods])
            self.iteration_status = iteration
        return self.best_food_source.position

    def get_agents(self):
        return self.agents
    
    def get_solution(self):
        return self.best_food_source.position
    
    def get_status(self):
        return self.iteration_status, self.scout_status


class _FoodSource:
    """
    Class that represents a food source (evaluated point). Useful to developers.
    Parameters
    ----------
    abc : Class
        A main class with variables and methods thats correlate food sources.
    Methods
    ----------
    evaluate_neighbor(evaluated_position = Position of evaluated point)
        Using evaluated position (partner position), generate a neighbor point,
        evaluate the "fit" value of it and applies greedy selection on it.
        If the "fit" value of neighbor point is better, permute this position with
        original food source's position and set trial counter of this food source
        to 0.
        If original food source "fit" value is better, mantain the original position
        and increases trial counter in 1.
        Trial counter is used to trigger scout event during iterations.
    """

    def __init__(self,abc):
        #When a food source is initialized, randomize a position inside boundaries and
        #calculate the "fit"
        self.abc = abc
        self.trial_counter = 0
        self.position = [rng.uniform(*self.abc.boundaries[i]) for i in range(len(self.abc.boundaries))]
        self.fit = self.abc.calculate_fit(self.position)
        
    def evaluate_neighbor(self,partner_position):
        #Randomize one coodinate (one dimension) to generate a neighbor point
        j = rng.randint(0,len(self.abc.boundaries))
        
        #eq. (2.2) [1] (new coordinate "x_j" to generate a neighbor point)
        xj_new = self.position[j] + rng.uniform(-1,1)*(self.position[j] - partner_position[j])

        #Check boundaries
        xj_new = self.abc.boundaries[j][0] if (xj_new < self.abc.boundaries[j][0]) else \
            self.abc.boundaries[j][1] if (xj_new > self.abc.boundaries[j][1]) else xj_new
        
        #Changes the coordinate "j" from food source to new "x_j" generating the neighbor point
        neighbor_position = [(self.position[i] if (i != j) else xj_new) for i in range(len(self.abc.boundaries))]
        neighbor_fit = self.abc.calculate_fit(neighbor_position)

        #Greedy selection
        if (neighbor_fit > self.fit):
            self.position = neighbor_position
            self.fit = neighbor_fit
            self.trial_counter = 0
        else:
            self.trial_counter += 1
#training using abc algorithm for Logistic Regression loss function
import math
X=X_sparse.toarray()
def LogRegLossFun(x):
    loss_fun=0
    h=0
    yhat=0
    for i in range(499):
        yhat=x[0]
        for j in range(0,1773):
            yhat+=X[i][j]*x[j+1]
        h=1.0/(1.0+math.exp(-yhat))
        loss_fun+=(-Y[i]*np.log(h))+(1-Y[i])*np.log(1-h)
    loss_fun/=500.0
    return loss_fun
limits=[]
for i in range(1774):
    limits.append((-1,1))
abc_obj = abc(LogRegLossFun, limits) #Load data
solution = abc_obj.fit() #Execute the algorithm

#If you want to get the obtained solution after execute the fit() method:
solution2 = abc_obj.get_solution()

#If you want to get the number of iterations executed and number of times that
#scout event occur:
iterations = abc_obj.get_status()[0]
scout = abc_obj.get_status()[1]

#If you want to get a list with position of all points (food sources) used in each iteration:
food_sources = abc_obj.get_agents()
print(solution)
def LogRegPredict(x_test):
    y_test=[]
    for i in range(499):
        yhat=solution[0]
        for j in range(1773):
            yhat+=X[i][j]*solution[j+1]
        h=1.0/(1.0+math.exp(-yhat))
        if h>0.5:
            y_test.append(1)
        else:
            y_test.append(0)
    return y_test
y=LogRegPredict(X)
y
#splitting the overall data into training setand test set based on a ratio randomly
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.15,train_size=.85,random_state=42)
print("Training set has {} samples".format(X_train.shape[0]))
print("Testing set has {} samples".format(X_test.shape[0]))
#building the model
from sklearn.linear_model import LogisticRegression
LogReg=LogisticRegression(solver='lbfgs')
#lbfgs is LR solver out of many other solvers but it works for multnomial option also so
#fitting the model
LogReg.fit(X_train,Y_train)
#predict dependent variables for training and test data
Y_predict=LogReg.predict(X_test)
Y_predict_train=LogReg.predict(X_train)
print(Y_predict)
print(Y_predict_train)
y_prob_train=LogReg.predict_proba(X_train)[:,1].reshape(1,-1)
print(y_prob_train)
y_prob=LogReg.predict_proba(X_test)[:,1]
y_prob
#results of performance
from sklearn.metrics import accuracy_score
score=accuracy_score(Y_test,Y_predict)#checking accuracy
print(score)
from sklearn.metrics import confusion_matrix
#confusion matrix
conf_matrix=confusion_matrix(Y_test,Y_predict)
print(conf_matrix)
tn,fp,fn,tp=conf_matrix[0][0],conf_matrix[0][1],conf_matrix[1][0],conf_matrix[1][1]
accuracy=(tn+tp)/(tn+tp+fp+fn)#tn true negative, fn false negative, tp true positive, fp false positive
print("accuracy {:0.2f}".format(accuracy))
sensitivity=tp/(tp+fn) 
#sensitivity is measure of probabability of correctly identifying spam
print("sensitivity {:0.2f}".format(sensitivity))
specificity=tn/(tn+fp)
#specificity is measure of probability of correctly identifying ham
print("specificity {:0.2f}".format(specificity))