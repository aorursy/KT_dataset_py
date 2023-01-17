import numpy as np



#the true classes of our dataset

y_true = np.random.binomial(n= 1, p= 0.05, size = 5000)

#the predicted probabilities

y_pred = np.random.random(size = 5000)

#Define the heaviside function

heaviside = np.vectorize(lambda x : 0 if x<0 else .5 if x == 0 else 1)



#the loop implementation that matches with the sum definition

def Loop_AUC(y_true, y_pred):

    #the predictions of our classifier for all the positive labeled data

    pos_pred = y_pred[ y_true == 1]

    P = len(pos_pred) #the number of the positive population

    

    #the predictions of our classifier for all the negative labeled data

    neg_pred = y_pred[ y_true == 0]

    N = len(neg_pred) #the number of the negative population

    

    AUC = 0

    for pos in pos_pred :

        for neg in neg_pred :

            AUC +=  heaviside(pos - neg)

    

    return AUC/(P*N)



print('The AUC of a random Classifier is :', Loop_AUC(y_true, y_pred))
#this is a more optimized approach that uses the broadcasting method of numpy that is much faster than

#the loop version



def Broadcasted_AUC(y_true, y_pred):

    #the predictions of our classifier for all the positive labeled data

    pos_pred = y_pred[ y_true == 1]

    

    #the predictions of our classifier for all the negative labeled data

    neg_pred = y_pred[ y_true == 0]

    #creates a matrix that have pairwise difference between all the observations

    pairwise_matrix = pos_pred[:, np.newaxis] - neg_pred

    transform = heaviside(pairwise_matrix)

    

    return transform.mean()



print('The AUC of a Random Classifier is :', Broadcasted_AUC(y_true, y_pred))
import matplotlib.pyplot as plt 

import seaborn as sns

sns.set()



X = np.linspace(-10,10,num= 100)

Y = heaviside(X)



plt.title('the Heaviside Function')

plt.plot(X,Y)

plt.show()
def param_sigmoid(x,alpha):

    return 1/(1+ np.exp(-alpha*x))



fig ,ax = plt.subplots(ncols=3, sharey = True, figsize = (12,7))



weak, normal_sigmoid, extreme_sigmoid = 0.1, 1, 10



ax[0].plot(X,param_sigmoid(X,weak))

ax[0].set_title('weak sigmoid')



ax[1].plot(X,param_sigmoid(X,normal_sigmoid))

ax[1].set_title('normal sigmoid')



ax[2].plot(X,param_sigmoid(X,extreme_sigmoid))

ax[2].set_title('extreme sigmoid')



plt.show()
def Rank_Statistic(y_true, y_pred):

    #the predictions of our classifier for all the positive labeled data

    pos_pred = y_pred[ y_true == 1]

    

    #the predictions of our classifier for all the negative labeled data

    neg_pred = y_pred[ y_true == 0]

    #creates a matrix that have pairwise difference between all the observations

    pairwise_matrix = pos_pred[:, np.newaxis] - neg_pred

    transform = param_sigmoid(pairwise_matrix, 10)

    

    return transform.mean()



AUC = Broadcasted_AUC(y_true, y_pred)

Rank_stat = Rank_Statistic(y_true, y_pred)



print('The Rank Statistic of Random Classifier is :', Rank_stat)

print("The difference between the AUC and it's differentiable estimation is :", np.abs(AUC-Rank_stat))