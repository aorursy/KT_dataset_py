import numpy as np

from sklearn.metrics import roc_auc_score



class Sim:

    def __init__(self, length, seed, name):

        self.name = name

        self.LENGTH = length

        self.PRIV_PUB_CUT = int(self.LENGTH * 0.3)

        np.random.seed(seed)

        self.PERFECT_SUB = np.random.rand(self.LENGTH)

        #Assumption

        #Imbalance of positive and negative classes in the test set is the same as in the training set

        #see https://www.kaggle.com/vpaslay/is-your-small-gini-significant

        self.TARGET = (self.PERFECT_SUB > 0.963552).astype(dtype=int)



    def gini(self,y_target, y_score):

        return 2 * roc_auc_score(y_target, y_score) - 1



    def gini_private(self,y_score):

        return self.gini(self.TARGET[self.PRIV_PUB_CUT:], y_score[self.PRIV_PUB_CUT:])



    def gini_public(self, y_score):

        return self.gini(self.TARGET[:self.PRIV_PUB_CUT], y_score[:self.PRIV_PUB_CUT])



    def evaluate_sub(self,sub):

        return self.gini (self.TARGET, sub ), self.gini_public ( sub ), self.gini_private ( sub )





    def evaluate_subs (self, subs):

        samples= subs.shape[1] 

        results = np.zeros((samples,3))



        for i in range ( samples ):

            sub = subs[:,i]

            results[ i, : ] = np.array( self.evaluate_sub(sub) )     



        return results



    def create_random_sub (self, naive_target):



        random_sub =  np.random.rand(self.LENGTH) 

        _t = ( np.random.rand(self.LENGTH) >  naive_target ).astype(dtype=int)





        return self.PERFECT_SUB + _t*(random_sub-self.PERFECT_SUB)





    def create_semi_random_subs (self, naive_target, noise=0.02, samples=5):

        #the naive assumption

        _t = ( np.random.rand(self.LENGTH) >  naive_target   ).astype(dtype=int)



        subs = np.zeros((self.LENGTH,samples))



        for i in range (samples):



            _n = np.maximum(_t,( np.random.rand(self.LENGTH) > 1.0 - noise ).astype(dtype=int))



            random_sub =  np.random.rand(self.LENGTH) 





            random_sub = self.PERFECT_SUB + _n*(random_sub - self.PERFECT_SUB)



            subs [:, i] =  random_sub



        return subs



    

TESTSET_LENGTH = 595212

    

sim_testset = Sim(TESTSET_LENGTH, seed=2017, name= "Testset")

sim_half_testset = Sim(int(TESTSET_LENGTH/2), seed=2017, name = "Half a testset")

sim_doubled_testset = Sim(2*TESTSET_LENGTH, seed=2017, name="Doubled testset")



simulations = [ sim_half_testset,sim_testset, sim_doubled_testset]



for sim in simulations:

    

    print(sim.name)

    print("\tgini for perfect score: {:f}".format(sim.gini ( sim.TARGET, sim.PERFECT_SUB)) )



    m = sim.evaluate_sub (sim.create_random_sub(0.28))

    print("\trandom sub")

    print("\tgini : {:f} {:f} {:f}".format ( m[0], m[1], m[2] ))



    print("\t10 semi random subs")



    subs = sim.create_semi_random_subs (0.284,  noise=0.05, samples=10)

    avg_subs = np.mean(subs,axis=1)



    m = sim.evaluate_sub (avg_subs)

    print("\tavg gini: {:f} {:f} {:f}".format ( m[0], m[1], m[2] ))

    mean=np.mean(sim.evaluate_subs(subs),axis=1)

    print ("\t    gini: {:f} {:f} {:f}".format( mean[0], mean[1], mean[2]))

import matplotlib.pyplot as plt

%matplotlib inline  





l=100

naive_target=0.28

noise=0.01

samples=5





for sim in simulations:

    



    avg_sub_res = np.zeros((l,3))

    single_sub_res = np.zeros((l*samples,3))

    avg_subs = np.zeros((sim.LENGTH,l))



    for i in range(l):

        semi_random_subs = sim.create_semi_random_subs (naive_target, noise=noise, samples=samples)



        for j in range(samples):

            single_sub_res[5*i+j,:] = sim.evaluate_sub (semi_random_subs[:,j])



        avg_subs [:, i] = np.mean(semi_random_subs,axis=1)

        avg_sub_res [i,:] = sim.evaluate_sub (avg_subs[:,i])





    plt.figure(figsize=(5,5))

    plt.title(sim.name)





    plt.scatter(single_sub_res[:,1], single_sub_res[:,2], marker='o', color='r',alpha=0.7,label='single sub')

    plt.scatter(avg_sub_res[:,1], avg_sub_res[:,2], marker='x', color='b',alpha=0.7,label='5 subs avg')





    plt.ylabel('Private LB')

    plt.xlabel('Public LB')

    plt.legend(loc='lower right')

    plt.show()



    print("single sub mean")

    print (np.mean(single_sub_res))



    print("5 subs mean")

    print (np.mean(avg_sub_res))
