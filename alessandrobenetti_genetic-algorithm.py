'''Python genetic algorithm for grouping the correlated signals'''

#We start by loading all the modules we need for the wrapping

import pandas as pd

import numpy as np

import random

from tqdm import tqdm

from sklearn.decomposition import PCA

from sklearn import preprocessing

import matplotlib.pyplot as plt

from IPython import display
df_val = pd.read_csv('/kaggle/input/enginecleanednormalized/Nasa_val.txt',header=None)

df_train = pd.read_csv('/kaggle/input/enginecleanednormalized/Nasa_train.txt',header=None)

df_train.head()
scaler = preprocessing.StandardScaler()

Train = np.array(df_train)

Train = scaler.fit_transform(Train)

df_train = pd.DataFrame(Train) #The normalized train:
normalizing = pd.DataFrame({'Mean':scaler.mean_, 'Std':scaler.var_**0.5})

normalizing.head()
def generate_parent(length, groups, min_size = 4):

    #This function is used to generate an array of 0 and n, used as the genes for our genetic algorithm,

    randBinList = lambda n: [random.randint(0,groups-1) for b in range(1,n+1)]

    genes = np.array(randBinList(length))

    #Let's generate bounded parents:

    for group_id in range(groups):

        Genes_vect = (group_id == genes)

        #Recursive to respect the limit of at least 4 sensor each group

        if sum(Genes_vect) < min_size:

            genes = generate_parent(length, groups)



    return genes
def get_fitness_wrapper(genes, group_id, val, train, normalizing, n_comp=1):

    #We have to isolate the grouped signals:

    Genes_vect = (group_id == genes)

    val_fun = val.iloc[:, Genes_vect] 

    train_fun = train.iloc[:, Genes_vect]

    norm_fun = normalizing.iloc[Genes_vect,:]

    

    #We transform them into numpy arrays

    val_group = np.array(val_fun)

    train_group = np.array(train_fun)

    norm_group = np.array(norm_fun)

    val_n = np.zeros(val_fun.shape)



    for isig in range(norm_fun.shape[0]):

            val_n[:,isig]=(val_group[:,isig]-norm_group[isig,0])/norm_group[isig,1];

    

    #PCA Reconstruction

    pca = PCA()

    pca.fit(train_group)

    eigen = pca.components_

    eigen = np.transpose(eigen[:n_comp])

    inveigen = np.transpose(eigen)



    Xhat_n = val_n@eigen@inveigen

    Xhat = np.zeros(val_fun.shape)



    for isig in range(norm_fun.shape[0]):

        Xhat[:,isig]=Xhat_n[:,isig]*norm_group[isig,1]+norm_group[isig,0];



    MSE = sum(sum((val_group-Xhat)**2)/len(Xhat))/val.shape[1]

        

    return MSE
genes = generate_parent(df_val.shape[1], 1)

MSE = get_fitness_wrapper(genes, 0, df_val, df_train, normalizing)

MSE
def mutate(child, groups, min_size=4):

    #This function is used to mutate a random gene in the chromosome, as it is important to explore all the space

    mutated_child = child

    genes = np.array(mutated_child)

    #We swap a single value if a mutation occours

    index = random.randrange(0, len(child))

    swap = random.randint(0,groups-1)

    genes[index] = swap

    child = genes 

    

    for group_id in range(groups):

        Genes_vect = (group_id == child)

        #Recursive to respect the limit of at least 4 sensor each group

        if sum(Genes_vect) < min_size:

            child = mutate(child, groups, min_size)

            

    return child
def breeder_scattered(parent1, parent2, groups, min_size=4):

    #Here we mix random element from the parents and make two child

    child = parent1 #We start from the first parent

    selectionscattered = generate_parent(len(parent1), 2) #Vector of 0 and 1, used as a mask

    mask = (1 == selectionscattered) #Boolean mask for our substitution

    child_d = np.array(child)

    child_d[mask] = parent2[mask]

    child = child_d      

    for group_id in range(groups):

        Genes_vect = (group_id == child)

        #Recursive to respect the limit of at least {min_size} sensors each group

        if sum(Genes_vect) < min_size:

            child = breeder_scattered(parent1, parent2, groups, min_size)

    

    return child
#As the name suggest, this is the core of the genetic algorithm:

def core(pop, gen, groups, df, df_train, normalizing, crossover_fr=0.8, elite=5, mutation_prob=0.1, min_size=4, n_comp=1):

    #First thing we have to generate the parents

    parents = list()

    for i in tqdm(range(pop), desc = 'Parents generation'):

        parents.append(generate_parent(df_val.shape[1], groups, min_size)) #list containing the parents of our problem

    #Initial state of the population

    fit_array = np.zeros(pop)

    for j in tqdm(range(len(parents))):

        for i in range(groups):

            fit_array[j] = fit_array[j]+get_fitness_wrapper(parents[j], i, df, df_train, normalizing, n_comp)

            

    #Setup of variables for later

    fit_mean_array = list()

    fit_min_array = list()

    gen_array = list()

    next_gen = parents

    next_fit = fit_array  

    

    #Evolution

    for j in range(gen): 



        fit_array = next_fit

        parents = next_gen                                               

        next_fit = np.zeros(pop)

        next_gen = list()

        

        #ELITE

        elite_array = np.copy(fit_array)

        for i in range(elite):        

            if i != 0:

                while next_fit[i-1] == elite_array[elite_array.argmin()]: #We assure different elite to be passed on

                    elite_array = np.delete(elite_array, elite_array.argmin(), 0)

            

            next_fit[i] = elite_array[elite_array.argmin()]

            next_gen.append(parents[elite_array.argmin()])

            

        #CROSSOVER:

        Cross_fr = 0

        Cc = 1

        while Cross_fr < crossover_fr and len(next_gen) < pop:

            couples = random.sample(range(0, len(fit_array)),len(fit_array))

            child = breeder_scattered(parents[couples[Cc]], parents[couples[Cc+1]], groups, min_size)

            fit_child = np.zeros(1)

            for k in range(groups):

                fit_child = fit_child + get_fitness_wrapper(child, k, df, df_train, normalizing, n_comp)

                

            #Only the strong survive in this version:

            if fit_child < fit_array[couples[Cc+1]]:

                next_fit[Cc+i] = fit_child

                next_gen.append(child)

            else:

                next_fit[Cc+i] = fit_array[couples[Cc+1]]

                next_gen.append(parents[couples[Cc+1]])



            Cc +=1     

            Cross_fr = Cc/pop

            

        #MUTATION:

        while len(next_gen) < pop:

            parent1 = random.randint(0, len(parents)-1)

            #We mutate the vector with the probability {mutation_prob}

            if np.random.rand() < mutation_prob:

                child = mutate(parents[parent1], groups, min_size)

                fit_child = np.zeros(1)

                for k in range(groups):

                    fit_child = fit_child + get_fitness_wrapper(child, k, df, df_train, normalizing, n_comp)



                next_fit[len(next_gen)] = fit_child

                next_gen.append(child)

            else:

                next_fit[len(next_gen)] = fit_array[parent1]

                next_gen.append(parents[parent1])

                

        #This part in only for presentation purpose, it will update the graph while the GA is running

        plt.clf()

        display.clear_output(wait=True)



        fit_mean_array.append(np.mean(fit_array)) 

        fit_min_array.append(fit_array[fit_array.argmin()])

        gen_array.append(j) 

        

        label1 = "mean fitness "+str(round(fit_mean_array[-1],3))

        label2 = "best fitness "+str(round(fit_min_array[-1],3))

        

        plt.xlim((0,gen))            

        plt.plot(gen_array, fit_mean_array, 'r.', label = label1)

        plt.plot(gen_array, fit_min_array, 'k.', label = label2, markersize=5)

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.legend()

        plt.ylabel('MSE')

        plt.xlabel('Gen')

        plt.show()

        

        if round(fit_mean_array[-1],3) == round(fit_min_array[-1],3):

            print('Convergence reached, the GA is stopping')

            return parents, fit_array

        

    return parents, fit_array
'''Avoid bugs in tqdm module'''

try:

    tqdm._instances.clear()

except:

    pass

#We can finally run the GA, we first select the number of groups we require for or system:

groups = 3



#This will return the evolved population and their fitnesses, we can then select one of the element to group the signals:

pop, fit_vect = core(150, 60, groups, df_val, df_train, normalizing)
fit_vect[fit_vect.argmin()]
pop[fit_vect.argmin()]