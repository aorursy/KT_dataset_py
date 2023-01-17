import numpy as np
#size of an individual, i.e. length of the vector
size = 30

# this is the size of breeding pool and used to generate the population
n=5

# in making this is a triangle number we can breed 
# the top n individuals and get the same size population back
pop_size= sum(range(n+1))
print(pop_size)
# evaluates fitness of population

def eval_fit(pop):
    fit_vals = []
    for i in range(len(pop)):
        fit_vals.append(np.sum(pop[i]))
        
    return np.array(fit_vals)
        
# ranks population

def rank_pop(pop):
    ranked =  [ pop[i] for i in np.argsort(-eval_fit(pop))]
    return ranked
# crossovers

def cross_pop(pop):
    new_pop = []
    for i in range(n):
        for j in range(i,n):
            x = np.random.randint(low=int(size/4),high=3*int(size/4)) # crossover point between 1/4 and 3/4
            new_pop.append(np.concatenate([pop[i][:x],pop[j][x:]]))
    return new_pop
# mutations

def mut_pop(pop,k):       # 1/k is prob of mutating an individual
    for i in range(len(pop)):
        x = np.random.randint(0,k)
        if(x==0):
            y = np.random.randint(0,size)
            pop[i][y] = (pop[i][y]+1) %2
    return pop
                   
        
# creates a population
pop = []

for i in range(pop_size):    
    pop.append(np.random.randint(low=0,high=2, size=(size)))

    
# runs the algorithm and finds an optimum
m = 0
mut_prob = 3   # probability of a mutation in a given individual (i.e. 1/mut_prob)
best_fitn = np.amax(eval_fit(pop))

while(best_fitn < size and m<100):
        
    pop = rank_pop(pop)
    pop = cross_pop(pop)
    pop = mut_pop(pop,mut_prob)
    
    print("Generation: " + str(m))
    print(str(best_fitn) + " : " + str(100*best_fitn/size) + "%")
    #print(pop[0])

    best_fitn = np.amax(eval_fit(pop))
    m=m+1
  
print("\n")
print("Completed at generation: " + str(m))
print("Best fitness is: " + str(100*best_fitn/size) + "%")
pop = rank_pop(pop)
print("Best individual is: ")
pop[0]
