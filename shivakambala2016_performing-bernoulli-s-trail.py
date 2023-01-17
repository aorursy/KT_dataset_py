import numpy as np
def perform_Bernoulli_trail(n,p):
    np.random.seed(42)
    random_numbers = np.empty(n)
    heads = np.empty(n)
    for i in range(n):
        random_numbers[i] = np.random.random(1)
    for i in range(n):
        heads[i] = random_numbers[i] > p
    pro = heads.sum() / n
    return pro
        
    
    
k = perform_Bernoulli_trail(3,0.5)

