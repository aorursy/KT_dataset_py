# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
gift_types = ['horse', 'ball', 'bike', 'train', 'coal', 'book', 'doll', 'blocks', 'gloves']

ngift_types = len(gift_types)



horse, ball, bike, train, coal, book, doll, blocks, gloves = range(ngift_types)
nsample=10000
def gift_weights(gift, ngift, n=nsample):

    if ngift == 0:

        return np.array([0.0])

    np.random.seed(2016)

    if gift == horse:

        dist = np.maximum(0, np.random.normal(5,2,(n, ngift))).sum(axis=1)

    if gift == ball:

        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift))).sum(axis=1)

    if gift == bike:

        dist = np.maximum(0, np.random.normal(20,10,(n, ngift))).sum(axis=1)

    if gift == train:

        dist = np.maximum(0, np.random.normal(10,5,(n, ngift))).sum(axis=1)

    if gift == coal:

        dist = 47 * np.random.beta(0.5,0.5,(n, ngift)).sum(axis=1)

    if gift == book:

        dist = np.random.chisquare(2,(n, ngift)).sum(axis=1)

    if gift == doll:

        dist = np.random.gamma(5,1,(n, ngift)).sum(axis=1)

    if gift == blocks:

        dist = np.random.triangular(5,10,20,(n, ngift)).sum(axis=1)

    if gift == gloves:

        gloves1 = 3.0 + np.random.rand(n, ngift)

        gloves2 = np.random.rand(n, ngift)

        gloves3 = np.random.rand(n, ngift)

        dist = np.where(gloves2 < 0.3, gloves1, gloves3).sum(axis=1)

    return dist
epsilon = 1

max_type = np.zeros(ngift_types).astype('int')



for gift, gift_type in enumerate(gift_types):

    best_value = 0.0

    for j in range(1, 200):

        weights = gift_weights(gift, j, nsample)

        raw_value = np.where(weights <= 50.0, weights, 0.0)

        value = raw_value.mean()

        if value > best_value:

            best_value = value

        else:

            break

    max_type[gift] = j

max_type
def gift_distributions(gift, ngift, n=nsample):

    if ngift == 0:

        return np.array([0.0])

    np.random.seed(2016)

    if gift == horse:

        dist = np.maximum(0, np.random.normal(5,2,(n, ngift)))

    if gift == ball:

        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift)))

    if gift == bike:

        dist = np.maximum(0, np.random.normal(20,10,(n, ngift)))

    if gift == train:

        dist = np.maximum(0, np.random.normal(10,5,(n, ngift)))

    if gift == coal:

        dist = 47 * np.random.beta(0.5,0.5,(n, ngift))

    if gift == book:

        dist = np.random.chisquare(2,(n, ngift))

    if gift == doll:

        dist = np.random.gamma(5,1,(n, ngift))

    if gift == blocks:

        dist = np.random.triangular(5,10,20,(n, ngift))

    if gift == gloves:

        gloves1 = 3.0 + np.random.rand(n, ngift)

        gloves2 = np.random.rand(n, ngift)

        gloves3 = np.random.rand(n, ngift)

        dist = np.where(gloves2 < 0.3, gloves1, gloves3)

    for j in range(1, ngift):

        dist[:,j] += dist[:,j-1]

    return dist



distributions = dict()

    

for gift in range(ngift_types):

    distributions[gift] = gift_distributions(gift, max_type[gift])
def gift_distributions(gift, ngift):

    if ngift <= 0:

        return 0

    if ngift >= max_type[gift]:

        return 51

    return distributions[gift][:,ngift-1]



def gift_value(ntypes):

    weights = np.zeros(nsample)

    for gift in range(ngift_types):

        dist = gift_distributions(gift, ntypes[gift])

        weights += dist

    weights = np.where(weights <= 50.0, weights, 0.0)

    return weights.mean(), weights.std()
from collections import deque



def get_update_value(bag, bag_stats):

    if bag in bag_stats:

        bag_mean, bag_std = bag_stats[bag]

    else:

        bag_mean, bag_std = gift_value(bag)

        bag_stats[bag] = (bag_mean, bag_std)

    return bag_mean, bag_std



def gen_bags():

    bag_stats = dict()

    queued = dict()

    queue = deque()

    bags = []

    bag0 = (0,0,0,0,0,0,0,0,0)

    queue.append(bag0)

    queued[bag0] = True

    bag_stats[bag0] = (0,0)

    counter = 0

    try:

        while True:

            if counter % 1000 == 0:

                print(counter, end=' ')

            counter += 1

            bag = queue.popleft()

            bag_mean, bag_std = get_update_value(bag, bag_stats)

            bags.append(bag+(bag_mean, bag_std ))

            for gift in range(ngift_types):

                new_bag = list(bag)

                new_bag[gift] = 1 + bag[gift]

                new_bag = tuple(new_bag)

                if new_bag in queued:

                    continue

                new_bag_mean, new_bag_std = get_update_value(new_bag, bag_stats)

                if new_bag_mean > bag_mean:

                    queue.append(new_bag)

                    queued[new_bag] = True

                    

    except:

        return bags



    

bags = gen_bags()



nbags = len(bags)



bags = pd.DataFrame(columns=gift_types+['mean', 'std'], 

                    data=bags)



bags['var'] = bags['std']**2



bags = bags[bags[gift_types].sum(axis=1) >= 3].reset_index(drop=True)



bags.head()
bags.shape[0]
gifts = pd.read_csv('../input/gifts.csv')



for gift in gift_types:

    gifts[gift] = 1.0 * gifts['GiftId'].str.startswith(gift)



gifts.head()
allgifts = gifts[gift_types].sum()



allgifts
from docplex.mp.model import Model



def qcpmip_solve(gift_types, bags, std_coef):

    mdl = Model('Santa')



    rbags = range(bags.shape[0])

    x_names = ['x_%d' % i for i in range(bags.shape[0])]

    x = mdl.integer_var_list(rbags, lb=0, name=x_names)

    

    var = mdl.continuous_var(lb=0, ub=mdl.infinity, name='var')

    std = mdl.continuous_var(lb=0, ub=mdl.infinity, name='std')

    mean = mdl.continuous_var(lb=0, ub=mdl.infinity, name='mean')

                                  

    mdl.maximize(mean + std_coef * std)

    

    for gift in gift_types:

        mdl.add_constraint(mdl.sum(bags[gift][i] * x[i] for i in rbags) <= allgifts[gift])

        

    mdl.add_constraint(mdl.sum(x[i] for i in rbags) <= 1000)



    mdl.add_constraint(mdl.sum(bags['mean'][i] * x[i] for i in rbags) == mean)

    

    mdl.add_constraint(mdl.sum(bags['var'][i] * x[i] for i in rbags) == var)



    mdl.add_constraint(std**2 <= var)

    

    mdl.parameters.mip.tolerances.mipgap = 0.00001

    

    s = mdl.solve(log_output=False)

    assert s is not None

    mdl.print_solution()

    
qcpmip_solve(gift_types, bags, 0.0)
qcpmip_solve(gift_types, bags, 3.0)