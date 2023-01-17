import torch



# the number of entries in our database

num_entries = 5000



db = torch.rand(num_entries) > 0.5

db

#db.shape
# try project here!



def get_parallel_db(db, remove_index):

    return torch.cat((db[:remove_index], 

                      db[remove_index+1 :]))

# test 

# get_parallel_db(db, 4).shape
def get_parallel_dbs(db):

    parallel_dbs = list()



    for i in range(len(db)):

        parallel_db = get_parallel_db(db, i)

        parallel_dbs.append(parallel_db)

    return parallel_dbs





def create_db_and_parallels(num_entries):

    db = torch.rand(num_entries) > 0.5

    pdbs = get_parallel_dbs(db)

    return db, pdbs
test_db, test_pdbs = create_db_and_parallels(9) 

test_pdbs
db, pdbs = create_db_and_parallels(5000)
def query_sum(db):

    return db.sum()
full_db_result = query_sum(db)
sensitivity = 0

for pdb in pdbs:

    pdb_result = query_sum(pdb)

    

    db_distance = torch.abs(pdb_result - full_db_result)

    

    if(db_distance > sensitivity):

        sensitivity = db_distance
sensitivity
# try this project here!

def sensitivity(query, n_entries=1000):

    db, pdbs = create_db_and_parallels(n_entries)

    full_db_result = query(db)

    max_dist = 0

    

    for pdb in pdbs:

        pdb_result = query(pdb)



        db_distance = torch.abs(pdb_result - full_db_result)



        if db_distance > max_dist:

            max_dist = db_distance

        

    return max_dist
def query_mean(db):

    return db.float().mean()
sensitivity(query_mean)
# try this project here!



def query(db, threshold=5):

    return (db.sum() > threshold).float()


for i in range(10):

    sens_f = sensitivity(query, n_entries=10)

    print(sens_f)
# try this project here!

db12, _ = create_db_and_parallels(12)



pdb = get_parallel_db(db12, remove_index=10)

db12[10]

sum(db12)
# differencing attack using the sum query

sum(db12)- sum(pdb)
# differencing attack using the mean query

(sum(db).float() / len(db)) - (sum(pdb).float() / len(pdb))
# differencing attack using the threshold query

(sum(db).float() > 5) - (sum(pdb).float()  > 5)

# try this project here!

db12



def query_skew(db):

    true_result = torch.mean(db.float())

    first_coinflip = (torch.rand(len(db)) < 0.5).float()

    second_coinflip = (torch.rand(len(db)) < 0.5).float()

    

    augm_db = db.float() * first_coinflip + (1 - first_coinflip) * second_coinflip

    

    # deskew the result

    deskewed_result = torch.mean(augm_db.float()) * 2 -0.5

    

    return deskewed_result, true_result

db, pdb = create_db_and_parallels(10)

private, true_result = query_skew(db)

print('with noise ' + str(private))

print('without noise ' + str(true_result))
db, pdb = create_db_and_parallels(100)

private, true_result = query_skew(db)

print('with noise ' + str(private))

print('without noise ' + str(true_result))
db, pdb = create_db_and_parallels(1000)

private, true_result = query_skew(db)

print('with noise ' + str(private))

print('without noise ' + str(true_result))
db, pdb = create_db_and_parallels(10000)

private, true_result = query_skew(db)

print('with noise ' + str(private))

print('without noise ' + str(true_result))
db, pdb = create_db_and_parallels(100*1000)

private, true_result = query_skew(db)

print('with noise ' + str(private))

print('without noise ' + str(true_result))
# try this project here!

def query_noise(db, noise=0.3):

    true_result = torch.mean(db.float())

    first_coinflip = (torch.rand(len(db)) < noise).float()

    second_coinflip = (torch.rand(len(db)) < 0.5).float()

    

    augm_db = db.float() * first_coinflip + (1 - first_coinflip) * second_coinflip

    

    skewed_result = torch.mean(augm_db.float())

    

    # deskew the result

    deskewed_result = skewed_result * 2 -0.5

    

    return deskewed_result, true_result
db, pdb = create_db_and_parallels(10)

private, true_result = query_noise(db, 0.1)

print('with noise ' + str(private))

print('without noise ' + str(true_result))
db, pdb = create_db_and_parallels(10)

private, true_result = query_noise(db, 0.2)

print('with noise ' + str(private))

print('without noise ' + str(true_result))
db, pdb = create_db_and_parallels(10)

private, true_result = query_noise(db, 0.3)

print('with noise ' + str(private))

print('without noise ' + str(true_result))
db, pdbs = create_db_and_parallels(100)



def query(db):

    return torch.sum(db.float())



def M(db):

    query(db) + noise



query(db)
# try this project here!

import numpy as np



db, pdbs = create_db_and_parallels(100)



def laplacian_mechanism(db, query, sensitivity=1, epsilon=0.0001):

    beta = sensitivity/epsilon

    noise = torch.tensor(np.random.laplace(0, beta, 1))

    return query(db) + noise

laplacian_mechanism(db, query_sum, 1)
laplacian_mechanism(db, query_mean, 0.01)
import numpy as np
num_teachers = 10 # we're working with 10 partner hospitals

num_examples = 10000 # the size of OUR dataset

num_labels = 10 # number of lablels for our classifier
preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int).transpose(1,0) # fake predictions
new_labels = list()

for an_image in preds:



    label_counts = np.bincount(an_image, minlength=num_labels)



    epsilon = 0.1

    beta = 1 / epsilon



    for i in range(len(label_counts)):

        label_counts[i] += np.random.laplace(0, beta, 1)



    new_label = np.argmax(label_counts)

    

    new_labels.append(new_label)
# new_labels
labels = np.array([9, 9, 3, 6, 9, 9, 9, 9, 8, 2])

counts = np.bincount(labels, minlength=10)

query_result = np.argmax(counts)

query_result
from syft.frameworks.torch.differential_privacy import pate
num_teachers, num_examples, num_labels = (100, 100, 10)

preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int) #fake preds

indices = (np.random.rand(num_examples) * num_labels).astype(int) # true answers



preds[:,0:10] *= 0



data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5)



assert data_dep_eps < data_ind_eps



data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5)

print("Data Independent Epsilon:", data_ind_eps)

print("Data Dependent Epsilon:", data_dep_eps)
preds[:,0:50] *= 0
data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5, moments=20)

print("Data Independent Epsilon:", data_ind_eps)

print("Data Dependent Epsilon:", data_dep_eps)
import torchvision.datasets as datasets

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
train_data = mnist_trainset.train_data

train_targets = mnist_trainset.train_labels
test_data = mnist_trainset.test_data

test_targets = mnist_trainset.test_labels