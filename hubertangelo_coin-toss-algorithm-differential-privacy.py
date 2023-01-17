import torch

from numpy import random

def create_db(quantity,probability):

        return torch.rand(quantity)<probability # Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)

orig_db = create_db(10000,0.7) # 10000 values with a 0.7 probability of a 1 for each value

x = torch.mean(orig_db.float())

print("Mean of original database is: ",x)

print()

first_flip = (torch.rand(len(orig_db))>0.5).float()

second_flip = (torch.rand(len(orig_db))>0.5).float()

altered_db = (orig_db.float() * first_flip) + (1 - first_flip) * second_flip

x = torch.mean(altered_db.float())

print("Mean of altered database is: ",x*2-0.5) #de-skewing