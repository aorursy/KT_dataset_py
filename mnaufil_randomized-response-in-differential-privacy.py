import torch

from numpy import random

def create_db(quantity,probability):

        return torch.rand(quantity)<probability

original_db = create_db(10000,0.70)

x = torch.mean(original_db.float())

print("Mean of original database is: ",x)

print()

first_coin = (torch.rand(len(original_db))>0.5).float()

second_coin = (torch.rand(len(original_db))>0.5).float()

tricked_db = (original_db.float() * first_coin) + (1 - first_coin) * second_coin

x = torch.mean(tricked_db.float())

print("Mean of tricked database is: ",x*2-0.5) #de-skewing