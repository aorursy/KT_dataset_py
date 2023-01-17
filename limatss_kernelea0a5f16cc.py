import random

import numpy as np

import matplotlib.pyplot as plt

import statistics

from collections import Counter
def one_roll():

  four_choices = []

  for x in range(4):

    four_choices.append(random.randint(1,6))

  four_choices.remove(min(four_choices))  

  return sum(four_choices)
def character_roll():

  atributes =[]

  for x in range(6):

    atributes.append(one_roll())

  return atributes
def is_a_valid_character(atributes):

  return sum(atributes) >= 64 and max(atributes) >=14
def atributes_to_cost(atributes):

  cost = 0

  for att in atributes:    

    if (att <= 8) : cost+=-2

    if (att == 9) : cost+=-1

    if (att == 10) : cost+=0

    if (att == 11) : cost+=1

    if (att == 12) : cost+=2

    if (att == 13) : cost+=3

    if (att == 14) : cost+=4

    if (att == 15) : cost+=6

    if (att == 16) : cost+=8

    if (att == 17) : cost+=11

    if (att == 18) : cost+=14    

  return cost
costs = []

for x in range(10000):

  character = character_roll()  

  if (is_a_valid_character(character)):

    costs.append(atributes_to_cost(character))



print(statistics.mean(costs))

print(statistics.median(costs))