import pandas as pd

import numpy as np

import random

X = []

#Load variable values into the list X 

#we keep only the value not the variables name since they are ordered so the number of the variable 

#is the index in the list

with open('../input/3SAT_Solution.txt','r') as f:

    for line in f:

        line = line.rstrip("\n")

        splited = line.split("=")

        X.append(bool(int(splited[-1])))

#Loading the clause in a matrix of dimension 218*3 

clause = []

with open('../input/3SAT_218dataset.txt','r') as f:

    for line in f:

        splited = line.split(",")

        del splited[-1]

        int_splited = [int(x) for x in splited]

        clause.append(int_splited)
#The function simple, we basically loop over the lines of our matrix

#For each line we loop over the colomns (the variable numbers) and we test if the variable will kept as it is

#or we will do a not of the variables (if it is of the form -x)

#For each line we do an or of the 3 variables and store the value in  partial_lateral, once we finish the line

#we do an and of the line to the previous lines in the partial_clause variables

#at the end of the loop of the lines we return partial_clause

def threeSatVerification(clause,X):

    partial_clause = True

    for i in range(len(clause)):

        partial_lateral = False

        for j in range(len(clause[i])):

            partial_lateral = partial_lateral or ( bool(X[clause[i][j]-1]) if (clause[i][j] > 0) else bool(not(X[-clause[i][j]-1])) )

        partial_clause = partial_clause and partial_lateral

    return partial_clause
#Function call

print(threeSatVerification(clause,X))