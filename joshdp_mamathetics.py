import numpy as np

import random

import pandas as pd

import heapq

#random.seed(5)

random.seed(0)

from collections import Counter

# Make the PhD class

def generate_phd_class(size):

  PhDs = pd.DataFrame(columns = ["Field", "O_Value"], index = range(size))

  for i in range(size):

    #25% are Anabra

    if i < p*size:

      PhDs.loc[i] = pd.Series({"Field": "Anabra", "O_Value": random.uniform(0, 1)})

    else:

    #75% are Algasis

      PhDs.loc[i] = pd.Series({"Field": "Algasis", "O_Value": random.uniform(0, 1)})

  return(PhDs)



def subjective_value(student, faculty):

  l = 0

  # for each faculty member

  for i in range(len(faculty)):

    #if faculty member and student are of the same field, add value

    if faculty.loc[i]["Field"] == student["Field"]:

      l+=(student["O_Value"]+t) 

    else:

      #otherwise subtract

      l+=(student["O_Value"]-t)

      #return mean of overall impression

  return l/len(faculty)



def subjective_value_each_student(PhD_Class, faculty):

  l = []

  for i in PhD_Class.index:

    l.append(subjective_value(PhD_Class.loc[i],faculty))

  return(l)

def find_best_two(subjective_values):

  indexes = heapq.nlargest(2, range(len(subjective_values)), key=subjective_values.__getitem__)

  return(indexes)
# p is the proportion that study Anabra, 1-p Algasis

p = 0.25

# n is the number of faculty members, 50 in the blog post, but 52 for divisibility reasons

n = 52

# k is the number of departments

k = 10

# t is the value factor

t = 0.01

# size of PhD class

size = 40

random.seed(0)

departments = dict()

for i in range(k):

  df = pd.DataFrame(columns=['Field', 'Age'], index=range(n)) 

  for j in range(n):

    #25% are Anabra

    if j < p*n:

      df.loc[j] = pd.Series({'Field': "Anabra", "Age": random.uniform(0, 1)})

    else:

      #75% are Algasis

      df.loc[j] = pd.Series({'Field': "Algasis", "Age": random.uniform(0, 1)})

  df["Age"] = df["Age"].astype('float64')

  departments[i] = df
proportion = pd.DataFrame(columns=range(k))

proportion.loc[0] = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

for y in range(800):

    PhDs = generate_phd_class(size)

    for i in range(k):

        d = departments[i]

        l = subjective_value_each_student(PhDs, d)

        indexes = find_best_two(l)

          # Add best two to department

        d.loc[n+1] = pd.Series({"Field":PhDs.loc[indexes[0]]["Field"], "Age":0})

        d.loc[n+2] = pd.Series({"Field":PhDs.loc[indexes[1]]["Field"], "Age":0})

        #drop those Phd Students

        PhDs.drop(indexes, inplace=True)

        #relabel the indices

        PhDs.index = range(len(PhDs))

        #drop oldest two

        d.drop(d["Age"].idxmax(), inplace=True)

        d.drop(d["Age"].idxmax(), inplace=True)

        #relabel the indices

        d.index = range(len(d))

        #add year to everyone's age

        d["Age"] = d["Age"]+1

    l = [] 

    for j in range(k):

        l.append(departments[j]["Field"].value_counts()["Algasis"]/52)

    proportion.loc[y+1] = l

for i in range(10):

  print(Counter(departments[i]["Field"]))
proportion.head()

import matplotlib.pyplot as plt

lines = proportion.plot.line(figsize=(12, 6)

                             ,title='Algasis Proportion over Time, t=0.01, Horizontal Line at 0.5'

                            )

lines.set_xlabel("Years")

lines.set_ylabel("Algasis Proportion")

plt.axhline(y=0.5,color='black',linestyle='--')