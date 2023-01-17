import json

f = open("/kaggle/input/wikipedia-mathematicians/mathematicians.json")

data = json.load(f)

f.close()

print(len(data))

print(data[0])
from random import randint

import numpy



# these do not _contribute

data = [item for item in data if item["links"]]



n = len(data)

print("n=", n)

lookup = dict((data[idx]["name"], idx) for idx in range(n))
# construct a stochastic matrix

A = numpy.zeros((n, n))



EPSILON = 1e-6



shift = EPSILON  # add on the diagonal 

for idx in range(n):

    item = data[idx]

    for name in item["links"]:

        jdx = lookup.get(name)

        if jdx is None:

            continue

        if idx != jdx:

            A[idx, jdx] = 1.0

            A[jdx, idx] = 1.0 # make symmetric

    A[idx, idx] = shift



assert numpy.allclose(A, A.transpose()) # yes it's symmetric ...



# row normalize to make a stochastic matrix

for idx in range(n):

    r = A[:, idx].sum()

    assert r>=EPSILON

    A[:, idx] /= r
# just use the power method



vec = numpy.zeros((n,))

for i in range(n):

    vec[i] = randint(1, 100)

vec[:] /= vec.sum()



for i in range(1000):

    vec = numpy.dot(A, vec)

    vec[:] /= vec.sum()
items = list(zip(list(range(n)), vec))

items.sort(key = lambda item : -abs(item[1]))



count = 1

for idx, v in items[:20]:

    name = data[idx]["name"]

    name = name.replace(" (mathematician)", "")

    print("%3d %24s %.4f"%(count, name, v*1000.))

    count += 1
