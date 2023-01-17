import json

f = open("/kaggle/input/wikipedia-people-network/output.json")

data = json.load(f)

print(len(data))
record = data[0]

print(record)
import numpy

from scipy.sparse import dok_matrix

SKIP_PAGES = """

Knut_Helle Jan_Eivind_Myhre Andreas_Holmsen Ingvar_Lars_Helle Knut_Kjeldstadli

""".split()



source = data

pages = []

lookup = {}

for item in source:

    name = item["name"]

    links = item["links"]

    dob = item["dob"]

    if name in SKIP_PAGES:

        print("skip", name)

        continue

    lookup[name] = len(pages)

    pages.append((name, links))



n = len(pages)

print("n=%d"%n)

shift = 1e-6  # add on the diagonal 

A = dok_matrix((n, n), dtype=numpy.float64)

for idx in range(n):

  page = pages[idx]

  jdxs = [lookup.get(name) for name in page[1]]

  jdxs = [jdx for jdx in jdxs if jdx is not None]

  for jdx in jdxs:

      if jdx is not None and idx != jdx:

          A[jdx, idx] = 1.0 / len(jdxs) # <--- normalized entry

  A[idx, idx] = shift

from random import randint, random



A = A.tocsr()



vec = numpy.zeros((n,))

vec[:] = 1./n

for i in range(n):

    vec[i] = randint(1, 100)

vec[:] /= vec.sum()



print("iterating...")

niter = 1000

for i in range(niter):

    vec = A.dot(vec)

    for j in range(n):

        vec[j] += 1e-6 * random() # add some fuzz

    vec[:] /= vec.sum()

    print(".", end="", flush=True)

print()



items = list(zip(list(range(n)), vec))

items.sort(key = lambda item : -abs(item[1]))



count = 1

for idx, v in items[:100]:

    name = pages[idx][0]

    print("%3d %24s %.4f"%(count, name, v*1000.))

    count += 1
