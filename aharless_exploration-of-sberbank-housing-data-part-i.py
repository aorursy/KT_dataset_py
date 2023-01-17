import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



train = pd.read_csv('../input/train.csv')

macro = pd.read_csv('../input/macro.csv')

test = pd.read_csv('../input/test.csv')



dfa = pd.concat([train, test])

dfa = dfa.merge(macro, on='timestamp', suffixes=['','_macro'])
def describe(varname="price_doc", df=dfa, minval=-1e20, maxval=1e20, 

             addtolog=1, nlo=8, nhi=8, dohist=True, showmiss=True):

  var = df[varname]



  print("DESCRIPTION OF ", varname, "\n")

  if (showmiss):

     print("Fraction missing = ", var.isnull().mean(), "\n")

  var = var[(var<=maxval) & (var>=minval)]

  if (nlo > 0):

    print("Lowest values:\n", var.sort_values().head(nlo).values, "\n")

  if (nhi > 0):

    print("Highest values:\n", var.sort_values().tail(nhi).values, "\n")



  if (dohist):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))



    print("Histograms of raw values and logarithm")

    var.plot(ax=axes[0], kind='hist', bins=100)

    np.log(var+addtolog).plot(ax=axes[1], kind='hist', bins=100, color='green', secondary_y=True)

    plt.show()
print(dfa.shape)

dfa.head()
print("The names of all 392 columns:\n\n", dfa.columns.values)
describe()
describe("full_sq")
describe("full_sq", minval=1.5, maxval=1000, showmiss=False)
describe("full_sq", minval=3, maxval=1000, nhi=0, showmiss=False)
describe("full_sq", minval=25, maxval=60, nhi=0, nlo=0, showmiss=False)

print("Mode is ", dfa.full_sq.mode().values[0] )
describe("life_sq")
describe("life_sq", minval=1.5, maxval=1000, nhi=0, showmiss=False)
describe("life_sq", minval=2.5, maxval=1000, nhi=0, nlo=0, showmiss=False)
describe("life_sq", minval=15, maxval=35, nhi=0, nlo=0, showmiss=False)
describe("floor")
describe("max_floor")
describe("num_room")
describe("kitch_sq")
describe("kitch_sq", maxval=12, nhi=0, nlo=0, showmiss=False)
describe("kitch_sq", minval=30, nhi=20, nlo=0, showmiss=False)
describe("kitch_sq", minval=4, maxval=150, nhi=0, nlo=0, showmiss=False)
describe("kitch_sq", minval=15, maxval=70, nhi=0, nlo=0, showmiss=False)
var = dfa.kitch_sq

var2 = var[((var>1)&(var<400))]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))

np.cbrt(np.log(var+1)).plot(ax=axes[0], kind='hist', bins=100, color='green', secondary_y=True)

np.cbrt(np.log(var2+1)).plot(ax=axes[1], kind='hist', bins=100, color='green', secondary_y=True)

plt.show()