# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from time import time;


def testLookupSpeed(dictSize):
    """
    returns performance in seconds per single lookup table test
    """
    randomInts = np.random.randint(0, 2**32, dictSize)
    lookupTable = set(randomInts)
    testInts = randomInts[0:100000:10]
    timeStart = time()
    # test elements on the list
    for i in testInts:
        found = i in lookupTable
    # test same number of elements (likely) not on the list
    for i in testInts:
        found = (i + 1) in lookupTable
    timeEnd = time()
    return (timeEnd - timeStart) / (len(testInts) * 2)

sizes = []
timesPerTest = []

for lookupTableSizeMagnitude in range(3, 9):
    lookupTableSize = 10 ** lookupTableSizeMagnitude
    sizes += [lookupTableSizeMagnitude]
    timesPerTest += [ testLookupSpeed(lookupTableSize) * 1e6 ];

plt.plot(sizes, timesPerTest, '-o')
plt.title( 'Time per Single Hash Lookup\nby the Lookup Table Size' )
plt.xlabel( 'Decimal Logarithm of the Lookup Table Size' )
plt.ylabel( 'Time per Single Lookup Test,\nMicroseconds' )
plt.show()