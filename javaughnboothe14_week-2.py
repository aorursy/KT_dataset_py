# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
import os
from numpy import min

min([3,4,5])




c=[3,4,5]  
c.append(6)
a= 3
if a > 3:
	print("a is greater than 3")
else:
	print("a is less than or equal to 3")
if 3 in c:
	print("3 is in c")
else:
	print("3 is not in c")
print("3 is in %s %d %3.2f" % ("my list",3,40.5678))

print("3 is in {0} {1} {2}".format("my list",3,40.5678) )

if 'h' in "gordon":
	print("h is in gordon")
else:
	print("- his is not in gordon , lets check for g")
	if 'g' in 'gordon':
		print("Hooray")
		if 1 == 3:
			print("something is wrong")
		print("Still executes")
for i in c:
    print(i)
d=(2,3,5)
print(("The collection has "+str(["%s" for i in d]).replace(']',''))%d)

"string a " + "string b"
range(3)
range(0,3)
for i in range(3):
    print (i)
for i in range(2,5):
    print(i)
for i in range(1,11,2):
    print(i)
    
np.arange(3)
list(np.arange(3))
[0,1,2] * 3
np.arange(3) *3
c

collection = [3,4,5]
[item *3 for item in collection]
def add(num1,num2=0,num3=0):
    """"
    add method
    """
    print("num1 = %f,num2=%f,num3=%f"%(num1,num2,num3))
    return num1+ num2 +num3
add(4,num3=6)
