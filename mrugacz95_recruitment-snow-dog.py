import os

import numpy as np

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def jaacard(a, b):

    pass
a = np.array([1, 0, 0, 1, 1, 1, 1])

b = np.array([0, 1, 1, 0, 1, 0, 0])

c = np.array([1, 1 ,0, 1, 1, 1, 1])
print(jaacard(a, a))

print(jaacard(a, b))

print(jaacard(a, c))