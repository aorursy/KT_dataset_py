# Import torch and other required modules

import torch

import numpy
# Example 1 - working (change this)

arr = numpy.array([1, 2, 3])

tor = torch.as_tensor(arr)

tor

# Example 2 - working

list = [3,5,7,9]

torch.as_tensor(list)
# Example 3 - breaking (to illustrate when it breaks)

a = numpy.array([1, 2, 3])

t = torch.as_tensor(a)

t[0] = -1

a

# Example 1 - working
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
# Example 1 - working
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
# Example 1 - working
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
# Example 1 - working
# Example 2 - working
# Example 3 - breaking (to illustrate when it breaks)
!pip install jovian --upgrade --quiet
import jovian
jovian.commit()