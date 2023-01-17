# This noteboot demonstrates how to unmangle the floating point numbers

# in the input data.



# The problem is that while the data is an unsigned 64-bit integer,

# it - i.e. its memory representation - has been interpreted as a double.

# The function 'unmangle_float' reverses this process.



# To see that this interpretation is correct, observe that the

# 'capacity_bytes' value listed for the first hard drive is

# 3,000,592,982,016 after demangling. Googling for 'HDS5C3030ALA630 bytes'

# returns a result that confirms this is the exact byte size for the device.
import numpy as np

import pandas as pd

import struct



data = pd.read_csv('../input/harddrive.csv', nrows=50)
data.head()
cap = data.head().capacity_bytes

cap
def unmangle_float(x):

    return struct.unpack('>Q', struct.pack('>d', x))[0]
cap.map(unmangle_float)
data.capacity_bytes = data.capacity_bytes.map(unmangle_float)

data