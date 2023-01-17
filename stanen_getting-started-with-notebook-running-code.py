a = 10

print(a)
import time

time.sleep(10)
import sys

from ctypes import CDLL

# this will crash a linux or Mac ystem

# equivalent calls can be made on Windows





# segfault



# dll = 'dylib' if sys.platform == 'darwin' else 'so.6'

# libc = CDLL("libc.%s" % dll)

# libc.time(-1) 
print("Hi, stdout")
from __future__ import print_function

print('Hi, stderr', file=sys.stderr)
import time, sys

for i in range(8):

    print(i)

    time.sleep(0.75)
for i in range(50):

    print(i)
for i in range(500):

    print(2**i - 1)