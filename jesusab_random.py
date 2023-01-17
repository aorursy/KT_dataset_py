# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
K = "How is the world doing?"
L = "The world is doing well"
print(K, L)

A = 1
B = 3
F = A
print(A, B)
print(F)

A = 2
print(A)

F = A
print(F)

print(K, L)
G = K
print(K)

K = "Just a hello?"
print(K)
print(G)
print(K)

v1 = "First test string"
v2 = "Second test string"
print(v1)
print(v2)
v1 = "First test string"
v2 = "Second test string"

temp1 = v1
v1 = v2
v2 = temp1
print(v1)
print(temp1)
print(v1)
print(v2)
#
#
#
# With internet turned on for work below:
#
#
#
!pip install netmiko
from netmiko import ConnectHandler

linux = {
    'device_type': 'linux',
    'host':   '3.80.187.178',
    'username': 'kevin',
    'password': 'S!mpl312',
}
c = ConnectHandler(**linux) # use of kwargs optional, could just use regular parameters

r1 = c.send_command("echo hello world  > hw.txt")
print(r1)

r2 = c.send_command("cat hw.txt")
print(r2)

r3 = c.send_command("ls -la")
print(r3)
import re

lines = r3.split("\n")

NAME_INDEX = 8

for item in lines:
#     parts = item.split(" ") # split on space
#     parts = re.split(' +', item) # use regex to split on more than one space
    parts = item.split() # no parameters does the right thing!
    print(parts)
    if len(parts) >= NAME_INDEX:
        file_name = parts[NAME_INDEX]
        print(file_name)
        if file_name == "hw.txt":
            print("YAY!!!")
#
#
#
# End of: With internet turned on for work
#
#
#
