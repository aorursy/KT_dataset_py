import os

#The OS module in python provides functions for interacting with the operating system.

#This module provides a portable way of using operating system dependent functionality.

#The *os* and *os.path* modules include many functions to interact with the file system.

import warnings

warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#UTF-8 is a variable-width character encoding standard 

#that uses between one and four eight-bit bytes to represent all valid Unicode code points.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.