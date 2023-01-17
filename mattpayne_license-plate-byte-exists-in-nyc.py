# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# str = open('../input/README.md', 'r').read()
# print(str)
lines = 0
accepted = open('../input/accepted-plates.csv','r')
for plate in accepted.readlines():
    lines = lines + 1
    #if (0 == (lines % 10000)):
    #    print("Read %d lines" % (lines))
    if "BYTE" in plate.upper():
        print("found BYTE at line %d" %(lines))
    
print("Read %d lines" % (lines))    