# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def close_to_power(number,powernumber,verbose=False):

    upper=1

    while number>upper:

        upper*=powernumber

    lower=upper/powernumber

    lowerpct=((number/lower)-1)

    upperpct=((number/upper)-1)*-1

    if verbose:print(f"{lower} {lowerpct} {upper} {upperpct} ")

    return(min(lowerpct,upperpct))
winlist=[]

for x in range(500):

    p2=2**x

    close=close_to_power(p2,10)

    if close<.024:

        winlist.append((p2,close))

        print(f"{x} {p2} {close}")