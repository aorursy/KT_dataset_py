# Why I'm doing this: to learn Python and pandas! I've been working in R for years but

# haven't scratched Python yet. 



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
# Import data

cereals = pd.read_csv('../input/cereal.csv')
# Describe the data. So whoa, this is different than R. 

# I was trying describe(cereals) but I've learned that's not how Python works 

cereals.describe()