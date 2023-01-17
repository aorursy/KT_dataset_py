# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 

import pandas as pd

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





# Any results you write to the current directory are saved as output.