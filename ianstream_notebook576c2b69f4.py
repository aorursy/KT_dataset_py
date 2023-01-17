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
genderclassmodel_url = "../input/genderclassmodel.csv"

genderclassmodel = pd.read_csv(genderclassmodel_url)



gendermodel_url = "../input/gendermodel.csv"

gendermodel = pd.read_csv(gendermodel_url)



test_url = "../input/test.csv"

test = pd.read_csv(test_url)



train_url = "../input/train.csv"

train = pd.read_csv(train_url)
genderclassmodel
gendermodel
test
train