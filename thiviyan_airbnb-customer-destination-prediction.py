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
age_gender_bkts = pd.read_csv("../input/age_gender_bkts.csv")

countries = pd.read_csv("../input/countries.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")

test_users = pd.read_csv("../input/test_users.csv")

train_users_2 = pd.read_csv("../input/train_users_2.csv")
age_gender_bkts.head(50)
countries.head(50)
sample_submission.head(50)
test_users.head(50)
train_users_2.head(50)