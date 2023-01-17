# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for visualisations

% matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Reading the data from file

data = pd.read_csv("../input/anonymous-survey-responses.csv")

data.head()
# Getting a short summary of data

data.describe()
sns.countplot(data['Just for fun, do you prefer dogs or cat?'], 

              hue = data["Do you have any previous experience with programming?"])
# I still want to see if theres something special abou people who have a whole lot of experience



pro = data [data["Do you have any previous experience with programming?"] == "I have a whole lot of experience"]
pro.describe()
sns.countplot(pro['Just for fun, do you prefer dogs or cat?'])