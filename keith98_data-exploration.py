# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

medications = pd.read_csv("../input/medications.csv", encoding="latin-1")

demographics = pd.read_csv("../input/demographic.csv", encoding="latin-1")

exams = pd.read_csv("../input/examination.csv", encoding="latin-1")

diets = pd.read_csv("../input/diet.csv", encoding="latin-1")

surveys = pd.read_csv("../input/questionnaire.csv", encoding="latin-1")
medications.info()
medications.RXDDRUG.value_counts()[:30]
demographics.head()
demographics.RIDAGEYR.hist()

plt.suptitle("Distribution of age")
demographics[["INDHHIN2"]].hist()

plt.suptitle("Household income")
exams.info()
exams.MGDCGSZ.hist()

plt.suptitle("Grip strength")