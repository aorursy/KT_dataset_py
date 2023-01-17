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

# Call libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

HR = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

HR.head()

sns.distplot(HR['Age'])

sns.countplot(HR.JobSatisfaction)

sns.barplot(x = 'JobSatisfaction' , y = 'JobInvolvement' , data = HR)

sns.barplot(x = 'WorkLifeBalance' , y = 'YearsAtCompany' , data = HR )

sns.boxplot(HR['PerformanceRating'], HR['PercentSalaryHike'])