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
math_students = pd.read_csv('../input/student-mat.csv')

math_students.info()
por_students = pd.read_csv('../input/student-por.csv')

por_students.info()
math_students.head()
from scipy import stats
# one way chi-square test for gender data

stats.chisquare(math_students['sex'].value_counts())
# one way chi-square test for romantics data

stats.chisquare(math_students['romantic'].value_counts())
# chi-square test between gender and romantic categories

contingencyTable = pd.crosstab(math_students['romantic'], math_students['sex'])

stats.chi2_contingency(contingencyTable)
# one way chi-square test for gender data

stats.chisquare(por_students['sex'].value_counts())
# one way chi-square test for romantics data

stats.chisquare(por_students['romantic'].value_counts())
# chi-square test between gender and romantic categories

new_contingencyTable = pd.crosstab(por_students['romantic'], por_students['sex'])

stats.chi2_contingency(new_contingencyTable)
import seaborn as sns

import matplotlib.pyplot as plt
plt.title('Portuguese students who are romantic')

sns.countplot(por_students['romantic'])

plt.show()
plt.title('Portuguese students gender')

sns.countplot(por_students['sex'])

plt.show()
plt.title('Math students who are romantic')

sns.countplot(math_students['romantic'])

plt.show()
plt.title('Math students gender')

sns.countplot(math_students['sex'])

plt.show()