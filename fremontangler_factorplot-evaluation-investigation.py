import numpy as np

import matplotlib.pyplot as plt
x2 = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

y2 = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
import pandas as pd



train = pd.read_csv('../input/train.csv')

test = pd.DataFrame({'Pclass': x2, 'Survived': y2})
test.head()
import seaborn as sns



sns.factorplot('Pclass', 'Survived', data=test)
test3 = pd.DataFrame(test)
test3
test3.describe()
# the point is exactly the mean(expectation) of y. For example, when x = 1, 

# we have (1, 0), (1, 3), (1, 3), and (1, 3), so the mean is (0 + 3 + 3 + 3) / 4 = 2.25,

# while the range is [0, 3]. The bar is from 0.75 to 3.0, why?



sns.factorplot(x='Pclass', y='Survived', data=test3)
# now use factorplot to determine if this feature is important

import seaborn as sns



sns.factorplot(x='Pclass', y='Survived', data=train)
sns.factorplot('Pclass', 'Survived', col='Sex', data=train)
sns.factorplot('Sex', 'Survived', col='Pclass', data=train)