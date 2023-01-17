import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
pd.set_option("display.precision", 2)
data = pd.read_csv('../input/train.csv', index_col='PassengerId')
data.head(5)
data.describe()
data[(data['Embarked'] == 'C') & (data.Fare > 200)].head()
data[(data['Embarked'] == 'C') & 
     (data['Fare'] > 200)].sort_values(by='Fare',
                               ascending=False).head()
def age_category(age):
    '''
    < 30 -> 1
    >= 30, <55 -> 2
    >= 55 -> 3
    '''
    if age < 30:
        return 1
    elif age < 55:
        return 2
    else:
        return 3
age_categories = [age_category(age) for age in data.Age]
data['Age_category'] = age_categories
data['Age_category'] = data['Age'].apply(age_category)
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here
# You code here