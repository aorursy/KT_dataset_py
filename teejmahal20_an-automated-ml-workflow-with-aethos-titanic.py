!pip install aethos
import numpy as np

import pandas as pd

import aethos as at
at.options.word_report = True

at.options.interactive_table = True
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
df = at.Data(train, target_field='Survived', report_name='titanic', split = True, test_split_percentage = 0.2)

df
df.x_train

df.x_test
# For specified column, return Pandas series

df['Age']
# For specified conditional operation in the square brackets, return a Pandas DataFrame

df[df['Age'] > 25]
# You can even run pandas functions on the Data object, however they only operate on your training data.

df.nunique() 
df.describe()
df.describe_column('Age')
df.describe_column('Age')['std']
df.data_report()
df.jointplot('Age', 'Fare', kind='hex', output_file='age_fare_joint.png')

df.pairplot(diag_kind='hist', output_file='pairplot.png')

df.histogram('Age', output_file='age_hist.png')
df.missing_values
df.replace_missing_mostcommon('Embarked')
df.replace_missing_median('Age')
df.drop('Cabin')
df.missing_values
def get_person(data):

    age, sex = data['Age'], data['Sex']

    return 'child' if age < 16 else sex

df.apply(get_person, 'Person')

df.drop('Sex')
df.onehot_encode('Embarked', 'Person', keep_col=False, drop=None)