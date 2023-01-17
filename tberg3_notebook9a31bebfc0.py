import pandas as pd

import numpy as np



tf = pd.read_csv('../input/test.csv', header=0)

tf.loc[tf['Sex'] == 'female', 'Survived'] = 1

tf.loc[tf['SibSp'] > 0, 'Survived'] = 1

tf.loc[tf['Parch'] > 0, 'Survived'] = 1

tf = tf.fillna(0)

tf = tf[['PassengerId', 'Survived']]



def convert(x):

    try:

        return x.astype(int)

    except:

        return x



tf = tf.apply(convert)

    

tf.to_csv('test.output.csv', index=False)
tf