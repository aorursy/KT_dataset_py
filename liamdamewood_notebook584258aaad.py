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
import pandas as pd
train = pd.read_csv('../input/train.csv')

X_data = train[[c for c in train.columns if c not in ('label')]]

y_data = train['label']
import pandas as pd

df = pd.read_csv('../input/test.csv')

print(df.shape)
def objective(kwargs):

    """

    The objective function.

    

    A standalone function that executes the learner and returns the score.

    """

    

    import time

    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import zero_one_loss

    from sklearn.model_selection import train_test_split

    from hyperopt import STATUS_OK

    

    x_data, x_holdout, y_data, y_holdout = train_test_split(

        mnist.data, mnist.target, 

        train_size=6.0/7.0, test_size=1.0/7.0,

        random_state=0, stratify=mnist.target,

    )

    

    x_train, x_test, y_train, y_test = train_test_split(

        x_data, y_data, 

        train_size=0.1, test_size=0.1, 

        random_state=1, stratify=y_data,

    )

    

    tic = time.time()

    clf = LogisticRegression(C=kwargs['C'], fit_intercept=kwargs['affine'] > 0, n_jobs=1)

    clf.fit(x_train, y_train)

    y_predicted = clf.predict(x_test)

    toc = time.time()



    return {

        'loss': zero_one_loss(y_test, y_predicted),

        'status': STATUS_OK,

        'eval_time': time.time(),

        'elapsed_time': toc - tic,

    }
print(objective(dict(C=1.0, affine=True)))