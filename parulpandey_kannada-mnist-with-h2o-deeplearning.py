# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import h2o

print(h2o.__version__)

from h2o.estimators.deeplearning import H2ODeepLearningEstimator



h2o.init(max_mem_size='16G')
train = h2o.import_file("/kaggle/input/Kannada-MNIST/train.csv")

test = h2o.import_file("/kaggle/input/Kannada-MNIST/test.csv")

submission = h2o.import_file("/kaggle/input/Kannada-MNIST/sample_submission.csv")
train.head()
x = train.columns[1:]

y = 'label'



train[y] = train[y].asfactor()
dl = H2ODeepLearningEstimator(input_dropout_ratio = 0.2, nfolds=3)

dl.train(x=x, y=y, training_frame=train)
dl.model_performance(xval=True)
preds = dl.predict(test)

preds['p1'].as_data_frame().values.shape
preds
sample_submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

sample_submission.shape
sample_submission['label'] = preds['predict'].as_data_frame().values

sample_submission.to_csv('H2O_DL.csv', index=False)

sample_submission.head()