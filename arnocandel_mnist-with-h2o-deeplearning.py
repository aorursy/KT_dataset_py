## This Python 3 environment comes with many helpful analytics libraries installed

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
%%time

train = h2o.import_file("../input/digit-recognizer/train.csv")

test = h2o.import_file("../input/digit-recognizer/test.csv")
train.head()
x = train.columns[1:]

y = 'label'

# For binary classification, response should be a factor

train[y] = train[y].asfactor()
dl = H2ODeepLearningEstimator(input_dropout_ratio = 0.2, nfolds=3)

dl.train(x=x, y=y, training_frame=train)
h2o.download_pojo(dl)  # production code in Java, more at http://docs.h2o.ai/h2o/latest-stable/h2o-docs/productionizing.html#about-pojos-and-mojos
dl.model_performance(xval=True)
preds = dl.predict(test)

preds['p1'].as_data_frame().values.shape
preds
sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

sample_submission.shape
sample_submission['Label'] = preds['predict'].as_data_frame().values

sample_submission.to_csv('h2o_dl_submission_1.csv', index=False)

sample_submission.head()