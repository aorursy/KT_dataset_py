# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/jigsaw-toxic-comment-classification-challenge"))

print(os.listdir("../input/obscene-with-h2o-automl"))

print(os.listdir("../input/identity-hate-with-h2o-automl"))

print(os.listdir("../input/toxic-with-h2o-automl"))

print(os.listdir("../input/insult-with-h2o-automl"))

print(os.listdir("../input/threat-with-h2o-automl"))

print(os.listdir("../input/severe-toxic-with-h2o-automl"))
obscene_submission = pd.read_csv("../input/obscene-with-h2o-automl/obscene_submission.csv")

identity_hate_submission = pd.read_csv("../input/identity-hate-with-h2o-automl/identity_hate_submission.csv")

toxic_submission = pd.read_csv("../input/toxic-with-h2o-automl/toxic_submission.csv")

insult_submission = pd.read_csv("../input/insult-with-h2o-automl/insult_submission.csv")

threat_submission = pd.read_csv("../input/threat-with-h2o-automl/threat_submission.csv")

severe_toxic_submission = pd.read_csv("../input/severe-toxic-with-h2o-automl/severe_toxic_submission.csv")
obscene_submission.head()
identity_hate_submission.head()
toxic_submission.head()
insult_submission.head()
threat_submission.head()
severe_toxic_submission.head()
submission = threat_submission.copy()

submission['toxic'] = toxic_submission['toxic']

submission['severe_toxic'] = severe_toxic_submission['severe_toxic']

submission['obscene'] = obscene_submission['severe_toxic']

submission['threat'] = threat_submission['severe_toxic']

submission['insult'] = insult_submission['severe_toxic']

submission['identity_hate'] = identity_hate_submission['severe_toxic']

del submission['Unnamed: 0']

submission.head()
submission.to_csv('submission.csv', index=False)