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
tfidf_ridge = pd.read_csv("/kaggle/input/tf-idf-ridge-regression-stacking/Submission.csv")

ent_emb = pd.read_csv("/kaggle/input/entity-embeddings-042868/ENTITY_EMBEDDINGS_0.42868.csv")

ftrl = pd.read_csv("/kaggle/input/guess-my-price-ftrl/submission.csv")

submission = pd.DataFrame(tfidf_ridge['train_id'])

submission['price'] = tfidf_ridge['price']*0.5 + ent_emb['price']*0.3 + ftrl['price']*0.2

submission.to_csv("submission.csv", index = False)

submission.head()