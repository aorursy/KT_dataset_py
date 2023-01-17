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
pd.read_csv('../input/clicks_train.csv').head()
print (pd.read_csv('../input/clicks_train.csv').columns)

print (pd.read_csv('../input/documents_categories.csv').columns)

print (pd.read_csv('../input/documents_entities.csv').columns)

print (pd.read_csv('../input/documents_meta.csv').columns)

print (pd.read_csv('../input/documents_topics.csv').columns)

print (pd.read_csv('../input/events.csv').columns)

print (pd.read_csv('../input/page_views_sample.csv').columns)

print (pd.read_csv('../input/promoted_content.csv').columns)

print (pd.read_csv('../input/sample_submission.csv').columns)
df = pd.read_csv('../input/clicks_train.csv')
pd.summary(df)