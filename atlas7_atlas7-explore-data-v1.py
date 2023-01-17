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
df_train = pd.read_csv('../input/en_train.csv')
print(repr(df_train.head(20)))
df_train.columns = ["sentence_id", "token_id", "token_class", "before", "after"]

print(repr(df_train.head(20)))
df_test = pd.read_csv('../input/en_test.csv')
print(repr(df_test.head(20)))
df_sample_submission = pd.read_csv('../input/en_sample_submission.csv')
print(repr(df_sample_submission.head(20)))
print("Unique sentences: {:,d}".format(df_train.sentence_id.unique().size))

print("Unique token classes: {:,d}".format(df_train.token_class.unique().size))

print("Unique before: {:,d}".format(df_train.before.unique().size))

print("Unique after: {:,d}".format(df_train.after.unique().size))
print('sentences per class...')

print('======================')

print(repr(df_train.groupby(['token_class'])['sentence_id'].count()))
df_train[df_train['token_class'] == 'ADDRESS'].head(5)
def peek_tokens_by_class(token_classes, view_x):

    for token_class in token_classes:

        print(df_train[df_train['token_class'] == token_class].head(view_x))
# Run it! Let's peak 10 samples from each token_class.

peek_tokens_by_class(df_train.token_class.unique(), 10)