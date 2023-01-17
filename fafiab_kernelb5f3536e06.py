# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input/output"))

OUTPUT_PATH = '../input/output/'

JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'

def load_data():

    print('Load train and test data')

    train = pd.read_csv(os.path.join(OUTPUT_PATH,'domain_ft.csv'))

#     test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')

    return train

def submit(preds):

    print('Prepare submission')

    submission = pd.read_csv(os.path.join(JIGSAW_PATH,'sample_submission.csv'), index_col='id')

    submission['prediction'] = preds

    submission.reset_index(drop=False, inplace=True)

    submission.to_csv('submission.csv', index=False)

    print('saved to ')

sub=load_data()

pred=sub['prediction'].values

submit(pred)

# Any results you write to the current directory are saved as output.