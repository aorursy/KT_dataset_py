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
import statsmodels.formula.api as smf

def train_my_model(train_df):

    train_df.DefendersInTheBox[train_df.DefendersInTheBox.isna()] = train_df.DefendersInTheBox.mean()

    result=smf.ols("Yards ~ A + S + DefendersInTheBox + Distance", data=train_df).fit()

    return result
Mean_DefenderInTheBox = train_df.DefendersInTheBox.mean()
from kaggle.competitions import nflrush

env = nflrush.make_env()
# Training data is in the competition dataset as usual

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

result = train_my_model(train_df)

def make_my_predictions(result_model, test_df, sample_prediction_df):

    if test_df.DefenderInTheBox.isna():

        test_df.DefenderInTheBox = Mean_DefenderInTheBox

    out_yard = result_modell.predict(test_df)

    num = int(out_yard//1) 

    sample_prediction_df.loc[:,:] = 0    

    sample_prediction_df.loc[:,f"Yards{num}":] = 1

    return 
for (test_df, sample_prediction_df) in env.iter_test():

    predictions_df = make_my_predictions(result_model, test_df, sample_prediction_df)

    env.predict(predictions_df)



# env.write_submission_file()