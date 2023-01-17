import pandas as pd
import seaborn as sns
dftrain = pd.read_csv('../input/train.csv')
dftrain.head()
dftrain.shape
dfgendersub = pd.read_csv('../input/gender_submission.csv')
dfgendersub.head()
dfgendersub.shape
dftest = pd.read_csv('../input/test.csv')
dftest.head()
dftest.shape
dftrain.isnull().sum()
177/891
dftest.isnull().sum()
