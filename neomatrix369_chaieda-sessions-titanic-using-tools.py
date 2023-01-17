## Currently the below does not work, raised an issue with the maintainer, see https://github.com/tkrabel/bamboolib/issues/23 

# !pip install -U bamboolib>=1.4.1

# import bamboolib as bam
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import warnings

warnings.filterwarnings('ignore')



# Import library and dataset

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid", font_scale=1.75)



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



!pip install -U pandas-profiling

import pandas_profiling as pp



!pip install -U ppscore

import ppscore as pps



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import math





import matplotlib.pyplot as plt



# prettify plots

plt.rcParams['figure.figsize'] = [20.0, 5.0]

sns.set_palette(sns.color_palette("muted"))

sns.set_style("ticks")



%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
training_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

print("Column count:", len(training_dataset.columns))

training_dataset.dtypes
training_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

training_dataset
pp.ProfileReport(training_dataset)
training_dataset.columns
sensible_columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Cabin', 'Embarked']
pp.ProfileReport(training_dataset[sensible_columns])
test_dataset = pd.read_csv('/kaggle/input/titanic/test.csv')

test_dataset
pp.ProfileReport(test_dataset)
sensible_columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Cabin', 'Embarked']
pp.ProfileReport(test_dataset[sensible_columns])
all_features_predictors_dataframe = pps.predictors(training_dataset, "Survived")

all_features_predictors_dataframe
plt.figure(figsize=(20,6))

sns.barplot(data=all_features_predictors_dataframe, x="x", y="ppscore")
all_features_matrix_dataframe = pps.matrix(training_dataset)

all_features_matrix_dataframe
all_features_matrix_dataframe = all_features_matrix_dataframe[['x', 'y', 'ppscore']]

all_features_matrix_dataframe = all_features_matrix_dataframe.pivot(columns="x", index="y", values="ppscore")

all_features_matrix_dataframe
plt.figure(figsize=(20,5))

sns.heatmap(all_features_matrix_dataframe, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
print(f'The training dataset has these columns: \n   {list(training_dataset.columns)}')
# without Ticket, PassengerId, and Name columns

sensible_columns=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
some_features_predictors_dataframe = pps.predictors(training_dataset[sensible_columns], "Survived")

some_features_predictors_dataframe
plt.figure(figsize=(20,5))

sns.barplot(data=some_features_predictors_dataframe, x="x", y="ppscore")
some_features_matrix_dataframe = pps.matrix(training_dataset[sensible_columns])

some_features_matrix_dataframe
some_features_matrix_dataframe = some_features_matrix_dataframe[['x', 'y', 'ppscore']]

some_features_matrix_dataframe = some_features_matrix_dataframe.pivot(columns="x", index="y", values="ppscore")

some_features_matrix_dataframe
plt.figure(figsize=(20,5))

sns.heatmap(some_features_matrix_dataframe, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)