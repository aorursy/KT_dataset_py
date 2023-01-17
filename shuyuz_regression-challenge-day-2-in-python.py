# python3 



import numpy as np

import pandas as pd 

from pandas import read_csv



kaggle = read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv",encoding='latin-1')

stackOverflow = read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv",encoding='latin-1')
kaggle.head()
stackOverflow.head()
# has_compensation = 

kaggle.describe()
# your work goes here :)

stackOverflow = pd.read_csv("../input/so-survey-2017/survey_results_public.csv")
stackOverflow.head()
stackOverflow.dtypes
import matplotlib.pyplot as plt

import seaborn as sns

stackOverflow.describe()
# Plotting libraries

import seaborn as sns

from ggplot import *



x = stackOverflow['Respondent']

y = stackOverflow['Salary']



plt.scatter(x,y)
stackOverflow.dtypes.sample(10)

one_hot_encoded_so = pd.get_dummies(stackOverflow)
one_hot_encoded_so.head()
one_hot_encoded_so.dropna(axis=1, how='any')
import matplotlib.pyplot as plt



x = stackOverflow['Respondent'][:, np.newaxis]

y = stackOverflow['Salary']

plt.scatter(x,y)
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x,y)

y_pred = reg.predict(x)

plt.scatter(x, y)

plt.plot(x, y_pred, color='blue', linewidth=3)