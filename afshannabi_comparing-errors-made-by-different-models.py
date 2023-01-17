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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
predictions = pd.read_csv("../input/compare-errors-made-by-different-models/error_analysis.csv")

predictions.head()
sns.heatmap(predictions, cmap = "pink", cbar=False,  yticklabels=False)

plt.title('Comparison of Predictions made by 3 models')

plt.show()
#sort predictions

predictions = predictions.sort_values(by=['True Label', 'SVM_prediction', 'XGBoost_prediction', 'MLP_prediction' ])

predictions.head()
#make heatmap after sorting

sns.heatmap(predictions, cmap = "pink", cbar=False,  yticklabels=False)

plt.title('Comparison of Predictions made by 3 models')

plt.show()
#Keep only those samples for which at least one of the models makes an incorrect prediction

errors = predictions.loc[(predictions['True Label'] != predictions['SVM_prediction']) | 

                         (predictions['True Label'] != predictions['XGBoost_prediction']) | 

                         (predictions['True Label'] != predictions['MLP_prediction'])]
sns.heatmap(errors, cmap = "pink", cbar=False,  yticklabels=False)

plt.title('Incorrect predictions comparison for 3 models')

plt.show()