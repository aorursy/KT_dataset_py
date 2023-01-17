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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")
data.columns
data.head()
# We can drop Serial No. column

data = data.drop(["Serial No."], axis = 1)
print("Total Number of records {0}".format(data.shape[0]))
sns.heatmap(data.corr())
# plotting pair plot

sns.pairplot(data=data[["GRE Score", "TOEFL Score", "CGPA", "Chance of Admit "]])
# Checking the average change of admit for the non-research applicant

data[data["Research"] == 0]["Chance of Admit "].mean()

# Checking the average chance of admit for the research applicant

data[data["Research"] == 1]["Chance of Admit "].mean()
# Checking the average Chance of admit for GRE > 310

gre_bar = data[data["GRE Score"] > 315]

gre_bar[gre_bar["Research"] == 1]["Chance of Admit "].mean()
gre_bar[gre_bar["Research"] == 0]["Chance of Admit "].mean()
def gre_bar_315(x):

    if x > 315:

        return 1

    else:

        return 0
data["gre_bar_315"] = data["GRE Score"].apply(gre_bar_310)
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',

       'LOR ', 'CGPA', 'Research', "gre_bar_315"]
from sklearn.ensemble import RandomForestRegressor

train = data[:400]

test = data[400:]

x_train = train[features]

y_train = train['Chance of Admit ']

x_test = test[features]

y_test = test['Chance of Admit ']
rf = RandomForestRegressor()

rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_pred, y_test)

# print(np.sqrt(mean_squared_error(y_pred, y_test)))
print(rf.score(x_test, y_test))
print(np.sqrt(mean_squared_error(y_pred, y_test)))
feature_names = x_train.columns

importance_frame = pd.DataFrame()

importance_frame['Features'] = x_train.columns

importance_frame['Importance'] = rf.feature_importances_

importance_frame = importance_frame.sort_values(by=['Importance'], ascending=False)

importance_frame