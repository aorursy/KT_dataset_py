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
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns







from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



from statsmodels.api import OLS
df = pd.read_csv("../input/chicago-public-health-statistics/public-health-statistics-selected-public-health-indicators-by-chicago-community-area.csv")

df.head(5)
df.dtypes
x = df[['Birth Rate',

       'General Fertility Rate', 'Low Birth Weight',

       'Prenatal Care Beginning in First Trimester', 'Preterm Births',

       'Teen Birth Rate', 'Assault (Homicide)', 'Breast cancer in females',

       'Cancer (All Sites)', 'Colorectal Cancer', 'Diabetes-related',

       'Firearm-related', 'Infant Mortality Rate', 'Lung Cancer',

       'Prostate Cancer in Males', 'Stroke (Cerebrovascular Disease)',

       'Childhood Blood Lead Level Screening', 'Childhood Lead Poisoning',

       'Gonorrhea in Females', 'Tuberculosis',

       'Below Poverty Level', 'Crowded Housing', 'Dependency',

       'No High School Diploma']]

x.dtypes

x = x.fillna(x.mean())
y = df[['Unemployment']]

y = y.fillna(y.mean())
sns.heatmap(x, cmap="Greens")

x_train, x_test, y_train, y_test = train_test_split(

    x, y, train_size = 0.8, test_size = 0.2, random_state=6)
lm = LinearRegression()



model = lm.fit(x_train, y_train)



y_predict= lm.predict(x_test)
print("Train score:")

print(lm.score(x_train, y_train))

plt.scatter(y_test, y_predict)

plt.plot(range(35), range(35))



plt.xlabel("Unemployment: $Y_i$")

plt.ylabel("Predicted Unemployment: $\hat{Y}_i$")

plt.title("Actual Unemployment vs Predicted Unemployment")



plt.show()
# Make predictions

expected = y_train

predicted = model.predict(x_train)

# Summarize the fit of the model

mse = np.mean((predicted-expected)**2)

print (model.intercept_, model.coef_, mse) 

print(model.score(x_train, y_train))