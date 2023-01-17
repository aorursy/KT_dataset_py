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
employees = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
employees.head()
#Try it out.

# 1. Run the first line of code. 

# 2. Then run only the second line to see that the employees dataframe hasn't changed.

# 3. Then run the third and fourth line of code to see the new dataframe



employees.set_index("Attrition")

employees

employees = employees.set_index("Attrition")

employees
# Resetting the index is similar to changing the index

# Make sure you save the new dataframe to its own variable and use the function reset_index() to revert the index back to the original

employees = employees.reset_index()
# employees.shape provides how many rows and columns are in the dataframe in the format (rows, columns)

employees.shape
# Test it out

employees.iloc[0:3,0:4]
employees[employees.Attrition == "No"]
employees_select = employees[["Attrition", "Age", "DistanceFromHome", "WorkLifeBalance", "EnvironmentSatisfaction", "DailyRate",\

                              "YearsAtCompany", "YearsSinceLastPromotion"]]

employees_select
employees_select.groupby("Attrition").mean()
employees_attrition = employees[employees.DistanceFromHome <= 10.632911]

employees_attrition.groupby("Attrition").Attrition.size()["No"]/employees.groupby("Attrition").Attrition.size()["No"]
employees_attrition = employees[employees.DistanceFromHome >= 10.632911]

employees_attrition.groupby("Attrition").Attrition.size()["Yes"]/employees.groupby("Attrition").Attrition.size()["Yes"]
# Try this with another factor!
# Here is one example. plot.barh() is a horizontal bar plot

employees_select.groupby("Attrition").mean().iloc[:,0:4].sort_values("Attrition", ascending=False).plot.barh()