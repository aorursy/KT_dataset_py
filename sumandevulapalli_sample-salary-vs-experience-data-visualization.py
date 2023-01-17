import  numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
salary_data = pd.read_csv("../input/salary/Salary.csv")

salary_data.head()
sns.lineplot(x = salary_data["YearsExperience"] , y = salary_data["Salary"])
sns.regplot(x = salary_data["YearsExperience"] , y = salary_data["Salary"])
sns.scatterplot(x = salary_data["YearsExperience"] , y = salary_data["Salary"])