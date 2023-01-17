%matplotlib inline
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
salaries=pd.read_csv("../input/Salaries.csv")
salaries.columns
salaries.hist('TotalPay', by = 'Year', sharex = True, sharey = True)
salaries.Status.value_counts()
salaries.JobTitle.value_counts()[:10]