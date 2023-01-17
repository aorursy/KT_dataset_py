import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/retail-data-customer-summary-learn-pandas-basics/1_Sales.csv')
df
df.to_csv('1_Sales_output.csv',index=False)
studentdata = pd.read_excel('/kaggle/input/retail-data-customer-summary-learn-pandas-basics/1_StudentsScore.xlsx',sheet_name='data')
studentdata.head()
studentdata.to_excel('1_StudentsScore_output.xlsx',sheet_name='Sheet1')