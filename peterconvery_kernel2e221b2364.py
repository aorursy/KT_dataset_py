import pandas as pd
import os
print(os.listdir("../input"))
my_data = pd.read_csv('../input/covid19-global-forecasting-week-2/submission.csv')
print(my_data)
my_data.to_csv("My_submittion.csv",index=False)
