import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pandas_profiling import ProfileReport # Generates EDA Report 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read data using pandas 

df = pd.read_csv("/kaggle/input/framingham-heart-study-dataset/framingham.csv")
# create eda report 

eda_report = ProfileReport(df, title="EDA of Framingham Heart Study Dataset")
# show eda report 

eda_report 
# generate report with full_width 

full_width_report = ProfileReport(df, title="EDA of Framingham Heart Study Dataset", html={'style': {'full_width':True}})
# show report 

full_width_report
# you can export this report as html 

eda_report.to_file("report.html")

# or you can export full_width_report as html 

full_width_report.to_file("report.html")