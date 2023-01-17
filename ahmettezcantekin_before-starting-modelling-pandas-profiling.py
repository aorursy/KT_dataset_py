#Importing the necessary libraries

import pandas_profiling as pp

import pandas as pd
#read the train and test data

df_train = pd.read_csv('../input/titanic/train.csv')

df_test  = pd.read_csv('../input/titanic/test.csv')
%%time

pp.ProfileReport(df_train)
#save the results as html file!

prof_report=pp.ProfileReport(df_train)

prof_report.to_file(output_file='profiling_report.html')
%%time

#save the minimal results as html file!

min_prof_report=pp.ProfileReport(df_train, minimal=True)

min_prof_report.to_file(output_file='min_profiling_report.html')