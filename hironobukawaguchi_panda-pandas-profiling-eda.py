import pandas as pd

from pandas_profiling import ProfileReport
train = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')
profile = ProfileReport(train, title='Train Profiling Report')

# profile = ProfileReport(train, title='Titanic Train Minimal Profiling Report', minimal=True)
profile.to_file(output_file="ProfileReport_train.html")
profile.to_widgets()
profile.to_notebook_iframe()
test = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')
profile_test = ProfileReport(test, title='Test Profiling Report')

# profile_test = ProfileReport(train, title='Titanic Test Minimal Profiling Report', minimal=True)
profile_test.to_file(output_file="ProfileReport_test.html")
profile_test.to_widgets()
profile_test.to_notebook_iframe()