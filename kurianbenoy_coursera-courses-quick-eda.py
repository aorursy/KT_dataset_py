import pandas as pd

from pandas_profiling import ProfileReport
courses = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')

courses.head()
courses_profile = ProfileReport(courses)
courses_profile.to_widgets()
courses_profile.to_notebook_iframe()