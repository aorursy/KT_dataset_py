# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Read dataset
student_data = pd.read_csv("../input/2016-FCC-New-Coders-Survey-Data.csv", low_memory=False)
student_data.describe()
# Calculate number of features, -1 to not calculate the target column
n_features = student_data.shape[1] - 1

# Calculate students with fulltime job after
n_fulljob = student_data[student_data['BootcampFullJobAfter'] == 1].shape[0]

# Calculate students without fulltime job after
n_nofulljob = student_data[student_data['BootcampFullJobAfter'] == 0].shape[0]

# Calculate number of students who tried to get a job
n_students = n_fulljob + n_nofulljob

# TODO: Calculate success rate for getting a fulltime job
job_rate = n_fulljob * 100.0 / n_students

# Print the results
print("Total number of students who tried to get a job: " + str(n_students))
print("Number of features: " + str(n_features))
print("Number of students who got a fulltime job: " + str(n_fulljob))
print("Number of students who failed to get a fulltime job: " + str(n_nofulljob))
print("Success rate for getting a fulltime job in %: " + str(job_rate))