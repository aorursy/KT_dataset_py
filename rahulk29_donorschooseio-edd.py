# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for drawing plots 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# The resources Data
resources_data = pd.read_csv("../input/Resources.csv")

# Donors Data
donors_data = pd.read_csv("../input/Donors.csv")

# Projects Data
projects_data  = pd.read_csv("../input/Projects.csv")
# Donation Data
donations_data = pd.read_csv("../input/Donations.csv")
# Teachers Data
teachers_data = pd.read_csv("../input/Teachers.csv")
# Donors Data
donors_data.head()
donors_data.groupby("Donor State").count().plot(y=["Donor ID"],kind="Bar",figsize=(18,10))
yes_teachers = donors_data[donors_data["Donor Is Teacher"] == "Yes"]
no_teachers = donors_data[donors_data["Donor Is Teacher"] == "No"]
print(len(yes_teachers),len(no_teachers),donors_data.shape)
x = plt.bar(["yes","no"],[len(yes_teachers),len(no_teachers)],color = ['red', 'green'],width=0.5)
# naming the x-axis
plt.xlabel('Donors is Teacher')
# naming the y-axis
plt.ylabel('No.of Donors')
# plot title
plt.title('Donors is Teacher')
 
# function to show the plot
plt.show()
projects_data.head()
projects_data.groupby("Project Subject Category Tree").count().plot(y=["Project ID"],kind="Bar",color=["red"],figsize=(15,9),title="Total No.of projects with respect to subject")
sub_categories = projects_data["Project Subject Subcategory Tree"].value_counts()[0:35].plot(kind="Bar",figsize=(12,8),title="Top 35 Project Subject Sub Category ")

projects_data.groupby("Project Grade Level Category").count().plot(y=["Project ID"],kind="Bar",figsize=(12,7),title="Type of Grades applying for Donation")
projects_data["Project Resource Category"].value_counts().plot(kind="BAR",figsize=(12,7),title="Resoucres request from Projects")
projects_data["Project Current Status"].value_counts().plot(kind="BAR",figsize=(12,7))

school_data = pd.read_csv("../input/Schools.csv")
school_data.head()
school_data["School Metro Type"].value_counts().plot(kind="bar",title="Number of  Schools Types  applying the Projects")
school_data["School City"].value_counts()[0:35].plot(kind="bar",figsize=(12,9),title="Projects applications from different cities")