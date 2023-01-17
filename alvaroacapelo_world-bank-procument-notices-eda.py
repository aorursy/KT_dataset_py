# import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns; sns.set()
import datetime as DT
# load data
proc_df = pd.read_csv("../input/procurement-notices.csv")
# inspect top of data
proc_df.head()
# get general info about data
proc_df.info()
# tidying columns names up: convert " " to "_" + lowercase letters
proc_df.columns = proc_df.columns.str.replace(" ", "_").str.lower()
# converting Publication Date and Deadline Date as datetime
proc_df.loc[:, "publication_date"] = pd.to_datetime(proc_df.publication_date)
proc_df.loc[:, "deadline_date"] = pd.to_datetime(proc_df.deadline_date)
proc_df.head(10)
# data where due date was assigned
due_null = proc_df.deadline_date.isna()
due_before_today = proc_df.deadline_date < pd.Timestamp.today()
due_after_today = proc_df.deadline_date > pd.Timestamp.today()
# currently out: due date later than today or not assigned due date
# NaT may mean project is still out
print("total entries:         ", proc_df.id.count())
print("due date not assigned: ", due_null.sum())
print("due date before today: ", due_before_today.sum())
print("due date after today:  ", due_after_today.sum())
current = proc_df[due_null | due_after_today]
current.head(10)
# distribution by country
curr_by_country = current[['country_name', 'project_id']].groupby('country_name').count().reset_index()

# sort resulting DF in descending order for visualization
curr_by_country.sort_values('project_id', axis=0, ascending=False).head(10)
# group by deadline_date where due days are after today
due_deadlines = current[due_after_today].groupby('deadline_date')['project_id'].count()
due_deadlines = due_deadlines.reset_index()

# pplot graph
due_deadlines.plot('deadline_date','project_id', color='blue')
# for project without due date, distribution of days since publication
not_assigned = proc_df[due_null].loc[:, ['publication_date', 'country_name']]
not_assigned['days_published'] = (pd.Timestamp.today() - not_assigned['publication_date'])
not_assigned.sort_values('days_published', axis=0, ascending=False).head(10)
