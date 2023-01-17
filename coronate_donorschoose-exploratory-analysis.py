
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import datetime
import matplotlib.patches as mpatches


# Filepath to main training dataset.
train_path = '../input/train.csv'
test_path = '../input/test.csv'

# Read data and store in DataFrame.
train_data = pd.read_csv(train_path, sep=',', date_parser="project_submitted_datetime")
train_data.head()
train_data.teacher_prefix.unique()
prefixAceptance = train_data[["teacher_prefix","project_is_approved"]].groupby("teacher_prefix").mean()
prefixAceptance["prefix"] = prefixAceptance.index

genderDictionary = {"Ms.": "Female", "Mrs.":"Female", "Mr.":"Male", "Teacher":"Neutral", "Dr.":"Neutral", np.nan:"Neutral"  }
train_data["gender"] = train_data.teacher_prefix.map( genderDictionary )
genderAceptance = train_data[["gender","project_is_approved"]].groupby("gender").mean()

titleDictionary = {"Ms.": "Na", "Mrs.":"Na", "Mr.":"Na", "Teacher":"Teacher", "Dr.":"Dr.", np.nan:"Na"  }
train_data["title"] = train_data.teacher_prefix.map( titleDictionary )
titleAceptance = train_data[["title","project_is_approved"]].groupby("title").mean()
fig  = plt.figure(figsize=(20,7))
ax1 = plt.subplot(1,3,1)
sns.barplot(x = prefixAceptance.index, y =prefixAceptance.project_is_approved )
ax2 = plt.subplot(1,3,2)
sns.barplot(x = genderAceptance.index, y =genderAceptance.project_is_approved )
ax3 = plt.subplot(1,3,3)
sns.barplot(x = titleAceptance.index, y =titleAceptance.project_is_approved )

stateAceptance = train_data[["school_state","project_is_approved"]].groupby("school_state").mean()
stateAceptance["state"] = stateAceptance.index

fig = plt.figure( figsize=(20,10))
plt.title("Accpetance rate per State")
sns.barplot(x = stateAceptance.index, y =stateAceptance.project_is_approved )

stateAceptance = train_data[["school_state","project_is_approved"]].groupby("school_state").count()
stateAceptance["state"] = stateAceptance.index
stateAceptance = stateAceptance.sort_values( "project_is_approved", ascending=False)

fig = plt.figure( figsize=(20,10))
plt.title("Number of applications per State")
sns.barplot(x = stateAceptance.index, y =stateAceptance.project_is_approved )

train_data["project_submitted_datetime"] = pd.to_datetime( train_data.project_submitted_datetime )
train_data["date"] = train_data.project_submitted_datetime.apply( lambda x: x.date() )
train_data["month"] = train_data.project_submitted_datetime.apply( lambda x: x.month )
train_data["weekday"] = train_data.project_submitted_datetime.apply( lambda x: x.weekday )
train_data["year"] = train_data.project_submitted_datetime.apply( lambda x: x.year )

dateAcceptance = train_data[["date","project_is_approved"]].groupby("date").mean()
dateAcceptanceCount = train_data[["date","project_is_approved"]].groupby("date").count() 

fig = plt.figure( figsize=(20,6))
plt.title("Acceptance rate per date and number of applications")
ax1 = plt.subplot(1,1,1)
plt.plot(dateAcceptance  )
ax2 = plt.subplot(1,1,1)
ax2 = ax1.twinx()
plt.plot(dateAcceptanceCount, "red"  )
red_patch = mpatches.Patch(color='red', label='Total number of applications')
blue_patch = mpatches.Patch(color='blue', label='Acceptance rate')
plt.legend(handles=[blue_patch, red_patch])
monthAcceptance = train_data[["month","project_is_approved"]].groupby("month").mean()
monthAcceptanceCount = train_data[["month","project_is_approved"]].groupby("month").count() 

fig = plt.figure( figsize=(20,6))

plt.title("Acceptance trend")

ax1 = plt.subplot(1,1,1)
plt.plot(monthAcceptance  )
ax2 = plt.subplot(1,1,1)
ax2 = ax1.twinx()
plt.plot(monthAcceptanceCount, "red"  )

red_patch = mpatches.Patch(color='red', label='Total number of applications')
blue_patch = mpatches.Patch(color='blue', label='Acceptance rate')
plt.legend(handles=[blue_patch, red_patch])

postedAcceptance= train_data[["teacher_number_of_previously_posted_projects","project_is_approved"]].groupby("teacher_number_of_previously_posted_projects").mean()
postedAcceptanceCount = train_data[["teacher_number_of_previously_posted_projects","project_is_approved"]].groupby("teacher_number_of_previously_posted_projects").count() 
postedAcceptanceCount = postedAcceptanceCount.rename( columns= {"project_is_approved": "applications_count"})

postedAcceptance =  postedAcceptance.merge( postedAcceptanceCount, right_index=True, left_index= True)
postedAcceptance = postedAcceptance.sort_index( ascending= True)

postedAcceptance50 = postedAcceptance.head(50)



fig = plt.figure( figsize=(20,10))
fig.suptitle( "Distribution: acceptance rate and number of applications per number of past posted projects", fontsize = 20)

ax1 = plt.subplot(2,1,1)
plt.bar( postedAcceptance50.index, postedAcceptance50.project_is_approved,  color='g') 
ax2 = plt.subplot(2,1,1)
ax2 = ax1.twinx()
plt.bar( postedAcceptance50.index, postedAcceptance50.applications_count, color = 'orange' )
orange_patch = mpatches.Patch(color='orange', label='Count number of records per previously_posted_projects')
green_patch = mpatches.Patch(color='green', label='Acceptance rate')
plt.legend(handles=[green_patch, orange_patch])
postedAcceptance = postedAcceptance.sort_index( ascending= False)
postedAcceptance50 = postedAcceptance.head(50)

ax3 = plt.subplot(2,1,2)
plt.bar( postedAcceptance50.index, postedAcceptance50.project_is_approved,  color='g') 
ax4 = plt.subplot(2,1,2)
ax4 = ax3.twinx()
plt.bar( postedAcceptance50.index, postedAcceptance50.applications_count, color = 'orange' )
orange_patch = mpatches.Patch(color='orange', label='Count number of records per previously_posted_projects')
green_patch = mpatches.Patch(color='green', label='Acceptance rate')
plt.legend(handles=[green_patch, orange_patch])


categoryAceptance = train_data[["project_grade_category","project_is_approved"]].groupby("project_grade_category").mean()
categoryAceptance = categoryAceptance.sort_values( "project_is_approved", ascending=False)

fig = plt.figure( figsize=(20,10))
fig.suptitle( "Distribution acceptance rate and number of applications per school grade ", fontsize = 25)
plt.subplot(2,1,1)
sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )

categoryAceptance = train_data[["project_grade_category","project_is_approved"]].groupby("project_grade_category").sum()

plt.subplot(2,1,2)
sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )


train_data.columnsumns
categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").mean()
categoryAceptance = categoryAceptance.sort_values( "project_is_approved", ascending=False)
categoryAceptance = categoryAceptance.head(15)
categoryAceptanceindex = categoryAceptance.index

fig = plt.figure( figsize=(50,20))
fig.suptitle( "Distribution acceptance rate and number of applications per category ", fontsize = 50)
ax1 = plt.subplot(2,1,1)
ax1.set_title( "Acceptance rate per categort", fontsize = 40)

sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )

categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").sum()
categoryAceptance = categoryAceptance.loc[categoryAceptanceindex]

ax2 = plt.subplot(2,1,2)
ax2.set_title( "Number of records per category", fontsize = 40)

sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )


categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").sum()
categoryAceptance = categoryAceptance.sort_values( "project_is_approved", ascending=False)
categoryAceptance = categoryAceptance.head(15)
categoryAceptanceindex = categoryAceptance.index

categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").mean()
categoryAceptance = categoryAceptance.loc[categoryAceptanceindex]

fig = plt.figure( figsize=(50,20))
fig.suptitle( "Distribution acceptance rate and number of applications per category ", fontsize = 50)
plt.title("Category Accpetance")
ax1 = plt.subplot(2,1,1)
ax1.set_title( "Acceptance rate per category", fontsize = 40)

sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )

categoryAceptance = train_data[["project_subject_categories","project_is_approved"]].groupby("project_subject_categories").sum()
categoryAceptance = categoryAceptance.loc[categoryAceptanceindex]

ax2 = plt.subplot(2,1,2)
ax2.set_title( "Number of records per category", fontsize = 40)
sns.barplot(x = categoryAceptance.index, y =categoryAceptance.project_is_approved )


