import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
data.describe()
data[data.isnull() == 'True'].count()

#Number of males and females in the dataset
sns.set_style('whitegrid')
sns.countplot(x=data['gender'])
plt.title('Number of males and females in the dataset')

#Race/Ethnicity Pie plot
fig, ax1 = plt.subplots()
x = data.groupby(by=data['race/ethnicity']).count()['lunch']
ax1.pie(x = x, labels = x.index, autopct='%1.1f%%')
ax1.axis('equal')
ax1.set_title("Percentage of students belonging to different races")
plt.rcParams['font.size'] = 12.0
plt.show()
#Number of students who prepared who completed a test prep course
sns.countplot(x = data['test preparation course'])
male_means = (int(np.mean(data[data['gender'] == 'male']['math score'])),
              int(np.mean(data[data['gender'] == 'male']['reading score'])),
              int(np.mean(data[data['gender'] == 'male']['writing score'])))

female_means = (int(np.mean(data[data['gender'] == 'female']['math score'])), 
               int(np.mean(data[data['gender'] == 'female']['reading score'])),
               int(np.mean(data[data['gender'] == 'female']['writing score'])))

ind = np.arange(3)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, male_means, width,
                color='SkyBlue', label='Male')
rects2 = ax.bar(ind + width/2, female_means, width,
                color='IndianRed', label='Female')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Scores')
ax.set_title('Average scores by subject and gender')
ax.set_xticks(ind)
ax.set_xticklabels(('Math', 'Reading', 'Writing'))
ax.legend(loc=3)


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(rects1, "center")
autolabel(rects2, "center")

plt.show()
prep_means = (int(np.mean(data[data['test preparation course'] == 'completed']['math score'])),
              int(np.mean(data[data['test preparation course'] == 'completed']['reading score'])),
              int(np.mean(data[data['test preparation course'] == 'completed']['writing score'])))

noprep_means = (int(np.mean(data[data['test preparation course'] == 'none']['math score'])), 
               int(np.mean(data[data['test preparation course'] == 'none']['reading score'])),
               int(np.mean(data[data['test preparation course'] == 'none']['writing score'])))

ind = np.arange(3)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, prep_means, width,
                color='SkyBlue', label='Prep course')
rects2 = ax.bar(ind + width/2, noprep_means, width,
                color='IndianRed', label='No prep course')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Scores')
ax.set_title('Average scores by subject and preparation course')
ax.set_xticks(ind)
ax.set_xticklabels(('Math', 'Reading', 'Writing'))
ax.legend(loc=3)

autolabel(rects1, "center")
autolabel(rects2, "center")

plt.show()
prep_means = (int(np.mean(data[(data['test preparation course'] == 'completed') & (data['gender'] == 'male')]['math score'])),
              int(np.mean(data[(data['test preparation course'] == 'completed') & (data['gender'] == 'female')]['math score'])))

noprep_means = (int(np.mean(data[(data['test preparation course'] == 'none') & (data['gender'] == 'male')]['math score'])),
              int(np.mean(data[(data['test preparation course'] == 'none') & (data['gender'] == 'female')]['math score'])))

ind = np.arange(2)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, prep_means, width,
                color='SkyBlue', label='Prep course')
rects2 = ax.bar(ind + width/2, noprep_means, width,
                color='IndianRed', label='No prep course')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Scores')
ax.set_title('Average math scores by subject and preparation course')
ax.set_xticks(ind)
ax.set_xticklabels(('Male', 'Female'))
ax.legend(loc=3)

autolabel(rects1, "center")
autolabel(rects2, "center")

plt.show()
prep_means = (int(np.mean(data[(data['test preparation course'] == 'completed') & (data['gender'] == 'male')]['reading score'])),
              int(np.mean(data[(data['test preparation course'] == 'completed') & (data['gender'] == 'female')]['reading score'])))

noprep_means = (int(np.mean(data[(data['test preparation course'] == 'none') & (data['gender'] == 'male')]['reading score'])),
              int(np.mean(data[(data['test preparation course'] == 'none') & (data['gender'] == 'female')]['reading score'])))

ind = np.arange(2)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, prep_means, width,
                color='SkyBlue', label='Prep course')
rects2 = ax.bar(ind + width/2, noprep_means, width,
                color='IndianRed', label='No prep course')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Scores')
ax.set_title('Average reading scores by gender and preparation course')
ax.set_xticks(ind)
ax.set_xticklabels(('Male', 'Female'))
ax.legend(loc=3)

autolabel(rects1, "center")
autolabel(rects2, "center")

plt.show()
prep_means = (int(np.mean(data[(data['test preparation course'] == 'completed') & (data['gender'] == 'male')]['writing score'])),
              int(np.mean(data[(data['test preparation course'] == 'completed') & (data['gender'] == 'female')]['writing score'])))

noprep_means = (int(np.mean(data[(data['test preparation course'] == 'none') & (data['gender'] == 'male')]['writing score'])),
              int(np.mean(data[(data['test preparation course'] == 'none') & (data['gender'] == 'female')]['writing score'])))

ind = np.arange(2)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, prep_means, width,
                color='SkyBlue', label='Prep course')
rects2 = ax.bar(ind + width/2, noprep_means, width,
                color='IndianRed', label='No prep course')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Scores')
ax.set_title('Average writing scores by subject and preparation course')
ax.set_xticks(ind)
ax.set_xticklabels(('Male', 'Female'))
ax.legend(loc=3)

autolabel(rects1, "center")
autolabel(rects2, "center")

plt.show()
parent_edu = round(data.groupby(by = data['parental level of education']).mean(),1)
parent_edu
x = list(parent_edu.index)
y = round(parent_edu.mean(axis=1),1)

fig, ax = plt.subplots(figsize=(12,6))

rects = ax.bar(x, y)

ax.set_ylabel('Average Scores')
ax.set_title('Average scores by parent\'s education level')
ax.set_xticklabels((x))

autolabel(rects, 'center')

plt.show()
x = parent_edu.index
y1 = parent_edu['math score']
y2 = parent_edu['reading score']
y3 = parent_edu['writing score']

ind = np.arange(6)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, y1, width/2,
                color='SkyBlue', label='Math')
rects2 = ax.bar(ind, y2, width/2,
                color='IndianRed', label='Reading')
rects3 = ax.bar(ind + width/2, y3, width/2,
                color='Green', label='Writing')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Scores')
ax.set_title('Average scores by subject and parent\'s education level')
ax.set_xticks(ind)
ax.set_xticklabels((x))
ax.legend(loc=3)

autolabel(rects1, "center")
autolabel(rects2, "center")
autolabel(rects3, "center")

plt.show()
standard_means = (int(np.mean(data[data['lunch'] == 'standard']['math score'])),
              int(np.mean(data[data['lunch'] == 'standard']['reading score'])),
              int(np.mean(data[data['lunch'] == 'standard']['writing score'])))

reduced_means = (int(np.mean(data[data['lunch'] == 'free/reduced']['math score'])), 
               int(np.mean(data[data['lunch'] == 'free/reduced']['reading score'])),
               int(np.mean(data[data['lunch'] == 'free/reduced']['writing score'])))

ind = np.arange(3)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, standard_means, width,
                color='SkyBlue', label='Standard Lunch')
rects2 = ax.bar(ind + width/2, reduced_means, width,
                color='IndianRed', label='Free/Reduced lunch')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Scores')
ax.set_title('Average scores by subject and lunch type at school')
ax.set_xticks(ind)
ax.set_xticklabels(('Math', 'Reading', 'Writing'))
ax.legend(loc=3)

autolabel(rects1, "center")
autolabel(rects2, "center")

plt.show()
x = ['Test prep', 'No test prep']
y = [data[(data['lunch'] == 'free/reduced') & (data['test preparation course'] == 'completed')]['lunch'].count(),
    data[(data['lunch'] == 'free/reduced') & (data['test preparation course'] == 'none')]['lunch'].count()]

fig, ax = plt.subplots(figsize=(12,6))

rects = ax.bar(x, y)

ax.set_ylabel('Count')
ax.set_title('Students studying in schools that provide free/reduced price lunch by test prep course')
ax.set_xticklabels((x))

autolabel(rects, 'center')

plt.show()
race = round(data.groupby(by = data['race/ethnicity']).mean(), 1)
race
x = list(race.index)
y = round(race.mean(axis=1),1)

fig, ax = plt.subplots(figsize=(12,6))

rects = ax.bar(x, y)

ax.set_ylabel('Average Scores')
ax.set_title('Average scores by race/ethnicity')
ax.set_xticklabels((x))

autolabel(rects, 'center')

plt.show()
x = race.index
y1 = race['math score']
y2 = race['reading score']
y3 = race['writing score']

ind = np.arange(len(x))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(ind - width/2, y1, width/2,
                color='SkyBlue', label='Math')
rects2 = ax.bar(ind, y2, width/2,
                color='IndianRed', label='Reading')
rects3 = ax.bar(ind + width/2, y3, width/2,
                color='Green', label='Writing')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Scores')
ax.set_title('Average scores by subject and parent\'s education level')
ax.set_xticks(ind)
ax.set_xticklabels((x))
ax.legend(loc=3)

autolabel(rects1, "center")
autolabel(rects2, "center")
autolabel(rects3, "center")

plt.show()
