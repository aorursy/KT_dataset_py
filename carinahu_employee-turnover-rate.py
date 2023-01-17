import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Read Data

data = pd.read_csv("../input/HR_comma_sep.csv")

data.head()

#check data info, you can find out whether there are null values exist in each column

data.info()
missing_data = data.isnull().sum().sort_values(ascending=False)

missing_data
data.head()
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt



left = len(data[data["left"]==1])

stay = len(data[data["left"]==0])



#We could use matplotlib to plot it 

values = [left,stay]

labels = ["Left","Stay"]

colors = ['lightgreen','lightblue']

explode = [0.05,0]

fig = plt.figure(figsize = (4,4))

pie = plt.pie(values, labels = labels, colors = colors, explode = explode, autopct = '%4.2f%%')
#Could also use pandas plot it

left_pd = pd.DataFrame({"Index":["Left","Stay"], "Value":[left,stay]})

sum_pd = left_pd.groupby("Index").sum()

pie = sum_pd.plot(kind="pie", figsize = (4,4), explode = explode, subplots = 'True', colormap="Pastel1", autopct='%3.2f%%')
#rename 'sales' to 'dept' use rename function

data = data.rename(columns = {'sales': 'dept'})



#though department is a string column, it cannot be quantified, no need to convert to numeric

#but we shold not forget this column

dept_dic = {}

#get unique department names

dept_name = data['dept'].unique().tolist()

#assign unique id to every department

dept_id = np.arange(1,len(dept_name)+1)

for i,dept in enumerate(dept_name):

    dept_dic[dept] = dept_id[i]

    

data['dept_id'] = data['dept'].map(dept_dic)

data.head()
#convert salary to numeric value

salary_dic = {'low': 1, 'medium':2, 'high':3}

data['salary'] = data['salary'].map(salary_dic)

data.head()
#correlation matrix

#below codes are used to pick numeric cols only

#num_cols = data._get_numeric_data().columns

#num_data = data[num_cols]



corrmat = data.corr()

corrmat
f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmin = -1, vmax = 1, square=True, annot=True, fmt='.2f');
#below codes are used for a mask to cover upper right part

mask = np.zeros_like(corrmat)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 9))

with sns.axes_style("white"):

    ax = sns.heatmap(corrmat, vmin = -1, vmax = 1, square=True, annot=True, mask=mask);



corr_list = pd.DataFrame(corrmat['left'].sort_values().drop('left'))

corr_list
#add mapping for left and stay

left_dic = {1:'left', 0:'stay'}

data['left_str'] = data['left'].map(left_dic)



#define function for evaluation classification

def eva_group(x):

    for i in np.arange(1,11):

        if x > (i-1)/10 and x <= i/10:

            return i



data['eva_group_int'] = data['last_evaluation'].apply(lambda x: eva_group(x))

eva_data = data[['eva_group_int','left_str','left']].groupby(['eva_group_int','left_str']).count()

eva_data = eva_data.reset_index()

eva_data
#as no employee left with evaluation 4, we fill it manually

eva_data.loc[13] = [4,'left',0]

eva_data = eva_data.sort_values(by='eva_group_int')

eva_data
eva_data_left = eva_data[eva_data['left_str'] == 'left']

eva_data_stay = eva_data[eva_data['left_str'] == 'stay']

#count of stay, for horizontal and vertical

eva_stay_h = eva_data_stay['eva_group_int'].tolist()

eva_stay_v = eva_data_stay['left'].tolist()

#count of left, for horizontal and vertical

eva_left_h = eva_data_left['eva_group_int'].tolist()

eva_left_v = eva_data_left['left'].tolist()



#add them to the same bar, to make it easy to see how may employees left in each evaluation group

f, ax = plt.subplots(figsize=(10,5))

ax.bar(eva_stay_h, eva_stay_v, label='stay', bottom=eva_left_v)

ax.bar(eva_left_h, eva_left_v, label='left')

plt.legend()

plt.show()
#calculate according to z-score

eva_mean = data['last_evaluation'].mean()

eva_std = data['last_evaluation'].std()

data['eva_label'] = (data['last_evaluation'] - eva_mean)/eva_std



def eva_label(x):

    if x>1:

        return 'good'

    else:

        return 'normal'

    

data['eva_label'] = data['eva_label'].apply(lambda x: eva_label(x))

eva_data_zscore = data[['eva_label','left_str','left']].groupby(['eva_label','left_str']).count().reset_index()

eva_data_zscore
#compare of two definitions

print(eva_data_zscore)

print("=======================================")

print(eva_data)
#left & stay over the time spend in company

f, ax = plt.subplots(figsize=(12,7))

time_data = data[['time_spend_company','left_str','left']].groupby(['time_spend_company','left_str']).count().reset_index()

sns.barplot(x='time_spend_company', y='left', hue='left_str', data=time_data)

plt.legend(loc='upper right')
def pre_group(x):

    if x<5:

        return 'prema'



#we only label those good employees who left

data['pre_label'] = data[(data['left_str']=='left') & (data['eva_label']=='good')]['time_spend_company'].apply(lambda x: pre_group(x))

data['pre_label'] = data['pre_label'].fillna('z')

data['pre_label'].value_counts()
#num_cols = data._get_numeric_data().columns

data['total_label'] = data['left_str']+":"+data['eva_label']+","+data['pre_label']

data.groupby(['total_label']).mean().transpose().drop(['left','eva_group_int','dept_id'])
#good employee

group1_data = data[data['eva_label'] == 'good']

group1_corr = group1_data.corr()

group1 = pd.DataFrame(group1_corr['left'].drop(['left','dept_id']).sort_values())

group1
#normal employee

group2_data = data[data['eva_label'] == 'normal']

group2_corr = group2_data.corr()

group2 = pd.DataFrame(group2_corr['left'].drop(['left','dept_id']).sort_values())

group2
#how the left & stay employees spred over the department?

#dept_data_left = data[['dept_n','dept','left_str','left']].groupby(['dept_n','dept','left_str']).count().reset_index()

f, ax = plt.subplots(figsize=(10,6))

#sns.barplot(x="dept", y="left", hue='left_str', data=dept_data_left)

data.dept.value_counts().plot(kind='bar')

data[data['left_str']=='left'].dept.value_counts().plot(kind='bar', color='b')
#how the good & left employees spred over the department?

f, ax = plt.subplots(figsize=(12,6))

#employee over departments

data.dept.value_counts().plot(kind='bar')

#good employees over departments

data[data['eva_label']=='good'].dept.value_counts().plot(kind='bar', color='b')

#good & left employees over departments

data[(data['eva_label']=='good')&(data['left_str']=='left')].dept.value_counts().plot(kind='bar', color='r')

plt.legend({'all employees':'gb', 'left':'b', 'good & left':'r'})
#percentage of group 1 over left employee in each dept

dept_group1 = data[(data['eva_label']=='good')&(data['left_str']=='left')].groupby(['dept']).count()

dept_total = data.groupby(['dept']).count()

dept_left = data[(data['left_str']=='left')].groupby(['dept']).count()

group1_perc = pd.DataFrame(dept_group1['left']/dept_total['left']).reset_index().sort_values(by='left',ascending=False)

f, ax = plt.subplots(figsize=(10,5))

sns.barplot(x='dept',y='left', data=group1_perc.sort_values(by='left'))
#mean satisfaction over departments for good employees

dept_mean_data = data.groupby(['dept']).mean()

dept_mean_cols = dept_mean_data.columns.unique().tolist()

dept_mean_cols = dept_mean_cols[0:9]



for col in dept_mean_cols:

    dept_mean_data[col].sort_values().plot(kind='bar')

    plt.ylabel(col)

    plt.show()
g = sns.FacetGrid(data,col='left')

g.map(sns.boxplot, 'time_spend_company')
g