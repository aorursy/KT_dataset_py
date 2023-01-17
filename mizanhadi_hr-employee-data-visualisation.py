import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df=pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.describe()
labels_left=['did not leave','left']
sns.barplot(x='left',y='satisfaction_level',data=df,palette='Blues_d').set_title('satisfaction level vs left')
plt.xticks(np.arange(2), labels_left, rotation='horizontal')
plt.show()
#The above graph tells us that the employees who did not leave 
#were highly satisfied as compare to the employees who left.
sns.barplot(x='time_spend_company',y='satisfaction_level',data=df,palette='Blues_d').set_title('Satisfaction level vs No. of years spent in the company')
# The above graph tells us about the satisfacttion level of employees with
# respect to the number of years they spent in that company.
# The fresher employees i.e. employees which have spent 0-2 years in the
# company are the most satisfied. 
sns.barplot(x='time_spend_company',y='last_evaluation',data=df,palette='Blues_d').set_title('Last evaluation vs No. of years spent in the company')
# The above graph shows the performance of employees with respect to the
# the number of years spent in that company.
# The employee having an experience of 5 years have the best performance.
sns.barplot(y='promotion_last_5years',x='number_project',data=df,palette='Blues_d').set_title('Number of projects done vs Promotion in last 5 years')
# The above graph shows the promotion of employees with respect to number of projects they have done.
# The employees which have done 4 projects have had the most promotion in the last 5 years.
a=df['sales'].value_counts()
b=pd.DataFrame(a)
b.head()
plt.pie(b['sales'],labels=b.index,shadow=True,colors = ['#fc910d','#fcb13e','#239cd3','#1674b1','#ed6d50'],explode=[0.2,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
plt.title('Distribution of number of employees according to the different sectors')
plt.axis("equal")
fig1 = plt.gcf()
fig1.set_size_inches(6,6)
plt.show()
# The above graph show us the distribution of number of employees according to the 
# different sectors in that company.
# The 'Sales' sector has he highest number of employees
# followed by 'Technical' sector & 'Support' sector.
sns.barplot(x='average_montly_hours',y='salary',data=df,palette='Blues_d').set_title('Average monthly hours vs Salary')
# The above graph shows the salary of the employees with respect to average of number 
# of monthly hours spent.
# The graphs shows the salary is not affected by the number of hours spent in the company.
sns.barplot(x='time_spend_company',y='satisfaction_level',data=df,palette='Blues_d',hue='promotion_last_5years').set_title('Satisfaction level vs Time spent in company, with respect to promotion in last 5 years.')
# The above graphs shows the satisfaction level of employees with respect to number
# of hours spent in that company and also how the promotion has affected this.
# In general, the employees with a promotion in the last 5 years have a greater
# satisfaction level that those who didn't have a promotion.
# But this has not affected the freshers,that is the satisfaction level of 
# fresher employees is not affected by the promotion.
ax=sns.barplot(x='left',y='time_spend_company',data=df,palette='Blues_d').set_title('Number of people left/not left vs The number of years spent in company.')
plt.xticks(np.arange(2), labels_left, rotation='horizontal')
plt.show()
# This graph show the number of employees which have left or not left with respect to 
# number of years spent in that company.

g1=df['time_spend_company'].groupby([df['left']])
g2=g1.value_counts()
gd=pd.DataFrame(g2)
gd= gd.drop(0)
gd = gd.rename(columns={'time_spend_company': 'time_spend_company', 'time_spend_company': 'left_total'})
gd=gd.reset_index(level=[0,1])
gd
sns.barplot(x='time_spend_company',y='left_total',data=gd,palette='Blues_d').set_title('The people who left vs The number of years they spent in company')
# This graph shows how many people have left with respect to number of years spent in that company.
# The highest number of employee which have left are the one who have been in the company for 3 years.
gl=pd.DataFrame(df['left'].value_counts())
plt.pie(gl,labels=labels_left,shadow=True,colors = ['#1674b1','#ed6d50'],explode=[0.05,0.05],autopct='%.2f')
my_circle=plt.Circle( (0,0), 0.6, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.axis("equal")
fig = plt.gcf()
plt.title('Distribution of number of employees who left vs who did not leave')
fig.set_size_inches(6,6)
plt.show()
# This graph shows the distribution between the number of employees who have left and
# those who have not.
