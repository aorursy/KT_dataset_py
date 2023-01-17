import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from collections import Counter
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
import os
teachers = pd.read_csv('../input/Teachers.csv')
projects = pd.read_csv('../input/Projects.csv')
schools = pd.read_csv('../input/Schools.csv')
donations = pd.read_csv('../input/Donations.csv')
plt.figure(figsize=(15,7))
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="rural"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='b', alpha=0.5,label='Rural')
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="urban"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='g', alpha=0.5,label='Urban')
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="suburban"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='orange',alpha=0.5, label='Suburban')
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="unknown"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='yellow', alpha=0.5,label='Unknown')
plt.hist(schools['School Percentage Free Lunch'][schools['School Metro Type']=="town"], range=[0,100],bins=50, histtype='stepfilled', normed=True, color='cyan', alpha=0.5,label='Town')
plt.title("Free Lunch Distributions")
plt.xlabel("Percentage Free Lunch")
plt.ylabel("Percentage")
plt.legend(loc='upper left')
plt.show()
schools_perc = schools[schools['School Percentage Free Lunch']<=100]
x1=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="rural"])
x2=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="urban"])
x3=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="suburban"])
x4=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="unknown"])
x5=(schools_perc['School Percentage Free Lunch'][schools_perc['School Metro Type']=="town"])
x_dict = {'rural':stats.normaltest(x1).pvalue,
              'urban':stats.normaltest(x2).pvalue,
              'suburban':stats.normaltest(x3).pvalue,
              'unknown':stats.normaltest(x4).pvalue,
              'town':stats.normaltest(x5).pvalue}
for keys,values in x_dict.items():
    print(keys)
    print("p-value:"+str(values))
kruskaltest=stats.kruskal(x1,x2,x3,x4,x5)
kruskaltest
plt.figure(figsize=(10,7))
ax = sns.boxplot(y=schools['School Percentage Free Lunch'],x=schools['School Metro Type'])
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x) for x in vals],fontsize=15)
ax.set_ylabel('School Percentage Free Lunch',fontsize=16)
ax.set_xlabel('School Metro Type',fontsize=16)
plt.title('Distribution of Free Lunch Across School Metro Type',fontsize=18)
plt.show()
funded_df=pd.get_dummies(projects['Project Current Status'])
fund_df=pd.concat([projects['Project Grade Level Category'], funded_df], axis=1)
project_grades = fund_df.groupby(['Project Grade Level Category']).agg([np.sum])
project_grades.columns=list(['Expired','Fully_Funded','Live'])
project_grades=project_grades.reset_index(level=0)
project_grades=project_grades.drop(4)
project_grades
chi2_contingency(project_grades.drop(['Project Grade Level Category'],axis=1))
expec = chi2_contingency(project_grades.drop(['Project Grade Level Category'],axis=1))[3]
observed=np.array(project_grades.drop(['Project Grade Level Category'],axis=1))
chi2_contingency(project_grades.drop(['Project Grade Level Category'],axis=1))
results=pd.concat([project_grades['Project Grade Level Category'],pd.DataFrame(observed-expec)],axis=1)
#results.columns(['Expired','Fully Funded','Live'])
results.columns = ['Project Grade Level Category','Expired','Fully Funded','Live']
results
cat_df = pd.concat([projects['Project Subject Category Tree'], funded_df], axis=1)
project_cats = cat_df.groupby(['Project Subject Category Tree']).agg([np.sum])
project_cats.columns=list(['Expired','Fully_Funded','Live'])
project_cats=project_cats.reset_index(level=0)
project_cats['Proportion Expired'] = project_cats['Expired']/(project_cats['Fully_Funded']+project_cats['Live']+project_cats['Expired'])
project_cats['Proportion Success'] = 1-project_cats['Proportion Expired']
project_cats_sorted = project_cats.sort_values(by='Proportion Success',ascending=True).reset_index()
plt.figure(figsize=(10,20))
ax=project_cats_sorted.drop('index',axis=1).plot(kind='barh',y=['Proportion Success'],figsize=(10,20),color='teal',xlim=[0,1],position=0)
for i,v in enumerate(project_cats_sorted['Project Subject Category Tree']):
    plt.text(0,i,str(v)+','+str("{:.2f}%".format(project_cats_sorted['Proportion Success'][i]*100)),fontsize=16)
plt.title('Proportion of Successful Campaigns by Subject',fontsize=20)
plt.xlabel('Proportion',fontsize=14)
plt.show()
