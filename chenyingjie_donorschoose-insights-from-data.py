# basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.plotly as py
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv')
donations.head()
donors.head()
year = []
for each in donations['Donation Received Date']:
    year.append(each[0:4])
donations['Donation Received Year']=year
new_don = pd.merge(donors, donations,how='left', on=['Donor ID'])
new_don['Log Donation Amount']=np.log(new_don['Donation Amount'])
new_don.head()
sns.set_style("darkgrid")
sns.kdeplot(donations['Donation Amount'],shade = True).set_title("Donation Amount Distribution", fontsize=18)
g = sns.countplot(x='Donation Included Optional Donation', 
              data=new_don,palette="Paired",
                  hue ='Donation Received Year',
                  hue_order=['2012','2013','2014','2015','2016','2017','2018'])
g.set_title("Time Series: Donation Count", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g = sns.countplot(x='Donor Is Teacher', 
              data=new_don,palette="Paired",
                  hue ='Donation Received Year',
                  hue_order=['2012','2013','2014','2015','2016','2017','2018'])
g.set_title("Time Series: Donation Count", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g = sns.barplot(x='Donation Included Optional Donation', y='Donation Amount',
              data=new_don,palette="Paired",
                  hue ='Donation Received Year',
                  hue_order=['2012','2013','2014','2015','2016','2017','2018'])
g.set_title("Time Series: Donation Amount", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
plt.legend(loc='upper right')
g = sns.barplot(x='Donor Is Teacher', y='Donation Amount',
              data=new_don,palette="Paired",
                  hue ='Donation Received Year',
                  hue_order=['2012','2013','2014','2015','2016','2017','2018'])
g.set_title("Time Series: Donation Amount", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
plt.legend(loc='upper right')
city = new_don['Donor City'].value_counts()
state = new_don['Donor State'].value_counts()
sns.set_style('darkgrid')
g = sns.countplot(x='Donor City', hue='Donor Is Teacher',
              data=new_don[new_don['Donor City'].isin(city[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Donors Distribution by City", fontsize=18)
sns.set_style('darkgrid')
g = sns.countplot(x='Donor State', hue='Donor Is Teacher',
              data=new_don[new_don['Donor State'].isin(state[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Donors Distribution by State", fontsize=18)
sns.set_style('darkgrid')
city1 = new_don.groupby('Donor City')['Donation Amount'].agg('sum').sort_values().sort_values(ascending = False)
g = sns.boxplot(x='Donor City', y='Log Donation Amount', 
                data=new_don[new_don['Donor City'].isin(city1[:10].index.values)], palette="Paired")
g.set_title("Top 10 Cities: Log Donation Amount", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)
plt.show()
sns.set_style('darkgrid')
state1 = new_don.groupby('Donor State')['Log Donation Amount'].agg('sum').sort_values().sort_values(ascending = False)
g = sns.boxplot(x='Donor State', y='Log Donation Amount', 
                data=new_don[new_don['Donor State'].isin(state1[:10].index.values)], palette="Paired")
g.set_title("Top 10 States: Log Donation Amount", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)
plt.show()
sns.set_style('darkgrid')
city1 = new_don.groupby('Donor City')['Donation Amount'].agg('mean').sort_values().sort_values(ascending = False)
g = sns.boxplot(x='Donor City', y='Log Donation Amount', 
                data=new_don[new_don['Donor City'].isin(city1[:10].index.values)], palette="Paired")
g.set_title("Top 10 Cities: Log Donation Amount (mean value)", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)
plt.show()
sns.set_style('darkgrid')
state1 = new_don.groupby('Donor State')['Log Donation Amount'].agg('mean').sort_values().sort_values(ascending = False)
g = sns.boxplot(x='Donor State', y='Log Donation Amount', 
                data=new_don[new_don['Donor State'].isin(state1[:10].index.values)], palette="Paired")
g.set_title("Top 10 States: Log Donation Amount (mean value)", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)
plt.show()
ca = new_don[new_don['Donor State']=='California']
sns.set_style("darkgrid")
sns.kdeplot(ca['Log Donation Amount'],shade = True).set_title("California: Log Donation Amount Distribution", fontsize=18)
sns.set_style('darkgrid')
city1 = ca.groupby('Donor City')['Donation Amount'].agg('sum').sort_values().sort_values(ascending = False)
g = sns.boxplot(x='Donor City', y='Log Donation Amount', 
                data=ca[ca['Donor City'].isin(city1[:10].index.values)], palette="Paired")
g.set_title("Cities in California: Log Donation Amount", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)
plt.show()
ma = new_don[new_don['Donor State']=='Massachusetts']
sns.set_style("darkgrid")
sns.kdeplot(ma['Log Donation Amount'],shade = True).set_title("Massachusetts: Log Donation Amount Distribution", fontsize=18)
sns.set_style('darkgrid')
city1 = ma.groupby('Donor City')['Donation Amount'].agg('sum').sort_values().sort_values(ascending = False)
g = sns.boxplot(x='Donor City', y='Log Donation Amount', 
                data=ma[ma['Donor City'].isin(city1[:10].index.values)], palette="Paired")
g.set_title("Cities in Massachusetts: Log Donation Amount", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)
plt.show()
wa = new_don[new_don['Donor State']=='Washington']
sns.set_style("darkgrid")
sns.kdeplot(wa['Log Donation Amount'],shade = True).set_title("Washington: Log Donation Amount Distribution", fontsize=18)
sns.set_style('darkgrid')
city1 = wa.groupby('Donor City')['Donation Amount'].agg('sum').sort_values().sort_values(ascending = False)
g = sns.boxplot(x='Donor City', y='Log Donation Amount', 
                data=wa[wa['Donor City'].isin(city1[:10].index.values)], palette="Paired")
g.set_title("Cities in Washington: Log Donation Amount", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)
plt.show()
ha = new_don[new_don['Donor State']=='Hawaii']
sns.set_style("darkgrid")
sns.kdeplot(ha['Log Donation Amount'],shade = True).set_title("Hawaii: Log Donation Amount Distribution", fontsize=18)
sns.set_style('darkgrid')
city1 = ha.groupby('Donor City')['Donation Amount'].agg('sum').sort_values().sort_values(ascending = False)
g = sns.boxplot(x='Donor City', y='Log Donation Amount', 
                data=ha[ha['Donor City'].isin(city1[:10].index.values)], palette="Paired")
g.set_title("Cities in Hawaii: Log Donation Amount", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.legend(loc=1)
plt.show()
proj = pd.read_csv('../input/Projects.csv')
proj.head()
proj['Log Cost']=np.log(proj['Project Cost'])
year = []
for each in proj['Project Posted Date']:
    year.append(each[0:4])
proj['Posted Year']=year
sns.set_style('darkgrid')
sns.kdeplot(proj['Project Cost'],shade = True).set_title('project cost distribution',fontsize=18)
sns.set_style('darkgrid')
g = sns.countplot(x='Project Type', hue='Project Current Status',data=proj,palette="Paired").set_title("Project Count by Project Type", fontsize=18)
sns.set_style('darkgrid')
g = sns.boxplot(x='Project Type', y='Log Cost', hue='Project Current Status',data=proj,palette="Paired").set_title("Project Log Cost by Project Type", fontsize=18)
sns.set_style('darkgrid')
sns.pointplot(x="Posted Year", y="Log Cost", hue="Project Type", data=proj,palette="Paired").set_title("Project Log Cost (by Type): Time Series", fontsize=18)
sns.set_style('darkgrid')
g = sns.countplot(x='Project Grade Level Category', hue='Project Current Status',data=proj,palette="Paired").set_title("Project Count by Grade Level", fontsize=18)
sns.set_style('darkgrid')
g = sns.boxplot(x='Project Grade Level Category', y='Log Cost', hue='Project Current Status',data=proj,palette="Paired").set_title("Project Log Cost by Grade Level", fontsize=18)
plt.legend(loc='upper right')
sns.set_style('darkgrid')
sns.pointplot(x="Posted Year", y="Log Cost", hue="Project Grade Level Category", data=proj,palette="Paired").set_title("Project Log Cost (by Grade Level): Time Series", fontsize=18)
sns.set_style('darkgrid')
cates = proj['Project Subject Category Tree'].value_counts()
g = sns.countplot(x='Project Subject Category Tree', hue='Project Current Status',
              data=proj[proj['Project Subject Category Tree'].isin(cates[:10].index.values)],
                  palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("TOP 10 Subject Categories: Project Count", fontsize=18)
sns.set_style('darkgrid')
cost = proj.groupby('Project Subject Category Tree')['Project Cost'].agg('sum').sort_values(ascending = False)
g = sns.barplot(x='Project Subject Category Tree', y ='Project Cost',
                 hue='Project Grade Level Category',
              data=proj[proj['Project Subject Category Tree'].isin(cost[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("TOP 10 Subject Categories: Project Cost", fontsize=18)
sns.set_style('darkgrid')
plt.figure(figsize=(10,8))
sns.pointplot(x="Posted Year", y="Log Cost", hue="Project Subject Category Tree", data=proj[proj['Project Subject Category Tree'].isin(cost[:10].index.values)],palette="Paired").set_title("Project Log Cost (by Subject): Time Series", fontsize=18)
sns.set_style('darkgrid')
cates1 = proj['Project Resource Category'].value_counts()
g1 = sns.countplot(x='Project Resource Category', hue='Project Current Status',
              data=proj[proj['Project Resource Category'].isin(cates1[:10].index.values)],palette="Paired")
g1.set_xticklabels(g.get_xticklabels(),rotation=70)
g1.set_title("Project Count by Resource Category", fontsize=18)
sns.set_style('darkgrid')
cost = proj.groupby('Project Resource Category')['Project Cost'].agg('sum').sort_values(ascending = False)
g = sns.barplot(x='Project Resource Category', y ='Project Cost',
                 hue='Project Grade Level Category',
              data=proj[proj['Project Resource Category'].isin(cost[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Project Cost by Resource Categories", fontsize=18)
sns.set_style('darkgrid')
plt.figure(figsize=(10,8))
sns.pointplot(x="Posted Year", y="Log Cost", hue="Project Resource Category", data=proj[proj['Project Resource Category'].isin(cost[:10].index.values)],palette="Paired").set_title("Project Log Cost (by Resource Categories): Time Series", fontsize=18)
new_proj = pd.merge(proj, donations,how='left', on=['Project ID'])
new_proj.head()
sns.set_style('darkgrid')
d = new_proj.groupby('Project Type')['Donation ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='Project Type', hue='Project Current Status',
              data=new_proj,palette="Paired").set_title("Donation Count by Project Type", fontsize=18)
sns.set_style('darkgrid')
d = new_proj.groupby('Project Grade Level Category')['Donation ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='Project Grade Level Category', hue='Project Current Status',
              data=new_proj,palette="Paired").set_title("Donation Count by Grade Level", fontsize=18)
sns.set_style('darkgrid')
don = new_proj.groupby('Project Subject Category Tree')['Donation ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='Project Subject Category Tree', hue='Project Current Status',
              data=proj[proj['Project Subject Category Tree'].isin(don[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Donation Count by Subject Category", fontsize=18)
sns.set_style('darkgrid')
don1 = new_proj.groupby('Project Resource Category')['Donation ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='Project Resource Category', hue='Project Current Status',
              data=proj[proj['Project Resource Category'].isin(don1[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Donation Count by Resource Category", fontsize=18)
new_proj['Log Donation Amount']=np.log(new_proj['Donation Amount'])
sns.set_style('darkgrid')
sns.pointplot(x='Posted Year', hue='Project Type', y='Log Donation Amount', data=new_proj,
              palette="Paired").set_title("Donation Amount by Project Type: Time Series", fontsize=18)
sns.set_style('darkgrid')
sns.pointplot(x="Posted Year", y="Log Donation Amount", hue="Project Grade Level Category", data=new_proj,
              palette="Paired").set_title("Donation Amount by Grade Level: Time Series", fontsize=18)
sns.set_style('darkgrid')
amount = new_proj.groupby('Project Subject Category Tree')['Donation Amount'].agg('sum').sort_values(ascending = False)
g = sns.barplot(x='Project Subject Category Tree', y ='Donation Amount',
                 hue='Project Grade Level Category',
              data=new_proj[new_proj['Project Subject Category Tree'].isin(amount[:10].index.values)],
               palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Donation Amount by Subject", fontsize=18)
sns.set_style('darkgrid')
amount1 = new_proj.groupby('Project Resource Category')['Donation Amount'].agg('sum').sort_values(ascending = False)
g = sns.barplot(x='Project Resource Category', y='Donation Amount',
                 hue='Project Grade Level Category',
              data=new_proj[new_proj['Project Resource Category'].isin(amount1[:10].index.values)],
               palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Donations Amount by Resource Category", fontsize=18)
new_proj['gap']=new_proj['Project Cost']-new_proj['Donation Amount']
sns.set_style('darkgrid')
gap = new_proj.groupby('Project Subject Category Tree')['gap'].agg('sum').sort_values(ascending = False)
g = sns.barplot(x='Project Subject Category Tree', y='gap',
              data=new_proj[new_proj['Project Subject Category Tree'].isin(gap[:10].index.values)],
               palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Gap Distribution by Subject Category", fontsize=18)
sns.set_style('darkgrid')
gap1 = new_proj.groupby('Project Resource Category')['gap'].agg('sum').sort_values(ascending = False)
g = sns.barplot(x='Project Resource Category', y='gap',
              data=new_proj[new_proj['Project Resource Category'].isin(gap1[:10].index.values)],
               palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Gap Distribution by Resource Category", fontsize=18)
new_proj['prop']=new_proj['gap']/new_proj['Project Cost']
sns.set_style('darkgrid')
pro = new_proj.groupby('Project Subject Category Tree')['prop'].agg('mean').sort_values(ascending = False)
g = sns.barplot(x='Project Subject Category Tree', y='prop',
              data=new_proj[new_proj['Project Subject Category Tree'].isin(pro[:10].index.values)],
               palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Gap Proportion by Subject Category: ", fontsize=18)
sns.set_style('darkgrid')
pro = new_proj.groupby('Project Resource Category')['prop'].agg('mean').sort_values(ascending = False)
g = sns.barplot(x='Project Resource Category', y='prop',
              data=new_proj[new_proj['Project Resource Category'].isin(pro[:10].index.values)],
               palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=70)
g.set_title("Gap Proportion by Resource Category", fontsize=18)
a=new_proj['Log Cost'].sample(100)
b=new_proj['Log Donation Amount'].sample(100)
colors = np.random.rand(100)
area = np.pi * (15 * np.random.rand(100))**2
import matplotlib.pyplot as plt
plt.scatter(a, b, s=area, c=colors, alpha=0.5)
plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
names = proj["Project Title"].sample(500)
wordcloud = WordCloud(background_color="white",max_font_size=35, width=400, height=300).generate(' '.join(names))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title("Keywords from project title", fontsize=18)
plt.axis("off")
plt.show() 
names = proj["Project Essay"].sample(500)
wordcloud = WordCloud(background_color="white",max_font_size=35, width=400, height=300).generate(' '.join(names))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title("Keywords from project essay", fontsize=18)
plt.axis("off")
plt.show() 
names = proj["Project Short Description"].sample(500)
wordcloud = WordCloud(background_color="white",max_font_size=35, width=400, height=300).generate(' '.join(names))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title("Keywords from project short discription", fontsize=18)
plt.axis("off")
plt.show() 
names = proj["Project Need Statement"].sample(500)
wordcloud = WordCloud(background_color="white",max_font_size=35, width=400, height=300).generate(' '.join(names))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title("Keywords from project need statement", fontsize=18)
plt.axis("off")
plt.show() 
names = proj["Project Subject Category Tree"].sample(100)
wordcloud = WordCloud(background_color="white",max_font_size=35, width=400, height=300).generate(' '.join(names))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title("Keywords from project subject tree", fontsize=18)
plt.axis("off")
plt.show() 
names = proj["Project Subject Subcategory Tree"].sample(100)
wordcloud = WordCloud(background_color="white",max_font_size=35, width=400, height=300).generate(' '.join(names))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title("Keywords from project subject subtree", fontsize=18)
plt.axis("off")
plt.show() 
resource = pd.read_csv('../input/Resources.csv')
resource.head()
resource['Total Resource']=resource['Resource Quantity']*resource['Resource Unit Price']
sns.kdeplot(resource['Total Resource'],shade = True).set_title('Total Resource distribution',fontsize=12)
resource['Log Resource']=np.log(resource['Total Resource'])
sns.set_style('darkgrid')
ven = resource['Resource Vendor Name'].value_counts()
g = sns.countplot(x='Resource Vendor Name', 
              data=resource[resource['Resource Vendor Name'].isin(ven[:10].index.values)],
                  palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Project Count by Vendors", fontsize=12)
sns.set_style('darkgrid')
ven1 = resource.groupby('Resource Vendor Name')['Log Resource'].agg('sum').sort_values(ascending = False)
g = sns.boxplot(x='Resource Vendor Name', y='Log Resource',
              data=resource[resource['Resource Vendor Name'].isin(ven1[:10].index.values)]
                , palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Log Resource Amount by Vendor", fontsize=12)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
names = resource["Resource Item Name"].sample(500)
wordcloud = WordCloud(background_color="white",max_font_size=35, width=400, height=300).generate(' '.join(names))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.title("Keywords from Resource Item Name", fontsize=18)
plt.axis("off")
plt.show() 
new_resource = pd.merge(proj, resource,how='left', on=['Project ID'])
sns.set_style('darkgrid')
sns.pointplot(x="Posted Year", y="Total Resource", hue="Project Grade Level Category", data=new_resource,palette="Paired").set_title("Resource Amount (by Subject): Time Series", fontsize=18)
sns.set_style('darkgrid')
sns.pointplot(x="Posted Year", y="Total Resource", hue="Project Type", data=new_resource,palette="Paired").set_title("Resource Amount (by Type): Time Series", fontsize=18)
sns.set_style('darkgrid')
sns.pointplot(x="Posted Year", y="Total Resource", hue="Project Current Status", data=new_resource,palette="Paired").set_title("Resource Amount (by Current Status): Time Series", fontsize=18)
sns.barplot(x="Posted Year", y="Total Resource",
              data=new_resource,palette="Paired")
sns.set_style('darkgrid')
res = new_resource.groupby('Project Subject Category Tree')['Log Resource'].agg('sum').sort_values(ascending = False)
g = sns.barplot(x='Project Subject Category Tree', y ='Log Resource',
                 hue='Project Grade Level Category',
              data=new_resource[new_resource['Project Subject Category Tree'].isin(res[:10].index.values)],
               palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Log Resource Amount by Subject Category:", fontsize=18)
plt.legend(loc='upper right')
sns.set_style('darkgrid')
res1 = new_resource.groupby('Project Resource Category')['Log Resource'].agg('sum').sort_values(ascending = False)
g = sns.barplot(x='Project Resource Category', y='Log Resource',
                 hue='Project Grade Level Category',
              data=new_resource[new_resource['Project Resource Category'].isin(res1[:10].index.values)],
               palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Log Resource Amountby Resource Category", fontsize=18)
plt.legend(loc='upper right')
school = pd.read_csv('../input/Schools.csv')
school.head()
city = school['School City'].value_counts()
state = school['School State'].value_counts()
sns.set_style('darkgrid')
g = sns.countplot(x='School City', 
              data=school[school['School City'].isin(city[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("School Count by City", fontsize=12)
sns.set_style('darkgrid')
g = sns.countplot(x='School State', 
              data=school[school['School State'].isin(state[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("School Count by State", fontsize=18)
sns.set_style('darkgrid')
g = sns.violinplot(x='School Metro Type', y='School Percentage Free Lunch', data=school, palette="Paired")
g.set_title("Free Lunch Percentage by Metro Type", fontsize=18)
sns.set_style('darkgrid')
city = school['School City'].value_counts()
g = sns.barplot(x='School City', y ='School Percentage Free Lunch',
              data=school[school['School City'].isin(state[:50].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("School Percentage Free Lunch by City", fontsize=18)
sns.set_style('darkgrid')
state = school['School State'].value_counts()
g = sns.barplot(x='School State', y ='School Percentage Free Lunch',
              data=school[school['School State'].isin(state[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("School Percentage Free Lunch by State", fontsize=18)
new_sch = pd.merge(school, proj,how='left', on=['School ID'])
new_sch.head()
sns.set_style('darkgrid')
pro = new_sch.groupby('School City')['Project ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='School City',
              data=new_sch[new_sch['School City'].isin(pro[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Project Distribution by City", fontsize=18)
sns.set_style('darkgrid')
pro = new_sch.groupby('School State')['Project ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='School State',
              data=new_sch[new_sch['School State'].isin(pro[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Project Distribution by State", fontsize=18)
sns.set_style('darkgrid')
pro = new_sch.groupby('School Metro Type')['Project ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='School Metro Type',
              data=new_sch,palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Project Distribution by School Metro Type", fontsize=18)
teacher = pd.read_csv('../input/Teachers.csv')
teacher.head()
sns.set_style('darkgrid')
g = sns.countplot(x='Teacher Prefix', 
              data=teacher,palette="Paired")
g.set_title("Teachers Prefix", fontsize=18)
gender = []
for each in teacher['Teacher Prefix']:
    if each == 'Mrs.':
        gender.append('female')
    elif each == 'Mr.':
        gender.append('male')
    elif each == 'Ms.':
        gender.append('female')
    else:
        gender.append('unknown')
teacher['Gender']=gender
sns.set_style('darkgrid')
g = sns.countplot(x='Gender', 
              data=teacher,palette="Paired")
g.set_title("Teachers Gender", fontsize=18)
year = []
for each in teacher['Teacher First Project Posted Date']:
    year.append(each[0:4])
teacher['Teacher Posted Year']=year
g = sns.countplot(x='Teacher Posted Year', hue='Gender',
              data=teacher,palette="Paired")
g.set_title("Time Series: Project Posted Count", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
new_tea = pd.merge(teacher, proj,how='left', on=['Teacher ID'])
new_tea.head()
new_tea['Project Type'].unique()
tea_led = new_tea[new_tea['Project Type']=='Teacher-Led']
tea_led.head()
g = sns.countplot(x='Teacher Posted Year', hue='Project Grade Level Category',
              data=tea_led,palette="Paired")
g.set_title("Time Series: Project Grade Level Category", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=70)
sub = ['Applied Learning, Literacy & Language','Math & Science','Music & The Arts','Health & Sports']
g = sns.countplot(x='Teacher Posted Year', hue='Project Subject Category Tree',
              data=tea_led[tea_led['Project Subject Category Tree'].isin(sub)],palette="Paired")
g.set_title("Time Series: Project Resource Category", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=70)
res = ['Books','Technology','Supplies','Trips']
g = sns.countplot(x='Teacher Posted Year', hue='Project Resource Category',
              data=tea_led[tea_led['Project Resource Category'].isin(res)],palette="Paired")
g.set_title("Time Series: Project Resource Category", fontsize=18)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
sns.set_style('darkgrid')
don1 = new_tea.groupby('Project Grade Level Category')['Project ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='Project Grade Level Category', hue='Gender',
              data=new_tea[new_tea['Project Grade Level Category'].isin(don1[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Grade Level v.s. Gender", fontsize=18)
sns.set_style('darkgrid')
don = new_tea.groupby('Project Subject Category Tree')['Project ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='Project Subject Category Tree', hue='Gender',
              data=new_tea[new_tea['Project Subject Category Tree'].isin(don[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=30)
g.set_title("Subject Category v.s. Gender", fontsize=18)
sns.set_style('darkgrid')
don1 = new_tea.groupby('Project Resource Category')['Project ID'].agg('count').sort_values(ascending = False)
g = sns.countplot(x='Project Resource Category', hue='Gender',
              data=new_tea[new_tea['Project Resource Category'].isin(don1[:10].index.values)],palette="Paired")
g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_title("Resource Category v.s. Gender", fontsize=18)
