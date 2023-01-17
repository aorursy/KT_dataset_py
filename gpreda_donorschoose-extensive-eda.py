USE_WORDCLOUD = True
IS_LOCAL = False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
if (USE_WORDCLOUD):
    from wordcloud import WordCloud, STOPWORDS
    from nltk.corpus import stopwords
import os
if(IS_LOCAL):
    PATH="../input/donorschoose"
else:
    PATH = "../input"   
print(os.listdir(PATH))
donations = pd.read_csv(PATH+"/Donations.csv")
donors = pd.read_csv(PATH+"/Donors.csv", low_memory=False)
projects = pd.read_csv(PATH+"/Projects.csv", error_bad_lines=False, warn_bad_lines=False,)
resources = pd.read_csv(PATH+"/Resources.csv", error_bad_lines=False, warn_bad_lines=False)
schools = pd.read_csv(PATH+"/Schools.csv", error_bad_lines=False)
teachers = pd.read_csv(PATH+"/Teachers.csv", error_bad_lines=False)
print("donations -  rows:",donations.shape[0]," columns:", donations.shape[1])
print("donors -  rows:",donors.shape[0]," columns:", donors.shape[1])
print("projects -  rows:",projects.shape[0]," projects:", projects.shape[1])
print("resources -  rows:",resources.shape[0]," columns:", resources.shape[1])
print("schools -  rows:",schools.shape[0]," columns:", schools.shape[1])
print("teachers -  rows:",teachers.shape[0]," columns:", teachers.shape[1])
donations.head()
donors.head()
projects.head()
resources.head()
schools.head()
teachers.head()
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(donations)
missing_data(donors)
missing_data(projects)
missing_data(resources)
missing_data(schools)
missing_data(teachers)
donations_donors = donations.merge(donors, on='Donor ID', how='inner')
donations_donors.columns.values
tmp = donations_donors['Donor State'].value_counts()
df1 = pd.DataFrame({'State': tmp.index,'Number of donations': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'State', y="Number of donations",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = donations_donors.groupby('Donor State')['Donation Amount'].sum()
df1 = pd.DataFrame({'State': tmp.index,'Total sum of donations': tmp.values})
df1.sort_values(by='Total sum of donations',ascending=False, inplace=True)
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'State', y="Total sum of donations",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = donations_donors['Donor City'].value_counts().head(50)
df1 = pd.DataFrame({'City': tmp.index,'Number of donations': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'City', y="Number of donations",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = donations_donors.groupby('Donor City')['Donation Amount'].sum()
df1 = pd.DataFrame({'City': tmp.index,'Total sum of donations': tmp.values})
df1.sort_values(by='Total sum of donations',ascending=False, inplace=True)
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'City', y="Total sum of donations",data=df1.head(50))
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = donations_donors['Donor City'].value_counts().tail(50)
df1 = pd.DataFrame({'City': tmp.index,'Number of donations': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'City', y="Number of donations",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = donations_donors.groupby('Donor City')['Donation Amount'].sum()
df1 = pd.DataFrame({'City': tmp.index,'Total sum of donations': tmp.values})
df1.sort_values(by='Total sum of donations',ascending=False, inplace=True)
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'City', y="Total sum of donations",data=df1.tail(50))
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = np.log(donations['Donation Amount'] + 1)
plt.figure(figsize = (10,6))

s = sns.distplot(tmp,color="green")
plt.xlabel('log(Donation Amount)', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.title("Density plot of Donation Amount (log)")
plt.show();

tmp = donors['Donor Is Teacher'].value_counts()
df1 = pd.DataFrame({'Donor Is Teacher': tmp.index,'Number of donors': tmp.values})

tmp = donations_donors['Donor Is Teacher'].value_counts()
df2 = pd.DataFrame({'Donor Is Teacher': tmp.index,'Number of donations': tmp.values})

tmp = donations_donors.groupby('Donor Is Teacher')['Donation Amount'].sum()
df3 = pd.DataFrame({'Donor Is Teacher': tmp.index,'Total sum of donations': tmp.values})


plt.figure(figsize = (14,12))

plt.subplot(2,2,1)
s = sns.barplot(x = 'Donor Is Teacher', y="Number of donors",data=df1)
plt.title("Number of donors")

plt.subplot(2,2,2)
s = sns.barplot(x = 'Donor Is Teacher', y="Number of donations",data=df2)
plt.title("Number of donations")

plt.subplot(2,2,3)
s = sns.barplot(x = 'Donor Is Teacher', y="Total sum of donations",data=df3.tail(50))
plt.title("Total sum of donations")

plt.show();
state_order = pd.DataFrame(donations_donors.groupby('Donor State')['Donation Amount'].sum()).\
        sort_values(by='Donation Amount', ascending=False)
fig, ax1 = plt.subplots(ncols=1, figsize=(16,6))
s = sns.boxplot(ax = ax1, x="Donor State", y="Donation Amount", hue="Donor Is Teacher",
                data=donations_donors, palette="PRGn",showfliers=False, order=state_order.index)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
donations_donors['Donation Received Date'] = pd.to_datetime(donations_donors['Donation Received Date'])
donations_donors['Year'] = donations_donors['Donation Received Date'].dt.year
donations_donors['Month'] = donations_donors['Donation Received Date'].dt.month
donations_donors['Day'] = donations_donors['Donation Received Date'].dt.day
donations_donors['Weekday'] = donations_donors['Donation Received Date'].dt.weekday
donations_donors['Hour'] = donations_donors['Donation Received Date'].dt.hour
def plot_time_variation(feature):
    tmp = donations_donors.groupby(feature)['Donation Amount'].sum()
    tmp = tmp[~tmp.index.isin([2018])] 
    df1 = pd.DataFrame({feature: tmp.index,'Total sum of donations': tmp.values})
    
    tmp = donations_donors.groupby(feature)['Donation Amount'].mean()
    tmp = tmp[~tmp.index.isin([2018])] 
    df2 = pd.DataFrame({feature: tmp.index,'Mean value of donations': tmp.values})
    
    tmp = donations_donors.groupby(feature)['Donation Amount'].min()
    tmp = tmp[~tmp.index.isin([2018])] 
    df3 = pd.DataFrame({feature: tmp.index,'Min value of donations': tmp.values})
    
    tmp = donations_donors.groupby(feature)['Donation Amount'].max()
    tmp = tmp[~tmp.index.isin([2018])] 
    df4 = pd.DataFrame({feature: tmp.index,'Max value of donations': tmp.values})
    
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(14,4))
    s = sns.barplot(ax = ax1, x = feature, y="Total sum of donations",data=df1)
    s = sns.barplot(ax = ax2, x = feature, y="Mean value of donations",data=df2)
    plt.show();
    
    fig, (ax3, ax4) = plt.subplots(ncols=2,figsize=(14,4))
    s = sns.barplot(ax = ax3, x = feature, y="Min value of donations",data=df3)
    s = sns.barplot(ax = ax4, x = feature, y="Max value of donations",data=df4)
    plt.show();

def boxplot_time_variation(feature, width=16):
    fig, ax1 = plt.subplots(ncols=1, figsize=(width,6))
    s = sns.boxplot(ax = ax1, x=feature, y="Donation Amount", hue="Donor Is Teacher",
                data=donations_donors, palette="PRGn",showfliers=False)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show();
plot_time_variation('Year')
boxplot_time_variation('Year',8)
plot_time_variation('Month')
boxplot_time_variation('Month',10)
plot_time_variation('Day')
boxplot_time_variation('Day')
plot_time_variation('Weekday')
boxplot_time_variation('Weekday',6)
plot_time_variation('Hour')
boxplot_time_variation('Hour',12)
projects_schools = projects.merge(schools, on='School ID', how='inner')
projects_schools.columns.values
tmp = projects_schools['School State'].value_counts()
df1 = pd.DataFrame({'School State': tmp.index,'Number of projects': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'School State', y="Number of projects",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = projects_schools.groupby('School State')['Project Cost'].sum()
df1 = pd.DataFrame({'School State': tmp.index,'Total Projects Cost': tmp.values})
df1.sort_values(by='Total Projects Cost',ascending=False, inplace=True)
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'School State', y="Total Projects Cost",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
projects_schools['Project Posted Date'] = pd.to_datetime(projects_schools['Project Posted Date'])
projects_schools['Year Posted'] = projects_schools['Project Posted Date'].dt.year
projects_schools['Month Posted'] = projects_schools['Project Posted Date'].dt.month
projects_schools['Day Posted'] = projects_schools['Project Posted Date'].dt.day
projects_schools['Weekday Posted'] = projects_schools['Project Posted Date'].dt.weekday
projects_schools['Hour Posted'] = projects_schools['Project Posted Date'].dt.hour
projects_schools['Project Fully Funded Date'] = pd.to_datetime(projects_schools['Project Fully Funded Date'])
projects_schools['Year Funded'] = projects_schools['Project Fully Funded Date'].dt.year
projects_schools['Month Funded'] = projects_schools['Project Fully Funded Date'].dt.month
projects_schools['Day Funded'] = projects_schools['Project Fully Funded Date'].dt.day
projects_schools['Weekday Funded'] = projects_schools['Project Fully Funded Date'].dt.weekday
projects_schools['Hour Funded'] = projects_schools['Project Fully Funded Date'].dt.hour
def plot_project_time_variation(feature):
    tmp = projects_schools.groupby(feature)['Project Cost'].sum()
    tmp = tmp[~tmp.index.isin([2018])] 
    df1 = pd.DataFrame({feature: tmp.index,'Total sum of projects cost': tmp.values})
    
    tmp = projects_schools.groupby(feature)['Project Cost'].mean()
    tmp = tmp[~tmp.index.isin([2018])] 
    df2 = pd.DataFrame({feature: tmp.index,'Mean value of projects cost': tmp.values})
    
    tmp = projects_schools.groupby(feature)['Project Cost'].min()
    tmp = tmp[~tmp.index.isin([2018])] 
    df3 = pd.DataFrame({feature: tmp.index,'Min value of projects cost': tmp.values})
    
    tmp = projects_schools.groupby(feature)['Project Cost'].max()
    tmp = tmp[~tmp.index.isin([2018])] 
    df4 = pd.DataFrame({feature: tmp.index,'Max value of projects cost': tmp.values})
    
    fig, (ax1, ax2) = plt.subplots(ncols=2,figsize=(14,4))
    s = sns.barplot(ax = ax1, x = feature, y="Total sum of projects cost",data=df1)
    s = sns.barplot(ax = ax2, x = feature, y="Mean value of projects cost",data=df2)
    plt.show();
    
    fig, (ax3, ax4) = plt.subplots(ncols=2,figsize=(14,4))
    s = sns.barplot(ax = ax3, x = feature, y="Min value of projects cost",data=df3)
    s = sns.barplot(ax = ax4, x = feature, y="Max value of projects cost",data=df4)
    plt.show();
    
def boxplot_project_time_variation(feature,width=8):
    fig, ax1 = plt.subplots(ncols=1, figsize=(2*width,6))
    s = sns.boxplot(ax = ax1, x=feature, y="Project Cost",data=projects_schools, palette="viridis",showfliers=False)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show();
plot_project_time_variation('Year Posted')
boxplot_project_time_variation('Year Posted',4)
plot_project_time_variation('Month Posted')
boxplot_project_time_variation('Month Posted',5)
plot_project_time_variation('Day Posted')
boxplot_project_time_variation('Day Posted')
plot_project_time_variation('Weekday Posted')
boxplot_project_time_variation('Weekday Posted',4)
plot_project_time_variation('Year Funded')
boxplot_project_time_variation('Year Funded',4)
plot_project_time_variation('Month Funded')
boxplot_project_time_variation('Month Funded',4)
plot_project_time_variation('Day Funded')
boxplot_project_time_variation('Day Funded')
plot_project_time_variation('Weekday Funded')
boxplot_project_time_variation('Weekday Funded',4)
def plot_project_schools(feature):
    tmp = projects_schools.groupby(feature)['Project Cost'].sum()
    df1 = pd.DataFrame({feature: tmp.index,'Total sum of projects cost': tmp.values})
  
    tmp = projects_schools[feature].value_counts()
    df2 = pd.DataFrame({feature: tmp.index,'Number of projects': tmp.values})
    
    fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))
    sns.barplot(ax = ax1, x = feature, y="Total sum of projects cost",data=df1,order=tmp.index)
    sns.barplot(ax = ax2, x = feature, y="Number of projects",data=df2,order=tmp.index)
    plt.show();
    
def boxplot_project_schools(feature1, feature2, n=2):
    fig, ax1 = plt.subplots(ncols=1, figsize=(n*7,6))
    s = sns.boxplot(ax = ax1, x=feature1, y=feature2,data=projects_schools, palette="viridis",showfliers=False)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show();
plot_project_schools('School Metro Type')
boxplot_project_schools('School Metro Type','School Percentage Free Lunch',1)
boxplot_project_schools('School State','School Percentage Free Lunch')
def plot_wordcloud(feature,additionalStopWords=""):
    if(USE_WORDCLOUD):
        stopwords = set(STOPWORDS)
        stopwords.update(additionalStopWords)
        text = " ".join(projects_schools[feature][~pd.isnull(projects_schools[feature])].sample(50000))
        wordcloud = WordCloud(background_color='black',stopwords=stopwords,
                          max_words=1000,max_font_size=100, width=800, height=800,random_state=2018,
                         ).generate(text)
        fig = plt.figure(figsize = (12,12))
        plt.imshow(wordcloud)
    plt.title("Wordcloud with %s content" % feature, fontsize=16)
    plt.axis('off')
    plt.show()
plot_wordcloud('Project Title')
plot_wordcloud('Project Essay',["DONOTREMOVEESSAYDIVIDER", "students", "will"])
plot_wordcloud('Project Short Description')
plot_wordcloud('Project Need Statement', ["need"])
def plot_category(feature):
    tmp = projects[feature].value_counts().sort_values(ascending = False).head(20)
    df1 = pd.DataFrame({feature: tmp.index,'Number of projects': tmp.values})
    fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
    s = sns.barplot(ax = ax1, x = feature, y="Number of projects",data=df1)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show();
plot_category('Project Subject Category Tree')
plot_category('Project Subject Subcategory Tree')
plot_category('Project Grade Level Category')
plot_category('Project Resource Category')
tmp = resources['Resource Vendor Name'].value_counts().head(20)
df1 = pd.DataFrame({'Resource Vendor Name': tmp.index,'Number of orders': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'Resource Vendor Name', y="Number of orders",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = resources.groupby('Resource Vendor Name')['Resource Quantity'].sum().sort_values(ascending = False).head(20)
df1 = pd.DataFrame({'Resource Vendor Name': tmp.index,'Number of resources': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'Resource Vendor Name', y="Number of resources",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
resources['Resource Price'] = resources['Resource Quantity'] * resources['Resource Unit Price']
tmp = resources.groupby('Resource Vendor Name')['Resource Price'].sum().sort_values(ascending = False).head(20)
df1 = pd.DataFrame({'Resource Vendor Name': tmp.index,'Total price of resources ordered': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
s = sns.barplot(ax = ax1, x = 'Resource Vendor Name', y="Total price of resources ordered",data=df1)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show();
tmp = teachers['Teacher Prefix'].value_counts()
df1 = pd.DataFrame({'Teacher Prefix': tmp.index,'Number of teachers': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(6,6))
s = sns.barplot(ax = ax1, x = 'Teacher Prefix', y="Number of teachers",data=df1)
plt.show();
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
tmp = teachers['Teacher First Project Posted Date'].value_counts()
df1 = pd.DataFrame({'Teacher First Project Posted Date': tmp.index,'Number of teachers': tmp.values})
fig, (ax1) = plt.subplots(ncols=1, figsize=(16,6))
df1.plot(ax=ax1,x = 'Teacher First Project Posted Date', y="Number of teachers")
plt.show();
import datetime
projects_schools['Project Funding Duration'] = (pd.to_datetime(projects_schools['Project Fully Funded Date'] -\
                    projects_schools['Project Posted Date']) - pd.to_datetime('1970-01-01')).dt.days
boxplot_project_schools('School State','Project Funding Duration')
boxplot_project_schools('Year Posted','Project Funding Duration',n=1)
boxplot_project_schools('School Metro Type','Project Funding Duration',n=1)
donations_donors_projects_schools = donations_donors.merge(projects_schools, on='Project ID', how='inner')
donations_donors_projects_schools['Project Donation Duration'] = (pd.to_datetime(donations_donors_projects_schools['Donation Received Date'] -\
                    donations_donors_projects_schools['Project Posted Date']) - pd.to_datetime('1970-01-01')).dt.days
tmp = np.log(donations_donors_projects_schools.groupby(['Donor State', 'School State'])['Donation Amount'].sum())
df1 = tmp.reset_index()
matrix = df1.pivot('Donor State', 'School State','Donation Amount')
fig, (ax1) = plt.subplots(ncols=1, figsize=(16,16))
sns.heatmap(matrix, 
        xticklabels=matrix.index,
        yticklabels=matrix.columns,ax=ax1,linewidths=.1,cmap="YlGnBu")
plt.title("Heatmap with log(Donation Amount) per donor state and school state", fontsize=14)
plt.show()
tmp = donations_donors_projects_schools.groupby(['Project Resource Category', 'School Metro Type'])['Project Funding Duration'].mean()
df1 = tmp.reset_index()
matrix = df1.pivot('Project Resource Category', 'School Metro Type','Project Funding Duration')
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,6))
sns.heatmap(matrix, 
        xticklabels=matrix.columns,
        yticklabels=matrix.index,ax=ax1,linewidths=.1,cmap="Reds")
plt.title("Heatmap with mean(Project Funding Duration) per project resource category and school metro type", fontsize=14)
plt.show()
tmp = donations_donors_projects_schools.groupby(['Day Posted', 'Day Funded'])['Project Funding Duration'].mean()
df1 = tmp.reset_index()
matrix = df1.pivot('Day Posted', 'Day Funded','Project Funding Duration')
fig, (ax1) = plt.subplots(ncols=1, figsize=(12,12))
sns.heatmap(matrix, 
        xticklabels=matrix.columns,
        yticklabels=matrix.index,ax=ax1,linewidths=.1,cmap="Greens")
plt.title("Heatmap with mean(Project Funding Duration) per project day posted and project day funded", fontsize=14)
plt.show()
tmp = donations_donors_projects_schools[['Project Funding Duration', 'Project Cost', 'Project Donation Duration',\
                                         'Donation Amount', 'School Percentage Free Lunch']].copy()
corr = tmp.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plt.show()