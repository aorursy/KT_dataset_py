import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
pd.set_option('display.max_columns',100)

pd.set_option('display.max_rows',None)

df=pd.read_csv('../input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
df.head()
df.dtypes
for i in range(len(df)):

    df['Crawl Timestamp'][i]=df['Crawl Timestamp'][i].replace('+0000','')

    i+=1
df.head()
df['Crawl Timestamp']=pd.to_datetime(df['Crawl Timestamp'])
df['Crawl Timestamp'][:5]
df.isna().any()
sns.heatmap(df.isnull(),cbar=True,cmap='gnuplot')
cols=[ 'Job Title', 'Job Salary',

       'Job Experience Required', 'Key Skills', 'Role Category', 'Location',

       'Functional Area', 'Industry', 'Role']

empty_vals=[]

for col in cols:

    print('Number of missing values in {}: {}'.format(col,df[col].isna().value_counts()[1]))

    empty_vals.append(df[col].isna().value_counts()[1])

print('Total entries:{}'.format(len(df)))
missing_df=pd.DataFrame(columns=['Column','Missing values'])

missing_df['Column']=cols

missing_df['Missing values']=empty_vals

missing_df.sort_values(by='Missing values',inplace=True,ascending=False)

missing_df.index=missing_df.Column

missing_df.drop('Column',axis=1,inplace=True)

missing_df
my_colors = 'rgbkymc'  #red, green, blue, black, etc.

ax=missing_df.plot(kind='bar',figsize=(20,10),rot=90,width=0.8,color=my_colors)

ax.set_title("Number of missing values in the dataframe",size=20)

ax.set_ylabel('Number of missing values',size=18)

ax.set_xlabel('Column',size=18)





#For annotating the bars



for i in ax.patches:

    ax.text(i.get_x()+0.045,i.get_height()+2,str(round((i.get_height()), 2)),

            rotation=0,fontsize=15,color='black')

    

df.dropna(axis=0,inplace=True)
df.isna().any()
df.size
df['Job Title'].describe()
df['Job Title'].value_counts()[0:10]
df_temp=df.copy()
df_temp.loc[df_temp['Job Title'].str.contains('Planner', case=False), 'Cleaned Title'] = 'Planner'
df_temp.loc[df_temp['Job Title'].str.contains('Analyst', case=False), 'Cleaned Title'] = 'Analyst'

df_temp.loc[df_temp['Job Title'].str.contains('Analytics', case=False), 'Cleaned Title'] = 'Analyst'

df_temp.loc[df_temp['Job Title'].str.contains('Develop',case=False),'Cleaned Title']='Software/Web/App Developer'

df_temp.loc[df_temp['Job Title'].str.contains('Software',case=False),'Cleaned Title']='Software/Web/App Developer'

df_temp.loc[df_temp['Job Title'].str.contains('Web',case=False),'Cleaned Title']='Software/Web/App Developer'

df_temp.loc[df_temp['Job Title'].str.contains('App',case=False),'Cleaned Title']='Software/Web/App Developer'

df_temp.loc[df_temp['Job Title'].str.contains('Designer', case=False), 'Cleaned Title'] = 'Design and Creativity'

df_temp.loc[df_temp['Job Title'].str.contains('Animation', case=False), 'Cleaned Title'] = 'Design and Creativity'

df_temp.loc[df_temp['Job Title'].str.contains('Content', case=False), 'Cleaned Title'] = 'Design and Creativity'

df_temp.loc[df_temp['Job Title'].str.contains('Consultant', case=False), 'Cleaned Title'] = 'Consultancy'

df_temp.loc[df_temp['Job Title'].str.contains('Risk', case=False), 'Cleaned Title'] = 'Risk analyst'

df_temp.loc[df_temp['Job Title'].str.contains('Call', case=False), 'Cleaned Title'] = 'Customer service'

df_temp.loc[df_temp['Job Title'].str.contains('Support',case=False),'Cleaned Title']='Customer service'

df_temp.loc[df_temp['Job Title'].str.contains('Customer support',case=False),'Cleaned Title']='Customer service'

df_temp.loc[df_temp['Job Title'].str.contains('Engineer',case=False),'Cleaned Title']='Core engineering'

df_temp.loc[df_temp['Job Title'].str.contains('Tech',case=False),'Cleaned Title']='Core engineering'
df_temp.loc[df_temp['Job Title'].str.contains('Prof',case=False),'Cleaned Title']='Academic role'

df_temp.loc[df_temp['Job Title'].str.contains('Business',case=False),'Cleaned Title']='Business Developer/Intelligence'

df_temp.loc[df_temp['Job Title'].str.contains('Social Media',case=False),'Cleaned Title']='Public Relations'

df_temp.loc[df_temp['Job Title'].str.contains('HR',case=False),'Cleaned Title']='Human Resources'

df_temp.loc[df_temp['Job Title'].str.contains('HR Executive',case=False),'Cleaned Title']='Human Resources'

df_temp.loc[df_temp['Job Title'].str.contains('Manager',case=False),'Cleaned Title']='Managerial role'

df_temp.loc[df_temp['Job Title'].str.contains('Fresher',case=False),'Cleaned Title']='Fresher role'

df_temp.loc[df_temp['Job Title'].str.contains('Account',case=False),'Cleaned Title']='Accounting role'

df_temp.loc[df_temp['Job Title'].str.contains('Intern',case=False),'Cleaned Title']='Internships'

df_temp.loc[df_temp['Job Title'].str.contains('Placement',case=False),'Cleaned Title']='Placement & Liaison'

df_temp.loc[df_temp['Job Title'].str.contains('Liaison',case=False),'Cleaned Title']='Placement & Liaison'

df_temp.loc[df_temp['Job Title'].str.contains('Recruit',case=False),'Cleaned Title']='Placement & Liaison'

df_temp.loc[df_temp['Job Title'].str.contains('Data',case=False),'Cleaned Title']='Data Science'

df_temp.loc[df_temp['Job Title'].str.contains('Sale',case=False),'Cleaned Title']='Sales Executive'

df_temp.loc[df_temp['Job Title'].str.contains('Health',case=False),'Cleaned Title']='Health Care'

df_temp.loc[df_temp['Job Title'].str.contains('Quality',case=False),'Cleaned Title']='Quality Control'

df_temp.loc[df_temp['Job Title'].str.contains('Tele',case=False),'Cleaned Title']='Telemarketing'

df_temp['Cleaned Title'].value_counts()
df_temp['Cleaned Title'].isna().value_counts()
df_temp.dropna(inplace=True)
df_temp.drop('Job Title',axis=1,inplace=True)
df_temp=df_temp[['Uniq Id', 'Crawl Timestamp', 'Cleaned Title','Job Salary', 'Job Experience Required',

       'Key Skills', 'Role Category', 'Location', 'Functional Area',

       'Industry', 'Role']]
df_temp.rename(columns={'Cleaned Title':'Job Title'},inplace=True)
df_temp.head()
df_temp.reset_index(inplace=True,drop=True)

df=df_temp.copy() #Checkpoint
df_temp['Job Experience Required'].value_counts()[0:10]
for i in range(len(df_temp)):

    df_temp['Job Experience Required'][i]=df_temp['Job Experience Required'][i].replace('yrs','')

    df_temp['Job Experience Required'][i]=df_temp['Job Experience Required'][i].replace('years','')

    df_temp['Job Experience Required'][i]=df_temp['Job Experience Required'][i].replace('Years','')



    i+=1
df_temp['Job Experience Required'].value_counts()[0:10]
top_job_exp=df_temp['Job Experience Required'].value_counts()[0:10]

top_job_exp
exp_df=pd.DataFrame(top_job_exp)

exp_df.reset_index(inplace=True)
exp_df.rename(columns={'index':'Job Experience','Job Experience Required':'Count'},inplace=True)
exp_df
exp_df.loc[exp_df['Job Experience'].str.contains('2 - 5',case=False),'Sorted Experience']='Early Professionals'

exp_df.loc[exp_df['Job Experience'].str.contains('5 - 10',case=False),'Sorted Experience']='Expereinced Professionals'

exp_df.loc[exp_df['Job Experience'].str.contains('2 - 7',case=False),'Sorted Experience']='Early Professionals'

exp_df.loc[exp_df['Job Experience'].str.contains('3 - 8',case=False),'Sorted Experience']='Expereinced Professionals'

exp_df.loc[exp_df['Job Experience'].str.contains('1 - 3',case=False),'Sorted Experience']='Early Professionals'



exp_df.loc[exp_df['Job Experience'].str.contains('3 - 5',case=False),'Sorted Experience']='Early Professionals'



exp_df.loc[exp_df['Job Experience'].str.contains('1 - 6',case=False),'Sorted Experience']='Early Professionals'



exp_df.loc[exp_df['Job Experience'].str.contains('1 - 5',case=False),'Sorted Experience']='Early Professionals'



exp_df.loc[exp_df['Job Experience'].str.contains('0 - 1',case=False),'Sorted Experience']='Freshers'

exp_df.loc[exp_df['Job Experience'].str.contains('2 - 4',case=False),'Sorted Experience']='Early Professionals'



exp_cat=exp_df.copy()

exp_cat.drop('Job Experience',axis=1,inplace=True)

exp_cat.rename(columns={'Sorted Experience':'Experience category'},inplace=True)

exp_cat=exp_cat[['Experience category','Count']]

exp_cat
grouped_df=exp_cat.groupby('Experience category').sum()

grouped_df.reset_index(inplace=True)

grouped_df
locs_df=pd.DataFrame(df_temp['Location'])

locs_df.head()
locs_df['Count']=1

group_locs=locs_df.groupby('Location').sum().reset_index()
group_locs.sort_values(by='Count',ascending=False,inplace=True)
group_locs_top=group_locs.head(12)

group_locs_top
df_titles=pd.DataFrame(df_temp['Job Title'],columns=['Job Title','Count'])

                    
df_titles['Count']=1
df_titles=df_titles.groupby('Job Title').sum()

df_titles
df_titles.reset_index(inplace=True)

df_titles.sort_values('Count',ascending=False,inplace=True)

df_titles
sns.catplot('Job Title','Count',data=df_titles,kind='bar',aspect=2,height=6,palette='summer')

plt.xticks(rotation=90)

plt.xlabel('Job Title',size=15)

plt.ylabel('Number of jobs available',size=15)

plt.title('Distribution of job titles',size=25)
grouped_df['Count']=grouped_df['Count'].astype(int)

grouped_df
plt.figure(figsize=(10,8))

ax=sns.barplot('Experience category','Count',data=grouped_df)

plt.xlabel('Category',size=15)

plt.ylabel('Number of vacancies',size=15)

plt.title('Expereince wise vacancies',size=20)



for i in ax.patches:

    ax.text(i.get_x()+.25,i.get_height()+2.3,str(int((i.get_height()))),

            rotation=0,fontsize=15,color='black')
group_locs_top
plt.figure(figsize=(10,8))

ax=sns.barplot('Location','Count',data=group_locs_top,palette='winter')

plt.xlabel('Location',size=15)

plt.ylabel('Number of vacancies',size=15)

plt.title('Expereince wise vacancies',size=20)

plt.xticks(rotation=45)



for i in ax.patches:

    ax.text(i.get_x(),i.get_height()+2.3,str(int((i.get_height()))),

            rotation=0,fontsize=15,color='black')
from wordcloud import WordCloud, STOPWORDS



print ('Wordcloud is installed and imported!')
imp_words = df_temp['Role Category'].to_list()



wordcloud = WordCloud(width = 500, height = 500, 

                background_color ='white', 

                min_font_size = 10).generate(str(imp_words))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
imp_words = df_temp['Key Skills'].to_list()



wordcloud = WordCloud(width = 500, height = 500, 

                background_color ='green', 

                min_font_size = 10).generate(str(imp_words))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()