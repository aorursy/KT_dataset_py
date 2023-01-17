import numpy as np #for algebric calculations
import pandas as pd #essential for data reading,writing etc
import seaborn as sns #visualization library
import plotly.express as px #ploting parameter's
import plotly.express
import matplotlib #visualization library.
import matplotlib.pyplot as plt #visualization library.
import sys #for System-specific parameters and functions.
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 7) #plotting parameters size's.
import warnings
warnings.filterwarnings('ignore')

# display the version of import libraries

print('Python : {}'.format(sys.version[0:5]))
print('Numpy : {}'.format(np.__version__))
print('Pandas : {}'.format(pd.__version__))
print('Matplotlib : {}'.format(matplotlib.__version__))
print('Seaborn : {}'.format(sns.__version__))
print('Plotly.Express : {}'.format(plotly.__version__))
# Define the path of CSV file & import the dataset.

df=pd.read_csv('../input/jobs-on-naukricom/naukri_com-job_sample.csv') 
# Display the first 10 columns.

df.head(5)
# Display the number of rows and columns in the dataset.

nrow,ncol=df.shape
print(f'There are {nrow} rows and {ncol} colunms in the dataset')
#The total number of elements.
#This is equal to the row_count * column_coun.

print(df.shape[0] * df.shape[1])
# Display the information about DataFrame including the index dtype, columns and etc.  

df.info()
count_missing = df.isnull().sum()
percent_missing =  count_missing* 100 / df.shape[0]
missing_value_df = pd.DataFrame({'count_missing': count_missing,
                                 'percent_missing': percent_missing})

missing_value_df.style.background_gradient(cmap='Spectral')
unique_df = pd.DataFrame([[df[i].nunique(), df[i].unique()]for i in df.columns],
                         columns=['Count','unique'],
                         index=df.columns)
unique_df.style.background_gradient(cmap='magma')
# split method to split payrate min to max

pay_split = df['payrate'].str[0:-1].str.split('-', expand=True)
pay_split.head(10)
#remove space in left and right
pay_split[0] =  pay_split[0].str.strip()

#remove comma 
pay_split[0] = pay_split[0].str.replace(',', '')

#remove all character in two condition
# 1 remove if only character
# 2 if start in number remove after all character
pay_split[0] = pay_split[0].str.replace(r'\D.*', '')

#display 
pay_split[0].head(10)
#remove space in left and right 
pay_split[1] =  pay_split[1].str.strip()

#remove comma 
pay_split[1] = pay_split[1].str.replace(',', '')

#remove all character in two condition
# 1 remove if only character
# 2 if start in number remove after all character
pay_split[1] = pay_split[1].str.replace(r'\D.*','')

#display 
pay_split[1].head(10)
pay_split[0] = pd.to_numeric(pay_split[0], errors='coerce')
pay_split[1] = pd.to_numeric(pay_split[1], errors='coerce')
pay=pd.concat([pay_split[0], pay_split[1]], axis=1, sort=False)
# rename the columns into min payrate and max payrate.
pay.rename(columns={0:'min_pay', 1:'max_pay'}, inplace=True )
pay.head()
# min and max payarte store the value in the dataframe.

df=pd.concat([df, pay], axis=1, sort=False)
df.head(5)
# spliting the experience into min experience to max experience.

experience_split = df['experience'].str[0:-1].str.split('-', expand=True)
experience_split.head()
#remove space in left and right 
experience_split[0] =  experience_split[0].str.strip()

#remove comma 
experience_split[0] = experience_split[0].str.replace('yr', '')

#remove all character in two condition
# 1 remove if only character
# 2 if start in number remove after all character
experience_split[0] = experience_split[0].str.replace(r'yr', '')

#display 
experience_split[0].head()
#remove space in left and right 
experience_split[1] =  experience_split[1].str.strip()

#remove comma 
experience_split[1] = experience_split[1].str.replace('yr', '')

#remove all character in two condition
# 1 remove if only character
# 2 if start in number remove after all character
experience_split[1] = experience_split[1].str.replace(r'yr', '')

#display 
experience_split[1].head()
experience_split[0] = pd.to_numeric(experience_split[0], errors='coerce')
experience_split[1] = pd.to_numeric(experience_split[1], errors='coerce')
experience=pd.concat([experience_split[0], experience_split[1]], axis=1, sort=False)
# rename the cloumns to min and max experience

experience.rename(columns={0:'min_experience', 1:'max_experience'}, inplace=True)
experience.head()
# store the min and max experience in the dataframe.

df=pd.concat([df, experience], axis=1, sort=False)
#displat max and min experience

df.head(5)
# Display average payrate and average experience.
# min experience and max experience define the average experience.
# min payrate and max payrate define the average payrate.

df['avg_payrate']=(df['min_pay'].values + df['max_pay'].values)/2
df['avg_experience']=(df['min_experience'].values + df['max_experience'].values)/2
df.head(5)
df['postdate'].dtypes
# parse the dates, currently coded as strings, into datetime format

df['postdate'] = pd.to_datetime(df['postdate'])
# extract year from date

df['Year'] = df['postdate'].dt.year
df['Year'].head(5)
# extract month from date

df['Month'] = df['postdate'].dt.month
df['Month'].head(5)
df['Month'].tail(5)
# extract day from date

df['Day'] = df['postdate'].dt.day
df['Day'].head(5)
df['Day'].tail(5)
# drop the original Date variable

df.drop('postdate', axis=1, inplace = True)
df.head(5)
df['joblocation_address'].value_counts().head(10)
replacements = {
   'joblocation_address': {
      r'(Bengaluru/Bangalore)':'Bangalore',
      r'Bengaluru':'Bangalore',
      r'Bangalore':'Bangalore',
      r'Bangalore Bangalore':'Bangalore',
      r'Hyderabad/Secunderabad':'Hyderabad',
      r'Mumbai , Mumbai':'Mumbai',
      r'Noida': 'Delhi',
      r'Delhi': 'Delhi',
      r'Gurgaon': 'Delhi',
      r'Delhi/NCR(National Capital Region)':'Delhi',
      r'Delhi/NCR(National Capital Region) ':'Delhi',
      r' Delhi/NCR(National Capital Region) ':'Delhi',
      r' Delhi/NCR(National Capital Region)':'Delhi',
      r'DELHI(NATIONAL CAPITAL REGION)':'Delhi',
      r'Delhi,Delhi':'Delhi',
      r'Noida/Greater Noida':'Delhi',
      r'Ghaziabad': 'Delhi',
      r'Delhi/NCR(National Capital Region),Gurgaon':'Delhi',
      r'NCR,NCR': 'Delhi',
      r'Delhi/NCR':'Delhi', 
      r'Bangalore,Bangalore / Bangalore':'Bangalore',
      r'Bangalore,karnataka': 'Bangalore',
      r'Delhi NCR':'Delhi',
      r'Delhi':'Delhi',
   }
}
df.replace(replacements, regex=True, inplace=True)
joblocation_address = df['joblocation_address'].value_counts()
# filter and find unique() cities from data set

df.joblocation_address = df.joblocation_address.str.upper()
new_location =df.joblocation_address.str.strip().str.split(",", expand = True)[0].str.split(" ", expand = True)[0].value_counts().reset_index()
new_location.columns = ["Location", "Job_Opportunities"]
new_location = new_location[:10]
new_location.style.background_gradient(cmap = "PuOr")
# drop the original Date variable

df.drop('payrate', axis=1, inplace = True)
# drop the original Date variable

df.drop('experience', axis=1, inplace = True)
# drop the original Date variable

df.drop('uniq_id', axis=1, inplace = True)
# a general overview of data
# T means Transpose

df.describe().T
#Dataset Summary statistics - categorical variables

df.describe(include = ['object']).T
# find categorical variables

categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n', categorical)
# view the categorical variables

df[categorical].head()
count_missing = df[categorical].isnull().sum()
percent_missing =  count_missing* 100 / df.shape[0]
missing_value_df = pd.DataFrame({'count_missing': count_missing,
                                 'percent_missing': percent_missing})

missing_value_df.style.background_gradient(cmap='tab20b')
# find numerical variables

numerical = [var for var in df.columns if df[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)
# view the numerical variables

df[numerical].head()
# check missing values in numerical variables

count_missing = df[numerical].isnull().sum()
percent_missing =  count_missing* 100 / df.shape[0]
missing_value_df = pd.DataFrame({'count_missing': count_missing,
                                 'percent_missing': percent_missing})

missing_value_df.style.background_gradient(cmap='coolwarm')
# view summary statistics in numerical variables

print(round(df[numerical].describe()))
cor_mat = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
plt.title('Correlation Heatmap of Job Market Analysis of India')
sns.heatmap(cor_mat, ax=ax, annot=True,cmap="PuOr", center=0, linewidths=0.08,linecolor="magenta")
plt.show()
plt.figure(figsize=(10,6))
corr = df.corr()["min_pay"]
corr.sort_values().head(20)[:-3].plot(kind='bar', color= ['olive','fuchsia',
                                                                'limegreen','grey','yellow','aqua',
                                                                'lawngreen','red',])
corr.abs().sort_values(ascending=False)[3:]
#display the company names..highest to lowest

com_Category = df.company.str.lstrip().str.rstrip().value_counts().reset_index()
com_Category.columns = ["Company", " Number of Company"]
com_Category = com_Category[:10]
com_Category.style.background_gradient(cmap = "tab20c")
#pLot of companies

f,ax=plt.subplots(figsize=(17,7))
df['company'].value_counts().head(10).plot(kind = 'bar', color =['fuchsia','orange',
                                                                 'limegreen','yellow','grey','aqua',
                                                                 'lawngreen','deepskyblue','indigo','skyblue'])
plt.title('Bar Plot', fontsize=22)
plt.show()
# These data shows the biggest industry in the country.

ind_Category = df.industry.str.lstrip().str.rstrip().value_counts().reset_index()
ind_Category.columns = ["Industry", " Number of Industry"]
ind_Category = ind_Category[:10]
ind_Category.style.background_gradient(cmap = "Blues")
# Plot of Industries.

f,ax=plt.subplots(figsize=(18,7))
explode =(0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2)
df['industry'].value_counts().head(10).plot(kind = 'pie', startangle=60, explode = explode,autopct='%1.1f%%')
plt.title('Pie Chart', fontsize=22)
plt.show()
#diplay the jobtitle.

Category = df.jobtitle.str.lstrip().str.rstrip().value_counts().reset_index()
Category.columns = ["jobtitle", " Number of Jobtitle"]
Category = Category[:10]
Category.style.background_gradient(cmap = "Greens")
#scatter plot for jobtitle.

fig=px.scatter(df['jobtitle'].value_counts().head(10))
fig.show()
# display the skills

# These data shows the biggest industry in the country.

skills_Category = df.skills.str.lstrip().str.rstrip().value_counts().reset_index()
skills_Category.columns = ["Skills", " Number of Skills"]
skills_Category = skills_Category[:10]
skills_Category.style.background_gradient(cmap = "hot")

# horizontal plot of skills

f,ax=plt.subplots(figsize=(17,7))
df['skills'].value_counts().head(5).plot(kind = 'line', color = 'midnightblue')
plt.title(' Line Plot', fontsize=22)
plt.show()
# filter and find unique() cities from data set

df_joblocation_address = df.joblocation_address.str.upper()
new_location =df.joblocation_address.str.strip().str.split(",", expand = True)[0].str.split(" ", expand = True)[0].value_counts().reset_index()
new_location.columns = ["Location", "Job_Opportunities"]
new_location = new_location[:10]
new_location.style.background_gradient(cmap = "PuOr")
plt.figure(figsize = (17,7))
plt.bar(new_location.Location,new_location.Job_Opportunities,color = ['orange','fuchsia',
                                                                'limegreen','grey','yellow','aqua',
                                                                'lawngreen','deepskyblue','indigo','skyblue'])
plt.xlabel("Locations")
plt.ylabel("Job Opportunities")
plt.xticks(new_location.Location, rotation = "60")
plt.title("Bar Plot")
plt.show()
#display the relation between min exp. and min payrate.

f,ax=plt.subplots(figsize=(17,7))
sns.stripplot(x='min_experience', y='min_pay', data=df, jitter=False)
plt.title('Seaborn Stripplots', fontsize=22)
plt.show()
#display the relation between max exp. and max payrate.

sns.catplot(x='max_experience', y='max_pay', data=df , kind="boxen" , width=0.4 ,aspect=3)
plt.title('Seaborn Catplots using Boxen', fontsize=22)
plt.show()
#display the relation between average exp. and average payrate.

f,ax=plt.subplots(figsize=(20,8))
sns.stripplot(x='avg_experience', y='avg_payrate', data=df, jitter=False, size=6)
plt.title('Seaborn Stripplots', fontsize=22)
plt.show()
#display the relation between min, max exp. and min payarte.

sns.pairplot(df, size=7,aspect=1,
             x_vars=["min_experience","max_experience"],
             y_vars=["min_pay"], diag_kind="kde", 
             plot_kws=dict(s=50, edgecolor = 'darkgreen', color="aqua", linewidth=1.5),diag_kws=dict(shade=True))

#display the relation between min, max exp. and max payrate.

sns.pairplot(df, 
             size=7, aspect=1, 
             x_vars=["min_experience","max_experience"],
             y_vars=["max_pay"],
             kind="reg")
#display the max payrate and industries comparsion.

df[['max_pay','industry']].groupby(["industry"]).median().sort_values(by='max_pay',
                                                                        ascending=False).head(10).plot.bar(color='springgreen')
plt.title('Bar Plot', fontsize=22)
plt.show()
#display the min payrate and industries comparsion.

df[['min_pay','industry']].groupby(["industry"]).median().sort_values(by='min_pay',
                                                                        ascending=False).head(10).plot.bar(color='magenta')
plt.title('Horizontal Bar Plot', fontsize=22)
plt.show()
#display the average payrate and skills comparsion.

df[['avg_payrate','skills']].groupby(["skills"]).median().sort_values(by='avg_payrate',
                                                                  ascending=False).head(10).plot.bar(color='skyblue')
plt.title('Bar Plot', fontsize=22)
plt.show()
df[['avg_payrate','jobtitle']].groupby(["jobtitle"]).median().sort_values(by='avg_payrate',
                                                                        ascending=False).head(10).plot.bar(color='orange')
plt.title('Horizontal Bar Plot', fontsize=22)
plt.show()
# Position are available in minimum payrate

plt.figure(figsize=(10, 5))
df['min_pay'].hist(rwidth=0.9, bins=15, color='aqua')
plt.title('Minimum payment')
plt.xlabel('Payment')
plt.ylabel('Position')
plt.show()
# Position are available in maximum payrate
plt.figure(figsize=(10, 5))
df['max_pay'].hist(rwidth=0.8, bins=15, color='r')
plt.title('Maximum payment')
plt.xlabel('Payment')
plt.ylabel('Position')

plt.show()
# Number of position are available in the industry.

max_positions = df.loc[df.numberofpositions > 100].loc[:,['numberofpositions','industry']]
plt.figure(figsize=(15, 5))
explode =(0.1,0.12,0.2,0.1,0.2,0.05,0.1,0.3,0.5,1)
hist_position_value = pd.value_counts(max_positions.industry)
hist_position_value.index
hist_position_value[hist_position_value >1].plot(kind='pie',startangle=40,explode=explode, autopct='%1.1f%%')
plt.title('Pie Chart', fontsize=22)
plt.show()