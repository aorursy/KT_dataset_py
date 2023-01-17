import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.io as pio
import missingno as msno
import plotly.express as px
from scipy import stats
df=pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head(5)
df.info()
#Number of the rows and columns
df.shape
df.describe(include='object')
# As you see,you can see the top for each attributes.
# drop some attributes that we do not need(drop 2 columns)
df.drop(["Unnamed: 0","Job Description"],axis=1,inplace=True)
df.head()
# we want to change the name of the attributes
df.rename(columns={'Job Title':'job_title','Salary Estimate':"salary_estimate",'Rating':'rating',
       'Company Name':'company_name', 'Location':'location', 'Headquarters':'headquarters', 'Size':"size", 'Founded':"founded",
       'Type of ownership':'type_of_ownership', 'Industry':'industry', 'Sector':'sector', 'Revenue':'revenue', 'Competitors':'competitors',
       'Easy Apply':'easy_apply'},inplace=True)
df.head(5)
#use filter with some atrributes(Health care sector which is founded after 2010)
df[(df['founded']>2010) & (df['sector']=='Health Care')][:5]
#size of the employees which compare to easy apply
df['easy_apply'].unique()
easy_apply_size=pd.crosstab(df['easy_apply'],df['size'])
easy_apply_size
# Categorize the revenue with their amount
df['revenue'].value_counts()
#clean the Glassdoor est from salary_estimate
df['salary_estimate']=df['salary_estimate'].apply(lambda x: str(x).replace(' (Glassdoor est.)','')if ' (Glassdoor est.)' in str(x) else str(x))
df.head(5)
#clean the k from salary estimate
df["salary_estimate"]=df["salary_estimate"].apply(lambda x:str(x).replace('K','')if 'K' in str(x) else str(x))
df.head(5)
# clean the - form salary estimate
df["salary_estimate"]=df["salary_estimate"].apply(lambda x: str(x).replace('-',',') if '-' in str(x) else str(x))
df.head(5)
#clean the employess from size
df['size']=df['size'].apply(lambda x: str(x).replace('employees',',') if 'employees' in str(x) else str(x))
df.head(5)
#clean the usd from revenue
df['revenue']= df['revenue'].apply(lambda x: str(x).replace('(USD)','')if'(USD)'in str(x) else str(x))
df.head(5)
# change the -1 for easy_apply to False
df['easy_apply']=df['easy_apply'].replace(['-1'],'False')
df.head(5)
new = df["job_title"].str.split(",", n = 1, expand = True) 
# making separate first name column from new data frame 
df["job"]= new[0] 
df.drop(columns =["job_title"], inplace = True) 
df.head(3)
df.groupby ('easy_apply').mean().sort_values('rating',ascending=True)
pd.pivot_table(df,index=['sector','easy_apply'],values='founded')
#split the size into upper and  lower 
for i in range(df.shape[0]):
    size=df.loc[i,"size"]
    if "to" in size:
        lower,upper=size.split("to")
        lower=lower.strip()
        _, upper, _ = upper.split(" ")
        upper=upper.strip()
        lower=int(lower)
        upper=int(upper)
    elif "+" in size: 
        lower,_=size.split("+")
        lower=int(lower)
        upper=np.inf
    else:
        lower=np.nan
        upper=np.nan
    df.loc[i,'minimum size']=lower
    df.loc[i,'maximum size']=upper
df.head(5)
df['minimum size']=df['minimum size'].apply(lambda x: str(x).replace('.0','')if '.0' in str(x) else str(x))
df.head(4)
df['maximum size']=df['maximum size'].apply(lambda x: str(x).replace('.0','')if '.0' in str(x) else str(x))
df.head(4)
df['minimum size']=df['minimum size'].replace(['nan'],'-1')
df.head(5)
df['maximum size']=df['maximum size'].replace(['nan'],'-1')
df.head(10)
df['maximum size']=df['maximum size'].replace(['inf'],'1')
df.head(10)
titles=list(df.columns)
titles[12],titles[13],titles[14],titles[15]=titles[13],titles[14],titles[15],titles[12]
titles
df=df[titles]
df.head(5)
#Top 20 companies with most number of jobs
plt.rcParams['figure.figsize']=(10,5)
df['company_name'].value_counts().sort_values(ascending=False).head(20).plot.bar(color='red')
plt.title('Top 20 companies with number of their jobs')
plt.xlabel('Companies')
plt.ylabel('Counts')
plt.show()
# Top 5 companies with number of the jobs
df['company_name'].value_counts().head(5)
plt.rcParams['figure.figsize']=(10,5)
df['job'].value_counts().sort_values(ascending=False).head(20).plot.bar(color='pink')
plt.title('The first 20 jobs with their amount ')
plt.xlabel('Kind of jobs')
plt.ylabel('counts')
plt.show()
#The 10 most jobs with their amount
df['job'].value_counts().head(10)
plt.rcParams['figure.figsize']=(10,6)
df['location'].value_counts().sort_values(ascending=False).head(20).plot.bar(color='yellow')
plt.title('The first 20 loaction with their amounts')
plt.xlabel('Loacation')
plt.ylabel('Counts')
plt.show()
df['location'].value_counts().head(10)
plt.rcParams['figure.figsize']=(10,5)
df['industry'].value_counts().sort_values(ascending=False).head(20).plot.pie(y='industry',autopct="%0.1f%%")
plt.title('The first 20 industry with their amount')
plt.axis('off')
plt.show()
df['industry'].value_counts().head(10)
sns.countplot(data=df,y='easy_apply',palette='hls')
plt.title('Amount of the easy apply ')
plt.figure(figsize=(10,5))
plt.show()
sns.relplot('easy_apply','rating',data=df,kind='line',hue='sector')
sns.catplot('easy_apply','founded',data=df,kind='box')
sns.pairplot(df[['rating','founded']])
sns.jointplot('rating','founded',data=df,kind='kde',color='red')
fig,ax=plt.subplots(figsize=(10,5))
sns.countplot(df['rating'],hue=df['easy_apply'],ax=ax)
plt.xlabel('Salary Estimate')
plt.ylabel('Easy Apply')
plt.xticks(rotation=50)
plt.show()
nums=['founded']
for i in nums:
    sns.jointplot(x=df[i],y=df['rating'],kind='reg',color='red')
    plt.xlabel(i)
    plt.ylabel('Counts')
    plt.xticks()
    plt.show()
    
count_True=len(df[df['easy_apply']=='True'])
count_False=len(df[df['easy_apply']=='False'])
percentage_of_True=count_True/(count_True+count_False)
percentage_of_False=count_False/(count_True+count_False)
print('percentage of True is',percentage_of_True*100)
print('percentage of False is',percentage_of_False*100)


labels=['percentage_of_True','percentage_of_False']
values=[percentage_of_True,percentage_of_False]
plt.title('Percentage of easy apply')
plt.pie(values, labels=labels,autopct='%11.1f%%')
plt.show()
# The first 10 most popular jobs 
plt.rcParams['figure.figsize']=(10,5)
df['job'].value_counts().sort_values(ascending=False).head(10).plot.pie(y='job',autopct="%0.1f%%")
plt.title('The first 20 job with their amount')
plt.axis('off')
plt.show()
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
le=preprocessing.LabelEncoder()
df.columns
df.salary_estimate=le.fit_transform(df.salary_estimate)
df.industry=le.fit_transform(df.industry)
df.location=le.fit_transform(df.location)
df.headquarters=le.fit_transform(df.headquarters)
df.type_of_ownership=le.fit_transform(df.type_of_ownership)
df.sector=le.fit_transform(df.sector)
df.competitors=le.fit_transform(df.competitors)
df.job=le.fit_transform(df.job)
df.easy_apply=le.fit_transform(df.easy_apply)
df.revenue=le.fit_transform(df.revenue)
df.head(5)
#delete the company name for making the model
df=df.drop(columns=["size","company_name"])
df.head(5)
#No missing values
df.isnull().sum()
train_features = df.iloc[:80,:-1]
test_features = df.iloc[80:,:-1]
train_targets = df.iloc[:80,-1]
test_targets = df.iloc[80:,-1]
train_features
missing_val_count_by_column = (df.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
model = DecisionTreeClassifier(criterion='gini' )
model.fit(train_features,train_targets)
predictions=model.predict(test_features)
predictions
score=accuracy_score(test_targets,predictions)
score
model=RandomForestClassifier()
model.fit(train_features,train_targets)
predictions=model.predict(test_features)
predictions
score=accuracy_score(test_targets,predictions)
score
model=MLPClassifier()
model.fit(train_features,train_targets)
predictions=model.predict(test_features)
predictions
score=accuracy_score(test_targets,predictions)
score