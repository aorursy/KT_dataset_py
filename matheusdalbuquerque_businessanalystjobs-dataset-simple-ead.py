import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for cool graphics
import seaborn as sb # even cooler graphics

dataset = pd.read_csv('/kaggle/input/business-analyst-jobs/BusinessAnalyst.csv',na_values=[-1,'-1'])
dataset
dataset.info()
indexs_range = 0 # how many rows are mismacthing
for i in dataset['Unnamed: 0']:
    if not i.isdigit():
        indexs_range +=1

dt_fix = dataset.iloc[-indexs_range:,:] # sub dataset with the mischating rows
dt_fix.drop(['Competitors','Easy Apply'],axis=1,inplace=True) 
dt_fix.columns = dataset.columns[2:] #ignores the first 2 columns names

dataset.drop(dataset.index[-indexs_range:],axis=0,inplace=True)
dataset.drop(['Unnamed: 0','index'],axis=1,inplace=True)
dataset = dataset.append(dt_fix)
dataset
dataset[['MinSalary','MaxSalary']] = dataset['Salary Estimate'].str.split(r"[\D]+",expand=True).drop([0,3],axis=1)
dataset['MinSalary'] = dataset['MinSalary'].apply(lambda x: int(x) * 1000)
dataset['MaxSalary'] = dataset['MaxSalary'].apply(lambda x: int(x) * 1000)
dataset['AverageSalary'] = dataset[['MaxSalary','MinSalary']].mean(axis=1)
dataset.drop(['Salary Estimate','MinSalary','MaxSalary'],axis=1,inplace=True)
dataset['Rating'] = dataset['Rating'].astype(float)
dataset[['StateName','State']] = dataset['Location'].str.split(', ',expand=True).drop([2],axis=1)
dataset['State'] = dataset['State'].str.replace('Los Angeles','CA')
sb.countplot(y='State', data=dataset).set_title("Number of job offers by State")
plt.rcParams["figure.figsize"] = (20,5)
fig, ax =plt.subplots(1,2)
sb.boxplot(x='State',y='Rating',data=dataset,ax=ax[0]).set_title("Distribution of Ratings by State")
sb.boxplot(x='State',y='AverageSalary',data=dataset,ax=ax[1]).set_title("Distribution of AverageSalary by State")
sb.countplot(y='Sector',data=dataset).set_title("Number of job offers by Sector")
plt.rcParams["figure.figsize"] = (10,10)
plus100 = dataset['Sector'].map(dataset['Sector'].value_counts()) > 100 # More than 100 job offers
fig, ax = plt.subplots(2,1)
sb.boxplot(y='Sector',x='Rating',data=dataset[plus100],ax=ax[0]).set_title("Distribution of Ratings by Sector")
sb.boxplot(y='Sector',x='AverageSalary',data=dataset[plus100],ax=ax[1]).set_title("Distribution of AverageSalary by Sector")
sb.countplot(x='State',hue='Sector',data=dataset[plus100])
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud

stopWords = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

def get_wordcloud(series): #simple function to tokenize and plot a said column
    word_cloud = ''
    
    for job in series:
        tokens = tokenizer.tokenize(job)
        for token in tokens:
            if token not in stopWords:
                word_cloud += ''.join(token) + ' '

    wordcloud = WordCloud(height=500,margin=0,max_words=300,
                          colormap='Set1').generate(word_cloud) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

job_titles = dataset['Job Title'].apply(lambda x: x.lower())

get_wordcloud(job_titles)
job_descrip = dataset['Job Description'].apply(lambda x: x.lower())
get_wordcloud(job_descrip)