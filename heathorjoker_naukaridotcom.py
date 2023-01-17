import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(rc={'figure.figsize':(12,8.27)})

import re

import matplotlib.pyplot as plt
csv_path = '/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv'
ndf = pd.read_csv(csv_path)
ndf.drop(['Uniq Id','Crawl Timestamp'],axis=1,inplace=True)
#Location wise job salary

#-not much values for job salary
locdf = ndf['Location'].str.split(",",expand=True)

for i in range(16):

    locdf[i].fillna('GARBAGE',inplace=True)

total_cities = []

for i in range(16):

    total_cities = total_cities + locdf[i].unique().tolist()

cities = np.unique(np.array(total_cities))

for i in range(len(cities)):

    cities[i] = cities[i].strip()

err = []

for ct in cities:

    try:

        locdf['IN_'+ct] = ndf['Location'].str.contains(ct)

    except:

        err.append(ct)

removeCols = []

for col in locdf.columns[16:]:

    if locdf[col].sum() < 300:

        removeCols.append(col)

locdf.drop(removeCols,axis=1,inplace=True)

#remove redundant columns

redd_cols = ['IN_Delhi NCR','IN_Mumbai Suburbs','IN_Navi Mumbai','IN_']

locdf.drop(redd_cols,axis=1,inplace=True)

locCols = locdf.columns[16:]

for col in locCols:

    ndf[col] = locdf[col]
#Location wise industries 

#This function will plot industry in top distribution among top job cities

def plotIndustryLocationDistribution(industry):

    ndf['COL_'+industry] = ndf['Industry'].str.contains(industry,case=False,na=False)

    locations = ['Ahmedabad','Bengaluru','Chandigarh','Chennai','Delhi','Gurgaon','Hyderabad','Kolkata','Mumbai','Noida','Pune']

    sums = []

    for loc in locations:

        sums.append(ndf[ndf['COL_'+industry]]['IN_'+loc].sum())

    return locations,sums
fig, axes = plt.subplots(2, 2,sharex=True)



axes[1][1].tick_params(axis='x', rotation=90,pad=10)

axes[1][0].tick_params(axis='x', rotation=90,pad=10)



loc,sums = plotIndustryLocationDistribution('Education')

sns.barplot(x=loc,y=sums,ax=axes[0][0]).set_title('Education')



loc,sums = plotIndustryLocationDistribution('IT-Software')

sns.barplot(x=loc,y=sums,ax=axes[0][1]).set_title('IT-Software')



loc,sums = plotIndustryLocationDistribution('Media')

sns.barplot(x=loc,y=sums,ax=axes[1][0]).set_title('Media')



loc,sums = plotIndustryLocationDistribution('Advertising')

sns.barplot(x=loc,y=sums,ax=axes[1][1]).set_title('Advertising')



fig.text(0.5, -0.05, 'Cities', ha='center', va='center')

fig.text(0.06, 0.5, 'Job Counts', ha='center', va='center', rotation='vertical')



plt.suptitle('Location wise distribution of industries')
#top Skills Barplot

skdf = ndf['Key Skills'].str.split("|",expand=True)

for col in skdf.columns:

    skdf[col] = skdf[col].str.strip()

skillCountArray = skdf[0].value_counts()

totalClms = skdf.columns.stop

for i in range(1,185):

    skillCountArray = skillCountArray.add(skdf[i].value_counts(),fill_value=0)

cntArr = skillCountArray.sort_values(ascending=False)

x = cntArr.index[:20]

y = cntArr[x]

axs = sns.barplot(y=x,x=y)

axs.set_title('Top Skills Required Overall')

axs.set_ylabel('Skills')

axs.set_xlabel('Job Count')
#plot experience wise jobs
#looking at 'Job Experience Required' column, there are 5 types of experiences format 

# 1. x - y years

# 2. x years and above

# 3. X Years

# 4. Not Mentioned

# 5. NaN
#Removing type 4 and 5 values and stripping values

expSer = ndf['Job Experience Required'].dropna()

expSer.drop(expSer[expSer == 'Not Mentioned'].index,inplace=True)

expSer = expSer.str.strip()

#extracting type 1 to 3 values

expSertyp1 = expSer[expSer.str.contains('\d - \d')]

expSertyp2 = expSer[expSer.str.contains('\d years and above',flags=re.IGNORECASE)]

expSertyp3 = expSer[expSer.str.contains('^\d Years$')]

#considering 60 as maximum Age for work

#Define Experience Array

expJobs = [0 for i in range(61)]

#Fill Experience Array for Type 1

expdf = expSertyp1.value_counts().reset_index()

expdf['fiyr'] = expdf['index'].str.split("-",expand=True)[0].astype('int')

expdf['seyr'] = expdf['index'].str.split("-",expand=True)[1].str.strip().str.split(" ",expand=True)[0].astype('int')

expdf.rename({'Job Experience Required':'cnt'},axis=1,inplace=True)
def fillExpArrayTyp1(row):

    r1 = row['fiyr']

    r2 = row['seyr'] + 1

    for i in range(r1,r2):

        expJobs[i] = expJobs[i] + row['cnt']

tp = expdf.apply(fillExpArrayTyp1,axis=1)
#Fill Experience Array for Type 2

def fillExpArrayTyp2(row):

    r1 = int(row.split(" ")[0])

    r2 = 61

    for i in range(r1,r2):

        expJobs[i] = expJobs[i] + 1

tp = expSertyp2.map(fillExpArrayTyp2)
#Fill Experience Array for Type 3

def fillExpArrayTyp3(row):

    r1 = int(row.split(" ")[0])

    expJobs[r1] = expJobs[r1] + 1

tp = expSertyp3.map(fillExpArrayTyp3)
#plot exper Experience wise Job Distribution

x = [i for i in range(61)]

y = expJobs

fig = sns.lineplot(x = x,y=y,color='red',linewidth=2.5)

fig.set_xlabel('AGE')

fig.set_ylabel('Job Count')

fig.set_title('Experience wise Job Distribution')

fig.set_xticks(np.linspace(0,60,11), minor=False)
#From above plot we can conclude that we have more job opportunities in 0 to 6 years experience
sub = sns.countplot(y='Role Category',data=ndf,order=ndf['Role Category'].value_counts().iloc[:20].index,palette="Greens_d")

for p in sub.patches:

    width = p.get_width()

    plt.text(5+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{}'.format(width),

             ha='left', va='center')