
import os# import command is used for importing external modules

print(os.listdir("../input")) 

# Any results you write to the current directory are saved as output.
import numpy as np #used for scientific computation
import pandas as pd #used for data mugging and preprocessing
import matplotlib.pyplot as plt #data visualization library
from pandas import DataFrame as show # dataframe is the optimised structure used here to clean and analyse data
import seaborn as sns # stastical visualization library
import squarify #used to make square area plots
%matplotlib inline #used in jupyter notebook for interactive visualizations within notebook
df=pd.read_csv('../input/startup_funding.csv',dtype={
    
    'InvestmentType':'category'
    
})#read_csv function is used to import dataset 'startup_funding.csv'


df.head(4)#display first 4 rows of dataframe
df.tail(4)#display first 4 rows of dataframe
print("Information of total number of non-empty columns")
print("-------------------------------------------------")
print(df.info(null_counts=True))
print('Rows {rows} in total and Columns {columns} in total'.format(rows=df.shape[0],columns=df.shape[1]))
print("Columns and their datatypes")
df.dtypes #.dtypes are used to display datatypes of each column
print("Frequency count of missing values")
df.apply(lambda X:sum(X.isnull())) 
#apply function is used to do mapping column-wise
#apply function can apply tranformations to each column individually

plt.figure(figsize=(10,5)) #plt is the object of matplot lib and .figure() is used to show or change properties of graphs
sns.heatmap(df.isnull(),cmap='viridis',yticklabels=False,cbar=False)#heatmaps are matrix plots which can visualize data in 2D
plt.show()

print("Here we can see in date column error- '.' is there instead of '/'")
df[df['Date']=='12/05.2015']['Date']

df['AmountInUSD'].head(5)#head(n) displays n rows
df['CityLocation']=df['CityLocation'].fillna(value='NotSpecific')
df['IndustryVertical']=df['IndustryVertical'].fillna(value='Other')
import re#importing regular expressions
def convert_Slash(x):#converts citylocation where multiple citiescentres
    x=x.lower()#converting  whole data to lower case to avoid dublicate entries 
    if   re.search('/',x):
        return x.split('/')[0].strip()#converting multiple citycentres to single one 
    else :
        return x.strip()# removing extra spaces from left and right to reduce duplicate cities
df['CityLocation']=df['CityLocation'].apply(convert_Slash)
newdf=df.copy()#backup cleansed data
del newdf['Remarks']#remaks is deleted to overcome stability in analysis
del newdf['SNo']
print('Different categories of Inverstment Type before cleansing and removing duplicacy in categories are as follows- ')
newdf['InvestmentType'].value_counts().index# aggregating frequency count according to categories of investment type

print('Different categories of Inverstment Type after cleansing and removing duplicacy in categories are as follows- ')
newdf['InvestmentType']=newdf['InvestmentType'].apply(lambda x:x.replace(' ','').lower())#code to apply mapping to remove duplicate categories
newdf['InvestmentType'].value_counts().index
def rem_err_date(x):#function checks for error in format of date column in funding dataframe
    if re.search('.',x):#data column has formatting errors like '12/052015','13/042015' where backslash (/) is missing or at wrong position
        return x.replace('.','')
    return x

newdf['Date']=newdf['Date'].apply(rem_err_date)#applying user defined funciton to date column using apply() which maps u.d.f to each record of date column
newdf['Date'].replace('12/052015','12/05/2015',inplace=True)
newdf['Date'].replace('15/012015','15/01/2015',inplace=True)
newdf['Date'].replace('22/01//2015','22/01/2015',inplace=True)
newdf['Date'].replace('13/042015','13/04/2015',inplace=True)
newdf['Date']=pd.to_datetime(newdf['Date'],format='%d/%m/%Y')#d/m/y is the format and to_datetime() is used to convert the datatype of date column to "datetime" from string
print('processed datatype of Date column')
newdf.dtypes['Date']
def calculate_n_investors(x):#function to calculate record wise number of investors
    if  re.search(',',x) and x!='empty':
        return len(x.split(','))
    elif x!='empty':
        return 1
    else:
        return -1
newdf['numberofinvestors']=newdf['InvestorsName'].replace(np.NaN,'empty').apply(calculate_n_investors)#removing missing investors and replacing with 'empty'
n_inv2=newdf

n_inv=newdf['InvestorsName']
n_inv.fillna(value='None',inplace=True)
listed_n_inv=n_inv.apply(lambda x: x.lower().strip().split(','))
investors=[]
for i in listed_n_inv:
    for j in i:
        if(i!='None' or i!=''):
            investors.append(j.strip())
unique_investors=list(set(investors))

investors=pd.Series(investors)
unique_investors=pd.Series(unique_investors)
investors=list(investors[investors!=''])
unique_investors=list(unique_investors[unique_investors!=''])

for i in range(len(unique_investors)):
    for j in range(len(investors)):
        if(re.search(unique_investors[i],investors[j])):
            investors[j]=unique_investors[i]

def convert_AmountInUSD(x):
    if re.search(',',x):
        return (x.replace(',',''))
    return x
newdf['AmountInUSD']=newdf[newdf['AmountInUSD'].notnull()]['AmountInUSD'].apply(convert_AmountInUSD).astype('int')
newdf['AmountInUSD']=round(newdf['AmountInUSD'].fillna(np.mean(newdf['AmountInUSD'])))
newdf['AmountInUSD']=newdf['AmountInUSD'].astype('int')
newdf['InvestmentType'].fillna(method='bfill',inplace=True)#backward filling of null values
newdf.iloc[:,[1,2,3,4,6]]=newdf.iloc[:,[1,2,3,4,6]].applymap(lambda x: x.lower().replace(' ','') if pd.notnull(x) is True else x )

def check(x):
    if(pd.notnull(x)):
        return x.lower()
newdf.iloc[:,3]=newdf.iloc[:,3].apply(check)  

plt.figure(figsize=(10,5))
sns.heatmap(newdf.isnull(),cmap='viridis',yticklabels=False,cbar=False)
plt.show()
unique_startup_name=list(newdf['StartupName'].unique())
startupname=list(newdf['StartupName'])
for i in range(len(unique_startup_name)):
    for j in range(len(startupname)):
        if(re.search(unique_startup_name[i],startupname[j])):
            startupname[j]=unique_startup_name[i]

newdf['StartupName']=startupname
newdf.head(10)
newdf.head()
show(newdf.describe()['AmountInUSD'].astype(int))


print(newdf['StartupName'].nunique())

tp10fund=show(newdf.groupby('StartupName')['AmountInUSD'].sum().sort_values(ascending=False))
tp10fund.head(10)


from wordcloud import WordCloud, STOPWORDS


st=pd.Series(newdf.groupby('StartupName').sum()['AmountInUSD'].sort_values(ascending=False).head(40).index).head(30)
fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(20,20),dpi=300)

for char in range(len(st)):
    st[char]=st[char].replace(' ','_')

wc=WordCloud(max_words=20,stopwords=set(st),background_color='darkgrey',random_state=0).generate(str(st.values[:30]))
ax[0].axis('off')
ax[0].set_title('Most funded startups')
ax[0].imshow(wc)




c=np.array(newdf['IndustryVertical'].value_counts().head(30).index.unique())
for char in range(len(c)):
    c[char]=c[char].replace(' ','_')

               
wc=WordCloud(max_words=30,stopwords=set(c),background_color='plum',random_state=1).generate(str(c))

ax[1].axis('off')
ax[1].set_title('Most funded Industry Sector')

ax[1].imshow(wc)
plt.rcParams['axes.facecolor'] = 'white'




#details of top 10 funded startups
def find(x):
    if x in tp10fund.head(10).index:
        return True
    return False

n=newdf[newdf['StartupName'].apply(find)]
print('amount funded on top 10 startups')
n.describe().iloc[:,0]
pd.crosstab(n['StartupName'],columns=n['InvestmentType']).sort_values(by='privateequity',ascending=False)
newdf[newdf['StartupName']=='paytm']

cmi=show(newdf.groupby('StartupName')['numberofinvestors'].count().sort_values(ascending=False))
fig=plt.figure(figsize=(10,5))
sns.barplot(y='numberofinvestors',x='StartupName',data=cmi.reset_index().head())
plt.show()
cmi.head(10)
sns.kdeplot(data=cmi.reset_index()['numberofinvestors'],gridsize=20,)#most are  2 or 3 in number
plt.title('Density estimation for number of investors ')
plt.show()
top10=tp10fund.join(cmi)
sns.heatmap(top10.corr(),annot=True)
plt.title('Corelation Matrix')
plt.show()

show(investors)[0].nunique()
#investors with most funding
sh=show(investors)[0].unique()
cinvestors=show(investors)[0].value_counts()[2:]
cinvestors.head(10)
print("Top Investors in Frequency ")
plt.figure(figsize = (12,5))
bar= sns.barplot(x=cinvestors.index[:20],y=cinvestors.values[:20])
bar.set_xticklabels(bar.get_xticklabels(),rotation=70)
bar.set_title("top Investors by funding on multiple days ", fontsize=15)
bar.set_xlabel("", fontsize=12)
bar.set_ylabel("Frequency of Funding", fontsize=12)
plt.show()
Investors=cinvestors.index
arr =np.array(Investors)     
for investor in range(len(arr)):
    arr[investor]=arr[investor].strip()
    arr[investor]=arr[investor].replace(' ','_')
    arr[investor]=arr[investor].replace("'",'')
    arr[investor]=arr[investor].replace("",'')

Ind=show(arr)[0].apply(lambda x: x.strip().lower())
Ind=Ind.values
fig=plt.figure(figsize=(10,10),dpi=700)

wc=WordCloud(max_words=30,stopwords=set(Ind),background_color='hotpink',random_state=1).generate(str(Ind[:30]))
plt.axis('off')
plt.title('Top Investors by funding on multiple days')
plt.imshow(wc)
plt.show()


d=dict()#to store individual investors and funding amount in key-value pairs
for i in unique_investors:
    for j in range(len(listed_n_inv)):
        if i in listed_n_inv[j]:
            d[i]=newdf['AmountInUSD'][j]/len(listed_n_inv[j])#taking average of amount 
            
            
Investor_amount=pd.Series(d,name='Amount')

Investor_amount=show(Investor_amount,).reset_index().groupby('index').sum()['Amount'].sort_values(ascending=False).head(100)
Investor_amount=show(Investor_amount).reset_index()
Investor_amount.columns=["Investor","Amount"]

print('Top 10 Most funded Investors')
plt.figure(figsize=(12,7))
sns.barplot(y='Investor',x='Amount',data=Investor_amount.head(10))
print(Investor_amount.head(10))
plt.show()

top_industry_vertical={}
for i in Investor_amount['Investor'].head(20):
    for j in range(len(listed_n_inv)):
        if i in listed_n_inv[j]:
            top_industry_vertical[i]=newdf['IndustryVertical'][j]

plt.figure(figsize=(12,7))
sns.countplot(y=pd.Series(top_industry_vertical))
plt.title('Industry sector opted by top investors' )
print('top investor\'s favourite Industry  ')
print(pd.Series(top_industry_vertical))
plt.show()

newdf.groupby('InvestmentType').sum()['AmountInUSD']

newdf['InvestmentType'].value_counts()


# converting all industry vertical entries to lower to avoid category duplication
newdf['IndustryVertical']=newdf['IndustryVertical'].apply(lambda x:x.lower())

#in which sector there are most startups
d=newdf['IndustryVertical'].value_counts().head(5)
f=newdf.groupby('InvestmentType').sum()['AmountInUSD']
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,10))
labels=[d.index,f.index]
size=[d.values,f.values]
colors = [['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','pink'],['green','pink','red','yellow']]
plt.axis('equal')
explode = ((0.1, 0, 0, 0,0),(-0.5,0.5,0.1,0.1))
ax[0].pie(size[0],explode=explode[0], labels=labels[0], colors=colors[0],
        autopct='%1.1f%%', shadow=True, startangle=140)

ax[1].pie(size[1],explode=explode[1], labels=labels[1], colors=colors[1],autopct='%1.5f%%', shadow=True, startangle=140)
plt.show()
plt.figure(figsize=(14,5))
iv=newdf['IndustryVertical'].value_counts().head(10)
iv.plot.bar()

plt.title('Frequency  of Industry Vertical ')
plt.ylabel('Frequency')
plt.xlabel('Industry Vertical')
plt.show()
plt.figure(figsize=(17,12))
mean_amount = newdf.groupby('CityLocation').mean()["AmountInUSD"].astype('int').sort_values(ascending=False).iloc[1:].head(10)
squarify.plot(sizes=mean_amount.values,label=mean_amount.index, value=mean_amount.values,color=['crimson','seagreen','olive','hotpink','deepskyblue','grey','purple','lime','yellow','orange'])
plt.title('Distribution of Startups across Top cities')

sns.barplot(y='CityLocation',x='AmountInUSD',data=newdf[(newdf['InvestmentType']=='debtfunding')|(newdf['InvestmentType']=='crowdfunding')],estimator=np.sum,palette='coolwarm')

#average investment in banglore is most
plt.show()
#amehdabad is the market place for dept funding

dnewdf=newdf.set_index('Date')
dnewdf.head()
print('total number of unique startups funded in 2017 -'+str(len(dnewdf['2017']['StartupName'].unique())))
print('total number of unique startups funded in 2016 -'+str(len(dnewdf['2016']['StartupName'].unique())))
print('total number of unique startups funded in 2015 -'+str(len(dnewdf['2015']['StartupName'].unique())))

dnewdf['2017']['AmountInUSD'].sum()

dnewdf['2016']['AmountInUSD'].sum()



dnewdf['2015']['AmountInUSD'].sum()

plt.title('total funding amount')
dnewdf.resample('AS')['AmountInUSD'].sum().plot.bar()
plt.show()
q=dnewdf['AmountInUSD'].resample('AS').mean()
q.plot(kind='bar')
plt.title('average funding amount')

a=dnewdf['AmountInUSD'].resample('AS').sum()
fig,ax=plt.subplots(nrows=1,ncols=2)

fig.tight_layout(pad=3) # Or equivalently,  "plt.tight_layout()"
fig.set_figheight(11)
fig.set_figwidth(24)

explode = (0.1, 0.1, 0.1)
ax[0].pie(a,autopct='%1.1f%%',shadow=True,startangle=180,explode=explode,colors=['red','blue','green'],labels=['2015','2016','2017'])
ax[0].set_title('total funding amount per annum')

q.plot(color='y',ax=ax[1])
ax[1].set_title('average funding variation')
plt.show()

newdf['year']=newdf['Date'].dt.year
newdf['month']=newdf['Date'].dt.month
fig =plt.figure(figsize=(20,7))
fig.set_figheight
ts_month = newdf.groupby(['year', 'month']).agg({'AmountInUSD':'sum'})['AmountInUSD']
ts_month.plot(linewidth=4, color='crimson',marker="o", markersize=10, markerfacecolor='olive')
plt.ylabel('USD in Billions')
plt.xlabel('Month');
plt.title('Funding Variation Per Month from 2015-2017')

sns.set_context('poster',font_scale=1)
plt.figure(figsize=(20,6))
a=dnewdf['2015'].resample('MS').sum()['AmountInUSD'].plot()
a.set_title('total funding in 2015')
#insights of june-july and augest of 2015
fig2,axes = plt.subplots(nrows=1,ncols=3,figsize=(20,5))
a1=dnewdf['2015-06':'2015-8'].resample('d')['AmountInUSD'].sum().plot(ax=axes[0],lw=1)
a1.set_title('important months')
a2=sns.barplot(data=dnewdf['2015'].resample('Q').sum().reset_index(),y='AmountInUSD',x=['Q1','Q2','Q3','Q4'],ax=axes[1])
a2.set_title('Quaterly')
a3=dnewdf['2015'].resample('B')['AmountInUSD'].sum().plot(ax=axes[2],lw=1)
a3.set_title('            funding on business days')
a3.set_xticklabels('2016')
plt.show()
sns.set_context('poster',font_scale=1)


plt.figure(figsize=(22,6))


a=dnewdf['2016'].resample('MS').sum()['AmountInUSD'].plot()
a.set_title('total funding in 2016')
#insights of june-july and augest of 2015

fig2,axes = plt.subplots(nrows=1,ncols=3,figsize=(20,5))

a1=dnewdf['2016-06':'2016-10'].resample('d')['AmountInUSD'].sum().fillna(method='ffill').plot(ax=axes[0],lw=1)
a1.set_title('  june-july-augest-sept')

sns.barplot(data=dnewdf['2016'].resample('Q').sum().reset_index(),y='AmountInUSD',x=['Q1','Q2','Q3','Q4'],ax=axes[1])
a2.set_title('Quaterly')

a3=dnewdf['2016'].resample('B')['AmountInUSD'].sum().plot(ax=axes[2],lw=1)
a3.set_title('       funding by business days')

a3.set_xticklabels('2016')
plt.show()
sns.set_context('poster',font_scale=1)


plt.figure(figsize=(20,6))


a=dnewdf['2017'].resample('MS').sum()['AmountInUSD'].plot()
a.set_title('total funding in 2017')
#insights of june-july and augest of 2015

fig2,axes = plt.subplots(nrows=1,ncols=3,figsize=(20,5))
a1=dnewdf['2017-02':'2017-6'].resample('d')['AmountInUSD'].sum().fillna(method='ffill').plot(ax=axes[0],lw=1)
a1.set_title('funding by days')

a2=sns.barplot(data=dnewdf['2017'].resample('Q').sum().reset_index(),y='AmountInUSD',x=['Q1','Q2','Q3'],ax=axes[1])
a2.set_title('funding by quarter')

a3=dnewdf['2017'].resample('B')['AmountInUSD'].sum().plot(ax=axes[2],lw=1)
a3.set_title('          funding by business days')
a3.set_xticklabels('  2017')
plt.show()
fig,ax=plt.subplots(1,2,figsize=(20,5))

sns.set_context('paper',font_scale=2)
dft=dnewdf.resample('Q').sum()
dft.index=dft.reset_index()['Date'].apply(lambda x: x.date())
a=sns.barplot(data=dft.reset_index(),x='Date',y='AmountInUSD',ax=ax[0])

plt.sca(ax[0])
plt.xticks(rotation=90)
plt.title('')

sns.distplot(dnewdf.resample('q').sum()['AmountInUSD'],ax=ax[1])

plt.show()

fig,ax=plt.subplots(nrows=2,ncols=1)
fig.set_figheight(13)
fig.set_figwidth(15)
ax[0].set_ylabel('USD in Billions')
quarter=dnewdf['AmountInUSD'].resample('Q').sum().astype('int')
quarter[1:].plot(linewidth=4, color='hotpink', marker="o", markersize=10, markerfacecolor='lime',ax=ax[0])
fig.tight_layout(pad=3) # Or equivalently,  "plt.tight_layout()"
plt.title('quartly funding amount')
explodes = (0,0.1, 0., 0,0,0,0,0.1,0.1,0.1,0.1)
ax[1].pie(quarter[0:],autopct='%1.1f%%',startangle=240,explode=explodes,labels=['2015-1','2015-2','2015-3','2015-4','2016-1','2016-2','2016-3','2016-4','2017-1','2017-2','2017-3'])

plt.show()

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(25,5))

plt.sca(axes[0])
plt.title('2015')
sns.kdeplot(dnewdf.resample('q').mean()['AmountInUSD']['2015'],ax=axes[0])

plt.sca(axes[1])
plt.title('2016')
sns.kdeplot(dnewdf.resample('q').mean()['AmountInUSD']['2016'],ax=axes[1])

plt.sca(axes[2])
plt.title('2017')
sns.kdeplot(dnewdf.resample('q').mean()['AmountInUSD']['2017'],ax=axes[2])
plt.show()
plt.scatter(x=newdf['AmountInUSD'],y=newdf['InvestmentType'])
plt.title('InvestmentType Vs AmountInUSD')
plt.xlabel('amount')
plt.ylabel('type')
plt.show()
plt.figure(figsize=(15,8))
d2015=dnewdf['2015']
print(sns.stripplot(data=d2015,x='InvestmentType',y='AmountInUSD',jitter=True,hue='numberofinvestors'))

print(d2015['InvestmentType'].value_counts())
d2015[d2015['InvestmentType']=='privateequity'].sort_values(by='AmountInUSD',ascending=False).head(5)
plt.show()
plt.figure(figsize=(15,8))
d2016=dnewdf['2016']
print(sns.stripplot(data=d2016,x='InvestmentType',y='AmountInUSD',jitter=True,hue='numberofinvestors'))

print(d2016['InvestmentType'].value_counts())
d2016[d2016['InvestmentType']=='privateequity'].sort_values(by='AmountInUSD',ascending=False).head(2)
plt.figure(figsize=(15,8))
d2017=dnewdf['2017']
print(sns.stripplot(data=d2017,x='InvestmentType',y='AmountInUSD',jitter=True,hue='numberofinvestors'))

print(d2017['InvestmentType'].value_counts())
d2017[d2017['InvestmentType']=='privateequity'].sort_values(by='AmountInUSD',ascending=False).head(2)
