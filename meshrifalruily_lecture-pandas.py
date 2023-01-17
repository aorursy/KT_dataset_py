import pandas as pd
#data = pd.read_csv('heart.csv')
data = pd.read_excel('../input/heart-xlsx-file/heart.xlsx')

data
data.head(3)
data.tail(3)

data.columns
data.dtypes
data.axes
data.ndim
data.shape
data.size
data.info()
data.cp.value_counts()
data['cp'].value_counts()
data.values[1:5]
data[ (data.cp==0) & (data.chol<200) ]
data['age']
data.age
data [ ['age', 'cp','sex' ] ]
data[19:25] # 25 execluded 
data.loc[1]  # negative index not working 
data.loc[1:2] #3 end range is inclusive 
data.loc[:,'age']
data.loc[19:25,  'age']
data.loc[2:4, 'age']
data.loc[2:4, ['age', 'sex','chol' ] ]
data.iloc[-2] # specific row and can use negative idex
data.iloc[2, 3] # can get specifc cell 
#one columns for all rows 
data.iloc[:,0] 
data.iloc[2:5, 1:4] # end range exclusive same as using data[2:5]
data.head()
# rows 3-7 and column 5 to end
data.iloc[3:8,5:]
#data.iloc[start:end:step]
# rows 5,7,9 and columns 2,4
data.iloc[5:10:2,2:5:2]
# rows 5,8,9 and columns 2,4,8
data.iloc[ [5,8,9]  , [2,4,8]  ]
data.sort_values(by='age',   ascending=False)
# arrange using two columns
data.sort_values(by=['age','chol'],ascending=False)
# arrange using two columns
data.sort_values(by=['age','chol'],ascending=[True, False])
#apply sort & filter by sex
data[data.sex==1].sort_values(by=['age'] )
data.sort_values(by=['sex','age'], ascending=[False, True] )
data['target'].value_counts()
sorted_data = data[data.sex==1].sort_values(by=['sex','age'] )
sorted_data
data_with_missing = pd.read_excel('../input/heart-missing/heart_missing.xlsx')
data_with_missing.head()
data_with_missing.info()
data_with_missing.chol.isnull()
data_with_missing.isnull()
data_with_missing.age.sum()

data_with_missing.sum()
##count Nulls in each columns  
data_with_missing.isnull().sum()
#detect columns with Null 
data_with_missing.isnull().sum()>0
# solution 1 for missing: Fill by 0
data_with_missing.head()
data_with_missing.fillna(0)
data_with_missing.head()
#2- remove rows or colulns with Null 
# remove rows in case number of rows with Null is small
data_with_missing.dropna() # remove all rows contains Null
# data_with_missing.dropna() default of axis=0 - rows
data_with_missing
# remove columns in case number of Nulls in column is large e.g. 80% of columns is null
data_with_missing.dropna(axis=1) # axis=1 mean column
data_with_missing.head()
#solution 3: in case null in rows or columns about 30% 
#filling : by 0, or pecific value 
data_with_missing.chol.fillna(150)
#fill by mean value
patient_mean= data_with_missing.chol.mean()

data_filled = data_with_missing.copy()

data_filled.chol.fillna(patient_mean, inplace=True) # change in same place of memory of data_filled 
data_filled
data_filled
# generate new feature from existing features
data_filled['risk'] = data_filled['chol']/data_filled['trestbps']
data_filled
data.groupby('sex').mean()
data.groupby('sex').max()
data.groupby('cp').mean()
data.groupby(['sex', 'cp']).mean()
#grouping using more than one folumns
data.groupby(['sex', 'target']).mean()
#grouping using more than one folumns
data.groupby(['sex','cp' ,'target']).mean()
data.groupby(['sex','cp' ,'target']).mean()['chol']
data.groupby(['sex','cp' ,'target']).mean().to_csv('group_data.csv')
import bs4 as bs
import urllib.request as req
url = req.urlopen('https://pythonprogramming.net/parsememcparseface/')
source = url.read()
soup = bs.BeautifulSoup(source,'lxml')

# title of the page
print(soup.title)

# get attributes:
print(soup.title.text)

# get attributes:
for paragraph in soup.find_all('p'):
    print(paragraph.string)
    print(str(paragraph.text))


# get attributes:
for paragraph in soup.find_all('tr'):
    #print(paragraph.string)
    row = str(paragraph.text)
    print(row.split('\n'))
    #print(str(paragraph.text))
    
#data = pd.DataFrame(columns= )