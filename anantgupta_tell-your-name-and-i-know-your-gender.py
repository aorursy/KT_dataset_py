# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Are people becoming more creative in naming their children, or less
data = pd.read_csv('../input/NationalNames.csv')
data.head()
def getCreativeCount(year):
	priorNames=data.loc[data['Year']<year,['Name']]['Name'].str.lower().unique()
	curNames=data.loc[data['Year']==year,['Name']]['Name'].str.lower().unique()
	print(year)
	return(sum([True for i in curNames if i in priorNames]))

#creativeCount=map(getCreativeCount,data['Year'].unique())
#creativeData=pd.DataFrame(data['Year'].unique())
#creativeData['Count']=creativeCount
#import matplotlib.pyplot as plt


def getRatio(year):
	priorNames=data.loc[data['Year']<year,['Name','Count']]['Name'].str.lower().unique()
	curNames=data.loc[data['Year']==year,['Name']]['Name'].str.lower().unique()
	NameCount=data.loc[data['Year']==year,['Name','Count']].groupby('Name').sum()
	NameCount['Name']=NameCount.index.values
	NameCount['Name']=NameCount['Name'].str.lower()
	if sum(NameCount['Count']) > 0:
		#print("aaaaaaa")
		result1=[NameCount.loc[NameCount['Name']==i,['Count']]['Count'] for i in curNames if i not in priorNames]
		#print(result1)
		result2=[x for x in result1 if len(x) > 0] 
		#print(result2)
		if len(result1)>0:
			result3=sum([i[0] for i in result2])
			print("The sum is " + str(result3))
			return(  (float(result3) * 100)/ sum(NameCount['Count']) )
	else:
		return 0

creativeWeightedCount=map(getRatio,data['Year'].unique())
creativeData=data['Year'].unique()
creativeData=pd.DataFrame(creativeData)
creativeData.columns.values
creativeData['Year']=creativeData[0]
creativeData['Ratio']=creativeWeightedCount
creativeData=creativeData[1:]

import matplotlib.pyplot as plt
creativeData18=creativeData[1:10]
plt.figure()
plt.plot(map(lambda x:str(x),creativeData18['Year']),creativeData18['Ratio'])
ax = plt.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.get_xaxis().get_major_formatter().set_scientific(False)
plt.xlabel('Year Value')
plt.ylabel('Unique Name vs Population Ratio')
plt.title('Population Adjusted Unique Count YOY')