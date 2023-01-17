# Below libraries are imported
import pandas as pd
import matplotlib.pyplot as plt

# Command to read the input file
hdata=pd.read_csv('../input/who_suicide_statistics.csv')

# Filter the data to reflect records from 2011 to 2016
lastfydata=hdata['year'] >= 2011 

# Store the data in a newly created data frame and keeping the only two columns that we need to perform this analysis
fiveydata=pd.DataFrame(data=hdata[lastfydata],columns=['country','suicides_no'])

# Group the records based on country and get a total sum of suicides. The top ten countries are filtered to be shown.
fiveydata.groupby('country').suicides_no.sum().sort_values(ascending=False).head(10).plot.bar()

plt.title('Top 10 countries with the highest suicide count from 2011 - 2016')
plt.xlabel('Country')
plt.ylabel('Death Count')
teenfiveydata=pd.DataFrame(data=hdata[(hdata['year'] >= 2011) & (hdata['age'] == '5-14 years')],columns=['country','suicides_no','age'])
teenfiveydata.groupby('country').suicides_no.sum().sort_values(ascending=False).head(10).plot.bar()
plt.title('Top 10 countries with children under the age of 15 committing suicide (5-14 years) from 2011-2016')
plt.ylabel('Death Count')
plt.xlabel('Country')
mendeathdata = pd.DataFrame(data=hdata[(hdata['year'] >=2011) & (hdata['sex']=='male')],columns=['country','sex','suicides_no'])
mendeathdata.groupby('country').suicides_no.sum().sort_values(ascending=False).head(10).plot.bar()
plt.title('Top 10 contries with highest male suicide rate from 2011-2016')
plt.xlabel('Country')
plt.ylabel('Death Count')
femaledeathdata = pd.DataFrame(data=hdata[(hdata['year'] >=2011) & (hdata['sex']=='female')],columns=['country','sex','suicides_no'])
femaledeathdata.groupby('country').suicides_no.sum().sort_values(ascending=False).head(10).plot.bar()
plt.title('Top 10 contries with highest female suicide rate between 2011-2016')
plt.xlabel('Country')
plt.ylabel('Death Count')