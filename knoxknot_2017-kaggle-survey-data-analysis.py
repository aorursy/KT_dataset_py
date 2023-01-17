'''

import the libraries

'''

import numpy as np                 # for scientific calculations

import pandas as pd                # for loading and manipulation



import matplotlib.pyplot as pp     # for visualizations

import seaborn as sb               # for appealing visualizations

import missingno as mn             # for missing data visualization



from matplotlib import rcParams    # for setting figure specification
'''

set figure specifications

inorder the code line enables plot to show within the notebook, next sets the pyplot style to a ggplot theme, next 

sets seaborn plot to a whitegrid theme and the last sets the figure size to a 5 width by 4 height units (inches or pixel) 

'''

%matplotlib inline

pp.style.use('ggplot')

sb.set_style('whitegrid')

rcParams['figure.figsize'] = 5,4
'''

names = [give a header to the columns]

'''

cr = pd.read_csv('../input/conversionRates.csv',  names = ['ID' ,'originCountry' ,'exchangeRate'] )



''' Exchange Rate with US Dollars as the Reference'''

# view 10 rows of all columns on shuffle 

cr.sample(n=10)
# read the content of the RespondentTypeREADME - unix command 

!cat ../input/RespondentTypeREADME.txt
'''

Let's look at the questions asked and to which respondents

'''

sch = pd.read_csv('../input/schema.csv')

sch.set_index(['Question','Asked']).unstack()[:10]    # format the dataframe in a form easy to capture details
len(sch.Column), len(sch.Question) # accessing the length of Column and Question
ffr = pd.read_csv('../input/freeformResponses.csv', dtype='O')

# sample 10 columns of the first 10 rows

ffr.sample(n=10, axis=1).reindex(np.random.permutation(ffr.index))[:10]
mcr = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='latin-1')

mcr.sample(n=10,axis='columns').reindex(np.random.permutation(mcr.index))[:10]
mcr.info(verbose=True)
'''

Persons from about 53 countries participated in the survey

'''

len(mcr.Country.unique())
specific_countries = mcr.Country.unique()

persons_per_country = mcr.Country.value_counts()

persons_per_country
lbls = persons_per_country.index

pp.pie(persons_per_country,labels=lbls, autopct='%1.1f%%', shadow=True, startangle=10)

pp.title("Percentage or Respondents Per Country")
nr = mcr[mcr.Country == 'Nigeria']  # nr implies nigerian respondents

nr.set_index(['GenderSelect'])[:10]
nsex = nr.GenderSelect.value_counts()

pp.pie(nsex,labels=nsex.index, autopct='%1.1f%%', shadow=True, startangle=10)

pp.title('Percentage of Gender Respondents from Nigeria')

pp.show()
nr.Age.unique()
agebrkt = nr.Age

bins = [18, 25, 35, 50]     # define an age bracket for nigerian respondents 

labels = ['Young Professionals','Seniors', 'Experienced']

age_data = pd.cut(agebrkt, bins, labels).value_counts()





pp.pie(age_data, labels=labels, autopct='%1.1f%%', shadow=True, startangle=10)

pp.axis('equal')                     # equal aspect ratio ensures that pie is drawn as a circle.

pp.show()
yp = nr[nr.Age < 25 ]

len(yp)
len(yp)
len(yp[yp.GenderSelect=='Male']), len(yp[yp.GenderSelect=='Female']), len(yp[yp.GenderSelect=='A different identity'])
genemp = pd.crosstab(yp['GenderSelect'],yp['EmploymentStatus'])
genemp.plot(kind='bar')
yp.StudentStatus.value_counts()