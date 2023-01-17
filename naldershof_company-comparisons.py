import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline

# Any results you write to the current directory are saved as output.
complaints = pd.read_csv("../input/consumer_complaints.csv", low_memory=False)

# The amount of complaints each company has
grouped_by_company = complaints.company.value_counts()

# Check out the distribution
grouped_by_company.plot(kind='hist')

grouped_by_lcompany = grouped_by_company[grouped_by_company > 100]
print ('Original dataset has {} companies'.format(grouped_by_company.size)) 
print ('Strictly limited dataset has {} companies'.format(grouped_by_lcompany.size)) 
company_chart = grouped_by_lcompany.head(25).plot(kind='bar', title='Max Complaints')
# This dataframe is then only those companies which have received at least 100 complaints in this
# dataset. From there we can do some deeper digging into those various companies

lcompany_complaints = complaints[complaints.company.isin(grouped_by_lcompany.index)]
