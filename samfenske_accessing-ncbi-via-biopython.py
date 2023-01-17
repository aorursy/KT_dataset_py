# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from Bio import Entrez 
#specify that I'm using Biopython within some
#larger software suite

#always tell NCBI who you are
Entrez.email = 'email here'
#entrez.read() call directly parses XML into a python object, has only one key
direct_access=Entrez.read(Entrez.einfo())
direct_access
#obtain more specific information
pubmed=Entrez.read(Entrez.einfo(db='pubmed'))
pubmed
type(pubmed)
pubmed.keys()
pubmed['DbInfo'].keys()
pubmed['DbInfo']['Description']
pubmed['DbInfo']['MenuName']
pubmed['DbInfo']['Description']
pubmed['DbInfo']['DbBuild']
pubmed['DbInfo']['Count']
pubmed['DbInfo']['LastUpdate']
pubmed['DbInfo']['FieldList']
pd.DataFrame(pubmed['DbInfo']['FieldList']).head(10)
pd.DataFrame(pubmed['DbInfo']['LinkList'])
handle=Entrez.esearch(db='pubmed', term='Single-cell sequencing in stem cell biology')
record=Entrez.read(handle)
record.keys()
record['TranslationStack']
handle=Entrez.efetch(db='pubmed', id='27083874',rettype='fasta', retmode='json')
#here we can actually find a specific journal just by calling it from python, changing to json mode may make it easier to read
print(handle.read())
pm=Entrez.esearch(db='pubmed', term='Biomedical Engineering')
#live search returns 182,474
count =Entrez.read(pm)['Count']
count
#this gets me the count for pubmed as an integer
x={'Database':['PubMed'], 'Counts':[count]}
df=pd.DataFrame(data=x)
df.append({'Database':'Ex', 'Counts':count},ignore_index=True)
#this 'Ex' part is just me verifying that I am using the .append() method correctly
databases=Entrez.read(Entrez.einfo())
databases['DbList']
#this is the list we want to navigate through
df=pd.DataFrame()

for base in databases['DbList']:
    search=Entrez.esearch(db=base, term='Biomedical Engineering')
    #round this count for some reason won't round to integer (tried math.trunc(), round())
    count =Entrez.read(search)['Count']
    x={'Database':base, 'Counts':count}
    df=df.append(x,ignore_index=True)
df
df=pd.DataFrame()

for base in databases['DbList']:
    search=Entrez.esearch(db=base, term='Washington University in St. Louis')
    #round this count for some reason won't round to integer (tried math.trunc(), round())
    count =Entrez.read(search)['Count']
    x={'Database':base, 'Counts':count}
    df=df.append(x,ignore_index=True)
df