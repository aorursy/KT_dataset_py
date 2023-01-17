# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
len(df)
all_words = []

for i in range(len(df)):

    all_text = str(str(df.iloc[i].title) + ' '+str(df.iloc[i].abstract)).split()

    all_words.append(all_text)
def find_word(find_words):

    priority = []

    for ii,i in enumerate(all_words):

        if True in np.in1d(i, find_words):

            priority.append(ii)

    return priority
relevant = find_word(['farmers'])

for i in range(0, len(relevant)):

    print(i, df.iloc[i].title)

    print ()
df = df.astype({"abstract": str})



#for i in range(0, len(relevant)):

#    if df.iloc[i].abstract != 'nan':

        #print(i, "TITLE: ", df.iloc[i].title);

        #print("ABSTRACT: ", df.iloc[i].abstract);

        #print()

word = ['SARS-CoV-2']

relevant = find_word(word)

for i in range(0, len(relevant)):

    print(i, df.iloc[i].title)

    print ()
#for i in range(0, len(relevant)):

#    if df.iloc[i].abstract != 'nan':

#        print(i, "TITLE: ", df.iloc[i].title)

#        print("ABSTRACT: ", df.iloc[i].abstract)

#        print()
# Take information about (ACE2) gene in human
#PATH = ".//kaggle/input/disgenet-gene-disease/curated_gene_disease_associations.tsv/"

filename = "/kaggle/input/disgenet-gene-disease/curated_gene_disease_associations.tsv"

df = pd.read_csv(filename, sep = '\t')

df.head()



target = df.loc[df['geneSymbol'] == 'ACE2']

target
biogrid=pd.read_csv('/kaggle/input/new-biogrid/BIOGRID-ALL-3.5.182.tab2.txt', sep='\t')





genesymbol = ['ACE2']

#select only human genes

biogrid=biogrid.loc[(biogrid['Organism Interactor A']==9606) & (biogrid['Organism Interactor B']==9606)]

# look for the genes which interacts with at least one seed genes

bio = biogrid.loc[(biogrid['Official Symbol Interactor A'].isin(genesymbol)) | (biogrid['Official Symbol Interactor B'].isin(genesymbol))]

bio
#save a list of all the symbols in order to search their uniprot

syms=[]

syms.extend(bio['Official Symbol Interactor A'])

syms.extend(bio['Official Symbol Interactor B'])

#remove duplicates

sym_to_fix=list(set(syms))



# using join() 

# avoiding printing last comma 

print("The formatted output is : ") 

for gene in sym_to_fix:

    print(gene)

##print in order to search on ENRICHR