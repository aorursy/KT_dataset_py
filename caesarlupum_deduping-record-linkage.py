import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
from IPython.display import HTML

HTML('<iframe width="1280" height="720" src="https://www.youtube.com/embed/McsTWXeURhA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



pd.set_option('display.max_colwidth', -1)
import os

print(os.listdir("../input/sec-edgar-companies-list/"))
root = '../input/sec-edgar-companies-list/'



data = pd.read_csv(root + 'sec__edgar_company_info.csv',encoding='latin')

print('Size of data ',data.shape)
data.head()
data.select_dtypes('object').apply(pd.Series.nunique, axis=0)
!pip install fuzzywuzzy

from fuzzywuzzy import fuzz

from fuzzywuzzy import process
data.tail()
fuzz.ratio('ZZ GLOBAL LLC', 'ZZLL INFORMATION TECHNOLOGY, INC')
fuzz.ratio('ZZ GLOBAL LLC', 'ZZX, LLC')
fuzz.partial_ratio('ZZ GLOBAL LLC', 'ZZLL INFORMATION TECHNOLOGY, INC')
fuzz.partial_ratio('ZZ GLOBAL LLC', 'ZZX, LLC')
fuzz.token_sort_ratio('ZZ GLOBAL LLC', 'ZZLL INFORMATION TECHNOLOGY, INC')
fuzz.token_sort_ratio('ZZ GLOBAL LLC', 'ZZX, LLC')
fuzz.token_set_ratio('ZZ GLOBAL LLC', 'ZZLL INFORMATION TECHNOLOGY, INC')
fuzz.token_set_ratio('ZZ GLOBAL LLC', 'ZZX, LLC')
!pip install ftfy # amazing text cleaning for decode issues..
import re

from ftfy import fix_text



def ngrams(string, n=3):

    string = fix_text(string) # fix text

    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars

    string = string.lower()

    chars_to_remove = [")","(",".","|","[","]","{","}","'"]

    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'

    string = re.sub(rx, '', string)

    string = string.replace('&', 'and')

    string = string.replace(',', ' ')

    string = string.replace('-', ' ')

    string = string.title() # normalise case - capital at start of each word

    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single

    string = ' '+ string +' ' # pad names for ngrams...

    string = re.sub(r'[,-./]|\sBD',r'', string)

    ngrams = zip(*[string[i:] for i in range(n)])

    return [''.join(ngram) for ngram in ngrams]
print('All 3-grams in "McDonalds":')

ngrams('McDonalds')
from sklearn.feature_extraction.text import TfidfVectorizer



company_names = data['Company Name'].unique()

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)

tf_idf_matrix = vectorizer.fit_transform(company_names)
data.head()
print( tf_idf_matrix.shape, tf_idf_matrix[5] )

# Check if this makes sense:



ngrams('#1 PAINTBALL CORP')
t1 = time.time()

print(process.extractOne('Ministry of Justice', company_names[0:999])) #org names is our list of organization names

t = time.time()-t1

print("SELFTIMED:", t)

print("Estimated hours to complete for 1000 rows of  dataset:", (t*len(company_names[0:999]))/60/60)
##################

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

root2 = '../input/gov-names/'

clean_org_names = pd.read_excel(root2 + 'Gov Orgs ONS.xlsx')

clean_org_names = clean_org_names.iloc[:, 0:6]



org_name_clean = clean_org_names['Institutions'].unique()



print('Vecorizing the data - this could take a few minutes for large datasets...')

vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)

tfidf = vectorizer.fit_transform(org_name_clean)

print('Vecorizing completed...')



from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)



org_column = 'Company Name' #column to match against in the messy data

unique_org = set(data[org_column].values) # set used for increased performance

###matching query:

def getNearestN(query):

    queryTFIDF_ = vectorizer.transform(query)

    distances, indices = nbrs.kneighbors(queryTFIDF_)

    return distances, indices



import time

t1 = time.time()

print('getting nearest n...')

distances, indices = getNearestN(unique_org)

t = time.time()-t1

print("COMPLETED IN:", t)



unique_org = list(unique_org) #need to convert back to a list

print('finding matches...')

matches = []

for i,j in enumerate(indices):

    temp = [round(distances[i][0],2), clean_org_names.values[j][0][0],unique_org[i]]

    matches.append(temp)



print('Building data frame...')  

matches = pd.DataFrame(matches, columns=['Match confidence (lower is better)','Matched name','Origional name'])

print('Done') 
matches.head(10)
matches.sort_values('Match confidence (lower is better)')