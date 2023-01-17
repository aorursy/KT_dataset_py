import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/data-scientist-job-market-in-the-us/alldata.csv')
df.head(2)
# job-description printer, you pick the index number
def job_describe(n, df=df):
    print ('Position:',df.iloc[n]['position'])
    print('---------------')
    print(df.iloc[n]['description'])
    
job_describe(7)
#a handful of descriptions are NaN, remove these rows
df.dropna(axis=0,subset=['description'],inplace=True)
#make column with States
def extract_state (s):
    return s.split(',')[1].split(' ')[1]

df['state']=df['location'].apply(extract_state)
#extract tf-idf representation, and compute similarities
vectorizer=TfidfVectorizer(analyzer='word',
                           stop_words='english',ngram_range=(1,3),min_df=0)
vectors=vectorizer.fit_transform(df['description'])

kernel=linear_kernel(vectors,vectors)
#Sanity check. A dark square with an orange diagonal line is what you'd expect.
import matplotlib.pyplot as plt
from skimage.io import imshow

imshow(kernel)
plt.show()
#similar job finder
def similar_jobs (index, more=False, state=None):
    if state==None:
        if more==False:
            ind=kernel[index].argsort()[-2:-10:-1]
        elif more==True:
            ind=kernel[index].argsort()[-2:-20:-1]
        kers=[kernel[index,i] for i in ind]
        kers=np.array(kers)
        sims=df.iloc[ind]
        sims['similarity score']=kers
        return sims
    else:
        ind=kernel[index].argsort()[-2:-100:-1]
        if more==False:
            kers=[kernel[index,i] for i in ind]
            kers=np.array(kers)
            sims=df.iloc[ind]
            sims['similarity score']=kers
            return sims[sims['state']==state][0:10]
        else:
            kers=[kernel[index,i] for i in ind]
            kers=np.array(kers)
            sims=df.iloc[ind]
            sims['similarity score']=kers
            return sims[sims['state']==state][0:20]
#let's try it out
similar_jobs(7,more=True)