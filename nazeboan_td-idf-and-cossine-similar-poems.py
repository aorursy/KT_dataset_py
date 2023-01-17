## importing auxiliary libraries

import numpy as np 

import pandas as pd



## vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



## similarity metric

from sklearn.metrics.pairwise import cosine_similarity
## inspecting the data



df = pd.read_csv('/kaggle/input/poems-in-portuguese/portuguese-poems.csv')



## lets drop any NA content

df.dropna(subset=['Content'],inplace=True)



## reset index for organization purposes

df.reset_index(drop=True,inplace=True)

df.head()
%%time

tfvec = TfidfVectorizer(max_features=10000)

x = tfvec.fit_transform(df['Content'])

x
## creating the not so optimized function that calculates and finds 10 most similar poems



def find_similar(poem):

    

    simi = []



    for i in range(x.shape[0]):

        simi.append((i,cosine_similarity(x[0],x[i])[0][0]))



    simi.sort(key = lambda x: x[1],reverse=True)

    

    df_ret = df.iloc[np.array(simi[:10])[:,0],[0,1]]

    df_ret['similarity'] = np.array(simi[:10])[:,1]

    

    return df_ret
%%time

find_similar(x[0])
print(df[(df.Author == 'Cecília Meireles') & (df.Title == 'Retrato')].Content[0])
print(df[(df.Author == 'Fernanda Benevides') & (df.Title == 'Flagrante')].Content[12643])