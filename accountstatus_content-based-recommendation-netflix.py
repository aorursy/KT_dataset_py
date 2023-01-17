import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data=pd.read_csv('../input/netflix-shows/netflix_titles.csv')
data=data.fillna('')
data.head()
vectorizer = TfidfVectorizer(stop_words='english')
desc=data['description'].values
desc=list(desc)
des=vectorizer.fit_transform(desc)
des.shape
cosine_sim = linear_kernel(des,des)
cosine_sim.shape
cosine_sim[1]
def recommendation(name,cos=cosine_sim):
    idx=data[data['title']==name].index[0]
    top_24_idx = np.argsort(cos[idx])[-25:]
    top_24_idx=top_24_idx[::-1]
    top_24_idx=top_24_idx[1:]
    return data['title'].iloc[top_24_idx],top_24_idx
    
    
movie=input('Please enter the movie name:')
recommendation(movie)[0]
rec_movies,ind=recommendation(movie)
# Asking from users what they prefer (since users choices are of their own ...) Kinda like a filter ..
type=0
director=0
rating=0
listed=0
country=0
x=(input('If you wanna filter please enter yes else enter no:'))
if 'y' in x.lower():
    type=int(input('Type(movie/ Tv)'))
    director=int(input('You want same director :(1/0)'))
    rating=int(input('Do you want the same rating :(1/0)'))
    listed=int(input('Do you care about the same genre :(1/0)'))
    country=int(input('Do you want the movie of the same country:(1/0)'))
call=[type,director,country,rating,listed]

    
def recom(ind=ind,movie=movie,call=call):
    score=[]
    for i in ind:
        sc=0
        a=data.iloc[i][1]
        b=data.iloc[i][3]
        c=data.iloc[i][5]
        d=data.iloc[i][8]
        e=data.iloc[i][10]
        a1=data[data['title']==movie].drop(columns=['show_id','cast','date_added','release_year','duration','description','title'],axis=1)['type'].values[0]
        b1=data[data['title']==movie].drop(columns=['show_id','cast','date_added','release_year','duration','description','title'],axis=1)['director'].values[0]
        c1=data[data['title']==movie].drop(columns=['show_id','cast','date_added','release_year','duration','description','title'],axis=1)['country'].values[0]
        d1=data[data['title']==movie].drop(columns=['show_id','cast','date_added','release_year','duration','description','title'],axis=1)['rating'].values[0]
        e1=data[data['title']==movie].drop(columns=['show_id','cast','date_added','release_year','duration','description','title'],axis=1)['listed_in'].values[0]
        if call[0]==0:
            if a==a1:
                sc+=1
        else:
            if a==a1:
                sc+=4
        if call[1]==0:
            if b==b1:
                sc+=1
        else:
            if b==b1:
                sc+=4
        if call[2]==0:
            x=c.split(',')
            for op in x:
                if op in c1:
                    sc+=1
                    break
        else:
            x=c.split(',')
            for op in x:
                if op in c1:
                    sc+=4
                    break
            
        if call[3]==0:
            if d==d1:
                sc+=1
        else:
            if d==d1:
                sc+=4
        if call[4]==0:
            x=e.split(',')
            for op in x:
                if op in e1:
                    sc+=1
                    break
        else:
            x=e.split(',')
            for op in x:
                if op in e1:
                    sc+=4
                    break
        score.append(sc)
    return score
            
            
            
            
        
        
    
    


ans=recom()
print('The 12 best of the recommendations i found are :')
top_12_idx = np.argsort(ans)[-12:]
top_12_idx=top_12_idx[::-1]
for i in top_12_idx:
    print(rec_movies.values[i])
# Pretty different result i suppose
data[data['title']=='Transformers: Robots in Disguise']
data[data['title']=='Transformers Prime']
#Hope you all liked this  :)


