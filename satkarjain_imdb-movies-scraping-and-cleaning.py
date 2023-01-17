from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import pandas as pd
import re
import re
import time
a=[0,51,101,151]                                   #Movies_index
df=pd.DataFrame()
for f in range(1):
  print(f)
  time.sleep(5)
  url = "https://www.imdb.com/search/title/?title_type=feature&num_votes=10000,&languages=hi&sort=year,desc&start="+str(a[f])+"&ref_=adv_nxt"
  response = get(url)
  html_soup = BeautifulSoup(response.text, 'html.parser')  
  movie_containers = html_soup.find_all('div', class_ = 'lister-item mode-advanced')
  years = []
  imdb_ratings = []
  metascores = []
  votes = []
  plots=[]
  genre=[]
  lengths=[]
  
  starss=[]
  directors=[]
  
  names = []
  collections=[]
  imagelink=[]
  Budgets=[]
#director=[]
#stars=[]
# Extract data from individual movie container
  for container in movie_containers:

    
      q1=container.find('a', attrs={'href': re.compile("/title/")})
      image=q1.find('img', {'loadlate':re.compile('.jpg')}).get('loadlate')
      imagelink.append(image)
      
          

      name = container.h3.a.text
      names.append(name)
      b=container.find_all('strong')
      vote = b[0].text
      imdb_ratings.append(vote)
    
# The year
      year = container.h3.find('span', class_ = 'lister-item-year text-muted unbold').text
      years.append(year)
#     
# The Metascore
      #m_score = container.find('span', class_ = 'metascore').text
      #metascores.append(int(m_score))
      
# The number of votes
      b=container.find_all('span', attrs = {'name':'nv'})
      vote = b[0].text
      votes.append(vote)
      if len(b)==2:
        collection=b[1].text
        collections.append(collection)
      
      else:
        collections.append('0')
      par=container.find_all('p')
      length= par[0].find('span',class_='genre').text
      genre.append(length)

      length= par[0].find('span',class_='runtime').text
      lengths.append(length)

      plot= par[1].text
      plots.append(plot)
      stars_director=container.find_all('p')[2].text
      directors.append(stars_director)

         
     
  #imagelink    
  #print(imagelink)
  #print(len(names))
  test_df = pd.DataFrame({'movie': names,
  "Revenue (Millions)":collections,
  'Description':plots,
  'Runtime (Minutes)':lengths,
  'Rating':imdb_ratings,
  'Genre':genre,
  'Director':directors,
  'Imagelink':imagelink
  })

  df=pd.concat([df,test_df]) 

df.head()
def split(text):
  text=text.strip('\n')
  return text
df.head()
df.columns
def edit(df):

 # df.drop(columns=["Unnamed: 0"],inplace=True)
  df['Description']=df['Description'].apply(lambda x: split(x))
  a=df['Director'].str.split(':',expand=True)
  df['Director']=a[1]
  df['Stars']=a[2]
  a=df['Director'].str.split('Stars',expand=True)
  df.drop(columns=["Director"],inplace=True)
  df["Director"]=a[0]
  df["Runtime (Minutes)"]=df["Runtime (Minutes)"].apply(lambda x: re.sub('min','',x))
  df["Revenue (Millions)"]=df["Revenue (Millions)"].apply(lambda x: re.sub('[M$]+','',x))
  df['Stars']=df['Stars'].apply(lambda x: re.sub('[\n|]+','',x))
  df['Director']=df['Director'].apply(lambda x: re.sub('[\n|]+','',x))
  df['Genre']=df['Genre'].apply(lambda x: re.sub("[\n|]+","",x))
 # df['year']=df['year'].apply(lambda x: re.sub('[()IVTMovie]+','',x))
 # df.columns=['Title','Revenue (Millions)' ,'Description','Runtime (Minutes)','Rating',  'Genre',
  #      , 'Actors', 'Director']
 # df["Votes"]=df["Votes"].apply(lambda x: re.sub(',','',x))
 # df["Year"]=df["Year"].astype(int)
 # df['Year']=df['Year'].apply(lambda x: abs(x))
 # df["Metascore"]=df["Metascore"].astype(float)
 # df["Votes"]=df["Votes"].astype(int)
  df["Runtime (Minutes)"]=df["Runtime (Minutes)"].astype(int)
  df["Revenue (Millions)"]=df["Revenue (Millions)"].astype(float)
  return df

df2 = df.copy()
df1= edit(df2)
df1['Director']=df1['Director'].apply(lambda x: re.sub('Stars','',x))
df1.head()