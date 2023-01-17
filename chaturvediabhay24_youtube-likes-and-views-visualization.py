
df = pd.read_csv("../input/youtube-new/INvideos.csv")

df['category_name'] = np.nan

df.loc[(df["category_id"] == 1),"category_name"] = 'Film and Animation'
df.loc[(df["category_id"] == 2),"category_name"] = 'Cars and Vehicles'
df.loc[(df["category_id"] == 10),"category_name"] = 'Music'
df.loc[(df["category_id"] == 15),"category_name"] = 'Pets and Animals'
df.loc[(df["category_id"] == 17),"category_name"] = 'Sport'
df.loc[(df["category_id"] == 19),"category_name"] = 'Travel and Events'
df.loc[(df["category_id"] == 20),"category_name"] = 'Gaming'
df.loc[(df["category_id"] == 22),"category_name"] = 'People and Blogs'
df.loc[(df["category_id"] == 23),"category_name"] = 'Comedy'
df.loc[(df["category_id"] == 24),"category_name"] = 'Entertainment'
df.loc[(df["category_id"] == 25),"category_name"] = 'News and Politics'
df.loc[(df["category_id"] == 26),"category_name"] = 'How to and Style'
df.loc[(df["category_id"] == 27),"category_name"] = 'Education'
df.loc[(df["category_id"] == 28),"category_name"] = 'Science and Technology'
df.loc[(df["category_id"] == 29),"category_name"] = 'Non Profits and Activism'
df.loc[(df["category_id"] == 25),"category_name"] = 'News & Politics'

df["category_name"]=df["category_name"].fillna("Other")
category_to_views={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_views.keys()):
        category_to_views[row["category_name"]]+=row["views"]
    else:
        category_to_views[row["category_name"]]=row["views"]

        
for key in category_to_views.keys():
    category_to_views[key]=str(np.log(category_to_views[key]))
    
    
import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_views.keys()) [:6]
views = list(category_to_views.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Views(log) for different content") 
plt.show() 
###########################################################
category_to_likes={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_likes.keys()):
        category_to_likes[row["category_name"]]+=row["likes"]
    else:
        category_to_likes[row["category_name"]]=row["likes"]

        
for key in category_to_likes.keys():
    category_to_likes[key]=str(np.log(category_to_likes[key]))
    

import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_likes.keys()) [:6]
views = list(category_to_likes.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Likes(log) for different content") 
plt.show() 

import pandas as pd
import numpy as np
df = pd.read_csv("../input/youtube-new/USvideos.csv")

df['category_name'] = np.nan

df.loc[(df["category_id"] == 1),"category_name"] = 'Film and Animation'
df.loc[(df["category_id"] == 2),"category_name"] = 'Cars and Vehicles'
df.loc[(df["category_id"] == 10),"category_name"] = 'Music'
df.loc[(df["category_id"] == 15),"category_name"] = 'Pets and Animals'
df.loc[(df["category_id"] == 17),"category_name"] = 'Sport'
df.loc[(df["category_id"] == 19),"category_name"] = 'Travel and Events'
df.loc[(df["category_id"] == 20),"category_name"] = 'Gaming'
df.loc[(df["category_id"] == 22),"category_name"] = 'People and Blogs'
df.loc[(df["category_id"] == 23),"category_name"] = 'Comedy'
df.loc[(df["category_id"] == 24),"category_name"] = 'Entertainment'
df.loc[(df["category_id"] == 25),"category_name"] = 'News and Politics'
df.loc[(df["category_id"] == 26),"category_name"] = 'How to and Style'
df.loc[(df["category_id"] == 27),"category_name"] = 'Education'
df.loc[(df["category_id"] == 28),"category_name"] = 'Science and Technology'
df.loc[(df["category_id"] == 29),"category_name"] = 'Non Profits and Activism'
df.loc[(df["category_id"] == 25),"category_name"] = 'News & Politics'

df["category_name"]=df["category_name"].fillna("Other")
category_to_views={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_views.keys()):
        category_to_views[row["category_name"]]+=row["views"]
    else:
        category_to_views[row["category_name"]]=row["views"]

        
for key in category_to_views.keys():
    category_to_views[key]=str(np.log(category_to_views[key]))
    
    
import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_views.keys()) [:6]
views = list(category_to_views.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Views(log) for different content") 
plt.show() 
###########################################################
category_to_likes={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_likes.keys()):
        category_to_likes[row["category_name"]]+=row["likes"]
    else:
        category_to_likes[row["category_name"]]=row["likes"]

        
for key in category_to_likes.keys():
    category_to_likes[key]=str(np.log(category_to_likes[key]))
    

import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_likes.keys()) [:6]
views = list(category_to_likes.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Likes(log) for different content") 
plt.show() 


df = pd.read_csv("../input/youtube-new/GBvideos.csv")

df['category_name'] = np.nan

df.loc[(df["category_id"] == 1),"category_name"] = 'Film and Animation'
df.loc[(df["category_id"] == 2),"category_name"] = 'Cars and Vehicles'
df.loc[(df["category_id"] == 10),"category_name"] = 'Music'
df.loc[(df["category_id"] == 15),"category_name"] = 'Pets and Animals'
df.loc[(df["category_id"] == 17),"category_name"] = 'Sport'
df.loc[(df["category_id"] == 19),"category_name"] = 'Travel and Events'
df.loc[(df["category_id"] == 20),"category_name"] = 'Gaming'
df.loc[(df["category_id"] == 22),"category_name"] = 'People and Blogs'
df.loc[(df["category_id"] == 23),"category_name"] = 'Comedy'
df.loc[(df["category_id"] == 24),"category_name"] = 'Entertainment'
df.loc[(df["category_id"] == 25),"category_name"] = 'News and Politics'
df.loc[(df["category_id"] == 26),"category_name"] = 'How to and Style'
df.loc[(df["category_id"] == 27),"category_name"] = 'Education'
df.loc[(df["category_id"] == 28),"category_name"] = 'Science and Technology'
df.loc[(df["category_id"] == 29),"category_name"] = 'Non Profits and Activism'
df.loc[(df["category_id"] == 25),"category_name"] = 'News & Politics'

df["category_name"]=df["category_name"].fillna("Other")
category_to_views={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_views.keys()):
        category_to_views[row["category_name"]]+=row["views"]
    else:
        category_to_views[row["category_name"]]=row["views"]

        
for key in category_to_views.keys():
    category_to_views[key]=str(np.log(category_to_views[key]))
    
    
import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_views.keys()) [:6]
views = list(category_to_views.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Views(log) for different content") 
plt.show() 
###########################################################
category_to_likes={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_likes.keys()):
        category_to_likes[row["category_name"]]+=row["likes"]
    else:
        category_to_likes[row["category_name"]]=row["likes"]

        
for key in category_to_likes.keys():
    category_to_likes[key]=str(np.log(category_to_likes[key]))
    

import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_likes.keys()) [:6]
views = list(category_to_likes.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Likes(log) for different content") 
plt.show() 


df = pd.read_csv("../input/youtube-new/CAvideos.csv")

df['category_name'] = np.nan

df.loc[(df["category_id"] == 1),"category_name"] = 'Film and Animation'
df.loc[(df["category_id"] == 2),"category_name"] = 'Cars and Vehicles'
df.loc[(df["category_id"] == 10),"category_name"] = 'Music'
df.loc[(df["category_id"] == 15),"category_name"] = 'Pets and Animals'
df.loc[(df["category_id"] == 17),"category_name"] = 'Sport'
df.loc[(df["category_id"] == 19),"category_name"] = 'Travel and Events'
df.loc[(df["category_id"] == 20),"category_name"] = 'Gaming'
df.loc[(df["category_id"] == 22),"category_name"] = 'People and Blogs'
df.loc[(df["category_id"] == 23),"category_name"] = 'Comedy'
df.loc[(df["category_id"] == 24),"category_name"] = 'Entertainment'
df.loc[(df["category_id"] == 25),"category_name"] = 'News and Politics'
df.loc[(df["category_id"] == 26),"category_name"] = 'How to and Style'
df.loc[(df["category_id"] == 27),"category_name"] = 'Education'
df.loc[(df["category_id"] == 28),"category_name"] = 'Science and Technology'
df.loc[(df["category_id"] == 29),"category_name"] = 'Non Profits and Activism'
df.loc[(df["category_id"] == 25),"category_name"] = 'News & Politics'

df["category_name"]=df["category_name"].fillna("Other")
category_to_views={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_views.keys()):
        category_to_views[row["category_name"]]+=row["views"]
    else:
        category_to_views[row["category_name"]]=row["views"]

        
for key in category_to_views.keys():
    category_to_views[key]=str(np.log(category_to_views[key]))
    
    
import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_views.keys()) [:6]
views = list(category_to_views.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Views(log) for different content") 
plt.show() 
###########################################################
category_to_likes={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_likes.keys()):
        category_to_likes[row["category_name"]]+=row["likes"]
    else:
        category_to_likes[row["category_name"]]=row["likes"]

        
for key in category_to_likes.keys():
    category_to_likes[key]=str(np.log(category_to_likes[key]))
    

import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_likes.keys()) [:6]
views = list(category_to_likes.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Likes(log) for different content") 
plt.show() 


df = pd.read_csv("../input/youtube-new/DEvideos.csv")

df['category_name'] = np.nan

df.loc[(df["category_id"] == 1),"category_name"] = 'Film and Animation'
df.loc[(df["category_id"] == 2),"category_name"] = 'Cars and Vehicles'
df.loc[(df["category_id"] == 10),"category_name"] = 'Music'
df.loc[(df["category_id"] == 15),"category_name"] = 'Pets and Animals'
df.loc[(df["category_id"] == 17),"category_name"] = 'Sport'
df.loc[(df["category_id"] == 19),"category_name"] = 'Travel and Events'
df.loc[(df["category_id"] == 20),"category_name"] = 'Gaming'
df.loc[(df["category_id"] == 22),"category_name"] = 'People and Blogs'
df.loc[(df["category_id"] == 23),"category_name"] = 'Comedy'
df.loc[(df["category_id"] == 24),"category_name"] = 'Entertainment'
df.loc[(df["category_id"] == 25),"category_name"] = 'News and Politics'
df.loc[(df["category_id"] == 26),"category_name"] = 'How to and Style'
df.loc[(df["category_id"] == 27),"category_name"] = 'Education'
df.loc[(df["category_id"] == 28),"category_name"] = 'Science and Technology'
df.loc[(df["category_id"] == 29),"category_name"] = 'Non Profits and Activism'
df.loc[(df["category_id"] == 25),"category_name"] = 'News & Politics'

df["category_name"]=df["category_name"].fillna("Other")
category_to_views={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_views.keys()):
        category_to_views[row["category_name"]]+=row["views"]
    else:
        category_to_views[row["category_name"]]=row["views"]

        
for key in category_to_views.keys():
    category_to_views[key]=str(np.log(category_to_views[key]))
    
    
import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_views.keys()) [:6]
views = list(category_to_views.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Views(log) for different content") 
plt.show() 
###########################################################
category_to_likes={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_likes.keys()):
        category_to_likes[row["category_name"]]+=row["likes"]
    else:
        category_to_likes[row["category_name"]]=row["likes"]

        
for key in category_to_likes.keys():
    category_to_likes[key]=str(np.log(category_to_likes[key]))
    

import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_likes.keys()) [:6]
views = list(category_to_likes.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Likes(log) for different content") 
plt.show() 


df = pd.read_csv("../input/youtube-new/FRvideos.csv")

df['category_name'] = np.nan

df.loc[(df["category_id"] == 1),"category_name"] = 'Film and Animation'
df.loc[(df["category_id"] == 2),"category_name"] = 'Cars and Vehicles'
df.loc[(df["category_id"] == 10),"category_name"] = 'Music'
df.loc[(df["category_id"] == 15),"category_name"] = 'Pets and Animals'
df.loc[(df["category_id"] == 17),"category_name"] = 'Sport'
df.loc[(df["category_id"] == 19),"category_name"] = 'Travel and Events'
df.loc[(df["category_id"] == 20),"category_name"] = 'Gaming'
df.loc[(df["category_id"] == 22),"category_name"] = 'People and Blogs'
df.loc[(df["category_id"] == 23),"category_name"] = 'Comedy'
df.loc[(df["category_id"] == 24),"category_name"] = 'Entertainment'
df.loc[(df["category_id"] == 25),"category_name"] = 'News and Politics'
df.loc[(df["category_id"] == 26),"category_name"] = 'How to and Style'
df.loc[(df["category_id"] == 27),"category_name"] = 'Education'
df.loc[(df["category_id"] == 28),"category_name"] = 'Science and Technology'
df.loc[(df["category_id"] == 29),"category_name"] = 'Non Profits and Activism'
df.loc[(df["category_id"] == 25),"category_name"] = 'News & Politics'

df["category_name"]=df["category_name"].fillna("Other")
category_to_views={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_views.keys()):
        category_to_views[row["category_name"]]+=row["views"]
    else:
        category_to_views[row["category_name"]]=row["views"]

        
for key in category_to_views.keys():
    category_to_views[key]=str(np.log(category_to_views[key]))
    
    
import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_views.keys()) [:6]
views = list(category_to_views.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Views(log) for different content") 
plt.show() 
###########################################################
category_to_likes={}
for ind, row in df.iterrows():
    if (row["category_name"] in category_to_likes.keys()):
        category_to_likes[row["category_name"]]+=row["likes"]
    else:
        category_to_likes[row["category_name"]]=row["likes"]

        
for key in category_to_likes.keys():
    category_to_likes[key]=str(np.log(category_to_likes[key]))
    

import numpy as np 
import matplotlib.pyplot as plt  
  
category = list(category_to_likes.keys()) [:6]
views = list(category_to_likes.values()) [:6]
   
fig = plt.figure(figsize = (10, 5)) 
  
# creating the bar plot 
plt.bar(category, views, color ='maroon',  
        width = 0.4) 
  
plt.xlabel("Category") 
plt.ylabel("Views(log)") 
plt.title("Likes(log) for different content") 
plt.show() 
