# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

#print(check_output(["pip" , "install" ,"requests"]).decode("utf8"))







# Load data

df = pd.DataFrame()



#This is the standar csv, with the gravatar images for each repo     

df = pd.read_csv('../input/TopStaredRepositories.csv')

df.set_index(['Repository Name'])



df.head(2)





## Cleaning data

# We fill the emptpy URL cells

df['Url'] = "http://github.com/" +         df['Username'] + "/" + df['Repository Name']

# We add a final comma character for the tag string, it will be usefull when we tokenize

df['Tags'].fillna("", inplace=True)

df['Tags'] = df['Tags'] + ","



# We do not want uppercase on any label

df['Language'] = df.loc[:, 'Language'].str.lower()

# Copy a backup variable, so we can change our main dataframe

df_backup = df.copy(deep=True)

df.head(2)

# Generate tag list

mergedlist = []

for i in df['Tags'].dropna().str.split(","):

    mergedlist.extend(i)

tags = sorted(set(mergedlist))

# Encode languages in single column

just_dummies = pd.get_dummies(df['Language'])

for column in just_dummies.columns:

    if column not in df.columns:

        df[column] = just_dummies[column]

df.head(2)

for tag in tags:

    if tag not in df.columns:

        df[tag] = 0

    try:

        if len(tag) > 4 :

            df.loc[df['Repository Name'].str.contains(tag), tag] = 1

            df.loc[df['Description'].str.contains(tag), tag] = 1

        df.loc[df['Tags'].str.contains(tag + ","), tag] = 1

    except Exception:

        pass

# Remove columns not needed

df.set_index(['Repository Name'])

COLUMNS_TO_REMOVE_LIST = ['', 'Username', 'Repository Name', 'Description',

                          'Last Update Date', 'Language', 'Number of Stars', 'Tags', 'Url','Gravatar' ,'Unnamed: 0']

# Stop words: links to (https://github)

RAGE_TAGS_LIST = [ 'github','algorithms','learn','learning','http' ,'https']







for column in COLUMNS_TO_REMOVE_LIST + RAGE_TAGS_LIST:

    try:

        del df[column]

    except Exception:

        pass



df.columns = df.columns.str.lower()





print ("Our final label matrix for repo list is")

df.head(2)  





# Create correaltion Matrix

corr = df.corr()

corr.iloc[0:5][0:5]

corr['machine-learning'].dropna().sort_values().tail(5).head(4)
###

### We comment the code because kaggle kernels does not allow internet access.

### If you decide to run from local, the code is: 

###



# We ask for a given user, to find the reccomendation for this user



GITHUBUSER="jaimevalero"





# import requests

# from requests.auth import HTTPBasicAuth



# github_user = GITHUBUSER

# We have do not have internet access inside the notebook, 

# but the coude to query github api would be

#PERSONAL_TOKEN = "<REPLACE-FOR-YOUR-TOKEN>"



# Token is not mandatory, but there is a query rate limit, for those API calls which do not use token.  



#url = 'https://api.github.com/users/%s/repos?sort=updated' % github_user

#headers = {'content-type': 'application/json',

#           'Accept-Charset': 'UTF-8',

#           'Accept': 'application/vnd.github.mercy-preview+json'}

#r = requests.get(url,  headers=headers, auth=HTTPBasicAuth(

#    "jaimevalero", PERSONAL_TOKEN))



# i=0 # first repo from the list

#repo_names       = pyjq.all(".[%s] | .name"        % i,  json_response)

#repo_languages   = pyjq.all(".[%s] | .language"    % i,  json_response)

#repo_description = pyjq.all(".[%s] | .description" % i,  json_response)

#repo_topics      = pyjq.all(".[%s] | .topics"      % i,  json_response)



# Mocked values are

repo_names       = ["github-recommendation-engine"]

repo_languages   = ["Python"]

repo_description = ["A github repository suggestion system"]

repo_topics  = ["github","machine-learning","recommendation-system"]





# Error test, in case of no description given

if repo_description[0] is None: repo_description = ['nodescription']



# We add a new element to the end of the DataFrame

new_element = pd.DataFrame(0, [df.index.max() + 1], columns=df.columns)



label_list = repo_names[0].split('-')  + repo_languages + repo_description[0].replace(".", " ").replace(",", " ").split() + list(repo_topics)

print( "The labels from the repo are :" , label_list)



for j in (label_list):

    if j is not None:

        if j.lower() in df.columns:

            #print("Setting to 1", j.lower())

            new_element[j.lower()] = 1



# Concat new user repo dataframe to stared repos dataframe

df = pd.concat([df, new_element])

# Now user repo is on the last row of the label matrix

df.tail(2)

df_reduced = pd.DataFrame()

NUM_ELEMENTS=len(df)-1

user_repo = df.iloc[NUM_ELEMENTS:]

for k in df.columns :

    existe = user_repo[k].values[0]

    if existe > 0 : df_reduced[k] = df[k]



df = df_reduced.copy(deep=True)



# Remember user repo is the last one

df.tail(10)



from scipy.spatial import distance

from scipy.spatial.distance import squareform, pdist



repos = list(df_backup['Username'] + "/" + df_backup['Repository Name'])

repos.extend(repo_names) # We add to the csv reponame list, the repo name from github 

print (repos[-1] ,len(repos), df.shape, df_backup.shape)

# We calculate the euclidean distance for the binary label matrix 3

res = pdist(df, 'euclidean')

df_dist = pd.DataFrame(squareform(res), index=repos, columns=repos)



print("""This is the euclidean distance matrix for 

     - the user repo (github-recommendation-engine) 

     - other eight repos :

      The lower the distance, stronger the similarity between repos

      """)





import seaborn as sns

import matplotlib.pyplot as plt



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df_dist.iloc[972:,972:] , annot=True, linewidths=.5, fmt= '.1f',ax=ax , cmap='viridis' )



result_array = []

i = df_dist.columns[-1]

# We change 0 for 1000, to filter when calculating minimun distances 

# (because obviously, the repo that has more similarity with the repo list is itself. )

df_dist.loc[df_dist[i] == 0, i] = 1000

# Get minimun distance

min = df_dist[i].min()

# Filter all repo within that minimun distance

closest_repos = df_dist[i][df_dist[i] == min].index, i, min

# print results

print ("Similar repos to %s are: " % repo_names )

for recomended_repo in (df_dist[i][df_dist[i] == min].index[0:12]):

    print (recomended_repo)

    

    