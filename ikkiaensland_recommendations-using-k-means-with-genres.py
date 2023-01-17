# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# First, we import some useful libraries

import numpy as num # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv # CSV files I/O

from sklearn.cluster import KMeans #For K-Means clustering



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
def dataInit():

    #Reads the .csv files in order to initialize the data

    

    #For the anime list

    fop = open(r'../input/anime.csv',encoding = 'utf-8')

    

    animeReader = csv.reader(fop, delimiter=',', quotechar='"')

    

    animeHeader = animeReader.__next__(); #To take out the header

    

    eofFlag = False #Flag for end of file

    

    animeInfo = [];

    

    while (eofFlag == False):

        try:

            animeInfo.append(animeReader.__next__()); 

        except StopIteration:        #Will be thrown at the end of the file

            eofFlag = True;

            

    fop.close();   

        

    #For the ratings list

    fop = open(r'../input/rating.csv',encoding = 'utf-8')

    

    ratingReader = csv.reader(fop, delimiter=',', quotechar='"')

    

    ratingHeader = ratingReader.__next__();

    

    eofFlag = False

    

    ratingInfo = [];

    

    while (eofFlag == False):

        try:

            ratingInfo.append(ratingReader.__next__());

        except StopIteration:        

            eofFlag = True;

            

    fop.close(); 

    

    return animeInfo, ratingInfo
def FindGenres(animeInfo):

    

    #Gets the possible relevant features. Namely:

    #   Genres: A binary variable

    #   Kinds: Whether it's a TV Show, OVA or Movie

    #   Number of Episodes

    #   Average Rating by the users

    

    genresList = [];

    kindList = [];

    

    for k in range(0,len(animeInfo)):

        genresk = animeInfo[k][2];         #Extracts the genres

        genresk = genresk.replace(" ","");

        genresk = genresk.split(","); 

        for m in range(0,len(genresk)):

            if not(genresk[m] in genresList):

                genresList.append(genresk[m]); #Check if they are in the genres list and add it if it is not

        kindk = animeInfo[k][3];

        if not(kindk in kindList):

            kindList.append(kindk); #Same with the kinds. There is only one value so it does not matter

        if animeInfo[k][4] == 'Unknown': #Some protection variables

            animeInfo[k][4] = '0';

        if animeInfo[k][5] == '':

            animeInfo[k][5] = '0';

    genreBinary = [[0 for x in range(0,len(genresList))] for y in range(0,len(animeInfo))];

    kindPos = [0 for y in range(0,len(animeInfo))];

    for k in range(0,len(animeInfo)):

        genresk = animeInfo[k][2];      #Extracts the genres (again)

        genresk = genresk.replace(" ","");

        genresk = genresk.split(",");

        for m in range(0,len(genresk)):

            binPos = genresList.index(genresk[m]); #Looks for the position in the Genres List

            genreBinary[k][binPos] = 1;            #assigning a 1 to the proper position

        kindPos[k] = kindList.index(animeInfo[k][3]); #Same with kinds, with 0, 1 or 2.

    

    featList = []; #To put everything in one list

    

    for k in range(0,len(animeInfo)):

        featk = [];

        featk.append(animeInfo[k][0]);

        featk.append(genreBinary[k]);

        featk.append(kindPos[k]);

        featk.append(int(animeInfo[k][4]))

        featk.append(float(animeInfo[k][5]))

        featList.append(featk)

    

    return genresList,featList,genreBinary;
def recommendAnime(dfRatings,dfAnimeInfo,user_id):

    #Use the clusters defined to recommend relevant animes by genre.

    

    dfRatingsk = dfAnimeInfo[dfAnimeInfo['ID'].isin(dfRatings['Anime ID'])] #Extracts the relevant animes

    dfRatingsklist = [float(i) for i in list(dfRatings['Rating'])] #Extracts the user ratings to be used later

    

    for k in range(0,len(dfRatingsklist)):

        if (dfRatingsklist[k] == -1.0):

            dfRatingsklist[k] = dfRatingsk.iloc[k,5] #If it is -1, we assume that the user rating is the average

    

    dfRatingsk['User Rating'] = dfRatingsklist #Append it to dfRatingsk

              

    dfClustersk = dfRatingsk['Cluster'].unique() #Get the unique clusters

    

    dfClustersScorek = num.zeros(len(dfClustersk),dtype=float) #Initializes the score variable

    

    dfClustersFreqk = dfRatingsk['Cluster'].value_counts() #Counts the frequency of the clusters

    

    for k in range(0,len(dfClustersk)):

        #In this case, the score will be the mean of the ratings (after substituting the -1s) times the

        #frequency of the cluster for a given user.

        dfClustersScorek[k] = dfRatingsk['User Rating'].loc[dfRatingsk['Cluster'] == dfClustersk[k]].mean()

        dfClustersScorek[k] = dfClustersScorek[k]*dfClustersFreqk.iloc[k]

    

    

    dfClustersFreqk = pd.DataFrame(dfClustersFreqk) #Convert it to DataFrame (it was Series)

    dfClustersFreqk['Score'] = dfClustersScorek #Append the scores

    dfClustersFreqk['Cluster'] = dfClustersFreqk.index #Assign the cluster numbers to the file

    dfClustersFreqk = dfClustersFreqk.sort_values(by='Score',ascending=False) #Sort it by the scores

    

    #The following will be the core for the recommendator. The output will be five or six animes

    #that will be (hopefully) relevant for the user. Since some have clusters with similar scores,

    #we weigh the three with the most score and assign a number of recommended animes to each one

    #So for example if one has a lot of weight compared to the other two, it will recommend

    #more animes from that cluster

    

    #Procedure is:

        #We get out the animes from the anime dataframe from a cluster

        #We filter out the ones that the user has already seen

        #We assign a score for each one, based on the average rating and the number of people who

        #has watched it. The latter is good to avoid obscure (but highly rated) animes.

        #We sort it by score and take the best n animes, where n is given by the weights (up to 6)

        #Repeat for the other two clusters and append the results

    

    if dfClustersFreqk.size/2 == 1: #Special case: only one cluster. No weighting

        bestClusterk = int(dfClustersFreqk.iloc[0,0])

        

        dfBestk = dfAnimeInfo[dfAnimeInfo['Cluster'] == bestClusterk]

        

        dfBestk = dfBestk[~dfBestk['ID'].isin(dfRatingsk['ID'])]

        

        dfBestk['Score'] = pd.to_numeric(dfBestk['Avg. Rating'])*num.log10(pd.to_numeric(dfBestk['Members']))

        dfBestk = dfBestk.sort_values(by='Score',ascending=False)

        dfBestk = dfBestk.iloc[0:6]

    elif dfClustersFreqk.size/2 == 2: #Special case: only two clusters

        dfClustersFreqk = dfClustersFreqk.iloc[0:2]

        sumScore = dfClustersFreqk.iloc[:,1].sum()

        

        bestClusterk1 = int(dfClustersFreqk.iloc[0,0])

        bestClusterNormScore1 = int(num.round(dfClustersFreqk.iloc[0,1]/sumScore*6))

        

        dfBestk1 = dfAnimeInfo[dfAnimeInfo['Cluster'] == bestClusterk1]

        

        dfBestk1 = dfBestk1[~dfBestk1['ID'].isin(dfRatingsk['ID'])]

        dfBestk1['Score'] = pd.to_numeric(dfBestk1['Avg. Rating'])*num.log10(pd.to_numeric(dfBestk1['Members']))

        dfBestk1 = dfBestk1.sort_values(by='Score',ascending=False)

        dfBestk1 = dfBestk1.iloc[0:bestClusterNormScore1]

        

        bestClusterk2 = int(dfClustersFreqk.iloc[1,0])

        bestClusterNormScore2 = int(num.round(dfClustersFreqk.iloc[1,1]/sumScore*6))

        

        dfBestk2 = dfAnimeInfo[dfAnimeInfo['Cluster'] == bestClusterk2]

        

        dfBestk2 = dfBestk2[~dfBestk2['ID'].isin(dfRatingsk['ID'])]

        dfBestk2['Score'] = pd.to_numeric(dfBestk2['Avg. Rating'])*num.log10(pd.to_numeric(dfBestk2['Members']))

        dfBestk2 = dfBestk2.sort_values(by='Score',ascending=False)

        dfBestk2 = dfBestk2.iloc[0:bestClusterNormScore2]

        

        dfBestk = dfBestk1.append(dfBestk2)

    else:

        dfClustersFreqk = dfClustersFreqk.iloc[0:3]

        sumScore = dfClustersFreqk.iloc[:,1].sum()

        

        bestClusterk1 = int(dfClustersFreqk.iloc[0,0])

        bestClusterNormScore1 = int(num.round(dfClustersFreqk.iloc[0,1]/sumScore*6))

        

        dfBestk1 = dfAnimeInfo[dfAnimeInfo['Cluster'] == bestClusterk1]

        

        dfBestk1 = dfBestk1[~dfBestk1['ID'].isin(dfRatingsk['ID'])]

        dfBestk1['Score'] = pd.to_numeric(dfBestk1['Avg. Rating'])*num.log10(pd.to_numeric(dfBestk1['Members']))

        dfBestk1 = dfBestk1.sort_values(by='Score',ascending=False)

        dfBestk1 = dfBestk1.iloc[0:bestClusterNormScore1]

        

        bestClusterk2 = int(dfClustersFreqk.iloc[1,0])

        bestClusterNormScore2 = int(num.round(dfClustersFreqk.iloc[1,1]/sumScore*6))

        

        dfBestk2 = dfAnimeInfo[dfAnimeInfo['Cluster'] == bestClusterk2]

        

        dfBestk2 = dfBestk2[~dfBestk2['ID'].isin(dfRatingsk['ID'])]

        dfBestk2['Score'] = pd.to_numeric(dfBestk2['Avg. Rating'])*num.log10(pd.to_numeric(dfBestk2['Members']))

        dfBestk2 = dfBestk2.sort_values(by='Score',ascending=False)

        dfBestk2 = dfBestk2.iloc[0:bestClusterNormScore2]

        

        bestClusterk3 = int(dfClustersFreqk.iloc[2,0])

        bestClusterNormScore3 = int(num.round(dfClustersFreqk.iloc[2,1]/sumScore*6))

        

        dfBestk3 = dfAnimeInfo[dfAnimeInfo['Cluster'] == bestClusterk3]

        

        dfBestk3 = dfBestk3[~dfBestk3['ID'].isin(dfRatingsk['ID'])]

        dfBestk3['Score'] = pd.to_numeric(dfBestk3['Avg. Rating'])*num.log10(pd.to_numeric(dfBestk3['Members']))

        dfBestk3 = dfBestk3.sort_values(by='Score',ascending=False)

        dfBestk3 = dfBestk3.iloc[0:bestClusterNormScore3]

        

        dfBestk = dfBestk1.append(dfBestk2)

        dfBestk = dfBestk.append(dfBestk3)   

    

    return dfBestk;
#Main program       

animeInfo, ratingInfo = dataInit(); #Initialization



genresList,featList, genresBinary = FindGenres(animeInfo); #Getting the features



kmeans = KMeans(n_clusters=45, random_state=0).fit(genresBinary) #Using K-means clustering for the genres in this case



kl = kmeans.labels_ #Getting the labels for each anime



#Convert everything to dataframes in order to speed up computations and simplify code

dfAnimeInfo = pd.DataFrame(animeInfo);

dfAnimeInfo.columns = ['ID','Name','Genre','Kind','Episodes','Avg. Rating','Members']

dfAnimeInfo['ID']=pd.to_numeric(dfAnimeInfo['ID'],errors='ignore')

dfAnimeInfo['Episodes']=pd.to_numeric(dfAnimeInfo['Episodes'],errors='ignore')

dfAnimeInfo['Avg. Rating']=pd.to_numeric(dfAnimeInfo['Avg. Rating'],errors='ignore')

dfAnimeInfo['Members']=pd.to_numeric(dfAnimeInfo['Members'],errors='ignore')

dfAnimeInfo['Cluster'] = kl;

dfAnimeInfo = dfAnimeInfo.sort_values(by='ID')



dfRatings = pd.DataFrame(ratingInfo);

dfRatings.columns = ['ID','Anime ID','Rating']

dfRatings = dfRatings.apply(pd.to_numeric)



#Get the user IDs from ratings file

user_ids = dfRatings['ID'].unique()

dfBest = pd.DataFrame()



for k in user_ids[0:2]: #Runs only ID 1 and 2

    dfRatingsRk = dfRatings.loc[dfRatings['ID'] == k]

    dfBestk = recommendAnime(dfRatingsRk,dfAnimeInfo,k)

    dfBestk['Rating ID'] = k

    dfBest = dfBest.append(dfBestk,ignore_index=True)



print(dfBest)
dfRatingsRk = dfRatings.loc[dfRatings['ID'] == 1]

print(dfAnimeInfo[dfAnimeInfo['ID'].isin(dfRatingsRk['Anime ID'])])
dfRatingsRk = dfRatings.loc[dfRatings['ID'] == 2]

print(dfAnimeInfo[dfAnimeInfo['ID'].isin(dfRatingsRk['Anime ID'])])
dfBest.to_csv(path_or_buf='result.csv',index=False,columns=['Rating ID','Name','Genre','Kind','Episodes'])