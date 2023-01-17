#Basic Sandbox

import os

import json

import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings("ignore")



#To generate a refid for each paper (dataset + bib_entries)

import hashlib #for sha1



#To build network and compute pagerank

import networkx as nx

import math as math



#For Data viz

import seaborn as sns

import matplotlib.pyplot as plt



from datetime import date

from datetime import timedelta
#Weight parameters for Approach 2 and 3 :



weights_InfluenceScore = [0.25, 0.25, 0.25, 0.25]
#1. Get the data

datafiles = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

            ifile = os.path.join(dirname, filename)

            if ifile.split(".")[-1] == "json":

                datafiles.append(ifile)

            

print("Number of Files Loaded: ", len(datafiles))





#Loading metadata csv file to get the publish time

metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")



#2. Creating of the two DataFrames:

#dfPaperList = df of Research Papers.. Variables: paper_id, paper_title, paper_authors

#dfCitationsFlat = df of all the citations . Variables: citationsId, paperId (where the citation is made),refid, title, year



authors = []



citationsFlat = []

citationsCount = 0



for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    paper_id = doc['paper_id']

    

    paper_authors = []



    for value in doc['metadata']['authors']:

        if len(doc['metadata']['authors']) == 0:

            paper_authors.append("NA")

        else:

            last = value["last"]

            first = value["first"]

            paper_authors.append(first+" "+last)



    authors.append({"paper_id": paper_id, "authors" : paper_authors})



    for key,value in doc['bib_entries'].items():

        refid = key

        title = value['title'].lower()

        year = value['year']

        venue = value['venue'] 

        SHATitleCitation = hashlib.sha1(title.lower().encode()).hexdigest() #



        if (len(title) == 0):

            continue #there is noting we can do without any title



        citationsFlat.append({"citationId":citationsCount,\

                          "refid" : SHATitleCitation,\

                          "from": paper_id,\

                          "title": title.lower(),\

                          "year": year})

        citationsCount=citationsCount+1

        

#Conversion into DataFrame

dfCitationsFlat = pd.DataFrame(citationsFlat)

authorsDf = pd.DataFrame(authors)



metadata_extract = metadata[["sha", "title", "abstract", "publish_time"]].rename(columns = {"sha" : "paper_id"})

dfPaperList = pd.merge(metadata_extract, authorsDf, on = "paper_id", how = "left")



dfPaperList["year"] = 0

dfPaperList["refid"] = ""



for i in range(len(dfPaperList)):

    

    dfPaperList["refid"][i] =  hashlib.sha1(str(dfPaperList["title"][i]).lower().encode()).hexdigest()

     #NB: We are building a custom identifier based on papers titles to ensure identification will be consistent between the papers in the Research Dataset and the papers extracted from the bib entries.

     #Unfortunately a paperId is not present for citations and doi is not provided for the whole dataset but title seem to be present for ~98% of the dataset. To enable and ease indexing capabilities we are hashing with SHA   

    dfPaperList["year"][i] = str(dfPaperList["publish_time"][i])[:4]

    

    try:

        dfPaperList["authors"][i] = dfPaperList["authors"][i].split(";")

    except:

        continue

        

quotationPapersFreq = pd.DataFrame({"refid" : dfCitationsFlat["refid"].value_counts().index, 

                       "nbQuotations" : dfCitationsFlat["title"].value_counts().values}) 



paperToScore = pd.merge(dfPaperList,quotationPapersFreq, on = "refid", how = "left")

paperToScore["nbQuotations"] = paperToScore["nbQuotations"].fillna(0)





#Adding list of references by papers according to the refid

refList = pd.DataFrame({"references" : dfCitationsFlat.groupby('from')['refid'].apply(list)}) 

refList["paper_id"] = refList.index; cols = ["paper_id","references"] ; refList = refList[cols].reset_index(drop = True) #Reformatting the reflist by papers

datasetForScoring = pd.merge(paperToScore, refList, how='left', on = 'paper_id').reset_index(drop = True)



datasetForScoring = datasetForScoring[(datasetForScoring["authors"].isna() == False)].reset_index(drop = True)
#3. A few stats regarding number of papers loaded



print("Number of Papers in the CORD-19 dataset :",dfPaperList.shape[0])

#(05/14/2020) Number of Papers in the covid dataset : 63,571



print("Number of Citations found in the CORD-19 dataset :",dfCitationsFlat.shape[0])

#(05/14/2020) Number of Citations made in the covid dataset : 4,208,974



print("Citations with no title: ",sum(1 if x == "" else 0 for x in dfCitationsFlat["title"]))

#(05/14/2020) Citations with no title:  0



#How many duplicates? 

print("Number of duplicated research paper titles: ",len(dfPaperList["title"])-len(dfPaperList["title"].drop_duplicates()))

#(05/14/2020) Number of duplicated research paper titles:  1,421



print("Number of duplicated citations titles: ",len(dfCitationsFlat["title"])-len(dfCitationsFlat["title"].drop_duplicates()))

#(05/14/2020) Number of duplicated citations titles:  2,543,820



#Dataframe Visualization

print("Number of Papers that will be scored: ", datasetForScoring.shape[0])

datasetForScoring.head()

#1. Creating an author dataset + Computation of the author page rank using an author network

#Variables for author dataset: id, name, co-authors, number of points linked to quotations, paper_count, citations, average citations,co_author_avg_citations,h-index



author_data = {}

author_id = {

    'start': 1,

    'curr': 1

}



assigned_ids = {}



def create_author_data(train_data, author_data, author_id, assigned_ids):

    for i in range(len(train_data)):

        authors = train_data.authors[i]

    

        try:

            citations = train_data.nbQuotations[i]/len(authors) #Number of times a paper have been quoted divided by len authors

        except:

            continue



        for author in authors:

            names = author.split(' ')

            unique_name = names[0] + "_" + names[len(names)-1]

            if unique_name not in author_data:

                author_data[unique_name] = {

                    'num_citations': citations,

                    'paper_count': 1,

                    'name': unique_name,

                    'author_id': author_id['curr'],

                    'co_authors': {},

                    'citations': [train_data.nbQuotations[i]]

                }

                assigned_ids[unique_name] = author_id['curr']

                author_id['curr'] += 1



            else:

                author_data[unique_name]['num_citations'] += citations

                author_data[unique_name]['paper_count'] += 1

                author_data[unique_name]['citations'].append(train_data.nbQuotations[i])



            for co_author in authors:

                co_author_names = co_author.split(' ')

                co_author_unique_name = co_author_names[0] + "_" + co_author_names[len(co_author_names)-1]

                if co_author_unique_name != unique_name:

                    author_data[unique_name]['co_authors'][co_author_unique_name] = 1

                        

            

            

# call for each data file

create_author_data(datasetForScoring, author_data, author_id, assigned_ids)



# add average citations

for data in author_data:

    author_data[data]['average_citations'] = author_data[data]['num_citations'] / author_data[data]['paper_count']

    

# adding h-index

def get_h_index(citations):

    return ([0] + [i + 1 for i, c in enumerate(sorted(citations, reverse = True)) if c >= i + 1])[-1]



data_to_df = []

for data in author_data:

    each_author = author_data[data]

    co_authors = each_author['co_authors']

    co_author_ids = []

    co_author_avg_citations = 0

    for co_author in co_authors:

        co_author_avg_citations += author_data[co_author]['average_citations']

        co_author_ids.append(assigned_ids[co_author])

    each_author['co_authors'] = co_author_ids

    each_author['co_author_avg_citations'] = co_author_avg_citations/len(co_author_ids) if len(co_author_ids) != 0 else 0

    data_to_df.append(each_author)

    

authorsData = pd.DataFrame.from_dict(data_to_df, orient='columns')



authorsData['h_index'] = authorsData.apply(lambda x: get_h_index(x.citations), axis=1)
#2. Computation of authors page rank



### AUTHOR PAGE RANK ###

#Data Pre-processing: building the dataset on which the author network will be built

train = authorsData.copy().drop(columns=['num_citations', 'h_index','paper_count', 'citations']).dropna(axis = 0, subset=['co_authors'])

train = train[train.co_authors != '[]']

train['author_id'] = pd.to_numeric(train['author_id'])



# Building up the network to compute author page rank: 

G = nx.Graph()

for i in range(len(train)):

    auth = train.iloc[i]['author_id']

    for neighbor in train.iloc[i]['co_authors']:

        if G.has_edge(auth, neighbor):

            G.add_edge(auth, neighbor, weight = G[auth][neighbor]['weight']+1)

        else:

            G.add_edge(auth, neighbor, weight = 1)

            

score_authors = nx.pagerank(G, alpha=0.55, max_iter=100, tol=1.0e-6, nstart=None, weight='weight', dangling=None)



#Saving the page rank by author id

authorPRK = pd.DataFrame.from_dict(score_authors, orient = "index")

authorPRK["author_id"] = authorPRK.index

authorPRK.columns = ["pagerank_author", "author_id"]

authorPRK.to_csv("pagerank_author.csv",index = False)
#3. Computation of publication page rank



# Building up the network to compute the pagerank for publication

G1 = nx.Graph()

for i in range(len(datasetForScoring)):

# for i in range(100): #Only on a sample

    G1.add_node(datasetForScoring['refid'][i])

    auth = datasetForScoring['refid'][i]

    

    for e in list(str(datasetForScoring["references"][i]).lstrip("[").rstrip("]").replace(" ","").split(",")):

        try:

            if G1.has_edge(auth, e):

                G1.add_edge(auth, e, weight = G[auth][e]['weight']+1)

            else:

                G1.add_edge(auth, e, weight = 1)

        except:

            continue

        

score_publication = nx.pagerank(G1, alpha=0.85, tol=1.0e-6, nstart=None, weight=1, dangling=None)



#Saving the page rank by paper id

publiPRK = pd.DataFrame.from_dict(score_publication, orient = "index")

publiPRK["publication_id"] = publiPRK.index

publiPRK.columns = ["pageRankPublication", "publication_id"]

publiPRK["publication_id"] = publiPRK["publication_id"].str.replace("'","")

publiPRK = publiPRK.reset_index(drop = True)



publiPRK.to_csv("pagerank_publication.csv",index = False)



#Integration of the variable Page Rank for publication datasetForScoring

enhancedDatasetForScoring = pd.merge(datasetForScoring,publiPRK, left_on = "refid", right_on = "publication_id", how = "left").drop(columns= ["publication_id"])

enhancedDatasetForScoring = enhancedDatasetForScoring.drop_duplicates(subset='refid', keep="last") #Temporary patch to manage the case where twice Page rank for some publications
#4. Computation of Author Scoring



#Dataset to consolidate Author Page Rank and Publication Rank in a way to compute authorP2

dfAuthorP2 = pd.merge(authorsData[["author_id","name"]],authorPRK, on = "author_id", how = "left").reset_index(drop=True)

dfAuthorP2["name"] = dfAuthorP2["name"].str.replace("_"," ")



# Extract enhancedDatasetForScoring "paper_refid" &"paper_authors"

authorsfromDf = enhancedDatasetForScoring[["refid","authors"]].reset_index(drop = True)

# authorsfromDf = authorsfromDf[(authorsfromDf["authors"].isna() == False)]

authorsfromDf = pd.DataFrame(authorsfromDf.authors.tolist(), index = authorsfromDf.refid).stack().reset_index(level=1, drop=True).reset_index(name='authors')[['authors','refid']]



#Computing the sum of publication page rank for each paper

dfAuthorP2withPRPubli = pd.merge(authorsfromDf,publiPRK, left_on = "refid", right_on = "publication_id", how = "left").drop(columns = ["refid", "publication_id"]).groupby("authors").sum()

dfAuthorP2withPRPubli["authors"] = dfAuthorP2withPRPubli.index #Reformatting

dfAuthorP2withPRPubli = dfAuthorP2withPRPubli.reset_index(drop=True)



dfAuthorP2Final = pd.merge(dfAuthorP2,dfAuthorP2withPRPubli, left_on = "name", right_on = "authors", how = "left").drop(columns = "name")



# ######### Author Scoring #########

dfAuthorP2Final["pagerank_author_norm"] = (dfAuthorP2Final["pagerank_author"]-dfAuthorP2Final["pagerank_author"].mean())/dfAuthorP2Final["pagerank_author"].std()

dfAuthorP2Final["pagerank_publication_norm"] = (dfAuthorP2Final["pageRankPublication"]-dfAuthorP2Final["pageRankPublication"].mean())/dfAuthorP2Final["pageRankPublication"].std()



dfAuthorP2Final["authorP2"] = 0.25*dfAuthorP2Final["pagerank_author_norm"] + 0.75*dfAuthorP2Final["pagerank_publication_norm"]
#1. Data Preparation



    # Consolidate Author Score for each paper

authorP2Data = dfAuthorP2Final[["authors","authorP2"]]

# enhancedDatasetForScoring = enhancedDatasetForScoring[(enhancedDatasetForScoring["authors"].isna() == False)]

authorToPaper = pd.DataFrame(enhancedDatasetForScoring[["refid","authors"]].authors.tolist(), index=enhancedDatasetForScoring[["refid","authors"]].refid).stack().reset_index(level=1, drop=True).reset_index(name='authors')[['authors','refid']]



authorP2Conso = pd.merge(authorToPaper,authorP2Data, on = "authors", how = "left")



# Consolidate AuthorP2 for each paper as followed: 0.5 * Max page rank + 0.5 * average of the page rank of all the authors

maxAuthorScore = authorP2Conso.groupby('refid').agg({'authorP2': 'max'})

meanAuthorScore = authorP2Conso.groupby('refid').agg({'authorP2': 'mean'})



authorScoring = pd.merge(maxAuthorScore,meanAuthorScore, on = "refid", how = "inner").rename(columns = {"authorP2_x" : "maxAuthorScore","authorP2_y" : "meanAuthorScore"})

authorScoring["refid"] = authorScoring.index

authorScoring = authorScoring.reset_index(drop = True)



authorScoring["authorP2"] = 0.5*authorScoring["maxAuthorScore"] + 0.5*authorScoring["meanAuthorScore"]

authorScoring = authorScoring.drop(columns = ["maxAuthorScore","meanAuthorScore"])



#Integration of the variable authorP2 for datasetForScoring

DatasetReadyForScoring = pd.merge(enhancedDatasetForScoring,authorScoring, on = "refid", how = "left")



# Influence Score Computation Dataset Overview

DatasetReadyForScoring.head()
#2. Data Exploration 

    #2.1. nbQuotations variable

q25 = DatasetReadyForScoring["nbQuotations"].quantile(.25); q50 = DatasetReadyForScoring["nbQuotations"].quantile(.5)

q75 = DatasetReadyForScoring["nbQuotations"].quantile(.75); q100 = DatasetReadyForScoring["nbQuotations"].quantile(1)



print("Max:", q100 ) ; print("Top 25% above :", q75 ); print("Top 50% above:", q50); print("Top 75% above:", q25 )
    #2.2. Pagerank Publication

pq25 = DatasetReadyForScoring["pageRankPublication"].quantile(.25); pq50 = DatasetReadyForScoring["pageRankPublication"].quantile(.5)

pq75 = DatasetReadyForScoring["pageRankPublication"].quantile(.75); pq100 = DatasetReadyForScoring["pageRankPublication"].quantile(1)



print("Max:", pq100 ) ; print("Top 25% above :", pq75 ); print("Top 50% above:", pq50); print("Top 75% above:", pq25 )
    #2.3. Author P2

aq25 = DatasetReadyForScoring["authorP2"].quantile(.25); aq50 = DatasetReadyForScoring["authorP2"].quantile(.5)

aq75 = DatasetReadyForScoring["authorP2"].quantile(.75); aq100 = DatasetReadyForScoring["authorP2"].quantile(1)

    

print("Max:", pq100 ) ; print("Top 25% above :", pq75 ); print("Top 50% above:", pq50); print("Top 75% above:", pq25 )
#3. Score computation all data by approaches



########### APPROACH ONE ########### 

now = date.today() ; numberWeeks = 6

dateThresold = now - timedelta(days = numberWeeks*7)



influenceScoreData_A1 = DatasetReadyForScoring[(DatasetReadyForScoring["publish_time"].isna() == False)].reset_index(drop = True)



influenceScoreData_A1["Recency"] = 0



    #Assessing Recency of a paper

for i in range(len(influenceScoreData_A1)):

    try:

        if influenceScoreData_A1["publish_time"][i] > dateThresold:

            influenceScoreData_A1["Recency"][i] = 1

        else:

            influenceScoreData_A1["Recency"][i] = 0

    except:

        continue

        

    #Compute Final Score Approach 1          

influenceScoreData_A1["influenceScore"] = 0



for i in range(len(influenceScoreData_A1)):

        if influenceScoreData_A1["Recency"][i] == 1:

            influenceScoreData_A1["influenceScore"] = influenceScoreData_A1["authorP2"]

        else:

            influenceScoreData_A1["influenceScore"] = influenceScoreData_A1["pageRankPublication"]





            

            

            

            

            

########### APPROACH TWO ########### 

influenceScoreData_A2 = DatasetReadyForScoring.copy()



#Compute Score for Nb_quoted and year



influenceScoreData_A2["nbQuotationsScore"] = 0

influenceScoreData_A2["yearScore"] = 0



for i in range(len(influenceScoreData_A2)):

    

    #Nb_quoted

    if influenceScoreData_A2["nbQuotations"][i] < q25:

        influenceScoreData_A2["nbQuotationsScore"][i] = 0

    elif influenceScoreData_A2["nbQuotations"][i] < q50:

        influenceScoreData_A2["nbQuotationsScore"][i] = 0.25

    elif influenceScoreData_A2["nbQuotations"][i] < q75:

        influenceScoreData_A2["nbQuotationsScore"][i] = 0.5

    else:

        influenceScoreData_A2["nbQuotationsScore"][i] = 1

        

    #Year

    if influenceScoreData_A2["year"][i] == 2020:

        influenceScoreData_A2["yearScore"][i] = 1

    elif influenceScoreData_A2["year"][i] == 2019:

        influenceScoreData_A2["yearScore"][i] = 0.5

    else:

        influenceScoreData_A2["yearScore"][i] = 0



influenceScoreData_A2["pageRankPublication_norm"] = (influenceScoreData_A2["pageRankPublication"] - influenceScoreData_A2["pageRankPublication"].mean())/ influenceScoreData_A2["pageRankPublication"].std()

influenceScoreData_A2["authorP2_norm"] = (influenceScoreData_A2["authorP2"] - influenceScoreData_A2["authorP2"].mean())/ influenceScoreData_A2["authorP2"].std()





influenceScoreData_A2["influenceScore"] = 0.00



for i in range(len(influenceScoreData_A2)):

    influenceScoreData_A2["influenceScore"][i] = weights_InfluenceScore[0] * influenceScoreData_A2["nbQuotationsScore"][i] + weights_InfluenceScore[1] * influenceScoreData_A2["yearScore"][i] + weights_InfluenceScore[2] * influenceScoreData_A2["pageRankPublication_norm"][i] + weights_InfluenceScore[3] * influenceScoreData_A2["authorP2_norm"][i]



        

influenceScoreData_A2 = influenceScoreData_A2.sort_values(by = "influenceScore", ascending = False).reset_index(drop = True)            

            

        

                        

########### APPROACH THREE ########### 

influenceScoreData_A3 = DatasetReadyForScoring.copy()



influenceScoreData_A3["influenceScore"] = 0.00



for i in range(len(influenceScoreData_A3)):

    

    x1 = 0; x2 = 0; x3 = 0; x4 = 0

    

    #Nb_quoted

    if influenceScoreData_A3["nbQuotations"][i] < q25:

        x1 = 0

    elif influenceScoreData_A3["nbQuotations"][i] < q50:

        x1 = 0.25

    elif influenceScoreData_A3["nbQuotations"][i] < q75:

        x1 = 0.5

    else:

        x1 = 1

        

    #Year

    if influenceScoreData_A3["year"][i] == 2020:

        x2 = 1

    elif influenceScoreData_A3["year"][i] == 2019:

        x2 = 0.5

    else:

        x2 = 0

        

    #PageRank Publication

    if influenceScoreData_A3["pageRankPublication"][i] < pq25:

        x3 = 0

    elif influenceScoreData_A3["pageRankPublication"][i] < pq50:

        x3 = 0.25

    elif influenceScoreData_A3["pageRankPublication"][i] < pq75:

        x3 = 0.5

    else:

        x3 = 1

        

    #Author Scoring

    if influenceScoreData_A3["authorP2"][i] < aq25:

        x4 = 0

    elif influenceScoreData_A3["authorP2"][i] < aq50:

        x4 = 0.25

    elif influenceScoreData_A3["authorP2"][i] < aq75:

        x4 = 0.5

    else:

        x4 = 1

        

    influenceScoreData_A3["influenceScore"][i] = weights_InfluenceScore[0] * x1 + weights_InfluenceScore[1] * x2 + weights_InfluenceScore[2] * x3 + weights_InfluenceScore[3] * x4

       

influenceScoreData_A3 = influenceScoreData_A3.sort_values(by = "influenceScore", ascending = False).reset_index(drop = True)

########### DATA EXPORT - EACH PAPER ID WITH SCORE ########### 



#Exporting influenceScore by paper id - FIRST APPROACH

PaperScoring_A1 = influenceScoreData_A1[["paper_id","influenceScore"]]

PaperScoring_A1.to_csv("PaperScoring_A1.csv", index = False)



#Exporting influenceScore by paper id - SECOND APPROACH

PaperScoring_A2 = influenceScoreData_A2[["paper_id","influenceScore"]]

PaperScoring_A2.to_csv("PaperScoring_A2.csv", index = False)



#Exporting influenceScore by paper id - THIRD APPROACH

PaperScoring_A3 = influenceScoreData_A3[["paper_id","influenceScore"]]

PaperScoring_A3.to_csv("PaperScoring_A3.csv", index = False)



#Exporting final dataset with all consolidated by paper id - THIRD APPROACH

ConsolidatedDfwithScore = pd.merge(DatasetReadyForScoring, PaperScoring_A1, on = "paper_id", how = "left") #Adding Score from approach 1

ConsolidatedDfwithScore = pd.merge(ConsolidatedDfwithScore, PaperScoring_A2, on = "paper_id", how = "inner") #Adding Score from approach 2

ConsolidatedDfwithScore = pd.merge(ConsolidatedDfwithScore, PaperScoring_A3, on = "paper_id", how = "inner") #Adding Score from approach 3

ConsolidatedDfwithScore = ConsolidatedDfwithScore.rename(columns = {"influenceScore_x" : "ScoreApproach1","influenceScore_y" : "ScoreApproach2", "influenceScore" : "ScoreApproach3" })



ConsolidatedDfwithScore.to_csv("ConsolidatedDfwithScore.csv", index = False)
#Building up the function to compute the influence score given a paper_id (possiblity to choose the approach)



def articleScore(paper_id,approach = 2): #Default approach is 2

    try:

        if approach == 1: 

            x = influenceScoreData_A1[(influenceScoreData_A1["paper_id"] == paper_id)]



        elif approach == 2: 

            x = influenceScoreData_A2[(influenceScoreData_A2["paper_id"] == paper_id)]



        elif approach == 3: 

            x = influenceScoreData_A3[(influenceScoreData_A3["paper_id"] == paper_id)]



        return x.influenceScore.values[0]



    except:

        return 0

    
#1. Scoring Comparison

    #1.1. Final Score Distribution 

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,5))

sns.distplot(influenceScoreData_A1["influenceScore"], ax = ax1).set_title('Approach 1: Influence Score Distribution')

sns.distplot(influenceScoreData_A2["influenceScore"], ax = ax2).set_title('Approach 2: Influence Score Distribution')

sns.distplot(influenceScoreData_A3["influenceScore"], ax = ax3).set_title('Approach 3: Influence Score Distribution')

plt.show()
    #1.2. Top 1000 for each approaches - Deep diving the characteristics for each variables



top = 1000



top_A1 = influenceScoreData_A1[["title","year", "nbQuotations", "pageRankPublication", "authorP2", "influenceScore"]].sort_values(by = "influenceScore", ascending = False).head(top).reset_index(drop = True)

top_A2 = influenceScoreData_A2[["title","year", "nbQuotations", "pageRankPublication", "authorP2", "influenceScore"]].sort_values(by = "influenceScore", ascending = False).head(top).reset_index(drop = True)

top_A3 = influenceScoreData_A3[["title","year", "nbQuotations", "pageRankPublication", "authorP2", "influenceScore"]].sort_values(by = "influenceScore", ascending = False).head(top).reset_index(drop = True)



        #Distribution Visualization



fig, ax = plt.subplots(3,4,figsize=(30,10))



plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)



#Approach 1

sns.distplot(top_A1[(top_A1["nbQuotations"] != 0)]["nbQuotations"], ax = ax[0,0]).set_title('Approach 1 - Top '+str(top)+': nbQuotations Distribution') 

sns.distplot(top_A1["year"], ax = ax[0,1]).set_title('Approach 1 - Top '+str(top)+': Year Distribution')

sns.distplot(top_A1["pageRankPublication"], ax = ax[0,2]).set_title('Approach 1 - Top '+str(top)+': Pagerank Publication Distribution')

sns.distplot(top_A1["authorP2"], ax = ax[0,3]).set_title('Approach 1 - Top '+str(top)+': Author Score Distribution')



#Approach 2

sns.distplot(top_A2["nbQuotations"], ax = ax[1,0], color=".2").set_title('Approach 2 - Top '+str(top)+': nbQuotations Distribution')

sns.distplot(top_A2["year"], ax = ax[1,1], color=".2").set_title('Approach 2 - Top '+str(top)+': Year Distribution')

sns.distplot(top_A2["pageRankPublication"], ax = ax[1,2], color=".2").set_title('Approach 2 - Top '+str(top)+': Pagerank Publication Distribution')

sns.distplot(top_A2["authorP2"], ax = ax[1,3], color=".2").set_title('Approach 2 - Top '+str(top)+': Author Score Distribution')



#Approach 3

sns.distplot(top_A3["nbQuotations"], ax = ax[2,0], color="orange").set_title('Approach 3 - Top '+str(top)+': nbQuotations Distribution')

sns.distplot(top_A3["year"], ax = ax[2,1], color="orange").set_title('Approach 3 - Top '+str(top)+': Year Distribution')

sns.distplot(top_A3["pageRankPublication"], ax = ax[2,2], color="orange").set_title('Approach 3 - Top '+str(top)+': Pagerank Publication Distribution')

sns.distplot(top_A3["authorP2"], ax = ax[2,3], color="orange").set_title('Approach 3 - Top '+str(top)+': Author Score Distribution')



plt.show()
    #1.3. Does the top1000 of each approaches have common papers in their top?

print("% of Common papers of Approach 1 with Approach 2 :", round(len(pd.merge(top_A1, top_A2, how='inner', on=['title']))/top*100,2))

print("% of Common papers of Approach 1 with Approach 3 :", round(len(pd.merge(top_A1, top_A3, how='inner', on=['title']))/top*100,2))

print("% of Common papers of Approach 2 with Approach 3 :", round(len(pd.merge(top_A3, top_A2, how='inner', on=['title']))/top*100,2))
paper_id = "a80c3e9dfde9824cb8f54f9f382d0d601743ffc0" 

#this is find in CORD-19 dataset, e.g. 



score = articleScore(paper_id)

#return article Score, for default approach

print(score)





score = articleScore(paper_id,1)

#return article Score, for approach 1

print(score)



score = articleScore(paper_id,2)

#return article Score, for approach 2

print(score)



score = articleScore(paper_id,3)

#return article Score, for approach 3

print(score)





paper_id = "DOES NOT EXIST" 

score = articleScore(paper_id)

#return article Score, when paper is missing

print(score)

#Variable NbQuotations

sns.distplot(DatasetReadyForScoring["nbQuotations"]).set_title('NbQuotations Distribution')
#Variable year

sns.distplot(DatasetReadyForScoring["year"]).set_title('Year Distribution')
#Variable Page Rank Publication

sns.distplot(DatasetReadyForScoring["pageRankPublication"]).set_title('Page Rank Publication Distribution')
#Variable authorP2

sns.distplot(DatasetReadyForScoring["authorP2"]).set_title('Author Score Distribution')