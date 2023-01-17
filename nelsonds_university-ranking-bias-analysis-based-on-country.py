#importing libraries

import IPython

import numpy as np

import scipy as sp

import pandas as pd

import matplotlib

from matplotlib import pyplot as plt

matplotlib.style.use('ggplot')

%matplotlib inline

import re

import math

from scipy import stats

from sklearn import datasets

from sklearn.cluster import KMeans

import sklearn.metrics as sm

import matplotlib.patches as mpatches



# Setting options

pd.set_option('display.max_rows', 5000)

pd.set_option('display.max_columns', 5000)

pd.set_option('display.width', 5000)
# Loading data 

times_df = pd.read_csv('../input/timesData.csv')

cwur_df = pd.read_csv('../input/cwurData.csv')

shanghai_df = pd.read_csv('../input/shanghaiData.csv')
# Cleaning data



times_df = times_df.replace("École Normale Supérieure", "Ecole Normale Superieure")

times_df = times_df.replace("École Polytechnique", "Ecole Polytechnique")

times_df = times_df.replace("École Polytechnique Fédérale de Lausanne","Ecole Polytechnique Federale de Lausanne")

times_df = times_df.replace("ETH Zurich – Swiss Federal Institute of Technology Zurich",

                            "Swiss Federal Institute of Technology Zurich")

times_df = times_df.replace("King’s College London", "King's College London")

times_df = times_df.replace("Rutgers, the State University of New Jersey", "Rutgers University, New Brunswick")

times_df = times_df.replace("The University of Queensland", "University of Queensland")

times_df = times_df.replace("University of Göttingen", "University of Gottingen")

times_df = times_df.replace("University of Michigan", "University of Michigan, Ann Arbor")

times_df = times_df.replace("University of Minnesota", "University of Minnesota, Twin Cities")

times_df = times_df.replace("Paris-Sud University", "University of Paris-Sud")

times_df = times_df.replace("Washington University in St Louis", "Washington University in St. Louis")

times_df = times_df.replace("University of Massachusetts", "University of Massachusetts, Amherst")

times_df = times_df.replace("Wageningen University and Research Center", "Wageningen University and Research Centre")

times_df = times_df.replace("Indiana University", "Indiana University Bloomington")

times_df = times_df.replace("Paris Diderot University – Paris 7", "Paris Diderot University")

times_df = times_df.replace("KTH Royal Institute of Technology", "Royal Institute of Technology")

times_df = times_df.replace("Université Libre de Bruxelles", "University Libre Bruxelles")

times_df = times_df.replace("University of São Paulo", "University of Sao Paulo")

times_df = times_df.replace("Université Catholique de Louvain", "Catholic University of Louvain")

times_df = times_df.replace("Aix-Marseille University", "Aix Marseille University")



cwur_df = cwur_df.replace("University of Göttingen", "University of Gottingen")

cwur_df = cwur_df.replace("École normale supérieure - Paris", "Ecole Normale Superieure")

cwur_df = cwur_df.replace("École Polytechnique", "Ecole Polytechnique")

cwur_df = cwur_df.replace("Indiana University - Bloomington", "Indiana University Bloomington")

cwur_df = cwur_df.replace("Ludwig Maximilian University of Munich", "LMU Munich")

cwur_df = cwur_df.replace("Ohio State University, Columbus", "Ohio State University")

cwur_df = cwur_df.replace("Paris Diderot University - Paris 7", "Paris Diderot University")

cwur_df = cwur_df.replace("Pennsylvania State University, University Park", "Pennsylvania State University")

cwur_df = cwur_df.replace("Pierre-and-Marie-Curie University", "Pierre and Marie Curie University")

cwur_df = cwur_df.replace("Purdue University, West Lafayette", "Purdue University")

cwur_df = cwur_df.replace("Rutgers University-New Brunswick", "Rutgers University, New Brunswick")

cwur_df = cwur_df.replace("Swiss Federal Institute of Technology in Zurich", "Swiss Federal Institute of Technology Zurich")

cwur_df = cwur_df.replace("Swiss Federal Institute of Technology in Lausanne","Ecole Polytechnique Federale de Lausanne")

cwur_df = cwur_df.replace("Technion \xe2\x80\x93 Israel Institute of Technology", "Technion-Israel Institute of Technology")

cwur_df = cwur_df.replace("Texas A&M University, College Station", "Texas A&M University")

cwur_df = cwur_df.replace("University of Illinois at Urbana–Champaign", "University of Illinois at Urbana-Champaign")

cwur_df = cwur_df.replace("University of Pittsburgh - Pittsburgh Campus", "University of Pittsburgh")

cwur_df = cwur_df.replace("University of Washington - Seattle", "University of Washington")

cwur_df = cwur_df.replace("University of Wisconsin–Madison", "University of Wisconsin-Madison")

cwur_df = cwur_df.replace("Katholieke Universiteit Leuven", "KU Leuven")

cwur_df = cwur_df.replace("Ruprecht Karl University of Heidelberg", "Heidelberg University")

cwur_df = cwur_df.replace("London School of Economics", "London School of Economics and Political Science")

cwur_df = cwur_df.replace("University of Massachusetts Amherst", "University of Massachusetts, Amherst")

cwur_df = cwur_df.replace("Technion – Israel Institute of Technology", "Technion Israel Institute of Technology")

cwur_df = cwur_df.replace("University of Colorado Denver", "University of Colorado at Denver")

cwur_df = cwur_df.replace("Albert Ludwig University of Freiburg", "University of Freiburg")

cwur_df = cwur_df.replace("Université libre de Bruxelles", "University Libre Bruxelles")

cwur_df = cwur_df.replace("University of São Paulo", "University of Sao Paulo")

cwur_df = cwur_df.replace("Aix-Marseille University", "Aix Marseille University")

cwur_df = cwur_df.replace("Université catholique de Louvain", "Catholic University of Louvain")

cwur_df = cwur_df.replace("Trinity College, Dublin", "Trinity College Dublin")



shanghai_df = shanghai_df.replace("Arizona State University - Tempe", "Arizona State University")

shanghai_df = shanghai_df.replace("Ecole Normale Superieure - Paris", "Ecole Normale Superieure")

shanghai_df = shanghai_df.replace("Massachusetts Institute of Technology (MIT)", "Massachusetts Institute of Technology")

shanghai_df = shanghai_df.replace("Pennsylvania State University - University Park", "Pennsylvania State University")

shanghai_df = shanghai_df.replace("Pierre and Marie  Curie University - Paris 6", "Pierre and Marie Curie University")

shanghai_df = shanghai_df.replace("Purdue University - West Lafayette", "Purdue University")

shanghai_df = shanghai_df.replace("Rutgers, The State University of New Jersey - New Brunswick",

                                  "Rutgers University, New Brunswick")

shanghai_df = shanghai_df.replace("Technical University Munich", "Technical University of Munich")

shanghai_df = shanghai_df.replace("Texas A & M University", "Texas A&M University")

shanghai_df = shanghai_df.replace("Texas A&M University - College Station", "Texas A&M University")

shanghai_df = shanghai_df.replace("The Australian National University", "Australian National University")

shanghai_df = shanghai_df.replace("The Hebrew University of Jerusalem", "Hebrew University of Jerusalem")

shanghai_df = shanghai_df.replace("The Imperial College of Science, Technology and Medicine", "Imperial College London")

shanghai_df = shanghai_df.replace("The Johns Hopkins University", "Johns Hopkins University")                                

shanghai_df = shanghai_df.replace("The Ohio State University - Columbus","Ohio State University")

shanghai_df = shanghai_df.replace("The University of Edinburgh","University of Edinburgh")

shanghai_df = shanghai_df.replace("The University of Manchester", "University of Manchester")

shanghai_df = shanghai_df.replace("The University of Melbourne","University of Melbourne")

shanghai_df = shanghai_df.replace("The University of Queensland", "University of Queensland")

shanghai_df = shanghai_df.replace("The University of Texas at Austin", "University of Texas at Austin")

shanghai_df = shanghai_df.replace("The University of Texas Southwestern Medical Center at Dallas",

                                  "University of Texas Southwestern Medical Center")

shanghai_df = shanghai_df.replace("The University of Tokyo","University of Tokyo")

shanghai_df = shanghai_df.replace("The University of Western Australia", "University of Western Australia")

shanghai_df = shanghai_df.replace("University of California-Berkeley", "University of California, Berkeley")

shanghai_df = shanghai_df.replace("University of Colorado at Boulder", "University of Colorado Boulder")

shanghai_df = shanghai_df.replace("University of Michigan - Ann Arbor", "University of Michigan, Ann Arbor")

shanghai_df = shanghai_df.replace("University of Michigan-Ann Arbor", "University of Michigan, Ann Arbor")

shanghai_df = shanghai_df.replace("University of Paris Sud (Paris 11)", "University of Paris-Sud")

shanghai_df = shanghai_df.replace("University of Paris-Sud (Paris 11)", "University of Paris-Sud")

shanghai_df = shanghai_df.replace("University of Pittsburgh-Pittsburgh Campus", "University of Pittsburgh")

shanghai_df = shanghai_df.replace("University of Pittsburgh, Pittsburgh Campus", "University of Pittsburgh")

shanghai_df = shanghai_df.replace("University of Wisconsin - Madison", "University of Wisconsin-Madison")

shanghai_df = shanghai_df.replace("University of Munich","LMU Munich")

shanghai_df = shanghai_df.replace("Moscow State University", "Lomonosov Moscow State University")

shanghai_df = shanghai_df.replace("University of Massachusetts Medical School - Worcester",

                                  "University of Massachusetts Medical School")

shanghai_df = shanghai_df.replace("Joseph Fourier University (Grenoble 1)", "Joseph Fourier University")

shanghai_df = shanghai_df.replace("University Paris Diderot - Paris 7", "Paris Diderot University")

shanghai_df = shanghai_df.replace("University of Wageningen", "Wageningen University and Research Centre")

shanghai_df = shanghai_df.replace("The University of Texas M. D. Anderson Cancer Center",

                                  "University of Texas MD Anderson Cancer Center")

shanghai_df = shanghai_df.replace("Technion-Israel Institute of Technology", "Technion Israel Institute of Technology")

shanghai_df = shanghai_df.replace("Swiss Federal Institute of Technology Lausanne", "Ecole Polytechnique Federale de Lausanne")

shanghai_df = shanghai_df.replace("University of Frankfurt", "Goethe University Frankfurt")

shanghai_df = shanghai_df.replace("The University of Glasgow", "University of Glasgow")

shanghai_df = shanghai_df.replace("The University of Sheffield", "University of Sheffield")

shanghai_df = shanghai_df.replace("The University of New South Wales", "University of New South Wales")

shanghai_df = shanghai_df.replace("University of Massachusetts Amherst", "University of Massachusetts, Amherst")

shanghai_df = shanghai_df.replace("University of Goettingen", "University of Gottingen")

shanghai_df = shanghai_df.replace("The University of Texas at Dallas", "University of Texas at Dallas")

shanghai_df = shanghai_df.replace("The University of Hong Kong", "University of Hong Kong")

shanghai_df = shanghai_df.replace("The Hong Kong University of Science and Technology",

                                  "Hong Kong University of Science and Technology")

shanghai_df = shanghai_df.replace("Royal Holloway, U. of London", "Royal Holloway, University of London")

shanghai_df = shanghai_df.replace("Queen Mary, University of London", "Queen Mary University of London")

shanghai_df = shanghai_df.replace("Korea Advanced Institute of Science and Technology",

                                  "Korea Advanced Institute of Science and Technology (KAIST)")



# recast data type

times_df['international'] = times_df['international'].replace('-', np.nan)

times_df['international'] = times_df['international'].astype(float)

times_df['income'] = times_df['income'].replace('-', np.nan)

times_df['income'] = times_df['income'].astype(float)

times_df['total_score'] = times_df['total_score'].replace('-', np.nan)

times_df['total_score'] = times_df['total_score'].astype(float)



# fill in na values with mean in the year and impute total score for times data

for year in range(2011, 2017):

    inter_mean = times_df[times_df['year'] == year].international.mean()

    income_mean = times_df[times_df['year'] == year].income.mean()

    times_df.ix[(times_df.year == year) & (times_df.international.isnull()), 'international'] = inter_mean

    times_df.ix[(times_df.year == year) & (times_df.income.isnull()), 'income'] = income_mean

times_df.ix[times_df.total_score.isnull(), 'total_score'] = 0.3*times_df['teaching'] + 0.3*times_df['citations'

                        ] + 0.3*times_df['research'] + 0.075*times_df['international'] + 0.025*times_df['income']



# Rename columns

cwur_df.rename(columns={'institution': 'university_name'}, inplace=True)



print("Data Cleaned")
# Getting data in appropriate format



# replace ranking range to midpoint

def mid_rank(rank_string):

    rank = re.sub('=', '', rank_string)

    rank = rank.split('-')

    s = 0

    for each in rank:

        each = float(each)

        s = s + each

    return s/len(rank)



# replace ranking range for shanghai and times data

times_df['world_rank_tidy'] = times_df['world_rank'].apply(mid_rank)

shanghai_df['world_rank_tidy'] = shanghai_df['world_rank'].apply(mid_rank)



# get unique school and country using times and cwur data 

# Manually link countries for unique shanghai universities

shanghai_schools = pd.DataFrame([['Technion-Israel Institute of Technology', 'Israel'],

                   ['Swiss Federal Institute of Technology Lausanne', 'Switzerland']], columns=['university_name', 'country'])

school_country = cwur_df.drop_duplicates(['university_name', 'country'])[['university_name', 'country']].append(

    times_df.drop_duplicates(['university_name', 'country'])[['university_name', 'country']], ignore_index=True).append(

    shanghai_schools, ignore_index=True)

school_country['country'].replace(['United States of America', 'United States'], 'USA', inplace=True)

school_country['country'].replace(['United Kingdom'], 'UK', inplace=True)



# Manually replacing countries which were not present in our pivot for countires - cwur

school_country['country'][school_country['university_name'] == 'Technion-Israel Institute of Technology'] = 'Israel'

school_country['country'][school_country['university_name'] == 'Swiss Federal Institute of Technology Lausanne'] = 'Switzerland'

school_country = school_country.drop_duplicates(['university_name', 'country'])[['university_name', 'country']]

school_country = school_country.reset_index(drop=True)



# get ranking and score information by year

cwur_world_ranking = cwur_df[['university_name', 'country', 'world_rank', 'year']]

cwur_world_ranking = cwur_world_ranking.pivot(index = 'university_name', columns = 'year')['world_rank']

cwur_world_ranking.columns = ['cwur_2012_r', 'cwur_2013_r', 'cwur_2014_r', 'cwur_2015_r']

cwur_world_ranking = cwur_world_ranking.reset_index()



times_ranking = times_df[['university_name', 'country', 'world_rank_tidy', 'year']]

times_ranking = times_ranking.pivot(index = 'university_name', columns = 'year')['world_rank_tidy']

times_ranking.columns = ['times_2011_r', 'times_2012_r', 'times_2013_r', 'times_2014_r', 'times_2015_r', 'times_2016_r']

times_ranking = times_ranking.reset_index()



shanghai_ranking = shanghai_df[['university_name', 'world_rank_tidy', 'year']]

for y in range(2005, 2011):

    shanghai_ranking = shanghai_ranking[shanghai_ranking.year != y]

shanghai_ranking = shanghai_ranking.pivot(index = 'university_name', columns = 'year')['world_rank_tidy']

shanghai_ranking.columns = ['sh_2011_r', 'sh_2012_r', 'sh_2013_r', 'sh_2014_r', 'sh_2015_r']

shanghai_ranking = shanghai_ranking.reset_index()



# join ranking and score for all 3

rank_all = pd.merge(cwur_world_ranking, times_ranking, on = 'university_name', how = 'outer')

rank_all = pd.merge(rank_all, shanghai_ranking, on = 'university_name', how = 'outer')

rank_all = pd.merge(rank_all, school_country, on = 'university_name', how = 'left')



rank_all.head(2)
# Merging relevant data and computing pairwise ranking system difference for each university

# For universities which are not common in all ranking system, I am imputing a rank of 700



# Taking top 100 colleges from 3 ranking systems for the year 2015

top = 150

rank_analysis = rank_all[['university_name','country', 'times_2014_r', 'cwur_2014_r', 'sh_2014_r']]

ra_t = rank_analysis.sort_values(by='times_2014_r').head(top)

ra_c = rank_analysis.sort_values(by='cwur_2014_r').head(top)

ra_s = rank_analysis.sort_values(by='sh_2014_r').head(top)



# Rename columns

ra_c.rename(columns={'country': 'country_c', 'times_2014_r': 'times_2014_r_c',

                     'cwur_2014_r': 'cwur_2014_r_c', 'sh_2014_r': 'sh_2014_r_c'}, inplace=True)

ra_s.rename(columns={'country': 'country_s', 'times_2014_r': 'times_2014_r_s',

                     'cwur_2014_r': 'cwur_2014_r_s', 'sh_2014_r': 'sh_2014_r_s'}, inplace=True)



# Merging the data based on top 100 universities from each ranking

rank_analysis_sct = pd.merge(ra_t, 

                             pd.merge(ra_c, 

                              ra_s, on = 'university_name', how = 'outer'), 

                                    on = 'university_name', how = 'outer')



# Ensuring country column is not blank for universities not present in all 3 rankings

for i in range(len(rank_analysis_sct)):

    if pd.isnull(rank_analysis_sct.loc[i, 'country']):

        rank_analysis_sct.loc[i, 'country'] = str(rank_analysis[rank_analysis['university_name'] ==

            rank_analysis_sct.loc[i, 'university_name']].iloc[0]['country'])





# Ensuring rank column is not blank for universities not present in all 3 rankings

rank_analysis_sct['times_2014_r'] = rank_analysis_sct['times_2014_r'].replace(np.nan, rank_analysis_sct['times_2014_r_c'])

rank_analysis_sct['times_2014_r'] = rank_analysis_sct['times_2014_r'].replace(np.nan, rank_analysis_sct['times_2014_r_s'])



rank_analysis_sct['cwur_2014_r'] = rank_analysis_sct['cwur_2014_r'].replace(np.nan, rank_analysis_sct['cwur_2014_r_c'])

rank_analysis_sct['cwur_2014_r'] = rank_analysis_sct['cwur_2014_r'].replace(np.nan, rank_analysis_sct['cwur_2014_r_s'])



rank_analysis_sct['sh_2014_r'] = rank_analysis_sct['sh_2014_r'].replace(np.nan, rank_analysis_sct['sh_2014_r_c'])

rank_analysis_sct['sh_2014_r'] = rank_analysis_sct['sh_2014_r'].replace(np.nan, rank_analysis_sct['sh_2014_r_s'])



# Replacing nan items (universities which do not exist in ranking) with rank of 700 to ensure they are at farther distance

rank_analysis_sct['times_2014_r'] = rank_analysis_sct['times_2014_r'].replace(np.nan, 700).astype(int)

rank_analysis_sct['cwur_2014_r'] = rank_analysis_sct['cwur_2014_r'].replace(np.nan, 700).astype(int)

rank_analysis_sct['sh_2014_r'] = rank_analysis_sct['sh_2014_r'].replace(np.nan, 700).astype(int)



# Selecting only required columns

rank_analysis_sct = rank_analysis_sct[['university_name', 'country', 

                                        'times_2014_r', 'cwur_2014_r', 'sh_2014_r']]



# Creating columns for difference in ranking for each pair

rank_analysis_sct['t_c'] = rank_analysis_sct['times_2014_r'] - rank_analysis_sct['cwur_2014_r']

rank_analysis_sct['t_s'] = rank_analysis_sct['times_2014_r'] - rank_analysis_sct['sh_2014_r']

rank_analysis_sct['c_s'] = rank_analysis_sct['cwur_2014_r'] - rank_analysis_sct['sh_2014_r']



rank_analysis_sct.head(2)
# Checking the distribution of pairwise ranking difference



fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 6))

fig.text(0.04, 0.5, 'Number of Universities', va='center', rotation='vertical', fontsize =15)



plt.subplot(1,3,1)

plt.hist(rank_analysis_sct.t_c, color = 'purple', alpha = 0.4, range=[-400,800], bins=(25))

plt.axvline(0, color = 'purple', linestyle = 'dashed', linewidth = 2)

plt.xlabel('Times & CWUR')



plt.subplot(1,3,2)

plt.hist(rank_analysis_sct.t_s, color = 'purple', alpha = 0.4, range=[-400,800], bins=(25))

plt.axvline(0, color = 'purple', linestyle = 'dashed', linewidth = 2)

plt.xlabel('Times & Shanghai')



plt.subplot(1,3,3)

plt.hist(rank_analysis_sct.c_s, color = 'purple', alpha = 0.4, range=[-400,800], bins=(25))

plt.axvline(0, color = 'purple', linestyle = 'dashed', linewidth = 2)

plt.xlabel('CWUR & Shanghai')



plt.suptitle("Distribution of pairwise ranking difference", fontsize=20)



plt.savefig('plot_all_hist.jpg')

plt.show()
# Function to create logical clusters by hardcoding group memberships

# The groups are

# 1. University heavily biased towards ranking system 1 -> Pairwise difference greater than 216

# 2. University slightly biased towards ranking system 1 -> Diff less than 216 greater than 50

# 3. University in ranking system 1 and ranking system 2 not biased -> Pairwise diff less than +/- 50

# 4. University slightly biased towards ranking system 2 -> Diff greater than -216 less than -50

# 5. University heavily biased towards ranking system 2 -> Pairwise difference lesser than -216



def logical_cluster(pair_col, logical_cluster_col):

    rank_analysis_sct[logical_cluster_col] = 0

    for i in range(len(rank_analysis_sct)):

        if rank_analysis_sct.loc[i,pair_col] < -216: rank_analysis_sct.loc[i,logical_cluster_col] = 0

        elif rank_analysis_sct.loc[i,pair_col] < -50 and rank_analysis_sct.loc[i,pair_col] >= -216:

            rank_analysis_sct.loc[i,logical_cluster_col] = 1

        elif rank_analysis_sct.loc[i,pair_col] > -50 and rank_analysis_sct.loc[i,pair_col] < 50:

            rank_analysis_sct.loc[i,logical_cluster_col] = 2

        elif rank_analysis_sct.loc[i,pair_col] > 50 and rank_analysis_sct.loc[i,pair_col] <= 216:

            rank_analysis_sct.loc[i,logical_cluster_col] = 3

        elif rank_analysis_sct.loc[i,pair_col] > 216: rank_analysis_sct.loc[i,logical_cluster_col] = 4
# Creating logical clusters based on intervals obtained after eyeballing the data

logical_cluster('t_c', 't_c_cluster_logical')

logical_cluster('t_s', 't_s_cluster_logical')

logical_cluster('c_s', 'c_s_cluster_logical')
# Function to create K-means cluster

def kmeans_cluster(pair_col, knn_cluster_col, order):

    model = KMeans(n_clusters=5)

    k_mean = rank_analysis_sct[[pair_col]]

    model.fit(k_mean)

    pred = np.choose(model.labels_, order).astype(np.int64)  # Assigning correct labels

    rank_analysis_sct[knn_cluster_col] = pred  # Adding column of cluster information to dataset
# Creating kmeans clusters

np.random.seed(seed=1)

kmeans_cluster('t_c', 't_c_cluster_kmeans', [2, 4, 0, 1, 3])

kmeans_cluster('t_s', 't_s_cluster_kmeans', [2, 4, 0, 3, 1])

kmeans_cluster('c_s', 'c_s_cluster_kmeans', [2, 0, 1, 4, 3])
# Function to create scatter plot for pairwise clustering results

def bias_scatter(colormap, rank_diff, cluster, r1, r2, typ):  

    plt.scatter(rank_diff, rank_diff, c=colormap[cluster], s=40, alpha=0.6)

    plt.title('University Bias - '+ r1 + ' vs ' + r2 + ' (' + typ + ')', fontsize = 15)

    plt.xlabel('Difference')

    plt.ylabel('Difference')

    b1 = mpatches.Patch(color=colormap[0], label='Highly Favored by' + r1, alpha = 0.7)

    b2 = mpatches.Patch(color=colormap[1], label='Favored by' + r1, alpha = 0.7)

    b3 = mpatches.Patch(color=colormap[2], label='Neutral', alpha = 0.7)

    b4 = mpatches.Patch(color=colormap[3], label='Favored by' + r2, alpha = 0.7)

    b5 = mpatches.Patch(color=colormap[4], label='Highly Favored by Times' +r2, alpha = 0.7)

    plt.legend(handles=[b1, b2, b3, b4, b5], loc = 2)



    #plt.savefig('LogicalVsKMean.jpg')

    #plt.show()
# Plotting scatterplot

colormap_tc = np.array(['navy', 'skyblue', 'black','palegreen', 'green'])

colormap_ts = np.array(['navy', 'skyblue', 'black','coral', 'darkred'])

colormap_cs = np.array(['green', 'palegreen', 'black','coral', 'darkred'])



plt.figure(figsize=(12,22))

plt.subplot(3, 2, 1)

bias_scatter(colormap_tc, rank_analysis_sct.t_c, rank_analysis_sct['t_c_cluster_logical'], 'Times', 'CWUR', 'Logical')

plt.subplot(3, 2, 2)

bias_scatter(colormap_tc, rank_analysis_sct.t_c, rank_analysis_sct['t_c_cluster_kmeans'], 'Times', 'CWUR', 'K-means')

plt.subplot(3, 2, 3)

bias_scatter(colormap_ts, rank_analysis_sct.t_s, rank_analysis_sct['t_s_cluster_logical'], 'Times', 'Shanghai', 'Logical')

plt.subplot(3, 2, 4)

bias_scatter(colormap_ts, rank_analysis_sct.t_s, rank_analysis_sct['t_s_cluster_kmeans'], 'Times', 'Shanghai', 'K-means')

plt.subplot(3, 2, 5)

bias_scatter(colormap_cs, rank_analysis_sct.c_s, rank_analysis_sct['c_s_cluster_logical'], 'CWUR', 'Shanghai', 'Logical')

plt.subplot(3, 2, 6)

bias_scatter(colormap_cs, rank_analysis_sct.c_s, rank_analysis_sct['c_s_cluster_kmeans'], 'CWUR', 'Shanghai', 'K-means')

plt.savefig('plot_clusters_scatter.jpg')
# Function to create barplot for pairwise clustering results

def bias_bar(logical_col, knn_col, cm, r1, r2):

    logical_bias = rank_analysis_sct.groupby(logical_col).count()['university_name']

    kmeans_bias = rank_analysis_sct.groupby(knn_col).count()['university_name']

    

    x = logical_bias.index

    y1 = logical_bias.values

    y2 = kmeans_bias

    bar_width = 0.35

    opacity = 0.7

    

    rects1 = plt.bar([x[0], x[0]+0.4], [y1[0], y2[0]], bar_width,  alpha=opacity, color=cm[0], label='High Favor: ' + r1)

    rects2 = plt.bar([x[1], x[1]+0.4], [y1[1], y2[1]], bar_width, alpha=opacity, color=cm[1], label='Favor: ' + r1)

    rects3 = plt.bar([x[2], x[2]+0.4], [y1[2], y2[2]], bar_width, alpha=opacity, color=cm[2], label='Neutral')

    rects4 = plt.bar([x[3], x[3]+0.4], [y1[3], y2[3]], bar_width, alpha=opacity, color=cm[3], label='Favor: ' + r2)

    rects5 = plt.bar([x[4], x[4]+0.4], [y1[4], y2[4]], bar_width, alpha=opacity, color=cm[4], label='High favor: ' + r2)



    plt.text(x[0], y1[0], y1[0], ha='center', va='bottom', size=10)

    plt.text(x[1], y1[1], y1[1], ha='center', va='bottom', size=10)

    plt.text(x[2], y1[2], y1[2], ha='center', va='bottom', size=10)

    plt.text(x[3], y1[3], y1[3], ha='center', va='bottom', size=10)

    plt.text(x[4], y1[4], y1[4], ha='center', va='bottom', size=10)

    

    plt.text(x[0] + bar_width, y2[0], y2[0], ha='center', va='bottom', size=10)

    plt.text(x[1] + bar_width, y2[1], y2[1], ha='center', va='bottom', size=10)

    plt.text(x[2] + bar_width, y2[2], y2[2], ha='center', va='bottom', size=10)

    plt.text(x[3] + bar_width, y2[3], y2[3], ha='center', va='bottom', size=10)

    plt.text(x[4] + bar_width, y2[4], y2[4], ha='center', va='bottom', size=10)



    plt.xlabel('Bias')

    plt.ylabel('Univesities')

    #plt.title('Bias in University Pairs')

    plt.xticks(x + bar_width, ('Logical / KMeans', 'Logical / KMeans',

                               'Logical / KMeans', 'Logical / KMeans', 'Logical / KMeans'))



    plt.legend()

    plt.tight_layout()
# Plotting barplot

plt.figure(figsize=(9,12))

plt.subplot(3, 1, 1)

bias_bar('t_c_cluster_logical', 't_c_cluster_kmeans', colormap_tc, 'Times', 'CWUR')

plt.subplot(3, 1, 2)

bias_bar('t_s_cluster_logical', 't_s_cluster_kmeans', colormap_ts, 'Times', 'Shanghai')

plt.subplot(3, 1, 3)

bias_bar('c_s_cluster_logical', 'c_s_cluster_kmeans', colormap_cs, 'CWUR', 'Shanghai')

plt.savefig('plot_clusters_bar.jpg')
# Comparing K-mean classification to logical classification

y = rank_analysis_sct.t_c_cluster_logical



# Performance Metrics

print('Accuracy',sm.accuracy_score(y, rank_analysis_sct['t_c_cluster_kmeans']))



# Confusion Matrix

sm.confusion_matrix(y, rank_analysis_sct['t_c_cluster_kmeans'])
# Creating binary columns to determine if 2 systems agree on the ranking of University (based on cluster)

for i in range(len(rank_analysis_sct)):

    if rank_analysis_sct.loc[i,'t_c_cluster_kmeans'] in [1,2,3]: rank_analysis_sct.loc[i,'t_c_proximity'] = 1

    else: rank_analysis_sct.loc[i,'t_c_proximity'] = 0

    if rank_analysis_sct.loc[i,'t_s_cluster_kmeans'] in [1,2,3]: rank_analysis_sct.loc[i,'t_s_proximity'] = 1

    else: rank_analysis_sct.loc[i,'t_s_proximity'] = 0

    if rank_analysis_sct.loc[i,'c_s_cluster_kmeans'] in [1,2,3]: rank_analysis_sct.loc[i,'c_s_proximity'] = 1

    else: rank_analysis_sct.loc[i,'c_s_proximity'] = 0



# Creating column for aggregate trustworthiness of all 3 ranking systems for each University

# Score of 3 means all 3 ranking sytem pairs agree on ranking of a University and

# Score of 0 means that no pair of ranking system agrees on ranking of a University

rank_analysis_sct['impartiality_score'] = rank_analysis_sct['t_c_proximity'

                                      ] + rank_analysis_sct['t_s_proximity'] + rank_analysis_sct['c_s_proximity']



                                                                                                 

rank_analysis_sct.to_csv('resultsRankingAnalysis.csv')



# Summarizing results

assurance_summary = rank_analysis_sct[['university_name', 'impartiality_score']].groupby('impartiality_score').count()

assurance_summary.rename(columns={'university_name': 'Total Universities'}, inplace=True)

assurance_summary.sort_index(ascending = False)
# Preparing data for analyzing country bias

country_bias = pd.DataFrame(rank_analysis_sct.groupby('country').count().sort_values(by=

                                      'university_name',ascending = False)['university_name'])

country_bias = pd.DataFrame(list(country_bias['university_name'].values),

                            list(country_bias['university_name'].index))

country_bias.rename(columns={0: 'Total Universities'}, inplace=True)

print(country_bias)
# Creating function to compute bias based on the kmeans cluster affiliation of a university

def country_bias_calc(p_kmeans, p, bias_name, country_bias_tab):

    pkm1, pkm2 = p_kmeans[0]+'_cluster_kmeans', p_kmeans[1]+'_cluster_kmeans'

    

    bias_pair = pd.DataFrame(rank_analysis_sct[rank_analysis_sct[pkm1].isin(p[0]) &

                                               rank_analysis_sct[pkm2].isin(p[1])

                                              ].groupby('country').count()['university_name'])

    bias_pair = pd.DataFrame(list(bias_pair['university_name'].values),

                             list(bias_pair['university_name'].index))

    bias_pair.rename(columns={0: bias_name}, inplace=True)

    

    if country_bias_tab.empty: tab = country_bias

    else: tab = country_bias_tab

    country_bias_tab = pd.merge(tab, bias_pair, on=None,left_index=True, right_index=True, 

                                how = 'left')

    country_bias_tab[bias_name] = country_bias_tab[bias_name].replace(np.nan, 0)

    country_bias_tab[bias_name + ' %'] = country_bias_tab[bias_name] / country_bias_tab[

        'Total Universities'] * 100

    return country_bias_tab
# Computing country bias

country_bias_f = pd.DataFrame

country_bias_a = pd.DataFrame



country_bias_f = country_bias_calc(['t_c', 't_s'],[[0,1],[0,1]], 'Times Bias', country_bias_f)

country_bias_f = country_bias_calc(['t_c', 'c_s'],[[3,4],[0,1]], 'CWUR Bias', country_bias_f)

country_bias_f = country_bias_calc(['t_s', 'c_s'],[[3,4],[3,4]], 'Shanghai Bias', country_bias_f)



country_bias_a = country_bias_calc(['t_c', 't_s'],[[3,4],[3,4]], 'Times Bias', country_bias_a)

country_bias_a = country_bias_calc(['t_c', 'c_s'],[[0,1],[3,4]], 'CWUR Bias', country_bias_a)

country_bias_a = country_bias_calc(['t_s', 'c_s'],[[0,1],[0,1]], 'Shanghai Bias', country_bias_a)



# Uncomment below code to check for extreme bias



#country_bias_f = country_bias_calc(['t_c', 't_s'],[[0,0],[0,0]], 'Times Bias', country_bias_f)

#country_bias_f = country_bias_calc(['t_c', 'c_s'],[[4,4],[0,0]], 'CWUR Bias', country_bias_f)

#country_bias_f = country_bias_calc(['t_s', 'c_s'],[[4,4],[4,4]], 'Shanghai Bias', country_bias_f)



#country_bias_a = country_bias_calc(['t_c', 't_s'],[[4,4],[4,4]], 'Times Bias', country_bias_a)

#country_bias_a = country_bias_calc(['t_c', 'c_s'],[[0,0],[4,4]], 'CWUR Bias', country_bias_a)

#country_bias_a = country_bias_calc(['t_s', 'c_s'],[[0,0],[0,0]], 'Shanghai Bias', country_bias_a)
country_bias_a.head(2)
# Breaking the main tables into tables based on rankings to plot

t = 15  # Minimumum bias % for us to consider bias

u = 2  # Minimum universities in the ranking system to consider bias



bias_for_times = country_bias_f[(country_bias_f['Times Bias %'] >= t) & (country_bias_f['Total Universities'] > u)

        ].sort_values(by='Times Bias %', ascending = False)[['Total Universities', 'Times Bias', 'Times Bias %']]



bias_against_times = country_bias_a[(country_bias_a['Times Bias %'] >= t) & (country_bias_a['Total Universities'] > u)

        ].sort_values(by='Times Bias %', ascending = False)[['Total Universities', 'Times Bias', 'Times Bias %']]



bias_for_cwur = country_bias_f[(country_bias_f['CWUR Bias %'] >= t) & (country_bias_f['Total Universities'] > u)

        ].sort_values(by='CWUR Bias %', ascending = False)[['Total Universities', 'CWUR Bias', 'CWUR Bias %']]



bias_against_cwur = country_bias_a[(country_bias_a['CWUR Bias %'] >= t) & (country_bias_a['Total Universities'] > u)

        ].sort_values(by='CWUR Bias %', ascending = False)[['Total Universities', 'CWUR Bias', 'CWUR Bias %']]



bias_for_shanghai = country_bias_f[(country_bias_f['Shanghai Bias %'] >= t) & (country_bias_f['Total Universities'] > u)

        ].sort_values(by='Shanghai Bias %', ascending = False)[['Total Universities', 'Shanghai Bias', 'Shanghai Bias %']]



bias_against_shanghai = country_bias_a[(country_bias_a['Shanghai Bias %'] >= t) & (country_bias_a['Total Universities'] > u)

        ].sort_values(by='Shanghai Bias %', ascending = False)[['Total Universities', 'Shanghai Bias', 'Shanghai Bias %']]
# Function to create country bias bar plot

def bias_plot(b_for, b_against, b_name):

    

    def autolabel(rects, ht, m):

        cnt = 0

        for rect in rects:

            height = rect.get_height()

            if cnt < len(rects) and rect == rects1[cnt]:

                ht.append(height)

                cnt+=1

                #m.text(rect.get_x() + rect.get_width()/2., 

                #        height/2-0.5, '%d' % int(height), ha='center', va='bottom', fontsize=12)

            else:

                #m.text(rect.get_x() + rect.get_width()/2., 

                #         height/2-0.5, '%d' % int(height), ha='center', va='bottom', fontsize=12)

                

                if m==ax2 and cnt==0 and height/ht[cnt] > 0.85:

                    m.text(rect.get_x() + rect.get_width()/2., 

                        height-2, '%d' % (height/ht[cnt]*100)+'%', ha='center', va='bottom', fontsize=18)

                else:

                    m.text(rect.get_x() + rect.get_width()/2., 

                        height, '%d' % (height/ht[cnt]*100)+'%', ha='center', va='bottom', fontsize=18)

                cnt+=1

        return ht

    

    N = len(b_for)

    univ_total = np.array(b_for['Total Universities'])

    univ_bias_for = np.array(b_for[b_name + ' Bias'])

    ind = np.arange(N)

    width = 0.35

    

    fig, (ax1, ax2) = plt.subplots(2, figsize = (13,8))

    rects1 = ax1.bar(ind, univ_total, width, color='green')

    rects2 = ax1.bar(ind + width, univ_bias_for, width, color='lightgreen')

    ax1.set_ylabel('Count', fontsize=14)

    ax1.set_xticks(ind + width)

    ax1.set_xticklabels(b_for.index, fontsize=14)

    ax1.legend((rects1[0], rects2[0]), ('Total Universities', 

                                       'Universities biased for by ' + b_name), loc='upper left')

    ax1.spines['right'].set_color('none')

    ax1.spines['top'].set_color('none')

    ax1.yaxis.set_ticks_position('none')

    ax1.xaxis.set_ticks_position('none')



    ht = []

    ht = autolabel(rects1, ht, ax1)

    autolabel(rects2, ht, ax1)

    

    N = len(b_against)

    univ_total = np.array(b_against['Total Universities'])

    univ_bias_against = np.array(b_against[b_name + ' Bias'])

    ind = np.arange(N)

    

    rects1 = ax2.bar(ind, univ_total, width, color='firebrick')

    rects2 = ax2.bar(ind + width, univ_bias_against, width, color='salmon')

    ax2.set_ylabel('Count', fontsize=14)

    ax2.set_xticks(ind + width)

    ax2.set_xticklabels(b_against.index, fontsize=14)

    ax2.legend((rects1[0], rects2[0]), ('Total Universities',

                                       'Universities biased against by ' + b_name), loc='upper left')

    ax2.spines['right'].set_color('none')

    ax2.spines['top'].set_color('none')

    ax2.yaxis.set_ticks_position('none')

    ax2.xaxis.set_ticks_position('none')



    ht = []

    ht = autolabel(rects1, ht, ax2)

    autolabel(rects2, ht, ax2)

    

    plt.suptitle('Country-wise bias towards(green) and against(red) universities - ' + b_name, fontsize=20)

    plt.savefig('plot_'+b_name+'_bias.jpg')

    plt.show()
# Computing country bias for each ranking system pair

bias_plot(bias_for_times, bias_against_times, 'Times')

bias_plot(bias_for_cwur, bias_against_cwur, 'CWUR')

bias_plot(bias_for_shanghai, bias_against_shanghai, 'Shanghai')