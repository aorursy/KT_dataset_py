#imports



from ast import literal_eval

import pandas as pd

import networkx as nx

from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt

import community

from collections import defaultdict

#read house_members_116 from csv: members

members = pd.read_csv('/kaggle/input/house-of-representatives-congress-116/house_members_116.csv', index_col = 0)



#converts values of relevant columns to lists of strings rather than strings

members.committee_assignments = members.committee_assignments.apply(literal_eval)



members.head()
#read house_legislation_116 from csv: bills

bills = pd.read_csv('/kaggle/input/house-of-representatives-congress-116/house_legislation_116.csv', index_col = 0, parse_dates=["date_introduced"])



#converts values of relevant columns to lists of strings rather than strings

bills.cosponsors=bills.cosponsors.apply(literal_eval)

bills.subjects = bills.subjects.apply(literal_eval)

bills.committees = bills.committees.apply(literal_eval)

bills.related_bills = bills.related_bills.apply(literal_eval)



bills.head()
#areas: list of unique policy areas in bills

areas = bills.policy_area.unique()



#dic: dictionary with keys corresponding to unique policy areas with values initialized to 0

dic = {area: [0,0,0,0] for area in areas}





#iterate through bills and tally the bills by policy area

for index, row in bills.iterrows():

    #capture bills that were introduced by Democrats

    if members.loc[row.sponsor].current_party == 'Democratic':

        dic[row.policy_area][0] += 1

        #capture bills passed by Democrats

        if row.bill_progress == ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate'):

                dic[row.policy_area][1]+=1 

    #capture bills introduced by Republicans   

    elif members.loc[row.sponsor].current_party == 'Republican':

        dic[row.policy_area][2] += 1

        #capture bills passed by Republicans

        if row.bill_progress == ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate'):

                dic[row.policy_area][3]+=1 

    

#create dataframe from dictionary with rows representing policy_areas and columns representing bills introduced/passed by Democrats and Republicans

df = pd.DataFrame(dic.values(), index = dic.keys(), columns = ['Democratic', 'Democrats_Passed','Republican', 'Republicans_Passed']).sort_values('Democratic', ascending = False)



df.drop(np.nan, inplace = True)



#add row for 'Total' representing all bills regardless of policy area

df.loc['Total'] = [sum(df[x]) for x in df.columns]



#add column 'Total'representing the total bills introduced by policy area

df['Total'] = df.Democratic +df.Republican



#add column 'Total_Passed' represengint the total bills passed by policy area

df['Total_Passed'] = df.Democrats_Passed+df.Republicans_Passed



#add columns representing percentages of bills passed/introduced by Democrats, Republicans and overall 

df['D%_passed']=df.Democrats_Passed/df.Total_Passed

df['R%_passed']=df.Republicans_Passed/df.Total_Passed

df['D%_introduced'] = df['Democratic']/df['Total'] 

df['R%_introduced'] = df['Republican']/df['Total']

df['%_passed'] = 100*(df['Total_Passed']/df['Total'])



#fill nan values with 0

df.fillna(0, inplace=True)



#sort df by total introduced by policy area

df.sort_values(by=['Total'], ascending = False, inplace= True)



#limits the plots to the 25 most prolific polify areas as measured by number of bills introduced in the policy area

n=25

x= df.head(n).index

y1= df.head(n)['D%_introduced']

y2= df.head(n)['R%_introduced']

total = df.head(n)['Total']



#creates two subplots one for bills introduced and one for bills passed

fig, (ax1) = plt.subplots(nrows =1, ncols =1, figsize=(12, 4), squeeze = True)



#plot democrats/republican_introduced on ax1

ax1.bar(x, y1, label='Democrats_Introduced', alpha = .5)

ax1.bar(x, y2 ,bottom= y1,label= 'Republicans_Introduced', alpha = .5)





# add text annotation corresponding to the percentages introduced in each policy by Republicans and Democrats.

for xpos, ypos, yval in zip(x, y1/2, y1):

    if yval>0:

        ax1.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation=90)

for xpos, ypos, yval in zip(x, y1+y2/2, y2):

    if yval>0:

        ax1.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation= 90)

# add text annotation corresponding to the number of total bills introduced in each policy area

for xpos, ypos, total in zip(x, y1+y2, total):

    ax1.text(xpos, ypos +.05, total, ha="center", va="bottom", rotation= 90)



#rotate x_tick labels 90 degrees for ax1

for tick in ax1.get_xticklabels():

    tick.set_rotation(90)    



#set title and legend for ax1

ax1.set_title('Bills Introduced by Party and Policy Area (%)')

ax1.legend(loc = 'upper right')



#set y_lim for ax1 and hide the y-axis of ax1

ax1.set_ylim(0,1.5)

ax1.yaxis.set_visible(False)



#create a second y-axis on the first subplot ax3

ax3 = ax1.twinx()

#plot a line graph for total introduced by policy area using ax3

ax3.plot(x, df.head(n).Total, label = 'Total_introduced')

ax3.set_ylim(0,641)

ax3.legend(loc = 'upper left')

ax3.yaxis.set_visible(True)

ax3.set_ylabel('Bills Introduced')



#show plot

plt.show()
#creates two subplots one for bills introduced and one for bills passed

fig, (ax2) = plt.subplots(nrows =1, ncols =1, figsize=(12, 4), squeeze = True)



#resort df by total passed by policy area

df.sort_values(by=['Total_Passed'], ascending = False, inplace= True)



#limits plot to the 25 most prolific policy areas as measured by the total number of bills passed

x= df.head(n).index

y1= df.head(n)['D%_passed']

y2= df.head(n)['R%_passed']

y3= df.head(n)['Democratic']+ df.head(n)['Republican']

total = df.head(n)['Total_Passed']



#plot bars representing the number of bills passed by Democrats/Republicans by policy area

ax2.bar(x, y1, label='Democrats_Passed', alpha=.5)

ax2.bar(x, y2 ,bottom= y1,label= 'Republicans_Passed', alpha=.5)



#create a second axis on the second subplot for a line plot representing the percentage of bills pased by policy area

ax4 = ax2.twinx()

ax4.plot(x, df.head(n)['%_passed'], label = 'Percent_Passed')

ax4.set_ylim(0,30)

ax4.legend(loc = 'upper left')

ax4.yaxis.set_visible(True)

ax4.set_ylabel("Bills Passed (%)")



# add text annotation corresponding to the percentage passed by Democrats and Republicans.

for xpos, ypos, yval in zip(x, y1/2, y1):

    if yval>0:

        ax2.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation=90)

for xpos, ypos, yval in zip(x, y1+y2/2, y2):

    if yval>0:

        ax2.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation= 90)

# add text annotation corresponding to the total number of bills passed in each policy area

for xpos, ypos, total in zip(x, y1+y2, total):

    ax2.text(xpos, ypos +.05, total, ha="center", va="bottom", rotation= 90)

    

#rotate xlabels 90 degrees on second subplot

for tick in ax2.get_xticklabels():

    tick.set_rotation(90) 

#set title legend for ax2

ax2.set_title('Bills Passed by Party and Policy Area (%)')

ax2.legend(loc = 'upper right')



#set y limit and y_axis visibiity for ax2

ax2.set_ylim(0,1.5)

ax2.yaxis.set_visible(False)





#show plot

plt.show()
#committees: list of unique committees in the House of Representatives

committees =[]

Dems=[]

Reps=[]

D_passed = []

R_passed = []



#iterate through bills and append unique House committees to list: committees

for bill, row in bills.iterrows():

    for committee in row.committees:

        if (committee not in committees) and ('House' in committee):

            committees.append(committee)

    



#dic: dictionary with keys corresponding to unique committees with values initialized to 0

dic = {committee: [0,0,0,0] for committee in committees}





#iterate through bills and based on committees

for index, row in bills.iterrows():

    #iterate through committees associated with bill

    for committee in row.committees:

        #only consider House committees, exclude Senate committees

        if ('House' in committee): 

            if members.loc[row.sponsor].current_party == 'Democratic':

                #capture bills referrered to committee that are sponsored by Democrats

                dic[committee][0] += 1

                #capture bill name

                if index not in Dems:

                    Dems.append(index)

                #capture bills referred to committee sponsored by Democrats that have passed the House

                if row.bill_progress == ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate'):

                    dic[committee][1]+=1

                    #capture bill name

                    if index not in D_passed:

                        D_passed.append(index)

                    

                    

            elif members.loc[row.sponsor].current_party == 'Republican':

                #capture bills referred to committee that were sponsored by Republicans

                dic[committee][2] += 1

                #capture bill name

                if index not in Reps:

                    Reps.append(index)

                #capture bills referred to committee that were sponsored by Republicans which passed the House

                if row.bill_progress == ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate'):

                    dic[committee][3]+=1

                    if index not in R_passed:

                        R_passed.append(index)

    

        

#create Dataframe: df from dic with rows representing House committees and columns representing bills introduced/passed by Democrats or Republicans                 

df = pd.DataFrame(dic.values(), index = dic.keys(), columns = ['Democratic', 'Democrats_Passed','Republican', 'Republicans_Passed']).sort_values('Democratic', ascending = False)



#add row representing all bills regardless of committee to which they were referred

df.loc['Total'] = [len(Dems), len(D_passed), len(Reps), len(R_passed)]



# add column 'Total' representing all bills introduced regardless of the party of the bill sponsor

df['Total'] = df.Democratic +df.Republican



# add column 'Total_Passed' representing all bills passed regardless of the party of the bill sponsor

df['Total_Passed'] = df.Democrats_Passed+df.Republicans_Passed



#add percentage columns

df['D%_passed']=df.Democrats_Passed/df.Total_Passed

df['R%_passed']=df.Republicans_Passed/df.Total_Passed

df['D%_introduced'] = df['Democratic']/df['Total'] 

df['R%_introduced'] = df['Republican']/df['Total']

df['%_passed'] = 100*(df['Total_Passed']/df['Total'])



#fill nan values with 0

df.fillna(0, inplace=True)



#sort df by 'Total' column

df.sort_values(by=['Total'], ascending = False, inplace= True)





n=25

x= df.head(n).index



y1= df.head(n)['D%_introduced']

y2= df.head(n)['R%_introduced']



total = df.head(n)['Total']







fig, (ax1) = plt.subplots(nrows =1, ncols =1, figsize=(12, 4), squeeze = True)





ax1.bar(x, y1, label='Democrats_Introduced', alpha = .5)

ax1.bar(x, y2 ,bottom= y1,label= 'Republicans_Introduced', alpha = .5)





# add text annotation corresponding to the values of each bar.

for xpos, ypos, yval in zip(x, y1/2, y1):

    if yval>0:

        ax1.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation=90)

for xpos, ypos, yval in zip(x, y1+y2/2, y2):

    if yval>0:

        ax1.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation= 90)

# add text annotation corresponding to the "total" value of each bar

for xpos, ypos, total in zip(x, y1+y2, total):

    ax1.text(xpos, ypos +.05, total, ha="center", va="bottom", rotation= 90)



#rotate tick labels to 90 degrees

for tick in ax1.get_xticklabels():

    tick.set_rotation(90)    



#set title and legend for first subplot

ax1.set_title('Bills Introduced by Party and Committee (%)')

ax1.legend(loc = 'upper right')



#set y_lim and y_axis visibility

ax1.set_ylim(0,1.5)

ax1.yaxis.set_visible(False)



#add second y_axis to first subplot

ax3 = ax1.twinx()



#plot line graph representing total bills referred to each committee

ax3.plot(x, df.head(n).Total, label = 'Total_introduced')

ax3.set_ylim(0,641)

ax3.legend(loc = 'upper left')

ax3.yaxis.set_visible(True)

ax3.set_ylabel('Bills Referred to Committee')





#show plot

plt.show()
fig, (ax2) = plt.subplots(nrows =1, ncols =1, figsize=(12, 4), squeeze = True)





#sort df by total bills passed by committee

df.sort_values(by=['Total_Passed'], ascending = False, inplace= True)



#limits plot to the top 25 committees

x= df.head(n).index



#calculate passage percentages

y1= df.head(n)['D%_passed']

y2= df.head(n)['R%_passed']

y3= df.head(n)['Democratic']+ df.head(n)['Republican']

total = df.head(n)['Total_Passed']



#plot bar graph for second subplot for bills bassed by committee

ax2.bar(x, y1, label='Democrats_Passed', alpha=.5)

ax2.bar(x, y2 ,bottom= y1,label= 'Republicans_Passed', alpha=.5)



#create second second y_axis for the second subplot for line plot

ax4 = ax2.twinx()



#plot line plot for number of bills passed by committee

ax4.plot(x, df.head(n)['%_passed'], label = 'Percent_Passed')



#set y_limit, legebd location, visibility and label for the ax4

ax4.set_ylim(0,30)

ax4.legend(loc = 'upper left')

ax4.yaxis.set_visible(True)

ax4.set_ylabel('Bills Passed %')



# add text annotation corresponding to the percentages of bills passed that were referred to each committee by Democrats or Republicans

for xpos, ypos, yval in zip(x, y1/2, y1):

    if yval>0:

        ax2.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation=90)

for xpos, ypos, yval in zip(x, y1+y2/2, y2):

    if yval>0:

        ax2.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation= 90)

        

# add text annotation corresponding to the number of bills passed that were also referred to each of the different committees

for xpos, ypos, total in zip(x, y1+y2, total):

    ax2.text(xpos, ypos +.05, total, ha="center", va="bottom", rotation= 90)

    

#rotate x_tick labels by 90 degrees

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)

    

#set title and legend location for the second subplot

ax2.set_title('Bills Passed by Party and Committee (%)')

ax2.legend(loc = 'upper right')



#set y limit and y_axis visibiity for ax2

ax2.set_ylim(0,1.5)

ax2.yaxis.set_visible(False)



plt.show()
#create subplots, in this case just 1

fig, (ax5) = plt.subplots(nrows =1, ncols =1, figsize=(12, 4), squeeze = True)



#committees: list of unique committees in the House of Representatives

committees =[]



#iterate through members dataframe to capture all unique committees

for member, row in members.iterrows():

    #iterate through each committee assignment for each member

    for committee in row.committee_assignments:

        #capture committee if not already within list committees

        if (committee not in committees):

            committees.append(committee)

            

#create dictionary having keys corresponding to values in list committees

dic = {committee: [0,0,0] for committee in committees}





#iterate through dataframe members and tally the number of Democrats and Republicans in each committee

for index, row in members.iterrows():

    #iterate through committee assignments for each member

    for committee in row.committee_assignments:

        #for each committee in dic add 1 to the 1st value of the list if member is a Democrat

        if row.current_party == 'Democratic':

            dic[committee][0] += 1

        # else add 1 to the 2nd value of the list if member is a Democrat

        elif row.current_party == 'Republican':

            dic[committee][1] += 1

        # otherwise add 1 to the 3rd value of the list to capture independents

        else:

            dic[committee][2] += 1





#count: series storing the total number of Democrats, Republicans and Independents in the House

count = members.groupby('current_party').count().name     

    

# df: dataframe storing the number of Democrats, Republicans and Independents in each committee

df = pd.DataFrame(dic.values(), index = dic.keys(), columns = ['Democratic', 'Republican', 'Independent']).sort_values(by = 'Democratic', ascending = False).head(23)



#add row to df for total number of Democrats, Republicans and Independents in the House

df.loc['Total'] = [count.loc['Democratic'],  count.loc['Republican'], count.loc['Independent']]



#add column to df representng the total number of members of each committee

df['Total'] = df['Democratic']+df['Republican']+df['Independent']



#add columns representing fraction of each of the parties by committee

df['D%'] = df['Democratic']/df['Total']

df['R%'] = df['Republican']/df['Total']

df['I%'] = df['Independent']/df['Total']



#sort df in descending order by the percentage of Democrats within each committee

df.sort_values(by='D%', ascending = False, inplace= True)



#define variables for plotting and annotations

n=25

x= df.head(n).index

y1= df.head(n)['D%']

y2= df.head(n)['R%']

y3= df.head(n)['I%']

total = df.head(n)['Total']





#Plot bars for Democrats, Republicans and Independents with shared axis ax5

ax5.bar(x, y1, label='Democrats', alpha=.5)

ax5.bar(x, y2 ,bottom= y1,label= 'Republican', alpha=.5)

ax5.bar(x, y3 ,bottom= y1+y2,label= 'Independent', alpha=.5)



# add text annotation corresponding to the values of each bar.

for xpos, ypos, yval in zip(x, y1/2, y1):

    ax5.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation=90)

for xpos, ypos, yval in zip(x, y1+y2/2, y2):

    ax5.text(xpos, ypos, str(round(yval*100,0))+ "%", ha="center", va="center", rotation= 90)



# add text annotation corresponding to the "total" value of each bar

for xpos, ypos, total in zip(x, y1+y2, total):

    ax5.text(xpos, ypos, total, ha="center", va="bottom",)



#set y_limit

ax5.set_ylim(0,1.5)



#rotate x_tick labels by 90 degrees

for tick in ax5.get_xticklabels():

    tick.set_rotation(90)   



#set title, legend location and y_axis visibility

ax5.set_title('Committee Membership by Party (%)')

ax5.legend(loc = 'upper right')

ax5.yaxis.set_visible(False)



#show plot

plt.show()
# dic: with keys corresponding to members_ids from members dataframe and list for storing count of bills introduced and passed by a member

dic = {k:[0,0] for k in members.index}



#iterrate through bills and add count number of bills introduced/passed by each member

for index, row in bills.iterrows():

    dic[row.sponsor][1]+=1

    if row.bill_progress == ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate'):

        dic[row.sponsor][0]+=1



#num_passed: dataframe created from dic with index correpodning to member_id and columns representing the number of bills introduced/passed by eac member

num_passed = pd.DataFrame.from_dict(dic, orient = 'index',columns = ['num_passed', 'total_introduced'])



#create column for percentage of bills passed by each member

num_passed['percent_passed'] =( num_passed['num_passed']/num_passed['total_introduced'])*100



#sort num_passed

num_passed.sort_values(by = ['num_passed','percent_passed'], ascending = False, inplace = True)



#merge num_passed with members dataframe

num_passed.merge(members[['name', 'current_party', 'committee_assignments']], how = 'inner', left_index= True, right_index =True).head(10)
#set policy_area to focus on for second part of analysis

policy_area = 'Crime and Law Enforcement'





# MD: graph with directed edges from bill cosponsors to a bill sponsor

MD = nx.MultiDiGraph()

MD.add_nodes_from(members.index)

for index, row in bills.iterrows():

    sponsor = [row.sponsor for i in range(len(row.cosponsors))]

    zipped = zip(row.cosponsors, sponsor)

    zipped = list(zipped)

    

    #set edge attribute related to bill policy_area

    MD.add_edges_from(zipped, bill = index, policy_area = row.policy_area, bill_progress = row.bill_progress)



#drop nodes representing non-voting members, former members and the Speaker of the House

to_drop =['G000582', 'R000600', 'N000147', 'P000610', 'S001177', 'S001204', 'P000197', 'J000255']

MD.remove_nodes_from(to_drop)



# set node attrbute 'party'

nx.set_node_attributes(MD,  members.current_party.to_dict(), 'party')





#convert MD from a multi-directed graph to graph with weighted edges, with weights representing the number of edges between two nodes

G = nx.Graph()



for n, nbrs in MD.adjacency():

    for nbr, edict in nbrs.items():

        if (G.has_edge(n,nbr)) :

            G[n][nbr]['weight'] +=len(edict)

        else:

             G.add_edge(n, nbr, weight=len(edict))



#MDs: subgraph limited to edges filtered by a policy area == 'Crime and Law Enforcement'

MDs = nx.MultiDiGraph()



for u,v,d in MD.edges(data = True):

    if d['policy_area'] ==  policy_area:

        MDs.add_edge(u,v,bill = d['bill'])





#convert MDs from a multi-directed graph to graph with weighted edges, with weights representing the number of edges between the nodes

Gs = nx.Graph()



for n, nbrs in MDs.adjacency():

    for nbr, edict in nbrs.items():

        if (Gs.has_edge(n,nbr)) :

            Gs[n][nbr]['weight'] +=len(edict)

        else:

             Gs.add_edge(n, nbr, weight=len(edict))

                

# function for setting colors of nodes and edges

def get_paired_color_palette(size):

    palette = []

    for i in range(size*2):

        palette.append(plt.cm.Paired(i))

    return palette





#use louvain community detection algorithm to detect communities in G

communities =[]

louvain = community.best_partition(G, weight = 'weight', random_state=42)

for i in set(louvain.values()):

    nodelist = [n for n in G.nodes if (louvain[n]==i)]

    communities.append(nodelist)



#make plot using matplotlib, networkx spring_layout, set_colors using cluster_count and get_paired_color_pallette

clusters_count = len(set(louvain.values()))

plt.figure(figsize=(10, 10))

light_colors = get_paired_color_palette(clusters_count)[0::2]

dark_colors = get_paired_color_palette(clusters_count)[1::2]

g = nx.drawing.layout.spring_layout(G, weight = 'weight', seed = 42, threshold = .0000000001)



#iterate through each of the communities found by the Louvain algorithm and plot

for i in set(louvain.values()):

    nodelist = [n for n in G.nodes if (louvain[n]==i)]

    edgelist = [e for e in G.edges if ((louvain[e[0]]==i) or (louvain[e[1]]==i))]

    node_color = [light_colors[i] for _ in range(len(nodelist))]

    edge_color = [dark_colors[i] for _ in range(len(edgelist))]

    nx.draw_networkx_nodes(G, g, nodelist=nodelist, node_color=node_color, edgecolors='k', label = i)                                                                                                           

    nx.draw_networkx_edges(G, g, edgelist=edgelist, alpha=.2, edge_color=edge_color)



#set title, legend and show plot

plt.title('Louvain clustering: House of Representatives', fontdict={'fontsize': 25})

plt.legend()

plt.axis('off')

plt.show()
# instantiate dictionaries

community_members = defaultdict()

community_bills = defaultdict()



# iterate through each community found by Louvain algorithm

for i in range(3):

    dic ={}

    index = []

    #set community_of_interest

    community_of_interest = i



    #create subgraph with nodes limited to the community_of_interest

    subgraph = MD.subgraph(communities[community_of_interest])



    #create dataframe for each community with index corresponding to member_id and a column representing the in_degree centrality of each member, merge with members dataframe

    community_members[i] = pd.DataFrame.from_dict(nx.algorithms.centrality.in_degree_centrality(subgraph), orient = 'index', columns = ['centrality']).merge(members[['name','current_party', 'committee_assignments']], how = 'left', left_index = True, right_index = True).sort_values(by= 'centrality',ascending = False)

    

    

    # community_bills[i] is a subset of bills dataframe having a sponsor within the given community i

    community_bills[i] = bills.loc[[row.sponsor in communities[i] for index, row in bills.iterrows()]]

    

    # iterate through community_bills[i]

    for bill, row in community_bills[i].iterrows():

        #append bill to list index

        index.append(bill)

        #create dic key related to each bill and set initial value to 0

        dic[bill] = 0

        #iterate through cosponsors of each bill

        for cosponsor in row.cosponsors:

            #count number of bill cosponsors which are within the given community

            if cosponsor in community_members[i].index:

                dic[bill] += 1

        

    



    #create data frame from dic with bill as index and column representing the number of in_community cosponsor for each bill

    tally = pd.DataFrame.from_dict(dic, orient = 'index',columns = ['in_community_cosponsors'])

    

    #merge tally with community_bills[i]

    community_bills[i] = tally.merge(community_bills[i][['title', 'bill_progress']],how = 'outer', left_index =True, right_index= True)



#instantiate lists

l = []

index = []



# iterate through each community found by Louvain algorithm

for i in range(3):

    # community_i is a subset of members limited to members of the given community

    community_i = members.loc[communities[i]]

    #democrats: stores the number of democrats within the given community

    democrats = len(community_i.loc[community_i.current_party == 'Democratic'])

    #republicans: stores the number of republicans in the given community

    republicans = len(community_i.loc[community_i.current_party == 'Republican'])

    #indepenents: stores the number of independents in teh given community

    independents = len(community_i.loc[community_i.current_party == 'Independent'])

    #total_members: is the total number of members in the given community

    total_members = len(community_i)

    #total_bills: the total number of bills introduced by members of the given community

    total_bills = len(community_bills[i])

    #bills_passed: is the total number of bills which passed the house which were introduced by members of the given community

    bills_passed = len(community_bills[i].loc[community_bills[i].bill_progress ==  ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate')])

    #append list containing all values to list l

    l.append([democrats, republicans, independents, total_members, total_bills, bills_passed])

    #append index string to list index

    index.append('community_' +str(i))

    

    

#create dataframe df from lists l and index

df = pd.DataFrame(l, columns = ['democrats', 'republicans', 'independents', 'total_members', 'total_bills', 'bills_passed'],index = index)  

#caculate passage percentage for each community

df['percent_passed'] = (df['bills_passed']/df['total_bills'])*100

df.head()



#democrats: subset of community_0 which are Democrats

democrats = members.loc[communities[0]]

democrats = democrats.loc[democrats.current_party == 'Democratic']

democrats
#dataframe lists subsets of communities 1 and 2 which are Republican 

com_1 = members.loc[communities[1]]

com_2 = members.loc[communities[2]]



pd.concat([com_1.loc[com_1.current_party=='Republican'], com_2.loc[com_2.current_party=='Republican']], keys = [1,2])

#dataframe lists the ten members of each community with the largest in_degree centralities

df = pd.concat([community_members[0], community_members[1], community_members[2]], keys =

         [0,1,2])



df.index.rename(['community','member'], inplace = True)



df= df.merge(num_passed, how = 'inner', left_on = 'member', right_index=True).sort_values(['community','centrality'], ascending = False)



pd.concat([df.loc[2].head(10), df.loc[1].head(10), df.loc[0].head(10)], keys =

         [2,1,0])



#dataframe list the top ten members of each community with the highest number of bills passed

df = pd.concat([community_members[0], community_members[1], community_members[2]], keys =

         [0,1,2])



df.index.rename(['community','member'], inplace = True)



df= df.merge(num_passed, how = 'inner', left_on = 'member', right_index=True).sort_values(['community','num_passed','percent_passed', 'total_introduced'], ascending = False)



pd.concat([df.loc[2].head(5), df.loc[1].head(5), df.loc[0].head(5)], keys =

         [2,1,0])



#iterate through each community

for i in range(3):

    #sort bills introduced by each community by the number of in_community cosponsors

    community_bills[i].sort_values(by='in_community_cosponsors', ascending = False, inplace = True)

#concatenate the ten most supported bills of each community into one dataframe    

pd.concat([community_bills[0].head(10), community_bills[1].head(10), community_bills[2].head(10)], keys =

         ['community_0', 'community_1', 'community_2'])
# function for setting colors of nodes and edges

def get_paired_color_palette(size):

    palette = []

    for i in range(size*2):

        palette.append(plt.cm.Paired(i))

    return palette





#use louvain community detection algorithm to detect communities in Gs, network of bills limited to Crime and Law Enforcement

communities =[]

louvain = community.best_partition(Gs, weight = 'weight', random_state=42)

for i in set(louvain.values()):

    nodelist = [n for n in Gs.nodes if (louvain[n]==i)]

    communities.append(nodelist)



#make plot using matplotlib, networkx spring_layout, set_colors using cluster_count and get_paired_color_pallette

clusters_count = len(set(louvain.values()))

plt.figure(figsize=(10, 10))

light_colors = get_paired_color_palette(clusters_count)[0::2]

dark_colors = get_paired_color_palette(clusters_count)[1::2]

g = nx.drawing.layout.spring_layout(Gs, weight = 'weight', seed = 42, threshold = .0000000001)



#iterate through each of the communities found by the Louvain algorithm and plot

for i in set(louvain.values()):

    nodelist = [n for n in Gs.nodes if (louvain[n]==i)]

    edgelist = [e for e in Gs.edges if ((louvain[e[0]]==i) or (louvain[e[1]]==i))]

    node_color = [light_colors[i] for _ in range(len(nodelist))]

    edge_color = [dark_colors[i] for _ in range(len(edgelist))]

    nx.draw_networkx_nodes(Gs, g, nodelist=nodelist, node_color=node_color, edgecolors='k', label = i)                                                                                                           

    nx.draw_networkx_edges(Gs, g, edgelist=edgelist, alpha=.5, edge_color=edge_color)



#set title, legend and show plot

plt.title('Policy Area: Crime and Law Enforcement', fontdict={'fontsize': 25})

plt.legend()

plt.axis('off')

plt.show()
#instantiate dictionaries

community_members = defaultdict()

community_bills = defaultdict()



#iterate through communities found by Louvain algorithm

for i in range(3):

    dic ={}

    index = []

    #set community_of_interest

    community_of_interest = i



    #create subgraph with nodes limited to the community_of_interest

    subgraph = MDs.subgraph(communities[community_of_interest])



    #sort members of community_of_interest by in_degree centality

    community_members[i] = pd.DataFrame.from_dict(nx.algorithms.centrality.in_degree_centrality(subgraph), orient = 'index', columns = ['centrality']).merge(members[['name','current_party', 'committee_assignments', 'state']], how = 'left', left_index = True, right_index = True).sort_values(by= 'centrality',ascending = False)

    

    

    # Tally bills introduced by members of each community

    community_bills[i] = bills.loc[[(row.sponsor in communities[i] and row.policy_area == 'Crime and Law Enforcement') for index, row in bills.iterrows() ]]

    

    # iterate through community_bills[i]

    for bill, row in community_bills[i].iterrows():

        #append bill to list index

        index.append(bill)

        #create dic key related to each bill and set initial value to 0

        dic[bill] = 0

        #iterate through cosponsors of each bill

        for cosponsor in row.cosponsors:

            #count number of bill cosponsors which are within the given community

            if cosponsor in community_members[i].index:

                dic[bill] += 1

        

    



    #create data frame from dic with bill as index and column representing the number of in_community cosponsor for each bill

    tally = pd.DataFrame.from_dict(dic, orient = 'index',columns = ['in_community_cosponsors'])

    

    #merge tally with community_bills[i]

    community_bills[i] = tally.merge(community_bills[i][['title', 'bill_progress']],how = 'outer', left_index =True, right_index= True)



    

#instantiate lists

l = []

index = []



# iterate through each community found by Louvain algorithm

for i in range(3):

    

    # community_i is a subset of members limited to members of the given community

    community_i = members.loc[communities[i]]

    #democrats: stores the number of democrats within the given community

    democrats = len(community_i.loc[community_i.current_party == 'Democratic'])

    #republicans: stores the number of republicans in the given community

    republicans = len(community_i.loc[community_i.current_party == 'Republican'])

    #indepenents: stores the number of independents in teh given community

    independents = len(community_i.loc[community_i.current_party == 'Independent'])

    #total_members: is the total number of members in the given community

    total_members = len(community_i)

    #total_bills: the total number of bills introduced by members of the given community

    total_bills = len(community_bills[i])

    #bills_passed: is the total number of bills which passed the house which were introduced by members of the given community

    bills_passed = len(community_bills[i].loc[community_bills[i].bill_progress ==  ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate')])

    #append list containing all values to list l 

    l.append([democrats, republicans, independents, total_members, total_bills, bills_passed])

    #append index string to list index

    index.append('community_' +str(i))

    

    

#create dataframe df from lists l and index

df = pd.DataFrame(l, columns = ['democrats', 'republicans', 'independents', 'total_members', 'total_bills', 'bills_passed'],index = index)  

#caculate passage percentage for each community

df['percent_passed'] = (df['bills_passed']/df['total_bills'])*100

df.head()



#democrats: subset of community_0 which are Democrats

democrats = members.loc[communities[0]]

democrats = democrats.loc[democrats.current_party == 'Democratic']

democrats
#subsets of members which are limited to the members of communities 1 and 2

com_1 = members.loc[communities[1]]

com_2 = members.loc[communities[2]]



#concatenate Republican members of communities 1 and 2 into a single datafame

pd.concat([com_1.loc[com_1.current_party=='Republican'], com_2.loc[com_2.current_party=='Republican']], keys = [1,2])

#concatenate members of communites 1,2 and 3 into a single dataframe

df = pd.concat([community_members[0], community_members[1], community_members[2]], keys =

         [0,1,2])

#rename index

df.index.rename(['community','member'], inplace = True)

#sort values in each community by in_degree centrality

df.sort_values(['community','centrality'], ascending = False, inplace = True)

#limit each community to the only the ten most central members

pd.concat([df.loc[0].head(10), df.loc[1].head(10), df.loc[2].head(10)], keys =

         [0,1,2])

# dic: dictionary with keys corresponding to members_ids from members dataframe and list for storing count of bills introduced and passed by a member

dic = {k:[0,0] for k in members.index}



#iterate through bills and count number of bills introduced/passed by each member in the policy area of Crime and Law Enforcement

for index, row in bills.iterrows():

    if row.policy_area == 'Crime and Law Enforcement':

        dic[row.sponsor][1]+=1

        if row.bill_progress == ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate'):

            dic[row.sponsor][0]+=1

#num_passed: dataframe created from dic with index correpodning to member_id and columns representing the number of bills introduced/passed by each member

num_passed = pd.DataFrame.from_dict(dic, orient = 'index',columns = ['num_passed', 'total_introduced'])

#add column representing bill passage rates for each member

num_passed['percent_passed'] =( num_passed['num_passed']/num_passed['total_introduced'])*100

#sort num_passed by columns num_passed and percent_passed

num_passed.sort_values(by = ['num_passed','percent_passed'], ascending = False, inplace = True)



#df: concatenate members of communities 0,1 and 3 in a single dataframe

df = pd.concat([community_members[0], community_members[1], community_members[2]], keys =

         [0,1,2])



#rename index

df.index.rename(['community','member'], inplace = True)

#merge df with num_passed and sort values

df= df.merge(num_passed, how = 'inner', left_on = 'member', right_index=True).sort_values(['num_passed','percent_passed', 'total_introduced'], ascending =False)



#limit each community to top members as previously sorted

pd.concat([df.loc[1].head(6), df.loc[2].head(5), df.loc[0].head(5)], keys =

         [1,2,0])
#iterate through each community

for i in range(3):

    #sort bills of each community by in_community_cosponsors 

    community_bills[i].sort_values(by='in_community_cosponsors', ascending = False, inplace = True)

#concatenate most supported bills from each community to into a single dataframe   

pd.concat([community_bills[0].head(5), community_bills[1].head(5), community_bills[2].head(5)], keys =

         ['community_0', 'community_1', 'community_2'])
passed = ('Passed House' or 'Passed Senate' or 'Became Law' or 'To President' or 'Agreed to in House' or 'Agreed to in Senate')



#concatenate passed bills from each community into a single dataframe    

pd.concat([community_bills[i].loc[community_bills[i].bill_progress == passed] for i in range(3)], keys =

         ['community_0', 'community_1', 'community_2'])