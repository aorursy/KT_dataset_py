import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams
import seaborn as sb
import numpy as np
%matplotlib inline
rcParams['figure.figsize'] = 16,16
sb.set_style('whitegrid')
path_for_input = '../input/input-file/test_data3.csv'
inputfile = pd.read_csv(path_for_input,header=0,sep=',')
inputfile.head(15)
fundpositions_frame = inputfile.groupby(['parent_fund'])['weight'].sum().reset_index()
fundpositions_frame = fundpositions_frame.rename(columns={'weight': 'total_fund_part'})
fundpositions_frame.head(3)
start_frame = pd.merge(inputfile,fundpositions_frame, how='left',left_on = inputfile.parent_fund
                      , right_on=fundpositions_frame.parent_fund, suffixes= ("","_y"))
start_frame = start_frame.drop(['parent_fund_y','key_0'],axis=1)
start_frame
path_for_approved = '../input/cost-set/cost3.csv'
approved_file= pd.read_csv(path_for_approved,header=0,sep=',')
approved_file.head(5)
final_frame = pd.DataFrame()
final_frame = pd.merge(start_frame, approved_file, left_on = start_frame.parent_fund, right_on = approved_file.fund, how='outer')
# leaves are missing total fund part - therefore backfilling them with 0s
final_frame.total_fund_part = final_frame.total_fund_part.fillna(0)
G = nx.DiGraph()
for index, row in final_frame.iterrows():
    if pd.isnull(row['child_fund']) == False:
        G.add_edge(row['fund'],row['child_fund'],weight=row['weight'])
    G.node[row['fund']]['total_fund_part'] = row['total_fund_part']
    G.node[row['fund']]['direct_cost'] = row['cost']
    if row['approved'] == 'N':
        G.node[row['fund']]['direct_approved'] = 0
    else:
        G.node[row['fund']]['direct_approved'] = 1
graph_information = []
nodelist = []
portfolio = 'A'
nodelist.append(portfolio)
for n in G.nodes():
    if nx.has_path(G, portfolio , n) == True and portfolio != n:
        length = nx.shortest_path_length(G, portfolio, n)
        short_path = nx.shortest_path(G,portfolio,n)
        graph_information.append([portfolio,n,length,short_path[1:]])
        nodelist.append(n) # this is the new scope

a = pd.DataFrame(graph_information, columns=['portfolio','child','num_of_steps','short_path'])
a.head(10)
T = G.subgraph(nodelist)            
posT = nx.spring_layout(T)
rcParams['figure.figsize'] = 14,14
nx.draw_networkx_nodes(T,posT,T.nodes(),node_size=300,node_color='b',alpha=0.1)
nx.draw_networkx_edges(T,posT,T.edges(),width=0.8,edge_color='g')
labels = nx.get_node_attributes(T,'direct_approved')
labels2 = nx.get_edge_attributes(T,'weight')
nx.draw_networkx_nodes(T,pos=posT,font_size=14,edge_labels= labels,node_size=300,node_color='b',alpha=0.1)
nx.draw_networkx_labels(T,pos=posT,font_size=14)
nx.draw_networkx_edge_labels(T,pos=posT,font_size=14,edge_labels=labels2)
plt.title("Direct and Indirect investments of A",size=14)
plt.grid(False)
plt.axis('off')
plt.show()
def create_edgepairs(listin):
    n = 0
    newlist = []
    while n < len(listin)-1:
        newlist.append([listin[n],listin[n+1]])
        n = n+1
    return newlist


weight_dictionary = nx.get_edge_attributes(G, 'weight')
total_fund_part_dictionary = nx.get_node_attributes(G,'total_fund_part')

#Lookup how much the of its total assets the chosen portfolio is investing into funds
if portfolio in total_fund_part_dictionary:
            p_fund_part = total_fund_part_dictionary[portfolio]
else:
            p_fund_part = 'Not Applicable'

node_information = []
for n in G.nodes():
    if nx.has_path(G,portfolio,n ) == True and n != portfolio:
        length = nx.shortest_path_length(G, portfolio,n)
        short_path = nx.shortest_path(G,portfolio,n)
        #collect all possible paths
        pathes = list(nx.all_simple_paths(G,portfolio,n))
        #loop through all possible pathes
        for everypath in pathes:
            edgelist = create_edgepairs(everypath)
        edgelist =  create_edgepairs(short_path)
        for i,edgepairs in enumerate(edgelist):
            if i ==0:
                port_impact = weight_dictionary[edgepairs[0],edgepairs[1]]
            else:
                port_impact = port_impact * weight_dictionary[edgepairs[0],edgepairs[1]]
               
        if len(list(G.successors(n))) == 0:
            leaf = 'Leaf - bottom portfolio'
        else:
            leaf = 'Parent portfolio'
        #check if value is in dictionary
        
        node_information.append([portfolio, n ,G.node[n]['direct_cost'] ,
                                 p_fund_part, port_impact,length, leaf, short_path[1:]])

impact_frame = pd.DataFrame(node_information,columns=['portfolio','child','child_cost','total_fund_part',
                                              'port_impact','steps','position','short_path'])
impact_frame.head(16)

cost_portions = impact_frame.port_impact * impact_frame.child_cost
rcParams['figure.figsize'] = 10,8
colors = ['gold','yellowgreen','lightcoral','lightskyblue','lavender','lightpink',
          'aliceblue','palegoldenrod','lavenderblush','navajowhite','lightsalmon','lightblue']
explode = [0,0,0.1,0,0,0,0,0,0,0,0,0]
plt.pie(cost_portions,labels= impact_frame.child,colors=colors,
        shadow=True, 
        startangle = 291,explode = explode)
plt.title("Impact by portfolio",size=15)
plt.axis('equal')
plt.legend(round(impact_frame.port_impact * impact_frame.child_cost,2))
plt.show()
node_information = []
weight_dictionary = nx.get_edge_attributes(G, 'weight')
total_fund_part_dictionary = nx.get_node_attributes(G,'total_fund_part')

if portfolio in total_fund_part_dictionary:
            p_fund_part = total_fund_part_dictionary[portfolio]
else:
            p_fund_part = 'Not Applicable'

for n in G.nodes():
    if nx.has_path(G, portfolio,n ) == True and n != portfolio:
        length = nx.shortest_path_length(G, portfolio,n)
        #collect all possible paths
        short_path = nx.shortest_path(G,portfolio,n)
        pathes = list(nx.all_simple_paths(G,portfolio,n))
        #loop through all possible pathes
        for everypath in pathes:
            edgelist = create_edgepairs(everypath)
        for i,edgepairs in enumerate(edgelist):
            if i ==0:
                port_impact = weight_dictionary[edgepairs[0],edgepairs[1]] * (1-G.node[n]['total_fund_part'])
            else:
                port_impact = port_impact * weight_dictionary[edgepairs[0],edgepairs[1]] * (1-G.node[n]['total_fund_part'])
               
        if len(list(G.successors(n))) == 0:
            leaf = 'Leaf - bottom portfolio'
        else:
            leaf = 'Parent portfolio'
        #check if value is in dictionary
        
        node_information.append([portfolio,n, G.node[n]['direct_cost'],p_fund_part,
                                 port_impact, length, leaf, short_path[1:]])

impact_frame = pd.DataFrame(node_information,columns=['portfolio','child','child_cost','total_fund_part',
                                              'port_impact','steps','position','short_path'])
impact_frame.head(16)

cost_portions = impact_frame.port_impact * impact_frame.child_cost
rcParams['figure.figsize'] = 10,8
colors = ['gold','yellowgreen','lightcoral','lightskyblue','lavender','lightpink',
          'aliceblue','palegoldenrod','lavenderblush','lightcyan','lightsalmon','lightblue']
explode = [0,0,0,0,0,0,0,0,0,0,0,0.1]
plt.pie(cost_portions,labels = impact_frame.child,colors=colors,
        shadow=True, 
        startangle = 291,explode = explode)
plt.title("Impact by portfolio",size=15)
plt.legend(round(impact_frame.port_impact * impact_frame.child_cost,2))
plt.axis('equal')
plt.show()
leaves = []
F = G.subgraph(G).copy()

for eachnode in F.nodes():
    F.node[eachnode]['inherited_cost'] = 0
    F.node[eachnode]['approved_inheritance'] = 0
    F.node[eachnode]['total_approved'] = F.node[eachnode]['approved_inheritance'] + F.node[eachnode]['direct_approved']
    F.node[eachnode]['total_cost'] = F.node[eachnode]['direct_cost']*F.node[eachnode]['direct_approved']
    if list(F.successors(eachnode)) == []:
        leaves.append(eachnode)
        F.node[eachnode]['total_fund_part'] = 0
        F.node[eachnode]['approved_inheritance'] = 1   

counter = 1
output = []
next_level = []
while leaves != []: #while the list of leaves is not empty
    for leaf in leaves:
        direct_parents = list(F.predecessors(leaf))     #get the list of the parents
        for eachpar in direct_parents:
            # send your weight * approval to parents
            if weight_dictionary[eachpar,leaf] * (1- (F.node[leaf]['total_approved'])) < 0.05:
                # update approved_inheritance to your existing + your child's total *weight
                F.node[eachpar]['approved_inheritance'] = F.node[eachpar]['approved_inheritance']  + weight_dictionary[eachpar,leaf]  * F.node[leaf]['total_approved']
                # update inherited cost to existings + your weight * child's cost
                F.node[eachpar]['inherited_cost'] =  F.node[eachpar]['inherited_cost'] + weight_dictionary[eachpar,leaf] * F.node[leaf]['total_cost']
                # update the total_approved of the parent
                F.node[eachpar]['total_approved'] = F.node[eachpar]['approved_inheritance'] + (1-F.node[eachpar]['total_fund_part'])*F.node[eachpar]['direct_approved']
                # update the total costs    
                F.node[eachpar]['total_cost'] = F.node[eachpar]['total_cost'] + weight_dictionary[eachpar,leaf] * F.node[leaf]['total_cost']
                
        output.append([leaf,
                     F.node[leaf]['approved_inheritance'],
                     F.node[leaf]['direct_approved'],
                     F.node[leaf]['total_approved'],
                     F.node[leaf]['inherited_cost'],
                     F.node[leaf]['direct_cost'],
                     F.node[leaf]['total_cost'],
                     counter,
                     F.node[leaf]['total_fund_part']]) #add the leaf to the list
        F.remove_node(leaf) #remove the leaf from the graph
    for eachnode in F.nodes():
        if list(F.successors(eachnode)) == []:  
            # identify all the bottom leaves again (these were parents in the last round)
                next_level.append(eachnode)
    leaves = next_level
    counter = counter + 1
    next_level = []
    #print('Check finished on level ' + str(counter))
df = pd.DataFrame(output, columns=['portfolio','approved_inheritance','direct_approved','total_approved',
                                 'inherited_cost','direct_cost','total_cost','counter','total_fund_inhertied'])

df

leaves = []

#create the "helper-graph" again

F = G.subgraph(G).copy()

for eachnode in F.nodes():
    F.node[eachnode]['inherited_cost'] = 0
    F.node[eachnode]['approved_inheritance'] = 0
    F.node[eachnode]['total_approved'] = F.node[eachnode]['approved_inheritance'] + F.node[eachnode]['direct_approved']
    F.node[eachnode]['total_cost'] = F.node[eachnode]['direct_cost']*F.node[eachnode]['direct_approved']
    if list(F.successors(eachnode)) == []:
        leaves.append(eachnode)
        F.node[eachnode]['total_fund_part'] = 0
        F.node[eachnode]['approved_inheritance'] = 1   

info = [[F.node["A"]['direct_cost'],"A",F.node["A"]['direct_cost']/(1-F.node["A"]['total_fund_part'])]]
checklist = []
next_level = []
while leaves != []: #while the list of leaves is not empty
    for leaf in leaves:
        direct_parents = list(F.predecessors(leaf))     #get the list of the parents
        for eachpar in direct_parents:
            if weight_dictionary[eachpar,leaf] * (1- (F.node[leaf]['total_approved'])) < 0.05:
                F.node[eachpar]['approved_inheritance'] = F.node[eachpar]['approved_inheritance'] + weight_dictionary[eachpar,leaf]  * F.node[leaf]['total_approved']
                inh_passed = weight_dictionary[eachpar,leaf]  * F.node[leaf]['total_approved']
                
                F.node[eachpar]['inherited_cost'] =  F.node[eachpar]['inherited_cost'] + weight_dictionary[eachpar,leaf] * F.node[leaf]['total_cost']
                
                F.node[eachpar]['total_approved'] = F.node[eachpar]['approved_inheritance'] + (1-F.node[eachpar]['total_fund_part'])*F.node[eachpar]['direct_approved']
                
                F.node[eachpar]['total_cost'] = F.node[eachpar]['total_cost'] + weight_dictionary[eachpar,leaf] * F.node[leaf]['total_cost']
                total_inh_passed =  weight_dictionary[eachpar,leaf] * F.node[leaf]['total_cost']
                
                checklist.append([eachpar +" received total cost from " + leaf + " a value of " + str(total_inh_passed)
                              + " and it is " + str(F.node[leaf]['total_approved']*100) + "% signed off and " 
                              + eachpar + " invests " + str(weight_dictionary[eachpar,leaf]) + " into it" ])
                if eachpar == portfolio:
                    info.append([total_inh_passed,leaf,total_inh_passed/weight_dictionary[eachpar,leaf]])
        F.remove_node(leaf) #remove the leaf from the graph
    for eachnode in F.nodes():
        if list(F.successors(eachnode)) == []:  
            # identify all the bottom leaves again (these were parents in the last round)
                next_level.append(eachnode)
    leaves = next_level
    next_level = []
checklist
info_frame = pd.DataFrame(info, columns=['cost_impact','fund','normalized cost impact'])
checklist
opacity = 0.9
colors = ['gold','yellowgreen','lightcoral','lightskyblue','lavender','lightsalmon']
rcParams['figure.figsize'] = 18,14
plt.subplot(2,2,1)
plt.bar(info_frame.fund, info_frame.cost_impact, color = colors, width = 0.8, alpha = opacity)
plt.title("Cost Impact on A by each portfolio",size=16)
plt.subplot(2,2,2)
plt.bar(info_frame.fund, info_frame['normalized cost impact'],color = colors, width = 0.8, alpha = opacity)
plt.title("Cost impact on A normalized by invested weight",size=16)
plt.show()