import numpy as np

import pandas as pd

import cvxpy as cvx

import pylab as plt

import networkx as nx



def prepare_data(data,suffix):

    data=data.drop(data.columns[range(3,8)],axis=1)

    data=data.drop(data.columns[range(0,2)],axis=1)

    data=data.groupby('settlement_name_english').sum()

    data=data.add_suffix(suffix)

    return data



def join_data(x_data,y_data,x_title,y_title):

    M1=x_data.shape[1]

    M2=y_data.shape[1]



    n_votes_x=np.sum(x_data.values)

    n_votes_y=np.sum(y_data.values)

        

    data_joint=pd.merge(x_data,y_data, how='inner', left_index=True, right_index=True)

    x_data=data_joint[data_joint.columns[range(1,M1)]]

    y_data=data_joint[data_joint.columns[(M1+1):]]

    

    retained_votes_x=np.int(100*np.round(np.sum(x_data.values)/n_votes_x,2)) 

    print('considering only settlements in both elections, '+str(retained_votes_x)+'% of the '+str(x_title)+' votes retained')

    retained_votes_y=np.int(100*np.round(np.sum(y_data.values)/n_votes_y,2))

    print('considering only settlements in both elections, '+str(retained_votes_y)+'% of the '+str(y_title)+' votes retained')

    

    x_data=x_data.div(x_data.sum(axis=1),axis=0)

    x_data=x_data.mul(y_data.sum(axis=1),axis=0)

    avg_growth_factor=np.int(100*np.round(np.sum(x_data.values)/n_votes_x-1,2))

    print('Normalizing for average growth of +'+str(avg_growth_factor)+'% from '+str(x_title)+' to '+str(y_title))

          

    return x_data,y_data
x_data=pd.read_csv('../input/results_by_booth_2013 - english - v2.csv', encoding='iso-8859-1')

y_data=pd.read_csv('../input/results_by_booth_2015 - english - v3.csv', encoding='iso-8859-1')



x_data=prepare_data(x_data,'_2013')

y_data=prepare_data(y_data,'_2015')



[x_data,y_data]=join_data(x_data,y_data,'2013','2015')



M1=x_data.shape[1]

M2=y_data.shape[1]
def solve_transfer_coefficients(x_data,y_data):

    C=cvx.Variable(x_data.shape[1],y_data.shape[1])

    constraints=[0<=C, C<=1, cvx.sum_entries(C,axis=1)==1]

    objective=cvx.Minimize(cvx.sum_entries(cvx.square((x_data.values*C)-y_data.values)))

    prob=cvx.Problem(objective, constraints)

    prob.solve()

    C_mat=C.value

    

    misplaced_votes=np.sum(np.abs(x_data.values*C_mat-y_data.values))

    properly_placed_votes=np.int(100*np.round(1-misplaced_votes/np.sum(y_data.values),2))

    print('Transfer model properly account for '+str(properly_placed_votes)+'% of the votes')

    

    return C_mat



def major_parties(data,threshold,title):

    party_is_major=(data.sum(axis=0)/sum(data.sum(axis=0)))>threshold

    major_party_votes=np.sum(data.values[:,party_is_major],axis=0)



    votes_in_major_parties=np.int(100*np.round(np.sum(major_party_votes)/np.sum(data.values),2))

    print(str(votes_in_major_parties)+'% of the '+title+' votes are in major parties')

    

    major_party_votes=major_party_votes/sum(major_party_votes)

    M=sum(party_is_major)

    major_party_titles=[party_is_major.index.values[party_is_major==True][n][:-5] for n in range(0,M)]



    return party_is_major,major_party_votes, major_party_titles
C_mat=solve_transfer_coefficients(x_data,y_data)



party_threshold=0.02

transfer_threshold=0.01



[major_x,major_party_votes_x,major_party_titles_x]=major_parties(x_data,party_threshold,'2013')

M1=major_party_votes_x.shape[0]

[major_y,major_party_votes_y,major_party_titles_y]=major_parties(y_data,party_threshold,'2015')

M2=major_party_votes_y.shape[0]



C_mat=C_mat[:,major_y.values]

C_mat=C_mat[major_x.values,:]



vote_transfers=np.diag(major_party_votes_x)*C_mat

predicted_y=major_party_votes_x*C_mat

major_parties_error=np.sum(np.abs(major_party_votes_y-predicted_y))

major_parties_proper_votes=np.int(100*np.round(1-major_parties_error,2))

print('Transfer model properly accounts for '+str(major_parties_proper_votes)+'% of the votes for major parties')
def create_results_graph(major_party_titles_x,major_party_titles_y,vote_transfers):

    G1=nx.DiGraph()

    G2=nx.DiGraph()

    G=nx.DiGraph()



    for p1 in major_party_titles_x:

        G1.add_node(p1,node_color=1)

    for p2 in major_party_titles_y:

        G2.add_node(p2,node_color=0)



    G=nx.compose(G1,G2) # if nodes are in both G1 and G2, the color is from G2.

    for p2 in major_party_titles_y:

        if not (p2 in G1.nodes()): #different color for new nodes

            G.add_node(p2,node_color=0.5)

    

    for p1 in major_party_titles_x:

        for p2 in major_party_titles_y:

            v=vote_transfers[major_party_titles_x.index(p1),major_party_titles_y.index(p2)]

            if v>transfer_threshold:

                if not(p1==p2):

                    new_label=str(np.int(np.round(100*v)))+'%'

                    G.add_edge(p1,p2, label=new_label,arrowsize=2.0)



    party_sizes=np.transpose(np.round(100*sum(vote_transfers)))

    party_sizes=[np.int(party_sizes[n][0,0]) for n in range(0,vote_transfers.shape[1])]

    party_sizes=dict(zip(major_party_titles_y,party_sizes))

    new_labels=dict(zip(G2.nodes(),[node+" - "+str(party_sizes[node]) for node in G2.nodes()]))

    G=nx.relabel_nodes(G,new_labels)

    return G
G=create_results_graph(major_party_titles_x,major_party_titles_y,vote_transfers)

pos=nx.spring_layout(G.to_undirected(), dim=2, iterations=50000, weight=None)
colors=[nx.get_node_attributes(G,'node_color').get(node) for node in G.nodes()]

nx.draw_networkx(G, pos=pos, cmap=plt.get_cmap('jet'), node_color=colors, with_labels=True,font_size=24)

edge_labels = nx.get_edge_attributes(G,'label')

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=24)

fig = plt.gcf()

fig.set_size_inches(30, 40)

plt.show()