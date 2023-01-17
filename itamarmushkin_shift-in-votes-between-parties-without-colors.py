import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import cvxpy as cvx
import networkx as nx

plt.style.use('fivethirtyeight')
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold' 

df=pd.read_csv('../input/israeli_elections_results_1996_to_2015.csv',encoding='iso-8859-1')
def read_and_prepare_data(df,election_year):
    
    votes=df[df['year']==int(election_year)]
    votes=votes.drop(columns='year')
    votes=votes.drop(votes.columns[range(3,8)],axis=1)
    votes=votes.drop(votes.columns[range(0,2)],axis=1)
    votes=votes.drop(votes.columns[votes.sum()==0],axis=1) #clearing empty columns
    votes=votes[np.sort(votes.columns)]
    votes=(votes[(votes.sum(axis=1)>0)]) #clearing empty rows
    votes=votes.add_suffix(election_year)            
    return votes

def load_and_join_data(df,x_label,y_label):
    x_data=read_and_prepare_data(df,x_label)
    y_data=read_and_prepare_data(df,y_label)
    x_data=x_data.groupby('settlement_name_english'+x_label).sum()
    y_data=y_data.groupby('settlement_name_english'+y_label).sum()
    M=x_data.shape[1]
    data_joint=pd.merge(x_data,y_data, how='inner', left_index=True, right_index=True)
    x_data=data_joint[data_joint.columns[range(0,M)]]
    y_data=data_joint[data_joint.columns[M:]]
    x_data=x_data.div(x_data.sum(axis=1),axis=0)
    x_data=x_data.mul(y_data.sum(axis=1),axis=0)
    return x_data,y_data
def major_parties(votes,threshold,election_year,verbose):
    if 'settlement_name_english'+election_year in votes.columns:
        votes=votes.drop('settlement_name_english'+election_year,axis=1)
    party_is_major=(votes.sum(axis=0)/sum(votes.sum(axis=0)))>threshold
    major_party_votes=np.sum(votes.values[:,party_is_major],axis=0)
    votes_in_major_parties=np.int(100*np.round(np.sum(major_party_votes)/np.sum(votes.values),2))
    if verbose:
        print(str(votes_in_major_parties)+'% of the '+election_year+' votes are in major parties')
    major_party_votes=major_party_votes/sum(major_party_votes) #rescaling to ignore dropped data
    major_party_titles=[party_is_major.index.values[party_is_major==True][n][:-4] for n in range(0,sum(party_is_major))]
    return party_is_major,major_party_votes, major_party_titles
def solve_transfer_coefficients(x_data,y_data,alt_scale,verbose):
    C=cvx.Variable([x_data.shape[1],y_data.shape[1]])
    constraints=[0<=C, C<=1, cvx.sum(C,axis=1)==1]
    
    objective=cvx.Minimize(cvx.sum_squares((x_data.values*C)-y_data.values))
    prob=cvx.Problem(objective, constraints)
    prob.solve(solver='ECOS')
    if verbose:
        print (prob.status)
    if prob.status!='optimal': #sometimes we need to rescale the objective function for the computation to succeed. It's just a numeric thing.
        objective=cvx.Minimize(alt_scale*cvx.sum_squares((x_data.values*C)-y_data.values))
        prob=cvx.Problem(objective, constraints)
        prob.solve()
        if verbose:
            print(prob.status+'(with alt_scale)')
    C_mat=C.value

    if verbose:
        print(C_mat.min()) #should be above 0
        print(C_mat.max()) #should be below 1
        print(C_mat.sum(axis=1).min()) #should be close to 1
        print(C_mat.sum(axis=1).max()) #should be close to 1
    
    if verbose:
        misplaced_votes=np.sum(np.abs(np.matmul(x_data.values,C_mat)-y_data.values))
        properly_placed_votes=np.int(100*np.round(1-misplaced_votes/np.sum(y_data.values),2))
        print('Transfer model properly account for '+str(properly_placed_votes)+'% of the votes on the local level') #this counts the overall error per settlement
    
    return C_mat
election_labels=['1996', '1999', '2003', '2006', '2009', '2013', '2015']
party_threshold=0.019
link_threshold=0.01
transfer_threshold=0.005
alt_scale=1e-3

data_trace = dict(type='sankey',orientation = "v",
                  node = dict(pad = 15,thickness = 20,line = dict(color = "black",width = 0.5),label=[],color='black'),
                  link = dict(source=[],target=[],value=[])
                 )

layout =  dict(title = "Shift in votes between parties",width = 1180,height = 1544,font = dict(size = 14))

for election_index in range(0,len(election_labels)-1):
    x_label=election_labels[election_index]
    y_label=election_labels[election_index+1]
    [x_data,y_data]=load_and_join_data(df,x_label, y_label)
    [major_x,major_party_votes_x,major_party_titles_x]=major_parties(x_data,party_threshold,election_year=x_label,verbose=False)
    major_party_titles_x=[party+'_'+x_label for party in major_party_titles_x]
    [major_y,major_party_votes_y,major_party_titles_y]=major_parties(y_data,party_threshold,election_year=y_label,verbose=False)
    major_party_titles_y=[party+'_'+y_label for party in major_party_titles_y]

    C_mat=solve_transfer_coefficients(x_data[x_data.columns[major_x]],y_data[y_data.columns[major_y]],alt_scale,verbose=False)
    vote_transfers=np.matmul(np.diag(major_party_votes_x),C_mat)
    links=np.where(vote_transfers>transfer_threshold)

    major_parties_error=np.sum(np.abs(np.matmul(major_party_votes_x,C_mat)-major_party_votes_y))
    major_parties_correct_votes=np.int(100*np.round(1-major_parties_error,2))
    print('Transfer model properly accounts for '+str(major_parties_correct_votes)+'% of the votes for on a national level '+'from '+str(x_label)+' to '+str(y_label))
    
    data_trace['node']['label']=data_trace['node']['label']+major_party_titles_x
    
    if y_label==election_labels[-1]:
        data_trace['node']['label']=data_trace['node']['label']+major_party_titles_y
    
    if len(data_trace['link']['source'])==0:
        data_trace['link']['source']=links[0]
    else:
        data_trace['link']['source']=np.append(data_trace['link']['source'],links[0]+np.round(np.max(data_trace['link']['source']))+1)
    data_trace['link']['target']=np.append(data_trace['link']['target'],links[1]+np.round(np.max(data_trace['link']['source']))+1)    
    data_trace['link']['value']=np.append(data_trace['link']['value'],vote_transfers[links[0],links[1]])
fig = dict(data=[data_trace], layout=layout)
iplot(fig,validate=False)