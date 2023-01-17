import numpy as np

import pandas as pd

import cvxpy as cvx

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)
results_2019_raw = pd.read_csv("../input/votes per settlement 2019 - hebrew.csv", encoding='iso_8859_8')

results_2019_raw.set_index('שם ישוב',inplace=True)

results_2019_raw.head()
results_2015_raw = pd.read_csv("../input/votes per settlement 2015 - hebrew.csv",encoding='iso_8859_8')

results_2015_raw.set_index('שם ישוב',inplace=True)

results_2015_raw.head()
def normalize_results(results):

    party_results = results.sum()

    party_results = party_results.div(party_results.sum())

    return party_results



def threshold_parties(results,threshold=0.99):

    cumulative_normalized_results = normalize_results(results).sort_values(ascending=False).cumsum()

    parties_above_threshold = cumulative_normalized_results.index[cumulative_normalized_results<threshold]

    return parties_above_threshold



def group_other_parties(results,threshold=0.99):

    total_votes = results.sum(axis=1)

    parties_above_threshold = threshold_parties(results,threshold)

    other_votes = results.sum(axis=1)-results[parties_above_threshold].sum(axis=1)

    results = results[parties_above_threshold]

    results.loc[:,'other'] = other_votes

    return results



def prepare_data(results,suffix):

    return group_other_parties(results[results.columns[5:]]).rename(lambda x: x+suffix,axis=1,inplace=False)



results_2015 = prepare_data(results_2015_raw,'_2015')

results_2019 = prepare_data(results_2019_raw,'_2019')

results_2015.head()
joint = results_2015.join(results_2019)

joint.dropna(inplace=True, axis=0)

votes_diff = results_2019.loc[joint.index].sum(axis=1) - results_2015.loc[joint.index].sum(axis=1)

joint['no_vote_2019']=[max(-x,0) for x in votes_diff]

joint['no_vote_2015']=[max(x,0) for x in votes_diff]

parties_2015 = np.append(results_2015.columns,'no_vote_2015')

parties_2019 = np.append(results_2019.columns,'no_vote_2019')

joint = joint[parties_2015].join(joint[parties_2019])

joint.head()
# solving functions



def solve_transfer_coefficients(x1,x2):

    m1 = x1.shape[1]

    m2 = x2.shape[1]

    

    C=cvx.Variable(shape=(m1,m2))

    constraints=[0<=C, C<=1, cvx.sum(C,axis=1)==1]

    

    settlement_sizes = x1.sum(axis=1)

    settlements_number = x1.shape[0]

    

    r_2_1 = cvx.sum_squares((x1*C)-x2)/settlements_number/np.square(settlement_sizes).mean()

    objective=cvx.Minimize(r_2_1)

    prob=cvx.Problem(objective, constraints)



    r_2_1 = prob.solve(verbose = True, solver='OSQP')

    coeff_mat = C.value

    

    overall_result = [v/sum(v) for v in [np.mean(x2,axis=0)]][0]

    naive_estimation = np.outer(np.sum(x2,axis=1),overall_result)

    r_2_0 = np.mean(np.sum(np.square(naive_estimation-x2),axis=1)) /np.square(settlement_sizes).mean()

    r_2_score = 1-r_2_1/r_2_0

    print("The percentage of variance explained by the model is "+str(np.round(r_2_score,2)))

    

    return coeff_mat, r_2_score



def make_transfer_coef_matrix(coeff_mat,titles_1, titles_2, resolution_digits = 3):

    transfer_coeffs_dict = dict(zip(titles_2,np.round(coeff_mat.T,3)))

    transfer_coeffs_matrix = pd.DataFrame.from_dict(transfer_coeffs_dict)

    transfer_coeffs_matrix.set_index(titles_1,inplace=True)

    return transfer_coeffs_matrix



def make_transfer_matrix(coeffs_mat,titles_1,titles_2,x1, resolution_digits=3):

    foo = np.sum(x1,axis=0)/np.sum(x1)

    vote_transfers = np.round(np.matmul(np.diag(foo),coeffs_mat),resolution_digits)

    vote_transfers = pd.DataFrame.from_dict(dict(zip(titles_2,np.round(vote_transfers.T,resolution_digits))))

    vote_transfers.set_axis(titles_1,inplace=True)

    return vote_transfers
m1 = len(parties_2015)

x1 = joint[parties_2015].values



m2 = len(parties_2019)

x2 = joint[parties_2019].values



coeffs_mat, _ = solve_transfer_coefficients(x1,x2)

make_transfer_coef_matrix(coeffs_mat,parties_2015,parties_2019)
vote_transfers = make_transfer_matrix(coeffs_mat,parties_2015,parties_2019,x1)

vote_transfers
transfer_threshold=0.003

links=np.where(vote_transfers > transfer_threshold)



labels_english = ['Likud_15', 'Avoda_15', 'Joint', 'Lapid','Kahlon_15', 'Bait',

                  'Shas_15', 'Liberman_15', 'Gimel_15' ,'Meretz_15', 'Yachad', 'other_15', 'no_15',

                 'Likud_19', 'Kaholavan', 'Shas_19', 'Gimel_19', 'Hadash', 'Avoda_19', 'Liberman_19',

                  'UYamin', 'Meretz_19', 'Kahlon_19', 'Raam-Balad', 'NYamin','Zehut','other_19','no_19']



data = dict(

    type='sankey',

    node = dict(pad = 15, 

                thickness = 20, 

                line = dict(color = "black",width = 0.5),

                color='black',

                label=labels_english),

    link = dict(source=links[0],

                target=links[1]+max(links[0])+1,

                value=[vote_transfers.values[f[0],f[1]]*120 for f in zip(links[0],links[1])]),

    orientation = 'h'

)



layout =  dict(

    title = "Shift in votes between parties, from 2015 to 2019 elections",

    font = dict(size = 14)

)



fig = dict(data=[data], layout=layout)

iplot(fig,validate=False)