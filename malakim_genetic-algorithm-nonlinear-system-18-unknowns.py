def gene_score(arr_gene):    
    D1 = arr_gene[:,1]
    D2 = arr_gene[:,2]
    D3 = arr_gene[:,3]
    D4 = arr_gene[:,4]
    D5 = arr_gene[:,5]
    D6 = arr_gene[:,6]
    D7 = arr_gene[:,7]
    D8 = arr_gene[:,8]
    D9 = arr_gene[:,9]
    D10 = arr_gene[:,10]
    Q = arr_gene[:,11]
    R = arr_gene[:,12]
    S = arr_gene[:,13]
    T = arr_gene[:,14]
    U = arr_gene[:,15]
    V = arr_gene[:,16]
    W = arr_gene[:,17]
    X = arr_gene[:,18]
    
    score_1 = D2 + D4 - Q
    score_2 = D5 + D2 + 1 - R
    score_3 = D1 - D3 + D2 - S
    score_4 = D4 + D5 + 1 - T
    score_5 = D1 - D5 - U
    score_6 = D3 - D5 + D2 - V
    score_7 = D4 + 1 - W
    score_8 = D3 + D5 - X
    score_9 = D10 - D9 - D8 - Q
    score_10 = D8 + D9 - R
    score_11 = D8 * 2 - S
    score_12 = D6 + D9 - T
    score_13 = D10 - D8 - D7 - U
    score_14 = (D7 * D9) + D7 - V
    score_15 = D7 + D6 - W
    score_16 = (D8 * D9) + D7 - X
        
    score = np.absolute(score_1) \
    + np.absolute(score_2) \
    + np.absolute(score_3) \
    + np.absolute(score_4) \
    + np.absolute(score_5) \
    + np.absolute(score_6) \
    + np.absolute(score_7) \
    + np.absolute(score_8) \
    + np.absolute(score_9) \
    + np.absolute(score_10) \
    + np.absolute(score_11) \
    + np.absolute(score_12) \
    + np.absolute(score_13) \
    + np.absolute(score_14) \
    + np.absolute(score_15) \
    + np.absolute(score_16)

    return score
def create_init_generation(n_pop):
    arr_gene = np.random.randint(0, high=10, size=(n_pop, 19))
    arr_gene[:,0] = -1
    return arr_gene

def update_gene_score(arr_gene, gene_score):
    # ----- select best gene by score -----
    arr_gene[:,0] = gene_score(arr_gene)
    
    return arr_gene

def mate_best_gen(arr_gene, n_pop_tar, mate_best_pct, gene_score):
    arr_gene = np.unique(arr_gene, axis=0)
    # get population of best genes for mating to restore full population
    n_pop = arr_gene.shape[0]
    n_best_gene = int(n_pop_tar * mate_best_pct)
    if n_pop < n_best_gene:
        n_best_gene = n_pop
    n_loop = n_pop_tar // n_best_gene
    # ----- select best gene by score -----
    arr_gene = update_gene_score(arr_gene, gene_score)
    arr_gene = arr_gene[np.argsort(arr_gene[:, 0])]
    arr_best = arr_gene[:n_best_gene][:]
    # ----- cross mate best genes to generate whole population -----
    arr_pop = arr_best
    is_ans = False
    for i in range(n_loop):
        arr_mask1 = np.random.randint(0, high=2, size=arr_best.shape)
        arr_mask2 = 1 - arr_mask1
        arr_best_shuffle = arr_best.copy()
        np.random.shuffle(arr_best_shuffle)
        arr_tmp = arr_best * arr_mask1 + arr_best_shuffle * arr_mask2
        arr_tmp = update_gene_score(arr_tmp, gene_score)
        arr_pop = np.vstack((arr_pop, arr_tmp))
        #display('loop {}: best score = {}'.format(i+1, arr_tmp[0][0]))    
    arr_pop = np.vstack((arr_pop, create_init_generation(n_best_gene)))
    # ----- select best gene by score -----
    arr_gene = np.unique(arr_pop, axis=0)
    arr_gene = update_gene_score(arr_gene, gene_score)
    arr_gene = arr_gene[np.argsort(arr_gene[:, 0])]
    arr_gene = arr_gene[:n_pop_tar,:]
    best_ans = arr_gene[0][0]

    '''
    display('Best Answer: {}'.format(best_ans))
    display(arr_gene)
    '''
    if best_ans == 0:
        is_ans = True
    
    return is_ans, arr_gene
import numpy as np

# number of settlements
N_SETTLEMENT = 50
# number of generations for evolution for each settlement
N_GEN = 20
# size of population for each settlement
N_POP = 100000
# top percent of the best genes for mating to re-populate the next generations
MATE_BEST_PCT = 0.20
# number of top best genes to be retain and stored for each settlement
N_SETTLEMENT_BEST = 50
is_ans = False
arr_gene = create_init_generation(n_pop = N_POP)
arr_gene_best = np.zeros((0, arr_gene.shape[1]))
for i_settle in range(N_SETTLEMENT):
    arr_gene = create_init_generation(n_pop = N_POP)
    for i_gen in range(N_GEN):
        is_ans, arr_gene = mate_best_gen(
            arr_gene = arr_gene,
            n_pop_tar = N_POP, 
            mate_best_pct = MATE_BEST_PCT,
            gene_score = gene_score
        )
        best_score = arr_gene[0][0]
        #display(arr_gene[0,:])
        if is_ans:
            print('answer found')
            break
        #display(arr_gene.shape)
        display('settle[{}/{}]-gen[{}/{}]; best score = {}'.format(i_settle + 1, N_SETTLEMENT, i_gen + 1, N_GEN, best_score))
    arr_gene_best = np.vstack((arr_gene_best, arr_gene[:N_SETTLEMENT_BEST,:]))
    shape_b = arr_gene_best.shape
    arr_gene_best = np.unique(arr_gene_best, axis=0)
    shape_a = arr_gene_best.shape
    arr_gene_best = arr_gene[np.argsort(arr_gene_best[:, 0])]
    best_score = arr_gene_best[0][0]
    display('settle[{}][shape.before = {}, shape.after = {}]; best score = {}'.format(i_settle + 1, shape_b, shape_a, best_score))
    np.savetxt('output.csv', arr_gene_best, delimiter=',')
    if is_ans:
        break
print('Number of Settlements: {}\nPopulation Size: {}\nNumber of Evolutions per Settlement: {}\nTop Best Genes for Mating per Evolution: {}%'.format(N_SETTLEMENT, N_POP, N_GEN, MATE_BEST_PCT * 100))
import pandas as pd

dict_col_rename = {
    0: 'gene_score',
    1: 'D1',
    2: 'D2',
    3: 'D3',
    4: 'D4',
    5: 'D5',
    6: 'D6',
    7: 'D7',
    8: 'D8',
    9: 'D9',
    10: 'D10',
    11: 'Q',
    12: 'R',
    13: 'S',
    14: 'T',
    15: 'U',
    16: 'V',
    17: 'W',
    18: 'X',    
}
df_result = pd.DataFrame(arr_gene_best)
df_result = df_result.rename(columns=dict_col_rename)

gene_score_best = min(df_result.gene_score)
display('Best Gene Score: {}'.format(gene_score_best))

#display(df_result.query('gene_score == @gene_score_best'))
print(df_result.query('gene_score == @gene_score_best').to_string(index=False))

