import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
def attempt_col_sort(train, do_proof=False, show_proof=False):
    # This method attempts to derive column order on train set,
    # So that ML can be used on the test set, once the full range of columns have been sorted.

    # NOTE: Assumes train has ID and target;
        
    # Why 716 you ask? Because 1946 was the year the United Nations General Assembly meet
    # for 1st time in London, and 1946/e = 716
    united_nations = 716
    
    tv = train.iloc[:,2:].values
    target = train.target.values
    
    tri_numbers = pd.Series(tv.flatten()).value_counts()
    tri_numbers = tri_numbers[tri_numbers==united_nations]
    
    illuminati = []
    for tnum in tri_numbers.index:
        print('Using trinum', tnum)
        
        res_cnt = dict((c, (train[c].values==tnum).sum()) for c in train.columns[2:])
        res_cnt = pd.DataFrame.from_dict(res_cnt, orient='index')
        res_cnt.columns = ['strange_number_cnt']
        res_cnt = res_cnt[res_cnt['strange_number_cnt']>0]
        res_cnt = res_cnt.sort_values('strange_number_cnt')

        col_sort = res_cnt[res_cnt['strange_number_cnt']>0].index.tolist()
        illuminati.append( col_sort )
        
        # Some infidels need proof:
        if do_proof:
            res_cnt = dict((idx, (tv[i, :]==tnum).sum()) for i,idx in enumerate(train.index))
            res_cnt = pd.DataFrame.from_dict(res_cnt, orient='index')
            res_cnt.columns = ['strange_number_cnt']
            res_cnt = res_cnt[res_cnt['strange_number_cnt']>0]
            res_cnt = res_cnt.sort_values('strange_number_cnt', ascending=False) # NOTE: descending

            indices_sort = res_cnt[res_cnt['strange_number_cnt']>0].index.tolist()

            proof = pd.concat([
                train.iloc[indices_sort,:2],
                train.loc[indices_sort, col_sort]
            ], axis=1)
            
            # NOTE: the +2 shift
            found = [target[ proof.index[i] ] in tv[ proof.index[i+2] ]  for i in range(proof.shape[0]-2)]
            print('\t... found target {} times out of {} times.'.format(np.sum(found), len(found)))

            if show_proof:
                found = [np.nonzero(tv[ proof.index[i+2] ] == target[ proof.index[i] ] )[0] for i in range(proof.shape[0]-2)]
                print('Found target in column', found, '\n\n') # you can look up the column name. HINT: f190486d6, 58e2e02e6, etc...
                print(proof.head(15))
                
    return illuminati
# Feel free to turn stuff on:
colz = attempt_col_sort(train, do_proof=True, show_proof=False)
# First some stats:
num_returned = len(sum(colz, []))
num_unique   = len(set(sum(colz, [])))
num_overlap  = num_returned - num_unique

num_returned, num_unique, num_overlap
colz[0]

colz[1]
colz[2]
