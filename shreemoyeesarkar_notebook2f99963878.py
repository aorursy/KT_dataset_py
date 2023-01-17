from nltk.corpus import stopwords

import pandas as pd

import numpy as np

from sklearn.metrics import log_loss

from scipy.optimize import minimize



stops = set(stopwords.words("english"))
def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R

#LOAD TRAINSET, TESTSET

train = pd.read_csv("../input/train.csv")

train[ 'R' ] = train.apply( word_match_share, axis=1, raw=True )

print( train.head() ) 

test = pd.read_csv("../input/test.csv", index_col=False )

test['R'] = test.apply( word_match_share, axis=1, raw=True )

print( test.head() )

#Mean target

GLOBAL_MEAN = np.mean( train['is_duplicate'] ) 

print( 'Mean is_duplicated', GLOBAL_MEAN )



#OPTIMIZE FUNCTIONS

def minimize_train_log_loss( W ):

    train["prediction"] = GLOBAL_MEAN + train["R"] * W[0] + W[1]

    score = log_loss( train['is_duplicate'], train['prediction'] )

    print(  score , W )

    return( score )



res = minimize(minimize_train_log_loss, [0.00,  0.00], method='Nelder-Mead', tol=1e-4, options={'maxiter': 400})

W = res.x

print( 'Best weights: ',W )



#APPLY TO TESTSET

test["is_duplicate"] = test["R"] 

test[ ['test_id','is_duplicate'] ].to_csv("count_words_benchmark.csv", header=True, index=False)



test.shape[0]
def prob_sim(rows):

        q1=pd.Series(get_words(rows['question1']))

        q2=pd.Series(get_words(rows['question2']))

        n1=len([w for w in q1.values if w in q2.values])

        n2=len([w for w in q2.values if w in q1.values])

        n=len(q1)+len(q2)

        return (n1+n2)/n

test['prob']=test.apply(prob_sim,axis=1,raw=True)
test_dummy['prob']=test_dummy.apply(prob_sim,axis=1,raw=True)
test_dummy['prob']
GLOBAL_MEAN = np.mean( train['is_duplicate'] ) 

print( 'Mean is_duplicated', GLOBAL_MEAN )
test[['test_id','prob']].to_csv("sub1.csv",index=False,header=True)