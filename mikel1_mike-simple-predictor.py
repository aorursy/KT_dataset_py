# this Notebook uses a simple logit estimator - no frills!

# 

# 1) attempt: all the data but only most recent of duplicate content_id

# Version 7 - ROC AUC = 0.746 pafile.txt

#

# 2) attempt: last 10 content_id but only first of duplicates

# Version 8 - ROCAUC=0.716 pafile10.txt

#

# 3) attempt: last 20 content_id but only first of duplicates

# Version 10 - ROCAUC=0.651 pafile20.txt

#

# 4) attempt: p=0.5 to check ROC AUC computation

# Version 12 ROCAUC=0.5

#

# 5) attempt: last 30 content_id but only first of duplicates

# Version 13 - ROCAUC= 0.742 pafile30.txt

#

# 6) attempt: last 40 content_id but only first of duplicates

# Version 14 - ROCAUC=0.746 pafile40.txt

#

# 7) attempt: last 50 content_id but only first of duplicates

# Version 15 - ROCAUC= 0.749 pafile50.txt

#

# 8) attempt: last 60 content_id but only first of duplicates

# Version 16 - ROCAUC= 0.750 pafile60.txt

#

# 9) attempt: global variable for each n observations: 0.5, 0.6., 

# Version 17 - ROCAUC=0.5 size is 500K

#

# 10) attempt: last 70 content_id but only first of duplicates

# Version 19 - ROCAUC=  pafile70.txt

#

# 11) attempt: last 80 content_id but only first of duplicates

# Version 20 - ROCAUC=  pafile80.txt



# Version 22: chasing elusive scoring error pa60 ia80 - error

# Version 23: ia60 pa80 ROC = .752

# Version 24: with the files! but p=0.6 ROC=0.5 - worked

# Version 25: last 80 again - failed - trailing vbtab

# Version 26: last 80 again ROC = .752

# Version 27: last 70 again ROC = .751

# Version 28: this was 70 again - Oops! - maybe with a loop :-( ROC = .751

# version 29: "last90" is actually facets 80 +.02 for time: ROC = 0.663

# version 30: last90win ROC = 0.753

# version 31: last100win ROC =

# do the housekeeping



import numpy as np

import pandas as pd

import bisect

import math

import numbers

import decimal

import riiideducation

env = riiideducation.make_env()

persons = pd.read_csv("../input/persons/persons.txt",sep="\t")

pafile = pd.read_csv("../input/last100/pafile100.txt",sep="\t")

iafile = pd.read_csv("../input/last100/iafile100.txt",sep="\t")

# plogit = pafile['MEASURE'].tolist()

# getting length of list 

#length = len(plogit) 

#for i in range(length): 

#    print(i, plogit[i])

#    plogit[i] = plogit[i]+1

#    print(i, plogit[i])



# exit
# compute a probability for each data line



def prob(user, content):

    userseq = bisect.bisect_left(persons.iloc[:,1],user)

    if persons.iat[userseq,1] != user: userseq=0

    # global plogit    

    plogit = pafile.iat[userseq,1]

    # increments by .02 logits per response

    # print (userseq, plogit[userseq])

    # if userseq!=0: plogit[userseq] = plogit[userseq]+.02

    ilogit = iafile.iat[content,1]

    # logit = plogit[userseq]-ilogit

    logit = plogit-ilogit

    p= 1/(1+math.exp(-logit))

    # print(user,content, userseq,plogit[userseq],ilogit,logit,p)

    # print(user,content, userseq,plogit,ilogit,logit,p)

    return p   

    
# go down the data lines



def mikes(df):

    user = df['user_id']

    content=df['content_id']

    if df['content_type_id'] != 0: content = 0

    p = prob (user, content)

    df['timestamp']=p

    return df['timestamp']

# loop down the tests



iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:

     test_df['timestamp'] = test_df.apply(mikes,axis=1)

     test_df=test_df.rename(columns = {'timestamp':'answered_correctly'})   

     env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])      

        

print ("done - yeah!")
