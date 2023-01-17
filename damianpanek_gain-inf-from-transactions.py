# Get further transactions information, 

# range 1: 30 , from transaction.csv

# check if this variables are informative 

# cross_val_score /feature importance check 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import numpy as np 

import pandas as pd 

import gc; gc.enable() 



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/sample_submission_zero.csv")

transactions  = pd.read_csv("../input/transactions.csv")





train_iter = transactions.sort_values(by = ['transaction_date'], ascending = [False])



for i in range(1, 30, 1):



    

    train_sub = train_iter.drop_duplicates(subset= 'msno', keep = 'first')

    ind = train_sub.index

    

    train_iter =  train_iter.drop(ind, axis = 0)



    print("Wymiar train_iter  po", i, "iters", train_iter.shape[0])

    print("Wymiar train_sub   po ", i, "iters", train_sub.shape[0])



    

# Does not work in my local python dist

#    train_sub.rename(columns = {

#        "payment_method_id": "payment_method_id" + str(i) , 

#        "payment_plan_days": "payment_plan_days" + str(i) , 

##        "plan_list_price"  : "plan_list_price"   + str(i),

 #       "actual_amount_paid" : "actual_amount_paid" + str(i),

 #       "is_auto_renew"     : "is_auto_renew" + str(i),

 #       "transaction_date"  : "transaction_date"  + str(i),

 #       "membership_expire_date"  : "membership_expire_date"  + str(i),

 #       "is_cancel"  : "is_cancel"  + str(i)

 #   })   

  

    

    train_sub.columns = ['msno', 'payment_method_id' + str(i), 

                    'payment_plan_days' + str(i), 

                    'plan_list_price' + str(i), 

                    'actual_amount_paid' + str(i), 

                    'is_auto_renew' + str(i), 

                    'transaction_date' + str(i), 

                    'membership_expire_date' + str(i), 

                    'is_cancel' + str(i)]

    

    

    train = pd.merge(train, train_sub, on = 'msno', how = 'left')

    test  = pd.merge(test, train_sub, on = 'msno', how = 'left')

    gc.collect()



print("train  rows", train.shape[0], "train cols", train.shape[1])

print("test rows", test.shape[0], "test cols", test.shape[1])















# Any results you write to the current directory are saved as output.