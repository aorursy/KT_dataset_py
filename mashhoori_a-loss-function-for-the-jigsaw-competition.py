

import pandas as pd 

import numpy as np



def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, True, False)       

    

def convert_dataframe_to_bool(df, columns):        

    bool_df = df.copy()

    for col in columns:

        convert_to_bool(bool_df, col)

    return bool_df



identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

aux_target_columns = ['sexual_explicit', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

num_identity = len(identity_columns)



train = pd.read_csv('../input/train.csv')

train.fillna(0, inplace = True)

train = convert_dataframe_to_bool(train, ['target'] + identity_columns)





weights = np.ones((len(train),))



# Positive and negative examples get balanced weights in each part



# These samples participate in the over all AUC term

weights[train['target']]   =  1 / train['target'].sum()                

weights[~train['target']]   =  1 / (~train['target']).sum()

for col in identity_columns:

    hasIdentity = train[col]

    # These samples participate in the subgroup AUC and BNSP terms    

    weights[hasIdentity & train['target']]   +=  2 / (( hasIdentity &  train['target']).sum() * num_identity)

    # These samples participate in the subgroup AUC and BPSN terms

    weights[hasIdentity & ~train['target']]  +=  2 / (( hasIdentity & ~train['target']).sum() * num_identity)

    # These samples participate in the BPSN term

    weights[~hasIdentity & train['target']]  +=  1 / ((~hasIdentity &  train['target']).sum() * num_identity)

    # These samples participate in the BNSP term

    weights[~hasIdentity & ~train['target']] +=  1 / ((~hasIdentity & ~train['target']).sum() * num_identity)

    

    

    

weights = weights / weights.max()





y_train = train['target'].values

y_aux_train = train[aux_target_columns].values

y_combined =  np.concatenate((y_train.reshape((-1, 1)), weights.reshape((-1, 1)), y_aux_train.reshape((-1, len(aux_target_columns)))), axis = 1)

def custom_loss(data, targets):    

    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])

    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])

    return (bce_loss_1 * 1) + (bce_loss_2 * 1)