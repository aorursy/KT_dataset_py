import pandas as pd

import numpy as np

from collections import Counter
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv') 
train.head()
test.head()
sub.head()
train.info()
test.info()
sample = train.iloc[1]
sample['sequence']
sample['structure']
Counter(sample['structure'])
sample['predicted_loop_type']
len(sample['deg_Mg_pH10'])
bases = []

for i in range(len(train)):

    count_dict = Counter(train.iloc[i]['sequence'])

    bases.append(count_dict) 

bases = pd.DataFrame.from_dict(bases)

bases    
base_percent = bases.div(train['seq_length'],axis=0)

base_percent.columns = ['G_percent','A_percent','C_percent','U_percent']

base_percent['id'] = train['id']

base_percent
all_pairs = []

all_partners = []

for idx in range(len(train)):

    sample = train.iloc[idx]

    stack=[]

    pair_freq = {}

    partners = [-1 for i in range(sample['seq_length'])]

    for i in range(len(sample['structure'])):

        if sample['structure'][i] == '(':

                stack.append(i)

        elif sample['structure'][i] == ')':

                poped = stack.pop()

                pair = sample['sequence'][i] + sample['sequence'][poped]

                partners[i] = sample['sequence'][poped]

                partners[poped] = sample['sequence'][i]

                if pair not in pair_freq:

                    pair_freq[pair] = 1

                else:

                    pair_freq[pair] += 1

    all_pairs.append(pair_freq)     

    all_partners.append(partners)
all_pairs[:10]
all_partners_df = pd.DataFrame(all_partners)

all_partners_df
pairs_df = pd.DataFrame.from_dict(all_pairs)

pairs_df.fillna(0,inplace=True)

pairs_df['tot_pair'] = pairs_df.sum(axis=1)

pairs_df
pairs_percent = pairs_df.iloc[:,:-1].div(pairs_df['tot_pair'],axis=0)

pairs_percent['tot_pair_percent'] = pairs_df['tot_pair'].div(train['seq_length']/2,axis=0)

pairs_percent.columns = ['GU_percent','GC_percent','AU_percent','CG_percent','UA_percent','UG_percent','tot_pair_percent']

pairs_percent['id'] = train['id']

pairs_percent
loop_type = []

for i in range(len(train)):

    count_dict = Counter(train.iloc[i]['predicted_loop_type'])

    loop_type.append(count_dict)

loop_type[:10]
loop_type = pd.DataFrame(loop_type)

loop_type.fillna(0,inplace=True)

loop_type
loop_type_percent = loop_type.div(train['seq_length'],axis=0)

loop_type_percent.columns = ['E_percent','S_percent','H_percent','B_percent',

                            'X_percent','I_percent','M_percent']

loop_type_percent['id'] = train['id']

loop_type_percent
train.head()
cols = ['id','base','loop_type','paired_with','prev_base1','prev_base2','prev_base3'

        ,'prev_base4','prev_base5','prev_base6','next_base1','next_base2',

        'next_base3','next_base4','next_base5','next_base6']



extracted_features = pd.DataFrame(columns = cols)

all_target = pd.DataFrame(columns = ['id','reactivity','deg_Mg_pH10','deg_Mg_50C'])



for idx in train['index']:

    sample = train.iloc[idx]

    neighbor_bases = []

    

    for i in range(sample['seq_scored']):

        temp_neighbor_bases = [-1 for j in range(12)]

        for j in range(6):

            temp_neighbor_bases[j] = sample['sequence'][i-j-1]

            

        k = i

        for j in range(6,12):

            temp_neighbor_bases[j] = sample['sequence'][k+1]

            k +=1

            

        neighbor_bases.append(temp_neighbor_bases)

        neighbor_bases_df = pd.DataFrame(neighbor_bases,columns = cols[4:])

        

            

    neighbor_bases_df['id'] = sample['id']

    neighbor_bases_df['base'] = list(sample['sequence'][:68])

    neighbor_bases_df['loop_type'] = list(sample['predicted_loop_type'][:68])

    neighbor_bases_df['paired_with'] = list(all_partners_df.loc[idx][:68])

    

    target = pd.DataFrame(columns = ['id','reactivity','deg_Mg_pH10','deg_Mg_50C'])

    target['reactivity'] = sample['reactivity']

    target['deg_Mg_pH10'] = sample['deg_Mg_pH10']

    target['deg_Mg_50C'] = sample['deg_Mg_50C']

    target['id'] = sample['id']

    

    all_target = pd.concat([all_target,target],axis = 0, ignore_index = True)

    extracted_features = pd.concat([extracted_features,neighbor_bases_df], axis = 0, ignore_index = True )

    

extracted_features['is_paired'] = np.where(extracted_features['paired_with'] == -1,0,1)     
extracted_features
from functools import reduce



dfs = [extracted_features,base_percent,pairs_percent,loop_type_percent]

final_train_data = reduce(lambda left,right: pd.merge(left,right), dfs)

final_train_data
final_train_data.columns
dummy_columns = ['base', 'loop_type', 'paired_with', 'prev_base1', 'prev_base2',

                   'prev_base3', 'prev_base4', 'prev_base5', 'prev_base6', 'next_base1',

                   'next_base2', 'next_base3', 'next_base4', 'next_base5', 'next_base6',]



X_train = pd.DataFrame()

for col in dummy_columns:

    X_train = pd.concat([X_train,pd.get_dummies(final_train_data[col],prefix=col)],axis=1)



X_train = pd.concat([X_train, final_train_data[['is_paired', 'G_percent', 'A_percent', 'C_percent', 'U_percent',

                                               'GU_percent', 'GC_percent', 'AU_percent', 'CG_percent', 'UA_percent',

                                               'UG_percent', 'tot_pair_percent', 'E_percent', 'S_percent', 'H_percent',

                                               'B_percent', 'X_percent', 'I_percent', 'M_percent']]],axis=1)

X_train
y_train = all_target[['reactivity','deg_Mg_pH10','deg_Mg_50C']]

y_train
test.head()
def extract_test_features_func():

    bases = []

    for i in range(len(test)):

        count_dict = Counter(test.iloc[i]['sequence'])

        bases.append(count_dict) 

    bases = pd.DataFrame.from_dict(bases)

    base_percent = bases.div(test['seq_length'],axis=0)

    base_percent.columns = ['G_percent','A_percent','C_percent','U_percent']

    base_percent['id'] = test['id']





    all_pairs = []

    all_partners = []

    for idx in range(len(test)):

        sample = test.iloc[idx]

        stack=[]

        pair_freq = {}

        partners = [-1 for i in range(sample['seq_length'])]

        for i in range(len(sample['structure'])):

            if sample['structure'][i] == '(':

                    stack.append(i)

            elif sample['structure'][i] == ')':

                    poped = stack.pop()

                    pair = sample['sequence'][i] + sample['sequence'][poped]

                    partners[i] = sample['sequence'][poped]

                    partners[poped] = sample['sequence'][i]

                    if pair not in pair_freq:

                        pair_freq[pair] = 1

                    else:

                        pair_freq[pair] += 1

        all_pairs.append(pair_freq)     

        all_partners.append(partners)    

    all_partners_df = pd.DataFrame(all_partners)

    pairs_df = pd.DataFrame.from_dict(all_pairs)

    pairs_df.fillna(0,inplace=True)

    pairs_df['tot_pair'] = pairs_df.sum(axis=1)

    pairs_percent = pairs_df.iloc[:,:-1].div(pairs_df['tot_pair'],axis=0)

    pairs_percent['tot_pair_percent'] = pairs_df['tot_pair'].div(test['seq_length']/2,axis=0)

    pairs_percent.columns = ['GU_percent','GC_percent','AU_percent','CG_percent','UA_percent','UG_percent','tot_pair_percent']

    pairs_percent['id'] = test['id']







    loop_type = []

    for i in range(len(test)):

        count_dict = Counter(test.iloc[i]['predicted_loop_type'])

        loop_type.append(count_dict)

    loop_type = pd.DataFrame(loop_type)

    loop_type.fillna(0,inplace=True)

    loop_type_percent = loop_type.div(test['seq_length'],axis=0)

    loop_type_percent.columns = ['E_percent','S_percent','H_percent','B_percent',

                                'X_percent','I_percent','M_percent']

    loop_type_percent['id'] = test['id']







    cols = ['id','base','loop_type','paired_with','prev_base1','prev_base2','prev_base3'

            ,'prev_base4','prev_base5','prev_base6','next_base1','next_base2',

            'next_base3','next_base4','next_base5','next_base6']

    extracted_features = pd.DataFrame(columns = cols)

    for idx in test['index']:

        sample = test.iloc[idx]

        neighbor_bases = []



        for i in range(sample['seq_length']):

            temp_neighbor_bases = [-1 for j in range(12)]

            for j in range(6):

                temp_neighbor_bases[j] = sample['sequence'][i-j-1]

            k = i

            for j in range(6,12):

                temp_neighbor_bases[j] = sample['sequence'][(k+1) % sample['seq_length']]

                k +=1



            neighbor_bases.append(temp_neighbor_bases)

            neighbor_bases_df = pd.DataFrame(neighbor_bases,columns = cols[4:])        

        neighbor_bases_df['id'] = sample['id']

        neighbor_bases_df['base'] = list(sample['sequence'][:sample['seq_length']])

        neighbor_bases_df['loop_type'] = list(sample['predicted_loop_type'][:sample['seq_length']])

        neighbor_bases_df['paired_with'] = list(all_partners_df.loc[idx][:sample['seq_length']])

        extracted_features = pd.concat([extracted_features,neighbor_bases_df], axis = 0, ignore_index = True )

    extracted_features['is_paired'] = np.where(extracted_features['paired_with'] == -1,0,1)

    dfs = [extracted_features,base_percent,pairs_percent,loop_type_percent]

    final_test_data = reduce(lambda left,right: pd.merge(left,right), dfs)





    

    dummy_columns = ['base', 'loop_type', 'paired_with', 'prev_base1', 'prev_base2',

                       'prev_base3', 'prev_base4', 'prev_base5', 'prev_base6', 'next_base1',

                       'next_base2', 'next_base3', 'next_base4', 'next_base5', 'next_base6',]

    X_test = pd.DataFrame()

    for col in dummy_columns:

        X_test = pd.concat([X_test,pd.get_dummies(final_test_data[col],prefix=col)],axis=1)

    X_test = pd.concat([X_test, final_test_data[['is_paired', 'G_percent', 'A_percent', 'C_percent', 'U_percent',

                                                   'GU_percent', 'GC_percent', 'AU_percent', 'CG_percent', 'UA_percent',

                                                   'UG_percent', 'tot_pair_percent', 'E_percent', 'S_percent', 'H_percent',

                                                   'B_percent', 'X_percent', 'I_percent', 'M_percent']]],axis=1)

    

    return X_test
X_test = extract_test_features_func()
X_test
X_train.columns == X_test.columns
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, BatchNormalization

from keras.optimizers import Adam
model = Sequential()

model.add(Dense(512,kernel_initializer = 'uniform', input_dim = X_train.shape[1], activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))





model.add(Dense(64,kernel_initializer = 'uniform',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.2))





model.add(Dense(3, kernel_initializer = 'uniform',activation='linear'))





#Setting the Optimizer

opt=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)





# Compile the network :

model.compile(

    optimizer=opt,

    loss='mse',

    metrics=[keras.metrics.MeanSquaredError()])

model.summary()
model.fit(X_train, y_train, epochs=80, batch_size=64, validation_split = 0.2)
prediction = model.predict(X_test)

prediction
prediction = pd.DataFrame(prediction,columns = ['reactivity','deg_Mg_pH10','deg_Mg_50C'])

prediction
sub
prediction['id_seqpos'] = sub['id_seqpos']

prediction['deg_pH10'] = 0

prediction['deg_50C'] = 0
prediction = prediction[sub.columns]
prediction.to_csv('submission.csv',index=False)