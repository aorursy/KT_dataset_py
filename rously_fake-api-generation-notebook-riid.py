import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import random
from copy import deepcopy
import _pickle as pickle

def save(file,name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'wb')
    else:
        outfile = open(name+'.pickle', 'wb')
    pickle.dump(file, outfile, protocol=4)
    outfile.close
    
def load(name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'rb')
    else:
        outfile = open(name+'.pickle', 'rb')
    file = pickle.load(outfile)
    outfile.close
    return file
train = pd.DataFrame()
for chunk in pd.read_csv('../input/riiid-test-answer-prediction/train.csv', chunksize=1000000, low_memory=False):
    train = pd.concat([train, chunk], ignore_index=True)
class FakeDataGenerator:
    
    def __init__(self):
        '''
        self.data will be a dictionnary to iterate over the stored data
        self.all_rows will be the rows of the train set that are used by the generato
        self.data_index will be all the data available in the dataset        
        '''
        self.data = None
        self.all_rows = None
        self.data_index = None
        return None
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        sub = sample[['row_id', 'group_num']].copy()
        sub['answered_correctly'] = np.zeros(sub.shape[0])+0.5
        return (sample, sub)
    
    
    def load(self, save_name):
        self.data,self.all_rows = load(save_name)
        self.data_index = np.array(list(self.data.keys()))
    
    def build_from_train(self, train, n_users, beginner_rate = 0.3, save_name = 'fake_train_generator'):
        """
        train will be the training set you loaded
        n_users is a number of user from whom you will sample the data
        beginner_rate is the rate of these users who will begin their journey during test
        save_name : the name under which the item will be saved
        """
        
        ## Sampling a restricted list of users
        user_list = train['user_id'].unique()
        test_user_list = np.random.choice(user_list, size = n_users)
        train.index = train['user_id']
        test_data_non_filter = train.loc[test_user_list]
        test_data_non_filter.index = list(range(test_data_non_filter.shape[0]))
        
        ## building a dictionnary with all the rows and container id from a user
        dico_user = {}
        def agg(x):
            return [elt for elt in x]
        
        print("Generating user dictionnary")
        for user, frame in tqdm(test_data_non_filter.groupby('user_id'), total =test_data_non_filter['user_id'].nunique()):
            if frame.shape[0] > 0:
                dico_user[user] = {}

                dico_user[user]['min_indice'] = frame['task_container_id'].min()
                dico_user[user]['max_indice'] = frame['task_container_id'].max()

                r = random.uniform(0,1)
                if r < beginner_rate:
                    dico_user[user]['current_indice'] = dico_user[user]['min_indice']
                else:
                    dico_user[user]['current_indice'] = random.randint(dico_user[user]['min_indice'],dico_user[user]['max_indice']-2)

                row_ids = frame[['task_container_id','row_id']].groupby('task_container_id').agg(agg)
                row_ids = row_ids.to_dict()['row_id']
                dico_user[user]['row_ids'] = row_ids

        work_dico = deepcopy(dico_user)
        
        ## Choosing batch_data to generate
        work_dico = deepcopy(dico_user)
        batches = {}

        all_rows = []
        batch_number = 0
        
        print('Creating batches')
        while len(work_dico)> 1:

            size = random.randint(20,500)
            size = min(size, len(work_dico))


            batch = []

            users = np.random.choice(np.array(list(work_dico.keys())),replace = False,  size = size)

            for u in users:
                try:
                    batch.extend(work_dico[u]['row_ids'][work_dico[u]['current_indice']])
                    all_rows.extend(work_dico[u]['row_ids'][work_dico[u]['current_indice']])
                    work_dico[u]['current_indice'] += 1
                    if work_dico[u]['current_indice'] == work_dico[u]['max_indice']:
                        work_dico.pop(u)
                except:
                    work_dico.pop(u)

            batches[batch_number] = batch
            batch_number += 1
        
        ## building data

        data = {}
        
        print("Building dataset")
        test_data_non_filter.index = test_data_non_filter['row_id']
        for i in tqdm(batches):
            current_data = test_data_non_filter.loc[np.array(batches[i])]
            current_data['group_num'] = i

            current_data['prior_group_answers_correct'] = [np.nan for elt in range(current_data.shape[0])]
            current_data['prior_group_responses'] = [np.nan for elt in range(current_data.shape[0])]

            if i != 0:
                current_data['prior_group_answers_correct'].iloc[0] = saved_correct_answer
                current_data['prior_group_responses'].iloc[0] = saved_answer

            saved_answer = str(list(current_data[current_data['content_type_id'] == 0]['user_answer'].values))
            saved_correct_answer = str(list(current_data[current_data['content_type_id'] == 0]['answered_correctly'].values))
            current_data = current_data.drop(columns = ['user_answer', 'answered_correctly'])

            data[i] = current_data

        save((data,np.array(all_rows)) , save_name)
        
        self.data = data
        self.all_rows = np.array(all_rows)
        self.data_index = np.array(list(data.keys()))
        print('finished')
env = FakeDataGenerator()
env.build_from_train(train, 100, beginner_rate = 0.3, save_name = 'fake_train_generator')
env.load('fake_train_generator')
env.data_index
for idx in env.data_index:
    (test_df, sample_prediction_df) = env[idx]
(test_df, sample_prediction_df) = env[5]
test_df
%%timeit
(test_df, sample_prediction_df) = env[1]
