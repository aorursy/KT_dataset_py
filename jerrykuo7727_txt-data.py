import os
import pandas as pd
TRAINING_PATH = '../input/hw2data/hw2data/training/'
TESTING_PATH = '../input/hw2data/hw2data/testing/'
categories = [dirname for dirname in os.listdir(TRAINING_PATH) if dirname[-4:] != '_cut']
print(len(categories), str(categories))
category2idx = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job':  3, 'WomenTalk': 4,
                  'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}
%%time
train_list = []

for category in categories:
    category_idx = category2idx[category]
    category_path = TRAINING_PATH + category + '_cut/'
    
    for filename in os.listdir(category_path):
        filepath = category_path + filename
        
        with open(filepath, encoding='utf-8') as file:
            words = file.read().strip().split(' / ')
            train_list.append([words, category_idx])
train_df = pd.DataFrame(train_list, columns=["text", "category"])
print("Shape:", train_df.shape)
train_df.sample(5)
%%time
train_df.to_pickle('train.pkl')
%time
pickle_df = pd.read_pickle('train.pkl')
train_df.equals(pickle_df)
%%time
test_list = []

for idx in range(1000):
    filepath = TESTING_PATH + str(idx) + '.txt'
    
    with open(filepath, encoding='utf-8') as file:
        words = file.read().strip().split(' / ')
        test_list.append([idx, words])
test_df = pd.DataFrame(test_list, columns=["id", "text"])
print("Shape:", test_df.shape)
test_df.sample(5)
test_df.to_pickle('test.pkl')
pickle_df = pd.read_pickle('test.pkl')
test_df.equals(pickle_df)