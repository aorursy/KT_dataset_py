import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from tqdm import tqdm_notebook
#Количество раз, когда выигрыват несбалансированный датасет

unbalanced_wins = 0

games = 200



for _ in tqdm_notebook(range(games)):

    

    #Несбалансированный датасет из пальца

    class_list = np.random.rand(1, 100000)

    class_list = np.where(class_list > 0.9, 1, 0)



    feature_1 = np.random.rand(1, 100000)

    feature_1 = np.where( (class_list == 1) & (feature_1 < 0.9) , 1, 0 )



    feature_2 = np.random.rand(1, 100000)

    feature_2 = np.where( (class_list == 1) & (feature_2 < 0.9) , 1, 0 )



    dataset = pd.DataFrame()

    dataset['feature_1'] = feature_1[0]

    dataset['feature_2'] = feature_2[0]

    dataset['tgt'] = class_list[0]



    #Второй сбалансированный датасет из первого

    dataset_i = dataset[dataset.tgt == 0].sample(frac=1/9).iloc[:dataset['tgt'].sum()]

    dataset_b = pd.concat([dataset[dataset.tgt == 1], dataset_i])

    

    trainx, testx, trainy, testy = train_test_split(dataset.drop(columns=['tgt']), dataset['tgt'])

    mymodel = DecisionTreeClassifier()

    mymodel.fit(trainx, trainy)

    proby = mymodel.predict_proba(testx)[:,1]

    #ROC метрика несбаланс датасета

    a = roc_auc_score(testy, proby)



    trainx, testx, trainy, testy = train_test_split(dataset_b.drop(columns=['tgt']), dataset_b['tgt'])

    mymodel = DecisionTreeClassifier()

    mymodel.fit(trainx, trainy)

    proby = mymodel.predict_proba(testx)[:,1]

    #ROC метрика сбаланс датасета

    b = roc_auc_score(testy, proby)

    

    if a > b:

        unbalanced_wins += 1

        

print("Unblanaced dataset won:", unbalanced_wins, "out of", games, "games")