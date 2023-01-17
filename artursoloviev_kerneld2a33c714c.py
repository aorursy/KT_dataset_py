import pandas as pd

gold = pd.read_csv("../input/ysda-ml-2020-lab-1-soloviev-a-v/gold.csv")

lh = pd.read_csv("../input/ysda-ml-2020-lab-1-soloviev-a-v/lh.csv")

train = pd.read_csv("../input/ysda-ml-2020-lab-1-soloviev-a-v/train.csv")



print('gold.csv contains data like this:')

print(gold.loc[:3])



print('lh.csv contains data like this:')

print(lh.loc[:3])



print('train.csv contains data like this:')

print(train.loc[:3])
# Фильтрация gold.csv и lh.csv по полю player_0, 

# исходя из предположения, что если данные отсутствуют для player_0, то они отсутствуют и для прочих player_'ов. 

# Это предположение основано на пролистывании случайных фрагментов обоих csv - своего рода hold-out =)



filtered_gold = gold[gold.player_0 != -1]

filtered_lh = lh[lh.player_0 != -1]



print('After filtering absent data, gold.csv contains data like this:')

print(filtered_gold)



print('After filtering absent data, lh.csv contains data like this:')

print(filtered_lh)

filtered_gold_on_600_second = filtered_gold[filtered_gold['time'] == 600]

filtered_lh_on_600_second = filtered_lh[filtered_lh['time'] == 600]



gold_and_lh_on_600_second = pd.DataFrame.merge(filtered_gold_on_600_second, filtered_lh_on_600_second, \

                                               left_on = 'mid', right_on = 'mid', suffixes = ('_gold', '_lh'))



stats_squeeze = pd.DataFrame(\

    {'Team' : ['Radiant', 'Dire'],\

         'MaxGold' : [gold_and_lh_on_600_second[['player_0_gold', 'player_1_gold', 'player_2_gold', 'player_3_gold', 'player_4_gold']].max(),\

                      gold_and_lh_on_600_second[['player_5_gold', 'player_6_gold', 'player_7_gold', 'player_8_gold', 'player_9_gold']].max()],\

         'MinGold' : [gold_and_lh_on_600_second[['player_0_gold', 'player_1_gold', 'player_2_gold', 'player_3_gold', 'player_4_gold']].min(),\

                     gold_and_lh_on_600_second[['player_5_gold', 'player_6_gold', 'player_7_gold', 'player_8_gold', 'player_9_gold']].min()],\

         'SumOfGold' : [gold_and_lh_on_600_second[['player_0_gold', 'player_1_gold', 'player_2_gold', 'player_3_gold', 'player_4_gold']].sum(),\

                     gold_and_lh_on_600_second[['player_5_gold', 'player_6_gold', 'player_7_gold', 'player_8_gold', 'player_9_gold']].sum()],\

         'MaxLastHits' : [gold_and_lh_on_600_second[['player_0_lh', 'player_1_lh', 'player_2_lh', 'player_3_lh', 'player_4_lh']].max(),\

                 gold_and_lh_on_600_second[['player_5_lh', 'player_6_lh', 'player_7_lh', 'player_8_lh', 'player_9_lh']].max()],\

         'MinLastHits' : [gold_and_lh_on_600_second[['player_0_lh', 'player_1_lh', 'player_2_lh', 'player_3_lh', 'player_4_lh']].min(),\

                     gold_and_lh_on_600_second[['player_5_lh', 'player_6_lh', 'player_7_lh', 'player_8_lh', 'player_9_lh']].min()],\

         'SumOfLastHits' : [gold_and_lh_on_600_second[['player_0_lh', 'player_1_lh', 'player_2_lh', 'player_3_lh', 'player_4_lh']].sum(),\

                     gold_and_lh_on_600_second[['player_5_lh', 'player_6_lh', 'player_7_lh', 'player_8_lh', 'player_9_lh']].sum()]\

    }\

)



total_train_set = pd.DataFrame.merge(train.sort_values(by = 'mid'), gold_and_lh_on_600_second)

print(total_train_set)


