import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline
## 시각화 한글 인식

from matplotlib import font_manager, rc

font_fname = 'C:\\Windows\\Fonts\\malgun.ttf'

font_name = font_manager.FontProperties(fname=font_fname).get_name()

rc('font', family=font_name)
!dir
## 데이터 읽기

data = pd.read_csv('pokemon.csv')

print(data.shape)

data.head(3)
## column 기본 정보살피기

data.info()
val_length = str(len(data))

pm_change_length = str(len(data[data['percentage_male'].isnull()]))

                 

print("전체 변수 값의 개수 : {0}, 치환해야할 변수 값의 개수 : {1}".format(val_length, pm_change_length))
## 치환 변수들의 값 치환 : -1 (중성)

data.loc[data['percentage_male'].isnull(),'percentage_male'] = -1
## 결과확인 : 98개의 변수 치환 완료

print(data[data['percentage_male']==-1].shape)

data[data['percentage_male']==-1].head(3)
## type2 변수의 탐색

type2_nan = len(data[data['type2'].isnull()])

                 

print("전체 변수 값의 개수 : {0}, 치환해야할 변수 값의 개수 : {1}".format(str(len(data)), str(type2_nan)))



type2_dummy_view = pd.concat([data[data['type2'].isnull()].head(3), data[data['type2'].notnull()].head(3)])

type2_dummy_view
## height_m 변수의 탐색

height_m_nan = data[data['height_m'].isnull()]

                 

print("전체 변수 값의 개수 : {0}, 치환해야할 변수 값의 개수 : {1}".format(str(len(data)), str(len(height_m_nan))))
## weight_kg 변수의 탐색

weight_kg_nan = data[data['weight_kg'].isnull()]

                 

print("전체 변수 값의 개수 : {0}, 치환해야할 변수 값의 개수 : {1}".format(str(len(data)), str(len(weight_kg_nan))))
## 치환을 위한 상관성이 큰 변수 찾기

data.corr()[['height_m','weight_kg']].T
## 두 변수 모두 서로 +0.65의 다른 변수들과 비교할때 가장 높은 상관성을 가진다.

print(data.corr()['height_m'].rank(ascending=False)['weight_kg'])

print(data.corr()['weight_kg'].rank(ascending=False)['height_m'])
## 하지만 두 변수의 null값의 index가 동일하므로 상관변수를 통한 치환은 불가능

print(height_m_nan.index)

print(weight_kg_nan.index)
## 'base_total' 변수가 hegit_m, weight_kg 변수 모두와 2번째로 상관성이 높으므로 이 변수를 통해 값을 치환한다.

print(data.corr()['height_m'].rank(ascending=False)['base_total'])

print(data.corr()['weight_kg'].rank(ascending=False)['base_total'])
## base_total은 Metadata에서 정의되지 않았다.

## -> 데이터탐색 결과 무게, 키, 능력치(공격력, 수비력, 속도 등) 과 높은 상관성을 보인다.

## -> 탐색결과 : 'base_total' = attack','defense','hp','sp_attack','sp_defense','speed'의 합으로 보임

## 따라서 base_total을 통해 weight와 heigth의 유추가 신뢰성 있다 판별가능.

pd.DataFrame(data.corr()['base_total']).T



print("base_total 값 : {0} ".format(data.loc[1]['base_total']))

print("pokemoon 능력치 변수들 값의 합 : {0} ".format(data.loc[1][['attack','defense','hp','sp_attack','sp_defense','speed']].sum()))
## 치환 기준 value 구하기

heigth_notnull = data[data['height_m'].notnull()]

weight_notnull = data[data['weight_kg'].notnull()]



print(heigth_notnull.shape)

mean_value_height = (np.log(heigth_notnull['base_total']) - np.log(heigth_notnull['height_m'])).mean()

mean_value_weight = (np.log(weight_notnull['base_total']) - np.log(weight_notnull['weight_kg'])).mean()

print(mean_value_height)

print(mean_value_weight)
## Height NaN 값 치환

data.loc[data['height_m'].isnull(),'height_m'] = round(np.exp((abs(np.log(data['base_total']) - mean_value_height))),1)

## weight NaN 값 치환

data.loc[data['weight_kg'].isnull(),'weight_kg'] = round(np.exp((abs(np.log(data['base_total']) - mean_value_weight))),1)
## NaN였던 18번 index의 치환 값 확인

data.loc[18][['height_m','weight_kg']]
## NaN 값 치환확인 2

check_height_nan = data[data['height_m'].isnull()]

check_weight_nan = data[data['weight_kg'].isnull()]

                 

print("Height_m 변수의 NaN 값의 개수 : {0}".format(str(len(check_height_nan))))

print("Weight_kg 변수의 NaN 값의 개수 : {0}".format(str(len(check_weight_nan))))
data.describe()
## capture_rate 변수의 이상치 발견

print(data['capture_rate'][1])

print(data['capture_rate'][773])



print("바뀌기전 데이터 차원" + str(data.shape))

data = data[data['capture_rate']!='30 (Meteorite)255 (Core)']

print("바뀐 후 데이터 차원" + str(data.shape))
fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12)) = plt.subplots(3,4)

fig.set_size_inches(15,15)



## Boxplot 그리기

sns.boxplot(data=data['attack'], orient="h", palette="Set2",ax=ax1)

sns.boxplot(data=data['base_egg_steps'], orient="h", palette="Set2",ax=ax2)

sns.boxplot(data=data['base_happiness'], orient="h", palette="Set2",ax=ax3)

sns.boxplot(data=data['base_total'], orient="h", palette="Set2",ax=ax4)

sns.boxplot(data=data['defense'], orient="h", palette="Set2",ax=ax5)

sns.boxplot(data=data['experience_growth'], orient="h", palette="Set2",ax=ax6)

sns.boxplot(data=data['height_m'], orient="h", palette="Set2",ax=ax7)

sns.boxplot(data=data['hp'], orient="h", palette="Set2",ax=ax8)

sns.boxplot(data=data['sp_attack'], orient="h", palette="Set2",ax=ax9)

sns.boxplot(data=data['sp_defense'], orient="h", palette="Set2",ax=ax10)

sns.boxplot(data=data['speed'], orient="h", palette="Set2",ax=ax11)

sns.boxplot(data=data['weight_kg'], orient="h", palette="Set2",ax=ax12)



fig.suptitle("Numberic data의 Box-plot 확인", fontsize=16, color='blue')

data.to_csv('pokemon_clear_data.csv')
pokemon_data = pd.read_csv('pokemon_clear_data.csv', encoding='ISO-8859-1')

pokemon_data = pokemon_data.drop(['Unnamed: 0'],axis=1)

print(pokemon_data.shape)

pokemon_data.head(3)
data['type1'].unique()
def visualization(val,title):

    val.set_title(title, fontsize=18, weight='bold')

    val.tick_params(axis='both', which='major', labelsize=15)
fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11) = plt.subplots(11,1)

fig.set_size_inches(18,100)



sns.countplot(data=data, x='type1', ax=ax1)

visualization(ax1, '속성별 포캣몬 수')

sns.barplot(data=data, x='type1',y='base_total',ax=ax2)

visualization(ax2,'포캣몬 능력치')

sns.barplot(data=data, x='type1',y='against_fire',ax=ax3)

visualization(ax3,'불 상성')

sns.barplot(data=data, x='type1',y='against_water',ax=ax4)

visualization(ax4,'물 상성')

sns.barplot(data=data, x='type1',y='experience_growth',ax=ax5)

visualization(ax5,'필요경험치')



## 성비 데이터 변경(중성 제거)

data2 = data[data['percentage_male']!=-1]

sns.barplot(data=data2, x='type1',y='percentage_male',ax=ax6)

visualization(ax6,'성비(0~1 : 여성~남성)')

sns.barplot(data=data, x='type1',y='weight_kg',ax=ax7)

visualization(ax7,'몸무게')

sns.barplot(data=data, x='type1',y='height_m',ax=ax8)

visualization(ax8,'키')

sns.barplot(data=data, x='type1',y='is_legendary',ax=ax9)

visualization(ax9,'레전드 포캣몬 유무')

sns.barplot(data=data, x='type1',y='base_egg_steps',ax=ax10)

visualization(ax10,'진화 필요 걸음수')

sns.barplot(data=data, x='type1',y='base_egg_steps',ax=ax11)

visualization(ax11,'포획 성공율')

from sklearn.cluster import KMeans

import mglearn

from tqdm import tqdm
element_differ = ['against_bug', 'against_dark', 'against_dragon',

       'against_electric', 'against_fairy', 'against_fight', 'against_fire',

       'against_flying', 'against_ghost', 'against_grass', 'against_ground',

       'against_ice', 'against_normal', 'against_poison', 'against_psychic',

       'against_rock', 'against_steel', 'against_water']

cluster_value = data[element_differ]
## 속성(18개)을 기준으로 군집화

kmeans = KMeans(n_clusters=18, random_state=1)

kmeans.fit(cluster_value)
## 각각의 포캣몬 별로 "사전에 정의된 속성(Metadata) <-> K-means로 정의한 속성" 을 비교

cluster_table = pd.concat([ pokemon_data[['name','type1']], pd.DataFrame(kmeans.labels_)],axis=1)

cluster_table.columns = ['포캣몬이름','사전에 정의된 속성', '상성으로 군집화 한 속성']

cluster_table
## 각각의 군집(18개)가 실제 어떤 속성에 해당되는지 알아본다.

## 이때, 기준은 속성의 포캣몬이 "가장 많이 속한" 군집의 번호로한다.



dict_label = {}



for index in tqdm(pokemon_data['type1'].unique()):

    temp = cluster_table[cluster_table['사전에 정의된 속성']==index]['상성으로 군집화 한 속성'].value_counts().index[0]

    

    dict_label[index] = temp

    
dict_label.items()
## 각각의 사전적 속성의 포켓몬 개체 수

type1_count = pd.DataFrame(pokemon_data['type1'].value_counts()).reset_index()

type1_count.T
## 새로운 군집에 따른 포켓몬 개체 수

# 10,12,15,16 의 군집은 'other'란 속성으로 정의하여 확실한 사전적 속성이 없는 변수들이다.

cluster_label = pd.DataFrame(kmeans.labels_)

cluster_count = pd.DataFrame(cluster_label[0].value_counts()).reset_index()



for label, c_number in dict_label.items():

    cluster_count.loc[cluster_count['index']==c_number,'index'] = label

    

cluster_count.T
## Merge()를 사용하여 보기좋게 테이블 합치기

comp_type_table = pd.merge(type1_count, cluster_count, on='index')



## Table 전처리

comp_type_table.loc[comp_type_table['index']=='electric','index'] = 'electric_ground'

comp_type_table.loc[comp_type_table['index']=='psychic','index'] = 'psychic_ghost'

comp_type_table.loc[comp_type_table['index']=='normal','index'] = 'normal_ice_dark'

comp_type_table.columns=['new_type','기존 값','K-means 값']

comp_type_table.index = comp_type_table['new_type']

comp_type_table = comp_type_table.drop(['new_type'],axis=1)



comp_type_table.T
dict_label.items()
cluster_table2 = cluster_table.copy()

cluster_table2.columns = ['poke_name','raw_type','new_type']



## 기존 속성(raw_type)과 군집 속성(new_type)의 비교를 위해 raw_type을 int형인 raw_type2로 치환해준다.

for index, new_type in dict_label.items():

    cluster_table2.loc[cluster_table2['raw_type']==index ,'raw_type2'] = new_type



## 자료형을 int로 맞춰준다.

cluster_table2['raw_type2'] = cluster_table2['raw_type2'].apply(int)



print(cluster_table2.shape)

cluster_table2.head(3)
## 이종 or 동일로 "이종포켓몬"을 정의한다.

cluster_table2['check'] = cluster_table2['new_type']-cluster_table2['raw_type2']

cluster_table2.loc[cluster_table2['check']==0,'check'] = '순종'

cluster_table2.loc[cluster_table2['check']!='순종','check'] = '이종'



print(cluster_table2.shape)

cluster_table2.head(10)
fusion_pokemon = cluster_table2[cluster_table2['check'] == '이종']

print(fusion_pokemon.shape)

pure_pokemon = cluster_table2[cluster_table2['check'] == '순종']

print(pure_pokemon.shape)

pure_pokemon.head(3)
## merge를 위해 특정 칼럼 명 변경

fusion_pokemon=fusion_pokemon.rename(columns = {'poke_name':'name'})
## 기존 능력치 데이터와 추출한 '이종'포켓몬 데이터의 병합

## 병합기준 : 포켓몬이름

fusion_poke_hap = pd.merge(fusion_pokemon,pokemon_data, on ='name')



print(fusion_poke_hap.shape)

fusion_poke_hap.head(3)
dict_label2 = {'bug': 2,'dark_ice_normal':8,'dragon': 4,'electric_ground': 11,'fairy': 7,'fighting': 0,'fire': 5,'flying': 13,

               'ghost_psychic': 3,'grass': 6,'poison': 17,'rock': 14,'steel': 9,'water': 1,'other1':10, 'other2':12, 'other3':15,'other4':16}
for index, number in dict_label2.items():

    fusion_poke_hap.loc[fusion_poke_hap['new_type'] == number,'new_type'] = index



print(fusion_poke_hap.shape)

fusion_poke_hap.head(5)
## 전처리

for index in ['other1','other2','other3','other4']:

    fusion_poke_hap.loc[fusion_poke_hap['new_type'] == index, 'new_type'] = -1

    

fusion_poke_hap = fusion_poke_hap[fusion_poke_hap['new_type'] != -1]



print(fusion_poke_hap.shape)

fusion_poke_hap.head(5)
## merge를 위해 특정 칼럼 명 변경

pure_pokemon=pure_pokemon.rename(columns = {'poke_name':'name'})
## 기존 능력치 데이터와 추출한 '동종'포켓몬 데이터의 병합

## 병합기준 : 포켓몬이름

pure_poke_hap = pd.merge(pure_pokemon,pokemon_data, on ='name')



print(pure_poke_hap.shape)

pure_poke_hap.head(3)
dict_label2 = {'bug': 2,'dark_ice_normal':8,'dragon': 4,'electric_ground': 11,'fairy': 7,'fighting': 0,'fire': 5,'flying': 13,

               'ghost_psychic': 3,'grass': 6,'poison': 17,'rock': 14,'steel': 9,'water': 1}
## raw_type2는 사전적 속성을 숫자로 변경한것이다. 



for index, number in dict_label2.items():

    pure_poke_hap.loc[pure_poke_hap['raw_type2'] == number,'raw_type2'] = index



print(pure_poke_hap.shape)

pure_poke_hap.head(5)
## 능력치 비교 시각화

def visualization(val,title):

    val.set_title(title, fontsize=18, weight='bold')

    val.tick_params(axis='both', which='major', labelsize=15)

    val.set_xlim(0,600)

    val.set_ylabel('')



fig, (ax1,ax2) = plt.subplots(1,2)

fig.set_size_inches(18,10)





sns.barplot(data=pure_poke_hap.sort_values('raw_type2'), y='raw_type2',x='base_total',ax=ax1)

visualization(ax1,'순종 포켓몬 능력치')



sns.barplot(data=fusion_poke_hap.sort_values('new_type'), y='new_type',x='base_total',ax=ax2)

visualization(ax2,'이종 포켓몬 능력치')
## 순종, 이종 포켓몬 능력치 상세 비교

mean_pure = pd.DataFrame(pure_poke_hap.groupby('raw_type2')['base_total'].mean().apply(int))

mean_fusion = pd.DataFrame(fusion_poke_hap.groupby('new_type')['base_total'].mean().apply(int))

performance_comp = pd.concat([mean_pure, mean_fusion],axis=1)

performance_comp.columns = ['순종','이종']

performance_comp.T
## 능력치의 평균을 이용하므로 t-test를 통한 검정



import scipy.stats as stats

stats.ttest_ind(performance_comp['이종'], performance_comp['순종'])   
pokemon_data.columns
def visualization(val,title):

    val.set_title(title, fontsize=18, weight='bold')

    val.tick_params(axis='both', which='major', labelsize=15)

    if(val == ax1 or val==ax2):

        val.set_xlim(0,600)

    val.set_ylabel('')

    if(val == ax3 or val==ax4):

        val.set_xlim()
## 능력치 비교 시각화





fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2)

fig.set_size_inches(18,30)



## 능력치 비교

sns.barplot(data=pure_poke_hap.sort_values('raw_type2'), y='raw_type2',x='base_total',ax=ax1)

visualization(ax1,'순종 포켓몬 능력치')

sns.barplot(data=fusion_poke_hap.sort_values('new_type'), y='new_type',x='base_total',ax=ax2)

visualization(ax2,'이종 포켓몬 능력치')



##가성비 비교(가성비 = 능력치/진화필요 횟수)

test = pure_poke_hap['base_total']/pure_poke_hap['base_egg_steps']

sns.barplot(data=pure_poke_hap.sort_values('raw_type2'), y='raw_type2',x=test,ax=ax3)

visualization(ax3,'순종 포켓몬 가성비')

sns.barplot(data=fusion_poke_hap.sort_values('new_type'), y='new_type',x=test,ax=ax4)

visualization(ax4,'이종 포켓몬 가성비')



##성장성 비교(경험치 증가율) 높을 수록 미래의 성장성이 높다.(=키우기 힘들다.)

sns.barplot(data=pure_poke_hap.sort_values('raw_type2'), y='raw_type2',x='experience_growth',ax=ax5)

visualization(ax5,'순종 포켓몬 성장성')

sns.barplot(data=fusion_poke_hap.sort_values('new_type'), y='new_type',x='experience_growth',ax=ax6)

visualization(ax6,'이종 포켓몬 성장성')
## 순종, 이종 포켓몬 능력치 상세 비교

mean_pure = pd.DataFrame(pure_poke_hap.groupby('raw_type2')['base_total'].mean().apply(int))

mean_fusion = pd.DataFrame(fusion_poke_hap.groupby('new_type')['base_total'].mean().apply(int))

performance_comp = pd.concat([mean_pure, mean_fusion],axis=1)

performance_comp.columns = ['순종능력치','이종능력치']



performance_comp['compare'] = performance_comp['순종능력치']-performance_comp['이종능력치']

performance_comp['변동수준'] = abs(performance_comp['compare'])



for temp in performance_comp.index:

    if(performance_comp[performance_comp.index==temp]['compare'].values[0] > 0):

        performance_comp.loc[performance_comp.index==temp, 'compare'] = '순종승'

    else:

        performance_comp.loc[performance_comp.index==temp, 'compare'] = '이종승'

        

performance_comp.T
## 순종, 이종 포켓몬 가성비 상세 비교

pure_poke_hap['pure_gaseong'] = pure_poke_hap['base_total']/pure_poke_hap['base_egg_steps']

fusion_poke_hap['fusion_gaseong'] = fusion_poke_hap['base_total']/fusion_poke_hap['base_egg_steps']



mean_pure = pd.DataFrame(pure_poke_hap.groupby('raw_type2')['pure_gaseong'].mean()*100)

mean_fusion = pd.DataFrame(fusion_poke_hap.groupby('new_type')['fusion_gaseong'].mean()*100)

performance_comp = pd.concat([mean_pure, mean_fusion],axis=1)

performance_comp.columns = ['순종가성비','이종가성비']



performance_comp['compare'] = performance_comp['순종가성비']-performance_comp['이종가성비']

performance_comp['변동수준'] = abs(performance_comp['compare'])



for temp in performance_comp.index:

    if(performance_comp[performance_comp.index==temp]['compare'].values[0] > 0):

        performance_comp.loc[performance_comp.index==temp, 'compare'] = '순종승'

    else:

        performance_comp.loc[performance_comp.index==temp, 'compare'] = '이종승'





performance_comp.T
## 순종, 이종 포켓몬 능력치 상세 비교

## 높을수록 성장 경험치가 많이 필요하다 = 키우기 힘들다. but 후반에 강해질 확률이 높다.



mean_pure = pd.DataFrame(pure_poke_hap.groupby('raw_type2')['experience_growth'].mean().apply(int))

mean_fusion = pd.DataFrame(fusion_poke_hap.groupby('new_type')['experience_growth'].mean().apply(int))

performance_comp = pd.concat([mean_pure, mean_fusion],axis=1)

performance_comp.columns = ['순종성장성','이종성장성']



performance_comp['compare'] = performance_comp['순종성장성']-performance_comp['이종성장성']

performance_comp['변동수준'] = abs(performance_comp['compare'])



for temp in performance_comp.index:

    if(performance_comp[performance_comp.index==temp]['compare'].values[0] > 0):

        performance_comp.loc[performance_comp.index==temp, 'compare'] = '순종승'

    else:

        performance_comp.loc[performance_comp.index==temp, 'compare'] = '이종승'

        

performance_comp.T
final_pokemon = fusion_poke_hap[fusion_poke_hap['new_type'] == 'fighting']

print(final_pokemon.shape)

final_pokemon
final_pokemon['gaseong'] = final_pokemon['base_total']/final_pokemon['base_egg_steps']
final_pokemon[['name','base_total','experience_growth','gaseong','capture_rate']]