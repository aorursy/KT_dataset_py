'''

Acknowledgements



This dataset is publicly available for anyone to use under the following terms.



von Birgelen, Alexander; Buratti, Davide; Mager, Jens; Niggemann, Oliver: Self-Organizing Maps for Anomaly Localization and Predictive Maintenance in Cyber-Physical Production Systems. In: 51st CIRP Conference on Manufacturing Systems (CIRP CMS 2018) CIRP-CMS, May 2018.



Paper available open access: https://authors.elsevier.com/sd/article/S221282711830307X



IMPROVE has received funding from the European Union's Horizon 2020 research and innovation programme under Grant Agreement No. 678867



'''

print('\n')
# 파이썬의 데이터 분석 도구인 Pandas 를 불러옵니다.

# Pandas 를 쉽게 설명드리면, 파이썬으로 엑셀을 다룰 수 있는 도구라고 볼 수 있습니다.

# 이 도구를 앞으로는 pd라는 축약어로 사용하기로 합니다.



import pandas as pd


# 판다스의 read_csv로 NewBlade001.csv, WornBlade001.csv 파일을 읽어옵니다.

# 읽어온 데이터를 각각 new, old 이라는 이름의 변수에 할당합니다.



path = '../input/'



new = pd.read_csv(path+"NewBlade001.csv")

old = pd.read_csv(path+"WornBlade001.csv")
# 새로 불러온 변수 new의 첫 5 개 행을 확인합니다.



new.head()
# 새로 불러온 변수 old의 첫 5 개 행을 확인합니다.



old.head()
# .describe() 를 통해 우리는, 기초 통계량을 알 수 있습니다.



new.describe()
# 새 것과 헌 것의 기초 통계량만 잘 비교해도 수치상 다른 것들이 있죠?



old.describe()
# label 컬럼을 새로 만들어, new 는 0을, old 는 1 을 할당합니다.

# 섞였을 때 서로 구분하기 위함입니다.



new['label'] = 0

old['label'] = 1
# pd.concat() 안에 리스트 형태로, 묶고자 하는 데이터프레임들을 넣어주면 됩니다.

# 왼쪽(new) 의 아래에 오른쪽(old) 이 붙는다고 생각하시면 됩니다.

# 이어붙인 값을 = 을 통해 combined 라는 변수에 할당합니다.

combined = pd.concat([new, old])



# 잘 실행되었는지 확인하기 위해 describe() 를 사용합니다.

# 아래에 row 가 추가된 것이므로, count 행이 2048 에서 4096 으로 바뀐 것을 확인하면 됩니다.

combined.describe()
# combined.columns 는 combined 이라는 변수에 담긴 데이터프레임 값의 컬럼들 입니다.

# 이 위치에 = 을 통해 리스트로 내가 새롭게 정하고자 하는 컬럼들을 집어넣을 수 있습니다.

# 기존 컬럼들의 이름이 너무 길어서... 줄여서 넣습니다.



combined.columns = ['Time', 'Torq', "Cut_Err", "Cut_Pos", "Cut_Spd", 

                    "Film_Pos", "Film_Spd", "Film_Err", "label"]
# 기존에 있던 컬럼들의 상호 비교를 위해 모든 변수들과의 관계에 대한 산점도 행렬을 구합니다.

# from pandas.plotting import * 는 판다스 안의 산점도 행렬 함수를 불러오기 위함입니다.

# 산점도 행렬 함수 안에 들어가는 내용은 다음과 같습니다.



# combined -> 보려고 하는 모든 데이터가 있는 데이터프레임

# c -> color 의 약자로, 산점도 안에서 구분하고자 하는 정보, 지금은 combined['label'] 이고, 

# 이는 새 것과 헌 것을 구분합니다.

# figsize -> 보려고 하는 이미지의 크기 정보



from pandas.plotting import *

result = scatter_matrix(combined, 

                        c = combined['label'], 

                        figsize=(25, 25))
# 데이터 분석자의 상식과 배경지식을 동원하여, 기존 특징들로부터 도출할 수 있는 새로운 변수입니다.

# 속도라는 물리량이 있고, 초당 250회 획득하는 변수이므로 

# 해당 순간의 힘을 구할 수 있습니다.



# 미분의 개념을 적용합니다.

# 속도의 변화량을 시간변화량으로 나누어 힘을 구합니다.

# 엄밀하게는 비례상수인 m 이 들어가지만, 상수이기 때문에 무시합니다.



# 새로운 파생변수의 이름을 Force 가 아닌 Acc(가속도) 로 해도 무관합니다.
# .diff(1) 을 하여, 이전 row 와의 차이를 구하고

# 데이터 획득 주기인 0.004 초로 나눕니다.



# .loc[combined['label'] == 0] 은 new blade 안에서만 연산하기 위함입니다.

# .loc[combined['label'] == 1] 은 old blade 안에서만 연산하기 위함입니다.



# Blade 기준으로는 Cut_ 수식어를, 언와인더 기준으로는 Film_ 을 붙여줍니다.



combined['Cut_Force'] = combined["Cut_Spd"].loc[combined["label"] == 0].diff(1) / 0.004

combined['Cut_Force'] = combined["Cut_Spd"].loc[combined["label"] == 1].diff(1) / 0.004



combined['Film_Force'] = combined["Film_Spd"].loc[combined["label"] == 0].diff(1) / 0.004

combined['Film_Force'] = combined["Film_Spd"].loc[combined["label"] == 1].diff(1) / 0.004



# .fillna(0) 을 하는 이유는, 첫 번째 행(row) 의 경우 이전 값이 존재하지 않기 때문에 

# NaN 이 있으므로, 이를 채워주기 위함입니다.



combined.fillna(0, inplace=True)
# 잘 연산되었는지 확인하기 위해 .head() 로 첫 5 번째 행들을 알아봅니다.



combined[["Cut_Force", "Film_Force", 'label']].head()
# for 문으로 데이터프레임을 뽑아오면 컬럼명이 나오게 됩니다.

# 컬럼명을 키워드로 하여 해당 컬럼의 모든 열을 가져올 수 있습니다.

# 이 부분은 combined[i] 이고, 여기서 i 는 컬럼의 이름 입니다.





# 아래 작업을 모든 컬럼에 대해 반복합니다.

for i in combined:

    # 한 컬럼에서 최소값과 최대값을 구하고

    # 해당 컬럼의 모든 열 값에서 최소값을 뺍니다.

    # 그리고 이를 최대값으로 나누게 되면, 모든 값들이 0 에서 1 사이의 값으로 정규화 됩니다.

    minimum = combined[i].min()

    maximum = combined[i].max()

    # 최대값이 아닌, minimum 을 뺀 값으로 나누는 이유는 

    # 모든 열에서 minimum 을 뺀 시점에서 이미 최대값이 minimum 만큼 감소하기 때문입니다.

    combined[i] = (combined[i] - minimum) / (maximum - minimum)

    

combined.describe()
# 블레이드와 언와인더에 작용하는 해당 순간의 Force 정보를 만들었으니

# 이를 모든 변수들과 산점도 행렬을 통해 비교합니다.



# 여기에서 볼 사항은 



# Time 과 다른 변수들간의 그래프

# 새롭게 만든 Force 와 다른 변수들간의 그래프 입니다.



from pandas.plotting import *

result = scatter_matrix(combined, 

                        c = combined['label'], 

                        figsize=(25, 25))
# 데이터 시각화 도구인 matplotlib를 불러와 이를 줄여 plt라고 사용합니다.

import matplotlib.pyplot as plt



# Time 과 다른 변수들간의 산점도를 lineplot 으로 확인하고자 합니다.

# 그래프가 y 축만 다르고 나머지가 동일하기 때문에

# 반복을 줄이기 위해 그래프를 그려주는 함수를 만들어줍니다.



def line_plot(y, log=False):

    # y 값만 입력받는 함수입니다.

    # .loc[combined['label'] == 0] 

    x = "Time"

    x1 = combined[x].loc[combined['label'] == 0]

    y1 = combined[y].loc[combined['label'] == 0]



    x2 = combined[x].loc[combined['label'] == 1]

    y2 = combined[y].loc[combined['label'] == 1]

    

    plt.plot(x1, y1, label= y + "_New") 

    plt.plot(x2, y2, label= y + "_Old")

    

    if log:

        plt.yscale('log')

    plt.legend(loc='lower left')

    plt.show()
line_plot("Torq")
target_cols = ['Torq', "Cut_Err", "Cut_Pos", "Cut_Spd", "Cut_Force", 

               "Film_Pos", "Film_Spd", "Film_Err", "Film_Force"]



for target in target_cols:

    line_plot(target)
# 함수의 입력값으로 log=True 를 넣어서 로그스케일로 확인합니다.

line_plot("Cut_Force", log=True)
# matplotlib의 subplots를 사용합니다. 이 함수는 여러 개의 시각화를 한 화면에 띄울 수 있도록 합니다.

# 이번에는 2x2으로 총 4개의 시각화를 한 화면에 띄웁니다.



figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)



# 시각화의 전체 사이즈는 18x8로 설정합니다.

figure.set_size_inches(18, 15)



combined.plot.scatter(x = 'Film_Force', y = 'Film_Spd', c = 'label', colormap = 'viridis', ax=ax1)

combined.plot.scatter(x = 'Film_Force', y = 'Cut_Spd', c = 'label', colormap = 'viridis', ax=ax2)

combined.plot.scatter(x = 'Cut_Force', y = 'Film_Spd', c = 'label', colormap = 'viridis', ax=ax3)

combined.plot.scatter(x = 'Cut_Force', y = 'Cut_Spd', c = 'label', colormap = 'viridis', ax=ax4)