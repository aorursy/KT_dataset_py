'''

Acknowledgements



Source:



Pınar Tüfekci, Çorlu Faculty of Engineering, Namık Kemal University, TR-59860 Çorlu, Tekirdağ, Turkey

Email: ptufekci '@' nku.edu.tr



Heysem Kaya, Department of Computer Engineering, Boğaziçi University, TR-34342, Beşiktaş, İstanbul, Turkey

Email: heysem '@' boun.edu.tr



'''

print("")
# 파이썬의 데이터 분석 도구인 Pandas 를 불러옵니다.

# Pandas 를 쉽게 설명드리면, 파이썬으로 엑셀을 다룰 수 있는 도구라고 볼 수 있습니다.

# 이 도구를 앞으로는 pd라는 축약어로 사용하기로 합니다.

import pandas as pd



# matplotlib로 그래프를 그릴 때, 바로 화면에 보이게끔 만들어 줍니다.

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# 판다스의 read_excel로 Folds5x2_pp.xlsx 파일을 읽어옵니다.



# 이 데이터셋은 Sheet1 부터 Sheet5 까지 데이터가 나뉘어져 있습니다.

# 각 시트별로 따로따로 불러와 각기 다른 변수(Sheet1 ~ Sheet5)에 넣어줍니다.



path = '../input/'



path = path + 'Folds5x2_pp.xlsx'

Sheet1 = pd.read_excel(path, sheet_name='Sheet1')

Sheet2 = pd.read_excel(path, sheet_name='Sheet2')

Sheet3 = pd.read_excel(path, sheet_name='Sheet3')

Sheet4 = pd.read_excel(path, sheet_name='Sheet4')

Sheet5 = pd.read_excel(path, sheet_name='Sheet5')
# year 컬럼을 새로 만들어, 1년차는 1, 2년차는 2, ... 5년차는 5를 넣어줍니다.

# 섞였을 때 서로 구분하기 위함입니다.



Sheet1['year'] = 1

Sheet2['year'] = 2

Sheet3['year'] = 3

Sheet4['year'] = 4

Sheet5['year'] = 5
# pd.concat() 안에 리스트 형태로, 묶고자 하는 데이터프레임들을 넣어주면 됩니다.

# 왼쪽(Sheet) 의 아래에 오른쪽(Sheet2, Sheet3, Sheet4, Sheet5) 이 차례대로 붙는다고 생각하시면 됩니다.

# 이어붙인 값을 = 을 통해 combined 라는 변수에 할당합니다.



combined = pd.concat([Sheet1, Sheet2, Sheet3, Sheet4, Sheet5])
combined.describe()
# 기존에 있던 컬럼들의 상호 비교를 위해 모든 변수들과의 관계에 대한 산점도 행렬을 구합니다.

# from pandas.plotting import * 는 판다스 안의 산점도 행렬 함수를 불러오기 위함입니다.

# 산점도 행렬 함수 안에 들어가는 내용은 다음과 같습니다.



# combined -> 보려고 하는 모든 데이터가 있는 데이터프레임

# c -> color 의 약자로, 산점도 안에서 구분하고자 하는 정보, 지금은 combined['label'] 이고, 이는 새 것과 헌 것을 구분합니다.

# figsize -> 보려고 하는 이미지의 크기 정보



from pandas.plotting import *



features = ['AT', 'V', 'AP', 'RH']

scatter_matrix(combined[features], 

                c = combined['PE'], 

                figsize=(25, 25))



# 산점도 행렬은 이후 여러번 사용할 것이기 때문에, 반복을 줄이기 위해 함수로 만들어줍니다.



def scatter(data, label):

    scatter_matrix(data, c = label, figsize=(25, 25))
# 데이터 분석자의 상식과 배경지식을 동원하여, 기존 특징들로부터 도출할 수 있는 새로운 변수입니다.

# 현재 데이터셋의 변수들은 1시간 평균치의 물리량이므로, 

# 1시간 전으로부터 환경변수의 변화율을 구할 수 있습니다.
# .pct_change(1) 을 하여, 이전 row 와의 퍼센트 차이를 구할 수 있습니다.



target_dset = combined



target_dset['1d_AT'] = target_dset['AT'].pct_change(1)

target_dset['1d_V'] = target_dset['V'].pct_change(1)

target_dset['1d_AP'] = target_dset['AP'].pct_change(1)

target_dset['1d_RH'] = target_dset['RH'].pct_change(1)



# .fillna(0) 을 하는 이유는, 첫 번째 행(row) 의 경우 이전 값이 존재하지 않기 때문에 

# NaN 이 있으므로, 이를 채워주기 위함입니다.

combined_change = target_dset.fillna(0)



# 잘 연산되었는지 확인하기 위해 .head() 로 첫 5 번째 행들을 알아봅니다.

combined_change.head()
# for 문으로 데이터프레임을 뽑아오면 컬럼명이 나오게 됩니다.

# 컬럼명을 키워드로 하여 해당 컬럼의 모든 열을 가져올 수 있습니다.

# 이 부분은 combined[i] 이고, 여기서 i 는 컬럼의 이름 입니다.





# 아래 작업을 모든 컬럼에 대해 반복합니다.

for i in combined_change:

    if i == 'PE':

        continue

    # 한 컬럼에서 최소값과 최대값을 구하고

    # 해당 컬럼의 모든 열 값에서 최소값을 뺍니다.

    # 그리고 이를 최대값으로 나누게 되면, 모든 값들이 0 에서 1 사이의 값으로 정규화 됩니다.

    minimum = combined_change[i].min()

    maximum = combined_change[i].max()

    # 최대값이 아닌, minimum 을 뺀 값으로 나누는 이유는 

    # 모든 열에서 minimum 을 뺀 시점에서 이미 최대값이 minimum 만큼 감소하기 때문입니다.

    combined_change[i] = (combined_change[i] - minimum) / (maximum - minimum)

    

combined_change.describe()
# 각 환경변수들의 한 시간 이전 대비 변화율을 뽑아내었으니

# 이를 모든 변수들과 산점도 행렬을 통해 비교합니다.



# 여기에서 볼 사항은 

# 새롭게 만든 환경변수의 변화율이 어떤 분포를 가지는가 입니다.



from pandas.plotting import *

data = combined_change

scatter(data.drop(columns=['PE']), data['PE'])
# 새롭게 만든 환경변수의 변화율 변수만 따로 뽑아서 보겠습니다.

# 1d_AT 의 경우 우리가 파악하고자 하는 값들이 가장 최소 값 인근으로 쏠려 있음을 확인할 수 있습니다.



features = ['1d_AT', '1d_V', '1d_AP', '1d_RH']

scatter(combined_change[features], combined_change['PE'])
# 한 쪽으로 쏠려있는 데이터를 고루 퍼지게 만드는 방법에 대해 소개합니다.

# root 를 씌우는 것과 역수를 취하는 방법에 대해 다룹니다.



import numpy as np

target_col = "1d_AT"



combined_change[target_col + "_sqrt"] = np.sqrt(combined_change[target_col] 

                                                - combined_change[target_col].min() + 1)



combined_change[target_col + "_invert_1"] = np.power(combined_change[target_col] 

                                                     - combined_change[target_col].min() + 1, -1)



combined_change[target_col + "_invert_5"] = np.power(combined_change[target_col] 

                                                     - combined_change[target_col].min() + 1, -5)



combined_change[target_col + "_invert_10"] = np.power(combined_change[target_col] 

                                                      - combined_change[target_col].min() + 1, -10)
# -1 승부터 -10 승까지의 데이터를 보면서, 어떤 것을 선택할지 알아봅니다.

# 히스토그램이 정규분포에 가장 가까운 것을 선택합니다.



features = ['1d_AT', '1d_AT_sqrt', '1d_AT_invert_1', '1d_AT_invert_5', '1d_AT_invert_10', 'RH']

scatter(combined_change[features], combined_change['PE'])
# 나머지 파생변수들은 -1승을 취해줍니다.



target_col = "1d_V"

combined_change[target_col + "_invert_1"] = np.power(combined_change[target_col] 

                                                     - combined_change[target_col].min() + 1, -1)



target_col = "1d_RH"

combined_change[target_col + "_invert_1"] = np.power(combined_change[target_col] 

                                                     - combined_change[target_col].min() + 1, -1)



target_col = "1d_AP"

combined_change[target_col + "_invert_1"] = np.power(combined_change[target_col] 

                                                     - combined_change[target_col].min() + 1, -1)
# 원래 있던 변수와 새로 만든 파생변수들을 넣고, 산점도 행렬을 알아봅니다.



features = ['V',  

            'AP',

            'AT', 

            'RH',

            '1d_V_invert_1', 

            '1d_AP_invert_1', 

            '1d_AT_invert_10', 

            '1d_RH_invert_1']



scatter(combined_change[features], combined_change['PE'])
# 원래 있던 변수와, 새롭게 만든 모든 파생변수들을 넣고

# 발전소의 발전량을 알아보는 예측 모델을 만들기



features = ['V',  

            'AP',

            'AT', 

            'RH',

            '1d_V',

            '1d_AP',

            '1d_AT',

            '1d_RH',

            '1d_V_invert_1', 

            '1d_AP_invert_1', 

            '1d_AT_invert_10', 

            '1d_RH_invert_1']
# 데이터를 1000 개를 기점으로 잘라서 학습 / 예측 합니다.

test_slice = 10000
# 슬라이싱을 적용하여 데이터를 특정 기점으로 잘라줍니다.

# 잘랐을 때, 특징 모음의 경우 컬럼의 개수가 동일해야 합니다.



# X_train -> 학습할 데이터의 특징들의 모음

# Y_train -> 학습할 데이터의 레이블



X_train = data[features][:test_slice]

Y_train = data['PE'][:test_slice]

X_train.shape
# X_test -> 예측할 데이터의 특징 모음

# Y_test -> 예측할 데이터의 레이블



X_test = data[features][test_slice:]

Y_test = data['PE'][test_slice:]

X_test.shape
# 테스트 한 알고리즘을 평가할 점수를 정의합니다.

# RMSE(Root Mean Squared Error) 를 사용합니다.



from sklearn.metrics import make_scorer

import numpy as np



def accuracy(predict, actual):

    predict = np.array(predict)

    actual = np.array(actual)

    

    difference = actual-predict

    squared = difference ** 2

    root = np.sqrt(squared)

    mean = np.mean(root)

    

    score = np.mean(difference)

    

    return score



simple_score = make_scorer(accuracy)

simple_score
# Cross Validation 은 학습한 데이터 안에서의 점수를 평가하는 것입니다.

# model 은 어떤 알고리즘으로 학습할 것인지를 나타냅니다.

# RandomForestRegressor 는 의사결정나무를 여러 개 만들어서 서로 투표하는 방법을 사용합니다.



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(random_state=2000, n_estimators=100, n_jobs=8)

cross_val_score(model, 

                X_train, 

                Y_train, 

                cv=20, 

                scoring=simple_score).mean()
# model.fit -> 선택한 알고리즘에 학습할 데이터를 넣어서 학습시킨다는 의미입니다.

# model.predict -> 학습한 알고리즘에 예측할 데이터를 넣어서 결과를 예측하는 것입니다.

# accuracy -> 예측한 결과인 predict 와 실제 결과를 비교하여 점수를 환산합니다. 



model.fit(X_train, Y_train)

predict = model.predict(X_test)

accuracy(predict, Y_test)
# 학습(fit) 한 이후에는, 이 예측모델이 어떤 기준에 따라 판별했는지를 알 수 있습니다.

# 이는 의사결정나무 기반 알고리즘의 특징이며, 변수들의 우선순위를 파악하는 데 도움이 됩니다.



print ("Features sorted by their score:")

for i in sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), features), reverse=True):

    print(i)
# 1d_RH 가 1d_RH_invert_1 보다 높은 점수를 얻었으므로, 둘 중에 이를 선택합니다.

# 나머지 파생변수들은 invert 한 것이 그렇지 않은 경우보다 높은 점수를 얻었으므로, 이를 선택합니다.



feature_selected = ['V', 

                    'AP', 

                    'AT', 

                    'RH',

                    '1d_V_invert_1', 

                    '1d_AP_invert_1', 

                    '1d_AT_invert_10', 

                    '1d_RH']
# 어떤 컬럼을 선택할지 정했으므로, 선별된 특징들로 학습 및 검증 데이터를 만들어줍니다.

X_train = data[feature_selected][:test_slice]

Y_train = data['PE'][:test_slice]

X_train.shape
X_test = data[feature_selected][test_slice:]

Y_test = data['PE'][test_slice:]

X_test.shape
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(random_state=2000, n_estimators=100, n_jobs=8)

cross_val_score(model, 

                X_train, 

                Y_train, 

                cv=20, 

                scoring=simple_score).mean()
# 학습할 변수들을 선별하니 점수가 더 좋아졌음을 알 수 있습니다.



model.fit(X_train, Y_train)



predict = model.predict(X_test)

accuracy(predict, Y_test)
print ("Features sorted by their score:")

for i in sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), feature_selected), reverse=True):

    print(i)