# 타이타닉 데이터를 불러옵니다.

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
# 필요한 모듈을 불러옵니다.

import numpy as np                                # 파이썬 선형대수 라이브러리

from scipy.stats import chi2                      # 카이 제곱 분포 클래스

from scipy.stats import chisquare                 # 카이 제곱 검정 함수

from scipy.stats.contingency import expected_freq # 기대도수 구해주는 함수

import matplotlib.pyplot as plt                   # 데이터 시각화 라이브러리
# 타이타닉 데이터셋에서 무작위로 5개의 데이터를 가져와서 확인합니다.

# -->> 데이터의 직관적인 이해

train.sample(10)
# 관측치를 확인합니다.

ROW = 'Survived'

COL = 'Pclass'

crosstab = pd.crosstab(train[ROW], train[COL], margins=True)

crosstab
observation = np.array([

    [80, 97, 372],

    [136, 87, 119]

])

observation
expectation = np.array([

    [216 * 549 / 891, 184 * 549 / 891, 491 * 549 / 891],

    [216 * 342 / 891, 184 * 342 / 891, 491 * 342 / 891]

])

expectation
row_sum = crosstab.iloc[:-1, -1].values # Vector

col_sum = crosstab.iloc[-1, :-1].values # Vector

total_sum = crosstab.iloc[-1, -1]       # Scalar

expectation = row_sum.reshape(-1, 1) * col_sum.reshape(1, -1) / total_sum

expectation
# a.편차(관측값 - 기대값)를 구합니다.

difference = np.subtract(observation, expectation)

difference
# b.편차의 제곱을 구합니다.

diff_squared = np.power(difference,2)

diff_squared
# c.편차 제곱에 기대값을 나눕니다.

diff_sq_over_exp = np.divide(diff_squared, expectation)

diff_sq_over_exp
# d.총 합을 구합니다,

chi_squared = np.sum(diff_sq_over_exp)

chi_squared
df = 5
# 카이 제곱 분포의 확률 밀도 함수 시각화

plt.plot(chi2(df).pdf(np.arange(0, 30)))

plt.show()
# 카이 제곱 분포의 누적 분포 함수 시각화

plt.plot(chi2(df).cdf(np.arange(0, 30)))

plt.show()
# p_value = 1 - chi2(df).cdf(chi_squared)

p_value = chi2(df).sf(chi_squared)

p_value
observation = [

    [80, 97, 372],

    [136, 87, 119]

]

observation
expectation = expected_freq(observation)

expectation
chi_squared, p_value = chisquare(observation, expectation, axis=None)

print("카이제곱은", chi_squared)

print("p value는", p_value)