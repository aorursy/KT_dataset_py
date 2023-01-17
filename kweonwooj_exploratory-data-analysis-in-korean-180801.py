# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# 코드 1-1. Training Data 읽어오기

import pandas as pd
import numpy as np

trn = pd.read_csv('../input/train_ver2.csv')
# 코드 1-2. Training Data 데이터 미리보기

trn.head()
# 코드 1-3. 모든 변수 미리보기

for col in trn.columns:
    print('{}\n'.format(trn[col].head()))
# 코드 1-4. Training Data 데이터 .info() 함수로 상세보기

trn.info()
# 코드 1-5. 수치형 변수 살펴보기

num_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['int64', 'float64']]
trn[num_cols].describe()
# 코드 1-6. 범주형 변수 살펴보기

cat_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['O']]
trn[cat_cols].describe()
# 코드 1-7. 범주형 변수 고유값 출력해보기

for col in cat_cols:
    uniq = np.unique(trn[col].astype(str))
    print('-' * 50)
    print('# col {}, n_uniq {}, uniq {}'.format(col, len(uniq), uniq))
# 시각화 준비
import matplotlib
import matplotlib.pyplot as plt
# Jupyter Notebook 내부에 그래프를 출력하도록 설정
%matplotlib inline
import seaborn as sns
# 코드 1-8. 변수를 막대 그래프로 시각화하기

skip_cols = ['ncodpers', 'renta']
for col in trn.columns:
    # 출력에 너무 시간이 많이 걸리는 두 변수는 skip
    if col in skip_cols:
        continue
        
    # 보기 편하게 영역 구분 및 변수명 출력
    print('='*50)
    print('col : ', col)
    
    # 그래프 크기 (figsize) 설정
    f, ax = plt.subplots(figsize=(20, 15))
    # seaborn을 사용한 막대그래프 생성
    sns.countplot(x=col, data=trn, alpha=0.5)
    # show() 함수를 통해 시각화
    plt.show()
# 코드 1-9. 월별 금융 제품 보유 여부를 누적 막대 그래프로 시각화

# 날짜 데이터를 기준으로 분석하기 위하여, 날짜 데이터 별도 추출
months = np.unique(trn['fecha_dato']).tolist()
# 제품 변수 24개 추출
label_cols = trn.columns[24:].tolist()

label_over_time = []
for i in range(len(label_cols)):
    # 매월, 각 제품 변수의 총합을 label_sum에 저장
    label_sum = trn.groupby(['fecha_dato'])[label_cols[i]].agg('sum')
    label_over_time.append(label_sum.tolist())
    
label_sum_over_time = []
for i in range(len(label_cols)):
    # 누적 막대 그래프 형식으로 시각화 하기 위하여, 누적값을 계산
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))
    
# 시각화를 위한 색깔 지정
color_list = ['#F5B7B1','#D2B4DE','#AED6F1','#A2D9CE','#ABEBC6','#F9E79F','#F5CBA7','#CCD1D1']

# 시각화를 위한 준비
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    # 24개 제품에 대하여 Histogram 그리기
    sns.barplot(x=months, y=label_sum_over_time[i], color = color_list[i%8], alpha=0.7)

# 우측 상단에 Legend 추가하기
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))], label_cols, loc=1, ncol = 2, prop={'size':16})
# 코드 1-10. 누적 막대 그래프를 상대값으로 시각화하기

# label_sum_over_time의 값을 퍼센트 단위로 변환하기
label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))) * 100

# 앞선 코드와 동일한, 시각화 실행 코드
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha=0.7)
    
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))], \
           label_cols, loc=1, ncol = 2, prop={'size':16})