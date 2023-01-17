import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy import stats

from sklearn.datasets import load_boston

%matplotlib inline

# boston 데이터 세트 로드

boston=load_boston()



#boston 데이터 세트 DataFrame 변환

bostonDF=pd.DataFrame(boston.data, columns=boston.feature_names)





#boston 데이터 세트의 target 배열은 주택가격임,이를 Price 칼럼으로 DataFrame에 추가함

bostonDF['PRICE']=boston.target

print('Boston 데이터 세트크기:',bostonDF.shape)

bostonDF.head()
fig,axs=plt.subplots(figsize=(16,8),ncols=4,nrows=2)

lm_features=['RM','ZN','INDUS','NOX','AGE','PTRATIO','LSTAT','RAD']

for i,feature in enumerate(lm_features):

     row=int(1/4)

     col=i%4

    #사본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현

     sns.regplot(x=feature,y='PRICE',data=bostonDF,ax=axs[row][col])