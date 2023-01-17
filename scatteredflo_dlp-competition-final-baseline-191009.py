

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/dlp-private-competition-dataset-modificated/train.csv")

train = train.drop(['ID'],1)     # train의 ID를 Drop 시켜줌

train = train.reset_index(drop=True)     # train의 index를 초기화 시켜줌

print(train.shape)

train.head()
test = pd.read_csv("/kaggle/input/dlp-private-competition-dataset-modificated/test.csv")

test = test.drop(['ID'],1)     # train의 ID를 Drop 시켜줌

test = test.reset_index(drop=True)     # train의 index를 초기화 시켜줌

print(test.shape)

test.head()
submission = pd.read_csv("/kaggle/input/dlp-private-competition-dataset-modificated/submission.csv")

print(submission.shape)

submission.head()
test['Y'] = 9999

test.head()

# test셋에 Y값에 9999를 insert (나중에 분할때 Y값이 9999인 것들만 test로 빼주면 됨)
total = pd.concat([train,test],0)     # train과 test를 합쳐줌 

print(train.shape, test.shape, "--> ",total.shape)     # train과 test가 합쳐져서 total이 되는데 shape를 확인

total.head()     # total의 앞에 5개 행 확인(잘 합쳐졌나..)
total.tail()     # total의 마지막 5개 행 확인(잘 합쳐졌나..)
str_line = []

column_name = []

unique_list = []



for i in total.columns:

    total_list1 = list(total[i].unique())

    total_list2 = list((pd.DataFrame(total_list1).dropna())[0])

    str_data = int(str(type(total_list2[0])) == "<class 'str'>")

    str_line.append(str_data)

    column_name.append(i)



str_line = pd.DataFrame(str_line)

column_name = pd.DataFrame(column_name)

all_column = pd.concat([column_name, str_line],1)

all_column.columns = ['column_name', 'strn']



str_column = pd.DataFrame(all_column[all_column['strn'] == 1])

str_column = list(str_column['column_name'])



unique_count = []

unique_column = []



for i in str_column:

    total_unique = list(total[i].unique())

    total_unique = len(total_unique)

    unique_count.append(total_unique)

    unique_column.append(i)



unique_count = pd.DataFrame(unique_count)

unique_column = pd.DataFrame(unique_column)



if len(unique_count) == 0:

    print("#####  Dataset have no string data  #####")

else:

    unique_total = pd.concat([unique_column, unique_count],1)

    unique_total.columns = ['unique_column', 'unique_count']

    unique_total = unique_total.sort_values(["unique_count"], ascending=[False])

    # unique_total

    # 상위 4개 항목들은 따로 관리해야 할듯, 나머지는 OHE진행

    unique_column = unique_total.unique_column.values

    print(unique_column)     # column에 string(문자) 데이터가 있는 column만 출력해줌
from sklearn.preprocessing import LabelEncoder

cols = ['END_TM', 'A', 'B']     # 앞에서 문자형 변수가 속해있는 Column들을 Label Encoding함



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(total[c].values)) 

    total[c] = lbl.transform(list(total[c].values))     # unique_column을 전무 label_encoding함



print('Shape all_data: {}'.format(total.shape))

cate_col = ['END_TM', 'A', 'B']



for col in cate_col:

    total[col] = total[col].astype('category')

    

# END_TM, A, B는 Label Encoding으로 숫자로 바꾸었지만 원래는 category 값임,

# Label Encoding으로 1,2,3,4,5,... 으로 바뀌었기 때문에 LightGBM 모델은 그냥 연속적인 값으로 이해함

# 그렇기 때문에 해당 column을 숫자이지만 Category로 변환시켜줘야 함
pd.set_option('display.max_columns', 10000)

# head() 출력시에 전체로 보이게 만들어 줌, 넓이를 늘려줌, 화면



total.head()     # 날짜까지 전부 Label Encoding 되었음..(필요하다면 Label Encoding 전에 따로 Feature Engineering 진행)
test = total[total['Y'] == 9999]     # total['Y']가 9999인 값만 test로 뽑아줌

test.head()     # test에서는 Y값은 필요 없기 때문에 삭제해줘야함
test = test.drop(['Y'],1)

test.head() 
train = total[total['Y'] != 9999]     # total['Y']가 9999이 아닌 값만 train으로 뽑아줌

train.head()     # 잘 분리되었군..
y = pd.DataFrame(train['Y'])

y.head()
X = train.drop(['Y'],1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split (X,y, random_state=0)
import lightgbm as lgb

lgbm = lgb.LGBMRegressor (objective = 'regression', num_leaves=144,

                         learning_rate=0.005,n_estimators=720, max_depth=13,

                         metric='rmse', is_training_metric=True, max_bin=55,

                         bagging_fraction=0.8, verbose=-1, bagging_freq=5, feature_fraction=0.9)
lgbm.fit(X_train, y_train)
#=== MSE로 평가할 경우 아래와 같이 진행===#

from sklearn.metrics import mean_squared_error



pred_train = lgbm.predict(X_train)

pred_valid = lgbm.predict(X_valid)

print(mean_squared_error(pred_train, y_train))

print(mean_squared_error(pred_valid, y_valid))





# #=== House Price에서 RMSE 값을 구하는 경우 아래와 같이 진행===#

# from sklearn.metrics import mean_squared_error



# pred_train = lgbm.predict(X_train)

# pred_valid = lgbm.predict(X_valid)



# print(np.sqrt(mean_squared_error(np.log1p(y_train), np.log1p(pred_train))))

# print(np.sqrt(mean_squared_error(np.log1p(y_valid), np.log1p(pred_valid))))



# # 위에는 MSE이며, RMSE를 구할때는 해당으로 진행
pred_test = lgbm.predict(test)

pred_test
submission = submission.drop("Y",1)

pred_test = pd.DataFrame(pred_test)



submission_final = pd.concat([submission,pred_test],axis=1)



submission_final.columns = ['ID','Y']

submission_final.to_csv("submission_fianl.csv", index=False)

submission_final.tail()