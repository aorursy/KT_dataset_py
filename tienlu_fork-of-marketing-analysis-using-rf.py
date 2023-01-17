import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/train.csv', index_col = 'row_id')
test = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/test.csv', index_col = 'row_id')
user = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/users.csv')
print(user.head(10))
print(len(user))
user[['attr_1','attr_2']] = user[['attr_1','attr_2']].fillna(0)
#將attribute1跟attribute2中沒有資料的部分都填上0
#fill in zero for all NaN in attribute1 and attribute2
user['age'] = user['age'].fillna(round(user['age'].mean(),2))
#將age中NaN的部分填上平均年齡
#fill the average age for all NaN in column 'age'
user.head(10)
train = pd.merge(train, user, on = 'user_id')
train.head()
#合併train和整理過的user表格
#merge the table 'train' and processed 'user'
test = pd.merge(test, user, on = 'user_id')
test.head()
#合併test和整理過的user表格
#merge the table 'test' and processed 'user'
"""
def f(x):
    if x == 'Never open':
        return 808
    else:
        return int(x)
train['last_open_day'] = train['last_open_day'].apply(lambda x : f(x))
"""

train['last_open_day'] = train['last_open_day'].apply(lambda x : 808 if x == 'Never open' else int(x))
test['last_open_day'] = test['last_open_day'].apply(lambda x : 808 if x == 'Never open' else int(x))
train.head(10)
#將last_open_day中'Never open'(從未打開)替換成該欄位的最大值(假設成很久之前打開過)
#replace the value 'Never open' in 'last_open_day' column with the max value in the column(assume it was opened long ago)
train['last_login_day'] = train['last_login_day'].apply(lambda x : None if x == 'Never login' else int(x))
train['last_checkout_day'] = train['last_checkout_day'].apply(lambda x : None if x == 'Never checkout' else int(x))
train = train[train['last_login_day'] < 1500]
test['last_login_day'] = test['last_login_day'].apply(lambda x : None if x == 'Never login' else int(x))
test['last_checkout_day'] = test['last_checkout_day'].apply(lambda x : None if x == 'Never checkout' else int(x))
print(train.shape)

#將'last_login_day'和'last_checkout_day'中'Never login'及'Never checkout'替換成空值
#replace the 'Never login' and 'Never checkout' with None
#'last_login_day'中有超過一萬以上的值，應屬noise，因此將之剔除
#delete the value over 10000, which is possible noise
train['last_login_day'] = train['last_login_day'].fillna(train['last_login_day'].median())
train['last_checkout_day'] = train['last_checkout_day'].fillna(train['last_checkout_day'].median())
test['last_login_day'] = test['last_login_day'].fillna(test['last_login_day'].median())
test['last_checkout_day'] = test['last_checkout_day'].fillna(test['last_checkout_day'].median())
test.head(10)
#將剛剛替換的空值換成中位數
#replace the 'None' value with median
train['grass_date'] = pd.to_datetime(train['grass_date'])
train['grass_date'] = train['grass_date'].dt.weekday
train.head(10)
#由於日期欄位的時間皆為GMT+8，因此只取日期，並轉為星期
#time in the date column are all GMT+8, thus, take date only, and convert to weekdays
test['grass_date'] = pd.to_datetime(test['grass_date'])
test['grass_date'] = test['grass_date'].dt.weekday
test.head(10)
#由於日期欄位的時間皆為GMT+8，因此只取日期，並轉為星期
#time in the date column are all GMT+8, thus, take date only, and convert to weekdays
train = train.drop(columns = ['user_id', 'subject_line_length'])
test = test.drop(columns = ['user_id', 'subject_line_length'])
#使用者ID以及電郵主題長度對結果幾乎沒有影響，將之剔除
#drop 'user id' and 'subject line length'
def normalize(s):
    mi = s.min()
    ma = s.max()
    s = s.apply(lambda x: round((x - mi) / (ma - mi),4))
    return s
#定義常規化的函數
#define function normalization
need_normalize = ['open_count_last_10_days',
                  'open_count_last_30_days',
                  'open_count_last_60_days',
                  'login_count_last_10_days',
                  'login_count_last_30_days',
                  'login_count_last_60_days',
                  'checkout_count_last_10_days',
                  'checkout_count_last_30_days',
                  'checkout_count_last_60_days',
                  'age']
for i in need_normalize:
    train[i] = normalize(train[i])
    test[i] = normalize(test[i])
#將數值的欄位全都進行常規化
#apply to all numeric values
train.head(10)
train.info()
test.info()
train['country_code'] = train['country_code'].apply(str)
train['grass_date'] = train['grass_date'].apply(str)
test['country_code'] = test['country_code'].apply(str)
test['grass_date'] = test['grass_date'].apply(str)
y = train['open_flag']
features = ['country_code',
            'grass_date',
            'last_open_day',
            'last_login_day',
            'last_checkout_day',
            'open_count_last_10_days',
            'open_count_last_30_days',
            'open_count_last_60_days',
            'login_count_last_10_days',
            'login_count_last_30_days',
            'login_count_last_60_days',
            'checkout_count_last_10_days',
            'checkout_count_last_30_days',
            'checkout_count_last_60_days',
            'attr_1',
            'attr_2',
            'attr_3',
            'age',
            'domain']
X = pd.get_dummies(train[features])
test = pd.get_dummies(test[features])
test.isna().sum()
X.info()
test.info()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
#將測試資料分為測試以及驗證集
#split the train datasets into train and validation datasets
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=1)
model.fit(X_train, y_train)
val = model.predict(X_test)
sum(abs(val - y_test)) / len(y_test)
#設定簡易測試方式
#define an easy way to test the accuracy of the model
ans = model.predict(test)
len(ans)
#將訓練好的模型套用在測試集(test)上
#apply the trained model on the test datasets
sub = pd.read_csv('../input/open-shopee-code-league-marketing-analytics/sample_submission_0_1.csv')
print(sub.shape)
print(len(sub))
sub['open_flag'] = ans
#將預測結果寫入
#write in the prediction
sub.head(10)
sub.to_csv('sub.csv',index=False)
#將結果輸出成csv檔
#export as csv file