import numpy as np
import pandas as pd
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression
dfoff_train = pd.read_csv('../input/o2o/ccf_offline_stage1_train.csv')
dfoff_test = pd.read_csv('../input/o2o/ccf_offline_stage1_test_revised.csv')
dfon_train = pd.read_csv('../input/o2o/ccf_online_stage1_train.csv')
dfoff_train.head()
dfoff_train.info()
dfoff_test.head()
dfoff_test.info()
dfon_train.head()
dfon_train.info()
dfoff_train_null_coupon_count = np.sum(dfoff_train['Date_received'].isnull())
dfoff_train_coupon_count = dfoff_train['Date_received'].count()

print("数据总量：",dfoff_train_null_coupon_count+dfoff_train_coupon_count)
print("有优惠券数量：",dfoff_train_coupon_count)
print("无优惠券数量：",dfoff_train_null_coupon_count)

dfoff_train_null_consume_count= np.sum(dfoff_train['Date'].isnull())
dfoff_train_consume_count = dfoff_train['Date'].count()
print("消费的数量：",dfoff_train_consume_count)
print("没有消费的数量：",dfoff_train_null_consume_count)


dfoff_train_consume_no_coupon = ((dfoff_train['Date_received'].isnull() == False) & dfoff_train['Date'].isnull())
dfoff_train_consume_no_coupon = np.sum(dfoff_train_consume_no_coupon)

dfoff_train_consume_coupon = ((dfoff_train['Date_received'].isnull() == False) & (dfoff_train['Date'].isnull() == False))
dfoff_train_consume_coupon = np.sum(dfoff_train_consume_coupon)

dfoff_train_no_consume_coupon = (dfoff_train['Date_received'].isnull() & (dfoff_train['Date'].isnull() == False))
dfoff_train_no_consume_coupon = np.sum(dfoff_train_no_consume_coupon)

print("有优惠券，但是没有消费了的数量：",dfoff_train_consume_no_coupon)
print("有优惠券，又有消费了的数量：",dfoff_train_consume_coupon)
print("没有有优惠券，但是消费了的数量：",dfoff_train_no_consume_coupon)
train_off_userId = set(dfoff_train["User_id"])
test_off_userId = set(dfoff_test["User_id"])

print("出现在测试集但是没有出现在训练集的User_id：",test_off_userId-train_off_userId)
train_off_MerchantId = set(dfoff_train["Merchant_id"])
test_off_MerchantId = set(dfoff_test["Merchant_id"])

print("出现在测试集但是没有出现在训练集的Merchant_id：",test_off_MerchantId-train_off_MerchantId)
dfoff_train["Discount_rate"].unique()
dfoff_train["Distance"].unique()
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]

#获取折扣类型，0没有折扣，1满减，2打折
def getDiscountType(row):
    row = str(row)
    if(row == 'nan'):
        return 0
    elif(':' in row):
        return 1
    else:
        return 2

#转换成正常折扣
def convertRate(row):
    row = str(row)
    if(row == 'nan'):
        return 1.0
    elif(':' in row):
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)
    
#获取满
def getDiscountMan(row):
    row = str(row)
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

#获取减
def getDiscountJian(row):
    row = str(row)
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0

def getWeekday(row):
    row = str(row)
    if(row == 'nan'):
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1
    
def processData(df):
    
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['weekday'] = df['Date_received'].astype(str).apply(getWeekday)
    # weekday_type :  周六和周日为1，其他为0
    df['weekday_type'] = df['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
    
    tmpdf = pd.get_dummies(df['weekday'])
    tmpdf.columns = weekdaycols
    df[weekdaycols] = tmpdf


    print("discount_rate:\n",df['discount_rate'].unique())
    
    # convert distance
    df['distance'] = df['Distance'].fillna(-1)
    print("distance:\n",df['distance'].unique())
    
    return df
    
dfoff_train = processData(dfoff_train)
print("\n")
dfoff_test = processData(dfoff_test)
dfoff_train.head()
def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0

dfoff_train['label'] = dfoff_train.apply(label, axis = 1)
dfoff_train.label.unique()
# data split
print("-----data split------")
df = dfoff_train[dfoff_train['label'] != -1].copy()
train = df[(df['Date_received'] < 20160516)].copy()
valid = df[(df['Date_received'] >= 20160516) & (df['Date_received'] <= 20160615)].copy()
train.count()
valid.count()
original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
print("----train-----")
model = SGDClassifier(#lambda:
    loss='log',
    penalty='elasticnet',
    fit_intercept=True,
    max_iter=100,
    shuffle=True,
    alpha = 0.01,
    l1_ratio = 0.01,
    n_jobs=1,
    class_weight=None
)
model.fit(train[original_feature], train['label'])

print(model.score(valid[original_feature], valid['label']))
y_test_pred = model.predict_proba(dfoff_test[original_feature])
dftest = dfoff_test[['User_id','Coupon_id','Date_received']].copy()
dftest['label'] = y_test_pred[:,1]
dftest.to_csv('submit.csv', index=False, header=False)
dftest.head()
