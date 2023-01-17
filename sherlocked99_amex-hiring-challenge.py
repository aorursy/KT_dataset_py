import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go
#import all data

train = pd.read_csv("../input/amexpert-2019/train.csv")

camp_data = pd.read_csv("../input/amexpert-2019/campaign_data.csv")

coupon_item = pd.read_csv("../input/amexpert-2019/coupon_item_mapping.csv")

cust_demo = pd.read_csv("../input/amexpert-2019/customer_demographics.csv")

cust_trans = pd.read_csv("../input/amexpert-2019/customer_transaction_data.csv")

item_data = pd.read_csv("../input/amexpert-2019/item_data.csv")

test = pd.read_csv("../input/amexpert-2019/test_QyjYwdj.csv")
id_train = train['id']

id_test = test['id']
train.at[1050,'customer_id']
train.drop(columns = "id",inplace = True)

test.drop(columns = "id",inplace = True)
from scipy.stats import pearsonr

def correlation(x,y):

    corr, _ = pearsonr(x,y)

    print("Pearson Correlation coefficient: ",str(corr))    
#first transfer relevant ID information from item_data to coupon_item data

itembrand = []

itembrtype = []

itemcat = []



for i in coupon_item['item_id']:

    itembrand.append(item_data.loc[i-1][1])

    itembrtype.append(item_data.loc[i-1][2])

    itemcat.append(item_data.loc[i-1][3])



coupon_item['item_brand'] = pd.Series(itembrand)

coupon_item['item_brand_type'] = pd.Series(itembrtype)

coupon_item['item_category'] = pd.Series(itemcat)
coupon_item.set_index("coupon_id",inplace = True)

cust_demo.set_index("customer_id",inplace = True)

cust_trans.set_index("customer_id",inplace = True)

camp_data.set_index("campaign_id",inplace = True)
def split_by_dash(x):

    return x.split('-')

def split_by_plus(x):

    return x.split('+')
#convert range,family size and marital and children to integer

age_low = []

age_high = []

marstat = []

nochil = []

fam_size= []



for index,row in cust_demo.iterrows():

    if len(row['family_size'])>1 and row['family_size'][1] == '+':

        fam_size.append(int(split_by_plus(row['family_size'])[0]))

    else:

        fam_size.append(int(row['family_size']))

    

    if row['age_range'][2] == '-':

        age_low.append(int(split_by_dash(row['age_range'])[0]))

        age_high.append(int(split_by_dash(row['age_range'])[1]))

    elif not pd.isna(row['age_range']):

        age_low.append(int(split_by_plus(row['age_range'])[0]))

        age_high.append(100) #assuming maximum age expectancy to be 100



#0 for Single and 1 for Married



count = 0

for index,row in cust_demo.iterrows():

    if pd.isna(row['marital_status']):

        if fam_size[count]==1:

            marstat.append(0)

        else:

            marstat.append(1)

    else:

        if row['marital_status']=="Single":

            marstat.append(0)

        else:

            marstat.append(1)

            

    if pd.isna(row['no_of_children']):

        if marstat[count] == 0:

            nochil.append(0)

        else:

            if fam_size[count]-2 == -1:

                print(row['marital_status'],row['family_size'])

            nochil.append(fam_size[count]-2)

    else:

        if len(row['no_of_children'])>1 and row['no_of_children'][1]=='+':

            nochil.append(int(split_by_plus(row['no_of_children'])[0]))

        else:

            nochil.append(int(row['no_of_children']))

    count+=1
cust_demo['family_size'] = fam_size

cust_demo['age_low'] = age_low

cust_demo['age_high'] = age_high

cust_demo['no_of_children'] = nochil

cust_demo['marital_status'] = marstat
cust_demo.drop(columns = "age_range",inplace = True)
from datetime import date

campstart = []

campdays = []

for index,row in camp_data.iterrows():

    campstart.append(int(row['start_date'][3:5]))

    date_s = date(int(row['start_date'][6:8]),int(row['start_date'][3:5]),int(row['start_date'][0:2]))

    date_e = date(int(row['end_date'][6:8]),int(row['end_date'][3:5]),int(row['end_date'][0:2]))

    days = date_e-date_s

    campdays.append(days.days)
camp_data['days'] = campdays

camp_data['startmonth'] = campstart



camp_data.drop(columns = ['start_date','end_date'],inplace = True)
coupon_item.sort_index(inplace = True)

unique_couponid = coupon_item.index.unique()
diffids = []

diffbrands = []

difftypes = []

diffcat = []

for x in unique_couponid:

    df = coupon_item.loc[x]

    diffids.append(np.size(np.unique(np.array(coupon_item.loc[x]['item_id']))))

    diffbrands.append(np.size(np.unique(np.array(coupon_item.loc[x]['item_brand']))))

    difftypes.append(np.size(np.unique(np.array(coupon_item.loc[x]['item_brand_type']))))

    diffcat.append(np.size(np.unique(np.array(coupon_item.loc[x]['item_category']))))
coupon_data = {'coupon_id':unique_couponid,'diffids':diffids,'difftypes':difftypes,'diffcat':diffcat}

coupon_data = pd.DataFrame.from_dict(coupon_data)
coupon_data.set_index('coupon_id',inplace = True)
train['age_mean'] = list(np.zeros(78369,dtype = np.float64))

train['marital'] = ""

train['rented'] = list(np.zeros(78369,dtype = np.int8))

train['family'] = list(np.zeros(78369,dtype = np.int8))

train['income'] = list(np.zeros(78369,dtype = np.int8))



train['diffids'] = list(np.zeros(78369,dtype = np.int8))

train['difftypes'] = list(np.zeros(78369,dtype = np.int8))

train['diffcat'] = list(np.zeros(78369,dtype = np.int8))



train['camp_start'] = list(np.zeros(78369,dtype = np.int8))

train['days'] = list(np.zeros(78369,dtype = np.int8))

train['camp_type'] = ""



for index,row in train.iterrows():

    #print(row['customer_id'])

    if row['customer_id'] in cust_demo.index:

        train.at[index,'age_mean'] = (cust_demo.loc[row['customer_id']]['age_low'] + cust_demo.loc[row['customer_id']]['age_high']) / 2

        train.at[index,'marital'] = cust_demo.loc[row['customer_id']]['marital_status']

        train.at[index,'rented'] = cust_demo.loc[row['customer_id']]['rented']

        train.at[index,'family'] = cust_demo.loc[row['customer_id']]['family_size']

        train.at[index,'income'] = cust_demo.loc[row['customer_id']]['income_bracket']

    

    train.at[index,'diffids'] = coupon_data.loc[row['coupon_id']]['diffids']

    train.at[index,'difftypes'] = coupon_data.loc[row['coupon_id']]['difftypes']

    train.at[index,'diffcat'] = coupon_data.loc[row['coupon_id']]['diffcat']

    

    train.at[index,'camp_start'] = camp_data.loc[row['campaign_id']]['startmonth']

    train.at[index,'campdays'] = camp_data.loc[row['campaign_id']]['days']

    train.at[index,'camp_type'] = camp_data.loc[row['campaign_id']]['campaign_type']
train.drop(columns = ['marital'],inplace = True)

for index,row in train.iterrows():

    train.at[index,'camp_type'] = 0 if row['camp_type']=='X' else 1

    train.at[index,'campdays'] = int(train.at[index,'campdays'])
l = list(train['camp_type'])

l = pd.Series(l)

train.drop(columns = 'camp_type',inplace = True)

train['camp_type'] = l
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
y_train = train['redemption_status']

train.drop(columns = 'redemption_status',inplace = True)
model = XGBClassifier(learning_rate = 0.05, max_depth = 4)

train_x, test_x, train_Y, test_y = train_test_split(train,y_train,train_size=0.8)

model.fit(train_x,train_Y)

xgb_predict = model.predict(test_x)

    

print( "Train Accuracy :: ", accuracy_score(train_Y, model.predict(train_x)))

print( "Test Accuracy  :: ", accuracy_score(test_y, xgb_predict))
test['age_mean'] = list(np.zeros(50226,dtype = np.float64))

test['rented'] = list(np.zeros(50226,dtype = np.int8))

test['family'] = list(np.zeros(50226,dtype = np.int8))

test['income'] = list(np.zeros(50226,dtype = np.int8))



test['diffids'] = list(np.zeros(50226,dtype = np.int8))

test['difftypes'] = list(np.zeros(50226,dtype = np.int8))

test['diffcat'] = list(np.zeros(50226,dtype = np.int8))



test['camp_start'] = list(np.zeros(50226,dtype = np.int8))

test['days'] = list(np.zeros(50226,dtype = np.int8))

test['camp_type'] = ""



for index,row in test.iterrows():

    #print(row['customer_id'])

    if row['customer_id'] in cust_demo.index:

        test.at[index,'age_mean'] = (cust_demo.loc[row['customer_id']]['age_low'] + cust_demo.loc[row['customer_id']]['age_high']) / 2

        test.at[index,'rented'] = cust_demo.loc[row['customer_id']]['rented']

        test.at[index,'family'] = cust_demo.loc[row['customer_id']]['family_size']

        test.at[index,'income'] = cust_demo.loc[row['customer_id']]['income_bracket']

    

    test.at[index,'diffids'] = coupon_data.loc[row['coupon_id']]['diffids']

    test.at[index,'difftypes'] = coupon_data.loc[row['coupon_id']]['difftypes']

    test.at[index,'diffcat'] = coupon_data.loc[row['coupon_id']]['diffcat']

    

    test.at[index,'camp_start'] = camp_data.loc[row['campaign_id']]['startmonth']

    test.at[index,'campdays'] = camp_data.loc[row['campaign_id']]['days']

    test.at[index,'camp_type'] = camp_data.loc[row['campaign_id']]['campaign_type']
for index,row in test.iterrows():

    test.at[index,'camp_type'] = 0 if row['camp_type']=='X' else 1

    test.at[index,'campdays'] = int(test.at[index,'campdays'])
l = list(test['camp_type'])

l = pd.Series(l)

test.drop(columns = 'camp_type',inplace = True)

test['camp_type'] = l
xgbpred = model.predict(test)
xgbpred
submission = pd.DataFrame(id_test)

submission['redemption_status'] = xgbpred 

submission.to_csv("amex1.csv",index = False)