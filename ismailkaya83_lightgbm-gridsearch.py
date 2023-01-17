import numpy as np # linear algebra
import pandas as pd # data processing

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import gc
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, index_col='TransactionID')
    df = reduce_mem_usage(df)
    return df


def aggreg(columns, userid, aggr='mean'):
    """
       Grouping the selected variables according to "userids", taking the averages and 
       assigning them to the new variable according to userids
       
    """
    
    
    for col in columns:
        # create the new colunm name
        new_col_name = col+'_'+userid+'_'+aggr 
        df_temp = pd.concat([X_train[[userid, col]], X_test[[userid,col]]]) 
        df_temp.loc[df_temp[col]==-1,col] = np.nan 
        
        # grouping column by userid
        df_temp = df_temp.groupby(userid)[col].agg([aggr]).reset_index().rename(columns={aggr: new_col_name})
        df_temp.index = list(df_temp[userid]) 
        df_temp = df_temp[new_col_name].to_dict()  
        
        # Add these average values to Train and Test sets according to userid with the name "new_col_name"
        X_train[new_col_name] = X_train[userid].map(df_temp).astype('float32')
        X_test[new_col_name]  = X_test[userid].map(df_temp).astype('float32')
        
        # Writes -1 instead of "nan" values in newly created variables.
        X_train[new_col_name].fillna(-1,inplace=True)
        X_test[new_col_name].fillna(-1,inplace=True)
      


    
def aggreg_uniq(columns, userid):
    
    """
        Variables in columns are grouped by userid and unique values ​​in this column are counted and
        the total number of each unique value is assigned across the "userid" in Test and Train sets.
    """
    for col in columns:  
        df = pd.concat([X_train[[userid,col]],X_test[[userid,col]]],axis=0)
        uniq = df.groupby(userid)[col].agg(['nunique'])['nunique'].to_dict()
        
        X_train[col+'_count'] = X_train[userid].map(uniq).astype('float32')
        X_test[col+'_count'] = X_test[userid].map(uniq).astype('float32')
    
    
def num_positiv(X_train,X_test):
    
    """
       We increase each value by the minimum value in the Train and Test set, so there is no negative value 
       and the minimum value becomes 0. The purpose in doing this is when we assign -1 to NAN values, 
       it can be perceived as a separate class.
    """
    for f in X_train.columns:  
        
        if f not in ['TransactionAmt','TransactionDT',"isFraud"]: 
            mn = np.min((X_train[f].min(),X_test[f].min())) 
            X_train[f] -= np.float32(mn)  
            X_test[f] -= np.float32(mn)
            
            X_train[f].fillna(-1,inplace=True)  
            X_test[f].fillna(-1,inplace=True)  
            

def class_freq(cols):
    """ 
       The "class_freq" function normalizes the specified columns in the entered data sets,
       converts their types to "float32" and adds them to the data sets as a new variable with "_freq" extension.
    """
    
    for col in cols:
        df = pd.concat([X_train[col],X_test[col]])
        vc = df.value_counts(dropna=True).to_dict()  
        vc[-1] = -1  
        nm = col+'_freq' 
        X_train[nm] = X_train[col].map(vc)  
        X_test[nm] = X_test[col].map(vc) 
        del df; x=gc.collect()
        

        
def factorize_categoric():    
    
    """
       Factorizing process is performed for all categoric (object) variables, and
       factorize function keeps nan values as -1.
    """
    for col in X_train.select_dtypes(include=['category','object']).columns:
        df = pd.concat([X_train[col],X_test[col]])
        df,_ = df.factorize(sort=True)
        X_train[col] = df[:len(X_train)].astype('int32')
        X_test[col] = df[len(X_train):].astype('int32')
        del df; x=gc.collect()        
        

        

def user_id(col1,col2):
    
    """
       Converts the values ​​in 2 columns to string and combines them 
       with "_" to create a string type new variable.
       
    """
    us_id = col1+'_'+col2
    
    X_train[us_id] = X_train[col1].astype(str)+'_'+X_train[col2].astype(str)
    X_test[us_id] = X_test[col1].astype(str)+'_'+X_test[col2].astype(str)
%%time

print('Loading data...')

train_id = import_data("../input/ieee-fraud-detection/train_identity.csv")
print('\tSuccessfully loaded train_identity!')

X_train = import_data('../input/ieee-fraud-detection/train_transaction.csv')
print('\tSuccessfully loaded train_transaction!')

# Merging Train set and train_id
X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)
test_id = import_data('../input/ieee-fraud-detection/test_identity.csv')
print('\tSuccessfully loaded test_identity!')

X_test = import_data('../input/ieee-fraud-detection/test_transaction.csv')
print('\tSuccessfully loaded test_transaction!')

test_id.columns = train_id.columns

# Merging Test set and test_id 
X_test = X_test.merge(test_id, how='left', left_index=True, right_index=True)  


# TARGET
y_train = X_train['isFraud'].copy()  

pd.set_option('max_columns', None)

print('Data was successfully loaded!\n')
nan_groups={}
v_cols = ['V'+str(i) for i in range(1,340)]
for i in X_train.columns:
    nan_sum = X_train[i].isna().sum()
    try:
        nan_groups[nan_sum].append(i)
    except:
        nan_groups[nan_sum]=[i]

for i,j in nan_groups.items():
    print('The Sum of the NaN Values =',i)
    print(j)
    
    

non_group_list=list()
for i,j in nan_groups.items():
    if len(j)>5:
        if i != 0:
            non_group_list.append(i)
            
            
# Variable groups with a correlation value of more than 0.70 within the groups

# V1 - V11 
grp1 = [[1],[2,3],[4,5],[6,7],[8,9],[10,11]]
# V12 - V34
grp2 = [[12,13],[14],[15,16,17,18,21,22,31,32,33,34],[19,20],[23,24],[25,26],[27,28],[29,30]]
# V35 - V52
grp3 = [[35,36],[37,38],[39,40,42,43,50,51,52],[41],[44,45],[46,47],[48,49]]
# V53 - V74
grp4 = [[53,54],[55,56],[57,58,59,60,63,64,71,72,73,74],[61,62],[65],[66,67],[68],[69,70]]
# V74 - V94
grp5 = [[75,76],[77,78],[79,80,81,84,85,92,93,94],[82,83],[86,87],[88],[89],[90,91]]
# V95 - V107
grp6 = [[95,96,97,101,102,103,105,106],[98],[99,100],[104]]
# V107 - V123
grp7 = [[107],[108,109,110,114],[111,112,113],[115,116],[117,118,119],[120,122],[121],[123]]
# V124 - V137
grp8 = [[124,125],[126,127,128,132,133,134],[129],[130,131],[135,136,137]]
# V138 - V163
grp9 = [[138],[139,140],[141,142],[146,147],[148,149,153,154,156,157,158],[161,162,163]]
# V167 - V183
grp10 = [[167,168,177,178,179],[172,176],[173],[181,182,183]]
# V184 - V216
grp11 = [[186,187,190,191,192,193,196,199],[202,203,204,211,212,213],[205,206],[207],[214,215,216]]
# V217 - V238
grp12 = [[217,218,219,231,232,233,236,237],[223],[224,225],[226],[228],[229,230],[235]]
# V240 - V262
grp13 = [[240,241],[242,243,244,258],[246,257],[247,248,249,253,254],[252],[260],[261,262]]
# V263 - V278
grp14 = [[263,265,264],[266,269],[267,268],[273,274,275],[276,277,278]]
# V220 - V272
grp15 = [[220],[221,222,227,245,255,256,259],[234],[238,239],[250,251],[270,271,272]]
# V279 - V299
grp16 = [[279,280,293,294,295,298,299],[284],[285,287],[286],[290,291,292],[297]]
# V302 - V321
grp17 = [[302,303,304],[305],[306,307,308,316,317,318],[309,311],[310,312],[319,320,321]]
# V281 V315
grp18 = [[281],[282,283],[288,289],[296],[300,301],[313,314,315]]
# V322 - V339
grp19 = [[322,323,324,326,327,328,329,330,331,332,333],[325],[334,335,336],[337,338,339]]


grp_list = [grp1,grp2,grp3,grp4,grp5,grp6,grp7,grp8,grp9,grp10,
            grp11,grp12,grp13,grp14,grp15,grp16,grp17,grp18,grp19]




def clip_group(group,df):
    """
      Selects the higher number of unique values from the same correlated variables
      
    """
    clipped_list = []
    for i in group:
        maximum = 0; 
        V_num = i[0]
        for j in i:
            n = df['V'+str(j)].value_counts().count()
            if n>maximum:
                maximum = n
                V_num = j
            
        clipped_list.append(V_num)
    
        
    print('Variables in the clipped_list: ',clipped_list)
    return clipped_list



# V variables that were decided to be used in the model as a result of the correlation were kept in the V_clipped_cols variable.
V_clipped_cols = list()
for i in grp_list:
    for j in clip_group(i,X_train):
        V_clipped_cols.append("V"+str(j))
        

for i in range (1, 339):
    name = "V"+str(i)
    if name not in V_clipped_cols:
        X_train.drop("V"+str(i),axis=1, inplace=True)
        X_test.drop("V"+str(i),axis=1, inplace=True)
valid_card = pd.concat([X_train[['card1']], X_test[['card1']]])
valid_card = valid_card['card1'].value_counts()
valid_card_std = valid_card.values.std()

invalid_cards = valid_card[valid_card<=2]

valid_card = valid_card[valid_card>2]
valid_card = list(valid_card.index)

X_train['card1'] = np.where(X_train['card1'].isin(X_test['card1']), X_train['card1'], np.nan)
X_test['card1']  = np.where(X_test['card1'].isin(X_train['card1']), X_test['card1'], np.nan)

X_train['card1'] = np.where(X_train['card1'].isin(valid_card), X_train['card1'], np.nan)
X_test['card1']   = np.where(X_test['card1'].isin(valid_card), X_test['card1'], np.nan)


# Making values "nan" if a value is not common in the Train and Test set
for col in ['card2','card3','card4','card5','card6']: 
    X_train[col] = np.where(X_train[col].isin(X_test[col]), X_train[col], np.nan)
    X_test[col]  = np.where(X_test[col].isin(X_train[col]), X_test[col], np.nan)

col_1 = 'card1'
col_2 = 'P_emaildomain'
col_3 = 'addr1'


user_id(col_1,col_2)
user_id(col_1+'_'+col_2,col_3)
X_train.drop(col_1+'_'+col_2, axis = 1, inplace=True)
X_test.drop(col_1+'_'+col_2, axis = 1, inplace=True)

us_id = col_1 + '_' + col_2 + '_' + col_3
X_train.rename(columns={us_id: 'userid'}, inplace=True)
X_test.rename(columns={us_id: 'userid'}, inplace=True)
for df in [X_train,X_test]:

    df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
    df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]

    df['browser_id_31'] = df['id_31'].str.split(' ', expand=True)[0]
    df['version_id_31'] = df['id_31'].str.split(' ', expand=True)[1]
for df in [X_train,X_test]:

    df['TransactionAmt'] = df['TransactionAmt'].astype('float32')
    df['Trans_min_std'] = (df['TransactionAmt'] - df['TransactionAmt'].mean()) / df['TransactionAmt'].std()
X_train["lastest_browser"] = np.zeros(X_train.shape[0])
X_test["lastest_browser"] = np.zeros(X_test.shape[0])

def setBrowser(df):
    
    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1
    return df

X_train=setBrowser(X_train)
X_test=setBrowser(X_test)
us_emails = ['gmail', 'net', 'edu']

for df in [X_train,X_test]:
    for c in ['P_emaildomain', 'R_emaildomain']:

        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
# Predicting foreign countries by exchange rate
for df in [X_train,X_test]:
    
    df['TransactionAmt_decimal_lenght'] = df['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str.len()
    df['cents'] = (df['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')
p = 'P_emaildomain'
r = 'R_emaildomain'
unknown = 'email_not_provided'

def setDomain(df):
    df[p] = df[p].astype('str')
    df[r] = df[r].astype('str')
    
    df[p] = df[p].fillna(unknown)
    df[r] = df[r].fillna(unknown)
    
    df['email_check'] = np.where((df[p]==df[r])&(df[p]!=unknown),1,0)

    df[p+'_prefix'] = df[p].apply(lambda x: x.split('.')[0])
    df[r+'_prefix'] = df[r].apply(lambda x: x.split('.')[0])
    
    return df
    
X_train=setDomain(X_train)
X_test=setDomain(X_test)
# Listing dates between '2017-10-01' and '2019-01-01 
dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')

# US national holidays are listed between '2017-10-01' and '2019-01-01 
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")


# The variable of the hour of day, day of week and day of month and month of year were created.
for df in [X_train,X_test]:
    
    df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    df['_Weekdays'] = df['Date'].dt.dayofweek
    df['_Dayhours'] = df['Date'].dt.hour
    df['_Monthdays'] = df['Date'].dt.day
    df['_Yearmonths'] = (df['Date'].dt.month).astype(np.int8) 

    
    # Is the transaction done on holiday?
    df['is_holiday'] = (df['Date'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

    df.drop("Date", axis=1,inplace=True)
for col in ['ProductCD','M4']:
    temp_dict = X_train.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(columns={'mean': col+'_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col+'_target_mean'].to_dict()
    
    if col=='ProductCD':
        X_train['ProductCD_1'] = X_train[col].map(temp_dict)
        X_test['ProductCD_1']  = X_test[col].map(temp_dict)
    else:
        X_train['M4_1'] = X_train[col].map(temp_dict)
        X_test['M4_1']  = X_test[col].map(temp_dict)
        
        
# Dropping 'ProductCD' and 'M4'
X_train.drop(['ProductCD','M4'], axis=1,inplace=True)
X_test.drop(['ProductCD','M4'], axis=1,inplace=True)
for i in range(1,16):
    if i in [1,2,3,5,9]:
        continue
    X_train['D'+str(i)] =  X_train['D'+str(i)] - X_train.TransactionDT/np.float32(24*60*60)
    X_test['D'+str(i)] = X_test['D'+str(i)] - X_test.TransactionDT/np.float32(24*60*60)
for df in [X_train,X_test]:

    df['mean_last'] = df['TransactionAmt'] - df.groupby('userid')['TransactionAmt'].transform(lambda x: x.rolling(10, 1).mean())
    df['min_last'] = df.groupby('userid')['TransactionAmt'].transform(lambda x: x.rolling(10, 1).min())
    df['max_last'] = df.groupby('userid')['TransactionAmt'].transform(lambda x: x.rolling(10, 1).max())
    df['std_last'] = df['mean_last'] / df.groupby('userid')['TransactionAmt'].transform(lambda x: x.rolling(10, 1).std())

    df['mean_last'].fillna(0, inplace=True, )
    df['std_last'].fillna(0, inplace=True)

    df['TransactionAmt_to_mean_card_id'] = df['TransactionAmt'] - df.groupby(['userid'])['TransactionAmt'].transform('mean')
    df['TransactionAmt_to_std_card_id'] = df['TransactionAmt_to_mean_card_id'] / df.groupby(['userid'])['TransactionAmt'].transform('std')
    
    
    # Replaces infinite values with 999
    df = df.replace(np.inf,999)
factorize_categoric()

num_positiv(X_train,X_test)

class_freq(['addr1','card1','card2','card3','P_emaildomain'])

aggreg(['TransactionAmt','D4','D9','D10','D11','D15'],'userid','mean')

aggreg(['TransactionAmt','D4','D9','D10','D11','D15','C14'],'userid','std')

aggreg(['C'+str(x) for x in range(1,15) if x!=3],'userid','mean')

aggreg(['M'+str(x) for x in range(1,10) if x!=4],'userid','mean')

aggreg_uniq(['P_emaildomain','dist1','id_02','cents','C13','V314','V127','V136','V309','V307','V320'],'userid')
# Reducing the memory usage
X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)
# Dropping userid to prevent overfitting
X_train.drop("userid", axis=1, inplace=True)
X_test.drop("userid", axis=1, inplace=True)

# Dropping target variable from Train set
X_train.drop("isFraud", axis=1, inplace=True)
# Splitting Train set
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
# Parameter Tuning
model = lgb.LGBMClassifier()
param_dist = {"max_depth": [5,10,15],
              "learning_rate" : [0.1,0.15,0.3],
              "num_leaves": [32,150,200],
              "n_estimators": [300,400],
              'is_unbalance': [True],
              'boost_from_average': [False],
              'device': ['gpu'],
              'gpu_platform_id': [0],
              'gpu_device_id': [0],
              "random_state": [2]}


grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 
                                   verbose=10, n_jobs=-1)


grid_search.fit(X_train1, y_train1)
# Best values of the hyperparameters
max_depth_best = grid_search.best_estimator_.max_depth
num_leaves_best = grid_search.best_estimator_.num_leaves
n_estimators_best = grid_search.best_estimator_.n_estimators
learning_rate_best = grid_search.best_estimator_.learning_rate
model = lgb.LGBMClassifier(max_depth = max_depth_best,
                          num_leaves = num_leaves_best, 
                          n_estimators = n_estimators_best,
                          n_jobs = -1 , 
                          verbose = 1,
                          learning_rate= learning_rate_best,
                          eval_metric='auc',
                          nthread=4,
                          is_unbalance = True,
                          boost_from_average = False,
                          device = 'gpu',
                          gpu_platform_id = 0,
                          gpu_device_id = 0
                          )
                         
                          
model.fit(X_train1,y_train1)





# Confusion matrix and Classification report
pred1 = model.predict(X_test1)
fpr, tpr, thresholds = metrics.roc_curve(y_test1, pred1, pos_label=2)
metrics.auc(fpr, tpr)

print(metrics.confusion_matrix(y_test1, pred1))
print(metrics.classification_report(y_test1, pred1))
preds = model.predict_proba(X_test)[:,1]
sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
sample_submission.isFraud = preds
sample_submission.to_csv('sub_lgbm.csv',index=False)

