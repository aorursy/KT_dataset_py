from scipy import stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sys
#from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm
def cal_clean(df):
    #starts at saturday  so we can drop the weekday
    df=df.drop(columns=['weekday'])
    #fill na in events and Label enconding
    df['event_name_1']=df['event_name_1'].fillna('no_event')
    df['event_type_1']=df['event_type_1'].fillna('no_event')
    df['event_name_2']=df['event_name_2'].fillna('no_event')
    df['event_type_2']=df['event_type_2'].fillna('no_event')
    le = LabelEncoder()
    df['event_type_1'] = le.fit_transform(df.event_type_1)
    df['event_name_1'] = le.fit_transform(df.event_name_1)
    df['event_type_2'] = le.fit_transform(df.event_type_2)
    df['event_name_2'] = le.fit_transform(df.event_name_2)
    #can drop the date because already extracted is available
    df=df.drop(columns=['date'])
    #label encode the d
    #df['d'] = le.fit_transform(df.d)
    return df

#clean the sales details
def sale_clean(df):
    le = LabelEncoder()
    df['dept_id'] = le.fit_transform(df.dept_id)
    df['cat_id'] = le.fit_transform(df.cat_id)
    df['store_id'] = le.fit_transform(df.store_id)
    df['item_id'] = le.fit_transform(df.item_id)
    df['state_id'] = le.fit_transform(df.state_id)
    df=df.drop(columns=['id'])
    return df
#store wise data break up
def store_split(sale_df):
    new_sale=sale_df.groupby(['store_id'])
    df_st1=new_sale.get_group(0)
    df_st2=new_sale.get_group(1)
    df_st3=new_sale.get_group(2)
    df_st4=new_sale.get_group(3)
    df_st5=new_sale.get_group(4)
    df_st6=new_sale.get_group(5)
    df_st7=new_sale.get_group(6)
    df_st8=new_sale.get_group(7)
    df_st9=new_sale.get_group(8)
    df_st10=new_sale.get_group(9)
    x=[]
    return df_st1,df_st2,df_st3,df_st4,df_st5,df_st6,df_st7,df_st8,df_st9,df_st10
INPUT_DIR = '../input/m5-forecasting-accuracy/'
cal_df = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
sale_df = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
#clean the cal df
new_cal=cal_clean(cal_df)
#clean the sales data
new_sale=sale_clean(sale_df)
#split store wise
s0=pd.DataFrame()
s1=pd.DataFrame()
s2=pd.DataFrame()
s3=pd.DataFrame()
s4=pd.DataFrame()
s5=pd.DataFrame()
s6=pd.DataFrame()
s7=pd.DataFrame()
s8=pd.DataFrame()
s9=pd.DataFrame()
s0,s1,s2,s3,s4,s5,s6,s7,s8,s9=store_split(new_sale)
def day_to_row(sale_df,cal_df,start_day,end_day):
    df=pd.DataFrame()
    x=sale_df.columns.to_list()
    end_day+=start_day
    cnt=end_day-start_day
    start_day+=5
    with tqdm(total=cnt) as pbar:
        for i in range(cnt):
            temp = pd.DataFrame(columns=['item_id','event_name_1','event_type_1','event_name_2','event_type_2','snap','wday','month','year','sales'])
            temp['item_id']=sale_df['item_id']
            temp['sales']=sale_df[x[start_day+i]]
            t=cal_df[cal_df['d']==x[start_day+i]].values
            temp['event_name_1']=t[0][7]
            temp['event_type_1']=t[0][8]
            temp['event_name_2']=t[0][9]
            temp['event_type_2']=t[0][10]
            temp['snap']=t[0][11]
            temp['wday']=t[0][1]
            temp['month']=t[0][2]
            temp['year']=t[0][3]
            #print(x[start_day+i],temp.shape)
            df=pd.concat([df,temp], axis=0,ignore_index=True)
            pbar.update(1)
        print('Dates copied from:',x[start_day],'->',x[start_day+i],' Dataset shape',df.shape)
    return df
#start day range 1-1913
#end day -> Required day count
train=day_to_row(s0,new_cal,0,50)
val=day_to_row(s0,new_cal,50,5)
X=train.drop(columns=['sales'])
y=train['sales']
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X, y)
X_val=val.drop(columns=['sales'])
y_true=val['sales']
y_pred=clf.predict(X_val)
# model evaluation
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score
from math import sqrt
print("RMSSE",mean_absolute_error(y_true, y_pred))
print("MSE",mean_squared_error(y_true, y_pred))
print("RMSSE",sqrt(mean_squared_error(y_true, y_pred)))
print("Accuracy",accuracy_score(y_true, y_pred))