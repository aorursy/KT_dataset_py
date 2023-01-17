import numpy as np 

import pandas as pd 
from pathlib import Path 

p = Path('../input/competitive-data-science-predict-future-sales')

!ls {p}

sales=pd.read_csv(p/'sales_train.csv',nrows=10); sales.head()
if 'sales' in globals(): del sales

daily_sales = pd.read_csv(p/'sales_train.csv',usecols=['date','date_block_num','shop_id','item_id','item_cnt_day'])

daily_sales['date']=pd.to_datetime(daily_sales['date'],format="%d.%m.%Y")

print(f"Quick summary - daily_sales\n")

for cl in daily_sales:

    print(f"column:{cl:16s} type:{str(daily_sales[cl].dtype):16s} nunqiue:{str(daily_sales[cl].nunique()):8s}, \t(min,max):({daily_sales[cl].min()}, {daily_sales[cl].max()})")
# check for null/missing values

daily_sales.isnull().sum()
# keep only sales data from the last month to work with (date_block_num=33)

last_month_sales = daily_sales[daily_sales.date_block_num==33].copy()



# roll up the daily figures for each store for this last month

last_month_sales = last_month_sales.groupby(['shop_id','item_id']).item_cnt_day.sum().reset_index()

last_month_sales.rename(columns={'item_cnt_day':'item_cnt_month'},inplace=True)



display(last_month_sales.head())



# show stats for last month

for cl in last_month_sales:

    print(f"column:{cl:16s} type:{str(last_month_sales[cl].dtype):16s} nunqiue:{str(last_month_sales[cl].nunique()):8s}, \t(min,max):({last_month_sales[cl].min()}, {last_month_sales[cl].max()})")
sales_shop_ids = set(last_month_sales.shop_id); sales_item_ids = set(last_month_sales.item_id)



test = pd.read_csv(p/'test.csv')

test_shop_ids = set(test.shop_id); test_item_ids = set(test.item_id)



shops_in_sales_and_test = sales_shop_ids & test_shop_ids

shops_in_test_but_not_in_sales = test_shop_ids - sales_shop_ids

items_in_sales_and_test = sales_item_ids & test_item_ids

items_in_test_but_not_in_sales = test_item_ids - sales_item_ids



print(f"{len(shops_in_sales_and_test):,} shops in sales and test data ({len(shops_in_test_but_not_in_sales):,} test shops not in sales data)")

print(f"{len(items_in_sales_and_test):,} items in sales and test data ({len(items_in_test_but_not_in_sales):,} test items not in sales data)")



from IPython.display import HTML

import base64

import datetime

import json



class CopyForwardModel:

    def __init__(self, train_df=None, test_df=None, name=None,):

        """ 

        Initialise the Model Class.

        test_df: the test examples

        train_df: I've called this train data but really this 'model' is doing retreival

                  expect train_df as an input to be the daily sales data from `sales_train.csv`

        """

        # perform a rollup to monthly sales on that data

        self.train_df = self._rollupdailysales(train_df)

        self.test_df = test_df

        self.summary = {}

    

    def predict_sales(self, X_df, M=34):

        """

        Produce a prediction for given inputs:

        X_df: the inputs as a dataframe, one example per row

              in this case expect (shop_id,item_id)

        M:    target month for predictions (as date_block_num)

        """

        # make a prediction for the target month by copying forward previous 

        # months sales. If no information predcict zero sales

        

        # fetch monthly sales for month prior to target month

        if M>0:

            prev_sales = self.train_df[self.train_df.date_block_num == M-1]

        else:

            print("Invalid target month.")

            return None

        

        # copy forward by joining previous sales onto X_df

        predictions = X_df.merge(prev_sales[['shop_id', 'item_id','item_cnt_month']], on=['shop_id', 'item_id'], how='left')

        

        # missing values will be nan, assume these are zero sales

        predictions.fillna(0.,inplace=True)

        

        self.predictions = predictions

        return predictions

    

    def _rollupdailysales(self, daily_sales):

        """ 

        Take the daily sales data in a dataframe and roll up to monthly sales

        """

        # add up the daily figures for each store/item/month combination

        sales_by_month = daily_sales.groupby(['shop_id','item_id','date_block_num']).item_cnt_day.sum().reset_index()

        sales_by_month.rename(columns={'item_cnt_day':'item_cnt_month'},inplace=True)

        

        return sales_by_month

        

    

    def _create_download_link(self,df, title = "Download ", filename = "data.csv", include_index=False): 

        """

        Thanks to Racheal Tatman (https://www.kaggle.com/rtatman) for this snippet to create a download link in the notebook.

        """

        csv = df.to_csv(index=include_index)

        b64 = base64.b64encode(csv.encode())

        payload = b64.decode()

        html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

        html = html.format(payload=payload,title=title+filename,filename=filename)

        return HTML(html)



    def create_submission(self, print_summary=True):

        fname=f"submission_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

        preds=self.predict_sales(self.test_df)[['ID','item_cnt_month']]

        self.summary['predictions'] = {

                                        'count':preds['item_cnt_month'].count(),

                                        'range':{'min':preds.min(), 'max':preds.max()},

                                        'mean':preds['item_cnt_month'].mean(),

                                        'stdev':preds['item_cnt_month'].std(),

                                        'median':preds['item_cnt_month'].median()

                                        }



        if print_summary:

            print(json.dumps(json.loads(pd.DataFrame(self.summary).to_json()),indent=4))

            

        return self._create_download_link(preds,filename=fname)
train_df=pd.read_csv(p/'sales_train.csv')

test_df=pd.read_csv(p/'test.csv')

mymodel = CopyForwardModel(train_df=train_df, test_df=test_df)

mymodel.create_submission()
# merge previous sales with predicted and compare `item_cnt_month` values.

X=mymodel.train_df[mymodel.train_df.date_block_num == 33]

X.merge(mymodel.predict_sales(test_df, M=34), on=['shop_id','item_id'], how='outer')[['ID','shop_id','item_id','item_cnt_month_x','item_cnt_month_y']]
class CopyForwardModelWithClip(CopyForwardModel):

    def create_submission(self, print_summary=True):

        fname=f"submission_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

        preds=self.predict_sales(self.test_df)[['ID','item_cnt_month']]

 

        # add clipping to match competition rules

        preds['item_cnt_month']=preds['item_cnt_month'].clip(0,20)

        

        self.summary['predictions'] = {

                                        'count':preds['item_cnt_month'].count(),

                                        'range':{'min':preds.min(), 'max':preds.max()},

                                        'mean':preds['item_cnt_month'].mean(),

                                        'stdev':preds['item_cnt_month'].std(),

                                        'median':preds['item_cnt_month'].median()

                                        }



        if print_summary:

            print(json.dumps(json.loads(pd.DataFrame(self.summary).to_json()),indent=4))

            

        return self._create_download_link(preds,filename=fname)
train_df=pd.read_csv(p/'sales_train.csv')

test_df=pd.read_csv(p/'test.csv')

mymodel = CopyForwardModelWithClip(train_df=train_df, test_df=test_df)

mymodel.create_submission()