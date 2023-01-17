%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd 

from pathlib import Path

p=Path("/kaggle/input/competitive-data-science-predict-future-sales/")
!ls -lh /kaggle/input/competitive-data-science-predict-future-sales/
sample_submission = pd.read_csv(p/"sample_submission.csv")

sample_submission.head()
test = pd.read_csv(p/"test.csv")

display(test.head())



s1="equal" if sample_submission.ID.nunique() == sample_submission.ID.nunique() else "not equal"

s2="match" if set(sample_submission.ID)==set(sample_submission.ID) else "do not match"



print(f"Number of IDs in sample ({sample_submission.ID.count():,}) is {s1} to test ({test.ID.count():,})")

print(f"Values of unique IDs in sample and test {s2}")
print(test.columns)

print(f"unique shop_ids: {test.shop_id.nunique():,}")

print(f"unique item_ids: {test.item_id.nunique():,}")

print(f"unique IDs: {test.ID.nunique():,} ({test.shop_id.nunique()} * {test.item_id.nunique():,} = {test.shop_id.nunique()*test.item_id.nunique():,})")

print(f"\n#shop_id")

print(f"range of shop_id = [{test.shop_id.min()},{test.shop_id.max()}] (N={test.shop_id.max()-test.shop_id.min()+1})")

missing_shop_id = sorted(set(range(test.shop_id.min(),test.shop_id.max()+1))-set(test.shop_id))

print(f"shop_id missing from contiguous: {missing_shop_id} (N={len(missing_shop_id)})")

print(f"\n#item_id")

print(f"range of item_id = [{test.item_id.min()},{test.item_id.max()}] (N={test.item_id.max()-test.item_id.min()+1:,})")

missing_item_id = sorted(set(range(test.item_id.min(),test.item_id.max()+1))-set(test.item_id))

print(f"item_id missing from contiguous: (N={len(missing_item_id):,})")
f,ax = plt.subplots(1,1,figsize=(8,8))

W=150

Y=(22168//W)+1

x=np.zeros((len(range(0,W*Y)),));

x[test.item_id.unique()]=1

a=plt.imshow(np.reshape(x,(Y,W)))
f,ax=plt.subplots(1,1,figsize=(12,4))

_=test.groupby('shop_id').item_id.count().plot.bar(ax=ax, fontsize=16)

_=ax.set_ylabel('count of item_id', fontsize=16)
from IPython.display import HTML

import base64

import datetime

import json



class ConstantModel:

    def __init__(self, C=1.0, name=None, test_df=None):

        """ 

        Initialise the Model Class.

        test_df: a dataframe containing the test examples

        """

        self.test_df = test_df 

        self.C = C

        self.summary = {}

    

    def predict_sales(self, X_df, return_inputs=True):

        """

        Produce a prediction for given inputs:

        X: the inputs as a dataframe, one example per row

        """

        # we actually ignore any inputs for the constant prediction

        X_df['item_cnt_month'] = self.C

        if return_inputs:

            return X_df

        else:

            return X_df.Y

    

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

        

    
test_df=pd.read_csv(p/'test.csv')

mymodel = ConstantModel(C=1, test_df=test_df)

mymodel.create_submission()
test_df=pd.read_csv(p/'test.csv')

mymodel = ConstantModel(C=0.1, test_df=test_df)

mymodel.create_submission()