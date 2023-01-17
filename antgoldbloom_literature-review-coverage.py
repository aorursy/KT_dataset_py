import numpy as np 

import pandas as pd 

import datetime as dt

!cp -r /kaggle/input/CORD-19-research-challenge/Kaggle/target_tables/* /kaggle/working/

!rm -r /kaggle/working/0_table_formats_and_column_definitions/

!rm -r /kaggle/working/unsorted_tables/

!rm /kaggle/working/__notebook_source__.ipynb
df_metadata = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv',index_col=[0])

df_metadata['publish_time'] = pd.to_datetime(df_metadata['publish_time'])

df_metadata['Year'] = pd.DatetimeIndex(df_metadata['publish_time']).year

df_metadata = df_metadata[(df_metadata['publish_time'] >= '2020-02-01') & (df_metadata['publish_time'] <= dt.datetime.now())]

df_metadata['Month'] = pd.DatetimeIndex(df_metadata['publish_time']).month

df_metadata['pdf_json_files'].replace('', np.nan, inplace=True) # Full-text only

df_metadata.dropna(subset=['pdf_json_files'], inplace=True) # Full-text only
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

initialize = True

for dirname, _, filenames in os.walk('/kaggle/working/'):

    for filename in filenames:

        df_tmp = pd.read_csv(os.path.join(dirname, filename))

        df_tmp.rename(columns={df_tmp.columns[1]:'Date',df_tmp.columns[2]:'Study',df_tmp.columns[3]:'Study Link',df_tmp.columns[4]:'Journal'},inplace=True)

        if initialize == True:

            

            df_inreview = df_tmp[df_tmp.columns[1:5]]

            initialize = False

        else:

            df_inreview = df_inreview.append(df_tmp[df_tmp.columns[1:5]],sort=False)

        

# Any results you write to the current directory are saved as output.

df_inreview = df_inreview[['Date','Study','Journal']].drop_duplicates()

df_inreview['Month'] = pd.to_datetime(df_inreview['Date'], errors='coerce')

df_inreview['Month'] = pd.DatetimeIndex(df_inreview['Month']).month
df_coverage = df_metadata.groupby('Month').count()[['doi']].merge(df_inreview.groupby('Month')['Study'].count(),how='inner',left_index=True,right_index=True).rename(columns={'doi':'Total','Study':'In Lit Review'})

print("Covers {}% of the studies published since February 1 ({} of the {} papers)".format(np.round(df_coverage[df_coverage.index > 1].sum()[1]/df_coverage[df_coverage.index > 1].sum()[0],3)*100,df_coverage[df_coverage.index > 1].sum()[1],df_coverage[df_coverage.index > 1].sum()[0]))



print("Covers <a href=\"https://www.kaggle.com/antgoldbloom/literature-review-coverage/\" target=\"_blank\">{}%</a> of the studies published since February 1 ({} of the {} papers)".format(np.round(df_coverage[df_coverage.index > 1].sum()[1]/df_coverage[df_coverage.index > 1].sum()[0],3)*100,df_coverage[df_coverage.index > 1].sum()[1],df_coverage[df_coverage.index > 1].sum()[0]))
