import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))



df_kaggle_test = pd.read_hdf(

         '../input/save-hdf-full/test.hdf',

         key="test"

)
from datetime import datetime



def add_timestamps(df):

    datedictAS = np.load('../input/timestamps/AvSigVersionTimestamps.npy')[()]

    df['DateAS'] = df['AvSigVersion'].map(datedictAS)  



    datedictOS = np.load('../input/timestamps/OSVersionTimestamps.npy')[()]

    df['DateOS'] = df['Census_OSVersion'].map(datedictOS)  

    # BL timestamp

    def convert(x):

        try:

            d = datetime.strptime(x.split('.')[4],'%y%m%d-%H%M')

        except:

            d = np.nan

        return d

    df['DateBL'] = df['OsBuildLab'].map(convert)

    

add_timestamps(df_kaggle_test)
df_kaggle_test.query("DateAS<'2018-10-26' and DateOS>='2018-10-26'").shape
df_kaggle_test.query("DateAS<'2018-10-26' and DateBL>='2018-10-26'").shape
df_submission=pd.read_csv("../input/nffm-baseline-0-690-on-lb/nffm_submission.csv")

df_submission.hist(bins=30)

df_submission.head()
df_submission.to_csv("original_submission.csv",index=None)
print("using only AvSigVersion Date, private split rows:",

      df_submission.loc[(df_kaggle_test.DateAS>="2018-10-26")].shape)



print("using all Dates available, private split rows:",

      df_submission.loc[(df_kaggle_test.DateAS>="2018-10-26")

                  | (df_kaggle_test.DateBL>="2018-10-26")

                  | (df_kaggle_test.DateOS>="2018-10-26")].shape)
df_private_submission=df_submission.copy()

df_private_submission.loc[~ ((df_kaggle_test.DateAS>="2018-10-26")

                  | (df_kaggle_test.DateBL>="2018-10-26")

                  | (df_kaggle_test.DateOS>="2018-10-26")),"HasDetections"]=0

df_private_submission.hist(bins=30)

df_private_submission.head()

df_private_submission.to_csv("private_lb_submission.csv",index=None)
df_submission.loc[(df_kaggle_test.DateAS>="2018-10-26")

                  | (df_kaggle_test.DateBL>="2018-10-26")

                  | (df_kaggle_test.DateOS>="2018-10-26"),"HasDetections"]=0

df_submission.hist(bins=30)

df_submission.head()
df_submission.to_csv("public_lb_submission.csv",index=None)