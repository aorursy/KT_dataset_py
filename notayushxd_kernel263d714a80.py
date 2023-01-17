# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/competion_test.csv")
data.shape
data.drop(['id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_30','id_18','id_14','id_07','id_08','V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166']

  , axis = 1, inplace =True)
def make_day_feature(df, offset=0, tname='TransactionDT'):



    days = df[tname] / (3600*24)

    encoded_days = np.floor(days-1+offset) % 7

    return encoded_days

def make_hour_feature(df, tname='TransactionDT'):

    hours = df[tname] / (3600)

    encoded_hours = np.floor(hours) % 24

    return encoded_hours



data['hours'] = make_hour_feature(data)

data['weekday'] = make_day_feature(data, offset=0.58)

data['bothPR'] = data["P_emaildomain"] + ' ' +data["R_emaildomain"]
data["Proton_P"] = np.where(data['P_emaildomain']== "protonmail.com", 1, 0)

data["Proton_R"] = np.where(data['R_emaildomain']== "protonmail.com", 1, 0)

data.fillna(-9999, inplace=True)

data.head()
data.to_csv("new_data_ieee.csv", index = False)