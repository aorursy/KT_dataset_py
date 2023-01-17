# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
loan = pd.read_csv('../input/loan.csv',dtype={'desc':np.object,'verification_status_joint':np.object})

loan.shape

print ("We have %i rows and %i cols" % (loan.shape[0],loan.shape[1]))
loan.info(null_counts= True)


Personal_info = ['member_id',

                 'addr_state' ,

                 'annual_inc',

                 'dti','emp_length',

                 'emp_title',

                 'home_ownership',

                 'zip_code']

Loan_info = ['id',

            'application_type',

            'collection_recovery_fee',

            'desc',

            'funded_amnt',

            'funded_amnt_inv',

            'grade',

            'initial_list_status',

            'installment',

            'int_rate',

            'issue_d',

            'last_pymnt_amnt',

            'last_pymnt_d',

            'loan_amnt',

            'loan_status',

            'next_pymnt_d',

            'out_prncp',

            'out_prncp_inv',

            'policy_code',

            'purpose',

            'pymnt_plan',

            'recoveries',

            'sub_grade',

            'term',

            'title',

            'total_pymnt',

            'total_pymnt_inv',

             'total_rec_int',

             'total_rec_late_fee',

             'total_rec_prncp']

Joint_info = loan.columns[loan.columns.str.endswith('_joint')].tolist()

data = []

for i in loan.columns.tolist():

    # define the category of variable

    if i in Personal_info:

        category = 'personal information'

    elif i in Loan_info:

        category = 'loan information'

    elif i in Joint_info:

        category = 'unique for joint loan'

    else:

        category = 'credit records'

    

    # difine data level

    if i in loan.columns[loan.columns.str.endswith('_d')].tolist():

        level = 'date'

    elif loan[i].dropna().nunique() ==2:

        level = 'binary'

    elif loan[i].dtype == object:

        level = 'categorical'

    elif loan[i].dtype == float:

        level = 'numeric'

    else:

        level = 'id'

    

    # defining datatype

    dtype = loan[i].dtype

    f_dict = {'name':i,

            'category':category,

            'level':level,

            'dtype': dtype

            }

    data.append(f_dict)

    

metadata = pd.DataFrame(data,columns=['name','category','level','dtype'])

metadata
metadata.to_csv(metadata,index = False)
missing = loan.isnull().sum()

missing_ratio = missing[missing != 0]/loan.shape[0] * 100

missing_ratio