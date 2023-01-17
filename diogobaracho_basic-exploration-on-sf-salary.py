import numpy as np
import pandas as pd
sal = pd.read_csv('../input/Salaries.csv',

                  dtype={'EmployeeName':str,

                         'BasePay':np.float64,

                         'JobTitle':str,

                         'OvertimePay':np.float64,

                         'OtherPay':np.float64,

                         'TotalPay':np.float64,

                         'TotalPayBenefits':np.float64,

                         'Benefits':str,

                         'Agency':str,

                         'Status':str},

                  low_memory=False,

                  na_values="Not Provided")
sal['BasePay'].mean()
sal[sal['TotalPayBenefits']==sal['TotalPayBenefits'].max()]
sal.iloc[sal['TotalPayBenefits'].idxmin()]
sal.groupby('Year').mean()['BasePay']
sal['JobTitle'].value_counts().head(5)
sal.groupby(['Year']).max()
sal.head()
sal.describe()
sal.info()