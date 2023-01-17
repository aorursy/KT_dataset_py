# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



pd.set_option('display.max_columns', None)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

deal_level_data = pd.read_csv("../input/fina2390-level-data/deal_level_data.csv")

quarter_level_data = pd.read_csv("../input/fina2390-level-data/quarter_level_data.csv")
deal_level_data.sample(5)

for index, col in enumerate(deal_level_data.columns): 

    print(index, col) 
quarter_level_data.head(26)
for col in quarter_level_data.columns: 

    print(col) 
# starting from quarter -12 

def hasNumbers(inputString):

    return any(char.isdigit() for char in inputString)



# create a list contains the columns of quarter

my_quarter_level_col = []

for col in deal_level_data.columns: 

    if not hasNumbers(col) and "Acq_" not in col:

        my_quarter_level_col.append(col)

        print(col) 

print(len(my_quarter_level_col)) 
my_quarter_level_data = ['Deal_Number', 'Date_Announced', 'Year_Announced', 'Acquirer_Name_clean', 

                         'Acquirer_Primary_SIC', 'Acquirer_State_abbr', 'Acquirer_CUSIP', 

                         'Acquirer_Ticker', 'Target_Name_clean', 'Target_Primary_SIC', 'Target_State_abbr',

                         'Target_CUSIP', 'Target_Ticker', 'Attitude', 'quarter_to_the_event_date',

                         'quarter', 'Com_Net_Charge_Off', 'Com_Insider_Loan', 'Com_NIE', 'Com_NII',

                         'Com_NIM', 'Com_ROA', 'Com_Total_Assets', 'Com_AvgSalary', 'Com_EmployNum',

                         'Com_TtlSalary', 'Com_AvgSalary_log', 'Com_EmployNum_log', 'Com_TtlSalary_log',

                         'Tar_Net_Charge_Off', 'Tar_Insider_Loan', 'Tar_NIE', 'Tar_NII', 'Tar_NIM', 

                         'Tar_ROA', 'Tar_AvgSalary', 'Tar_EmployNum', 'Tar_TtlSalary', 'Tar_Total_Assets',

                         'Tar_AvgSalary_log', 'Tar_EmployNum_log', 'Tar_TtlSalary_log']



my_quarter_level_data = pd.DataFrame(columns=my_quarter_level_data)



# print(deal_level_data.iloc[0])

quarter_date_string = ['__12', '__11', '__10', '__9', '__8', '__7', '__6', '__5', '__4', '__3', '__2', "__1",

                '', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_10', '_11', "_12"]



quarter_date_number = [-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1,

                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



quarter_level_col = ['quarter', 'Tar_Net_Charge_Off', 'Tar_Insider_Loan', 'Tar_NIE', 'Tar_NII', 'Tar_NIM', 'Tar_ROA', 

                     'Tar_AvgSalary', 'Tar_EmployNum', 'Tar_TtlSalary', 'Tar_Total_Assets', 'Com_Net_Charge_Off', 

                     'Com_Insider_Loan', 'Com_NIE', 'Com_NII', 'Com_NIM', 'Com_ROA', 'Com_Total_Assets', 'Com_AvgSalary',

                     'Com_EmployNum', 'Com_TtlSalary', 'Com_AvgSalary_log', 'Com_EmployNum_log', 'Com_TtlSalary_log',

                     'Tar_AvgSalary_log', 'Tar_EmployNum_log', 'Tar_TtlSalary_log']



count = 0 



for index, row in deal_level_data.iterrows():

    for quarter in range(25):

        my_quarter_level_data.loc[count] = row[0:14]

        # my_quarter_level_data.at[quarter, 'quarter_to_the_event_date'] = quarter_date_number[quarter]

        for column in quarter_level_col:

            # 

            my_quarter_level_data.at[count, column] = row[column + quarter_date_string[quarter]]

        count += 1

    if index == 10:

        break

#print(my_quarter_level_data)


