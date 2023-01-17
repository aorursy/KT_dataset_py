import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"
pd.options.display.max_columns = 200
print(os.listdir("../input"))
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
application_train = pd.read_csv('../input/application_train.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
installments_payments = pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
application_test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
def get_info(df):
    print('Dataframe dimensions:', df.shape)
    print("file sample example:")
    display(df.head())

    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
                             .T.rename(index={0:'null values (%)'}))
    print("Data type and Null values")
    display(tab_info)
    return
get_info(application_train)
print("positive values: {:<5.2f}%".format(sum(application_train['TARGET'] == 1) / application_train.shape[0] * 100))
get_info(bureau)
get_info(POS_CASH_balance)
get_info(previous_application)
get_info(installments_payments)
