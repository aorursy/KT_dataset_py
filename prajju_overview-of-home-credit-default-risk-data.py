import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
application_test=pd.read_csv("../input/application_test.csv")
application_train=pd.read_csv("../input/application_train.csv")
bureau=pd.read_csv("../input/bureau.csv")
bureau_balance=pd.read_csv("../input/bureau_balance.csv")
credit_card_balance=pd.read_csv("../input/credit_card_balance.csv")
installments_payments=pd.read_csv("../input/installments_payments.csv")
POS_CASH_balance=pd.read_csv("../input/POS_CASH_balance.csv")
previous_application=pd.read_csv("../input/previous_application.csv")
sample_submission=pd.read_csv("../input/sample_submission.csv")
application_test.head()
application_train.head()
bureau.head()
bureau_balance.head()
credit_card_balance.head()
installments_payments.head()
POS_CASH_balance.head()
previous_application.head()
sample_submission.head()
from collections import Counter
count=Counter(application_train["TARGET"])
#Distribution of Target
import matplotlib.pyplot as plt
plt.pie([float(v) for v in count.values()], labels=[float(k) for k in count])
# NAME_CONTRACT_TYPE wise distribution of TARGET
application_train.groupby(["TARGET",'NAME_CONTRACT_TYPE']).size()

# gender wise distribution of TARGET
application_train.groupby(["TARGET",'CODE_GENDER']).size()
# no of unique SK_ID_CURR in bureau
len(set(bureau['SK_ID_CURR']))
# no of unique SK_ID_CURR in application train
len(set(application_train['SK_ID_CURR']))
# no of unique SK_ID_CURR in application test
len(set(application_test['SK_ID_CURR']))
# no of SK_ID_CURR which are present both in bureau and application train
len(set(bureau['SK_ID_CURR']).intersection(set(application_train['SK_ID_CURR'])))
# no of SK_ID_CURR which are present both in bureau and application test
len(set(bureau['SK_ID_CURR']).intersection(set(application_test['SK_ID_CURR'])))
import seaborn as sns
%matplotlib inline
fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(application_train.corr(), annot=True)
#columns with NaN in application train and thier percentages
a=pd.isnull(application_train).mean()
a[a!=0]
#columns with NaN in application test and thier percentages
a=pd.isnull(application_test).mean()
a[a!=0]
#outlier in REGION_POPULATION_RELATIVE
sns.boxplot("REGION_POPULATION_RELATIVE", data=application_train, palette="PRGn")

