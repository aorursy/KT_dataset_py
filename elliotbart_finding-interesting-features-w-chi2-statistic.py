# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Any results you write to the current directory are saved as output.
PATH="../input"
application_train = pd.read_csv(PATH+"/application_train.csv")
application_test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
from sklearn.feature_selection import chi2, mutual_info_classif
from scipy.stats import chi2_contingency
categorical = [f for f in application_train.columns if application_train[f].dtype == 'object']
numerical = [f for f in application_train.columns if application_train[f].dtype != 'object' and (application_train[f]>0).all()]

plt.figure(figsize=(18,6))

cont = [pd.crosstab(application_train[f], application_train.TARGET) for f in categorical]
chi2_list = [chi2_contingency(c)[0] for c in cont]
chi2_list, categorical = (list(t) for t in zip(*sorted(zip(chi2_list, categorical), reverse=True)))

chiscoredata = pd.DataFrame({'Features':categorical, 'Chi2 scores':chi2_list})
plt.subplot(1,2,1)
plt.title("Chi2 scores for categorical features")
sns.barplot(y='Features', x = 'Chi2 scores', data= chiscoredata)


cont = [pd.crosstab(pd.cut(application_train[f],10, duplicates='drop'), application_train.TARGET) for f in numerical]
chi2_list = [chi2_contingency(c)[0] for c in cont]
chi2_list, numerical = (list(t) for t in zip(*sorted(zip(chi2_list, numerical), reverse=True)))

chiscoredata = pd.DataFrame({'Features':numerical, 'Chi2 scores':chi2_list})
plt.subplot(1,2,2)
plt.title("Chi2 scores for binned numerical features")
sns.barplot(y='Features', x = 'Chi2 scores', data= chiscoredata)

plt.tight_layout()