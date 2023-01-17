import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

application_test = pd.read_csv("../input/application_test.csv")
application_train = pd.read_csv("../input/application_train.csv")
bureau = pd.read_csv("../input/bureau.csv",chunksize=1000)
bureau_balance = pd.read_csv("../input/bureau_balance.csv",chunksize=1000)
credit_card_balance = pd.read_csv("../input/credit_card_balance.csv",chunksize=1000)
installments_payments = pd.read_csv("../input/installments_payments.csv",chunksize=1000)
POS_CASH_balance = pd.read_csv("../input/POS_CASH_balance.csv",chunksize=1000)
previous_application = pd.read_csv("../input/previous_application.csv",chunksize=1000)
sample_submission = pd.read_csv("../input/sample_submission.csv",chunksize=1000)
for i in bureau:
    bureau=pd.DataFrame(i)
    break
for i in credit_card_balance:
    credit_card_balance=pd.DataFrame(i)
    break
for i in installments_payments:
    installments_payments=pd.DataFrame(i)
    break
for i in POS_CASH_balance:
    POS_CASH_balance=pd.DataFrame(i)
    break
for i in previous_application:
    previous_application=pd.DataFrame(i)
    break
for i in sample_submission:
    sample_submission=pd.DataFrame(i)
    break   
    
plt.figure(figsize=(12,6))
plt.subplot(121)
application_train["TARGET"].value_counts().plot(fontsize = 16,
                                        kind = 'pie',
                                        autopct = "%1.0f%%",
                                        colors = sns.color_palette(),
                                        labels=["1 - Non Defaulter","0 - Defaulter"],
                                       )
plt.title("Distribution of Target", fontsize=30)
plt.figure(figsize=(12,6))
plt.subplot(121)
application_train["NAME_EDUCATION_TYPE"].value_counts().plot(fontsize = 12,
                                        kind = 'pie',
                                        autopct = "%1.0f%%",
                                        colors = sns.color_palette(),
                                       )
plt.title("Distribution of NAME_EDUCATION_TYPE", fontsize=30)
plt.tight_layout()
sns.barplot(y="NAME_EDUCATION_TYPE", x="AMT_CREDIT", hue="TARGET", data=application_train)
plt.title("Education categories v/s CreditWorthiness", fontsize=20)
plt.xlim(0,1000000)
sns.barplot(x="CODE_GENDER", y="AMT_CREDIT", hue="TARGET", data=application_train)
plt.title("Credit v/s Gender", fontsize=20)
sns.barplot(y="NAME_INCOME_TYPE", x="AMT_CREDIT", hue="TARGET", data=application_train)
plt.title("INCOME Type v/s CreditWorthiness", fontsize=20)
plt.xlim(0,1250000)