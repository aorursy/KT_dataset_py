import numpy as np
import pandas as pd

# Reading of datasets
application_train = pd.read_csv('../input/application_train.csv')
application_test  = pd.read_csv('../input/application_train.csv')
bureau = pd.read_csv('../input/bureau.csv')
# Importing neccessary libraries to perform out the required EDA
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Distribution of the Target variable:
# How many clients have difficulties while repayment of the loan? 
sns.countplot(x="TARGET", data=application_train, palette="Blues")


# 1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample
# 0 - all other cases
# How Gender and Target variables are related to each other?
sns.barplot(x="CODE_GENDER", y="TARGET", data=application_train)
