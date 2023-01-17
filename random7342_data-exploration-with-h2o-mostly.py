# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import h2o
#connecting to cluster

h2o.init(strict_version_check=False)
data_csv = "/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv"

data = h2o.import_file(data_csv)
data.head()
data.describe()
cols_names = data.columns #because we know the data type for all the columns (they are all ints)

cols_names
# Overall percentage of defaulting



print(data['default.payment.next.month'].sum() / len(data['default.payment.next.month']), "%", sep="")
#Let's print out the unique values for each categorical data type - this can tell us if we have any missing data



not_categorical = ['ID',

 'LIMIT_BAL',

 'BILL_AMT1',

 'BILL_AMT2',

 'BILL_AMT3',

 'BILL_AMT4',

 'BILL_AMT5',

 'BILL_AMT6',

 'PAY_AMT1',

 'PAY_AMT2',

 'PAY_AMT3',

 'PAY_AMT4',

 'PAY_AMT5',

 'PAY_AMT6']



for col in cols_names:

    if col not in not_categorical:

        uniq_vals = h2o.as_list(data[col].unique(), use_pandas=False, header=False)

        uniq_vals = [val for sublist in uniq_vals for val in sublist] #flattening

        uniq_vals.sort()

        print(col + ": ", uniq_vals, "\n")
ed_counts = data[['EDUCATION']].table(data2=None,dense=True)



total = round(ed_counts['Count'].sum()) #rounding because of the python integer addition problem - see https://docs.python.org/2/tutorial/floatingpoint.html

print(total) #30000 should be the value



no_ed_missing = round(ed_counts[ed_counts['EDUCATION'] == 0]['Count'].sum())



print("Number of missing values:", no_ed_missing)

print("Percent missing: ", no_ed_missing*100/total, "%", sep="")
ma_counts = data[['MARRIAGE']].table(data2=None,dense=True)





total = round(ma_counts['Count'].sum()) #rounding because of the python integer addition problem - see https://docs.python.org/2/tutorial/floatingpoint.html

print(total) #30000 should be the value



no_ma_missing = round(ma_counts[ma_counts['MARRIAGE'] == 0]['Count'].sum())



print("Number of missing values:", no_ma_missing)

print("Percent missing: ", no_ma_missing*100/total, "%", sep="")
%matplotlib inline



#print(data[['EDUCATION']].table(data2=None,dense=True))



for col in cols_names:

    if col not in not_categorical:

        data[col].hist()
for col in cols_names:

    if col in not_categorical:

        data[col].hist()
#Will have to see if this is possible for certain repayment statuses



for i in range (1, 7):

    pay_var = 'PAY_AMT' + str(i)

    bill_var = 'BILL_AMT' + str(i)

    frac_var = 'FRACT_PAY' + str(i)



    temp = data[[pay_var, bill_var, 'default.payment.next.month']]

    temp[frac_var] = temp[pay_var] / temp[bill_var]



    #Now, we want a table of only when the PAY_AMT > BILL_AMT. Which means FRACT_PAY > 1.

    mask = temp[frac_var] > 1

    temp = temp[mask,:]



    #as can't easily figure out how to create a line plot with h2o (only hist())

    temp = temp.as_data_frame()

    #print(temp)



    print("Month", i, ": ", temp['default.payment.next.month'].sum()*100 / len(temp['default.payment.next.month']), "%", sep="")
#calculating percent of people who've paid minimum that defualted for each month



data.rename(columns={"PAY_0": "PAY_1"})



for i in range (1, 7):

    status_var = 'PAY_' + str(i)



    temp = data[[status_var, 'default.payment.next.month']]



    #Now, we want a table of only when the PAY_0 == 0.

    mask = temp[status_var] == 0

    temp = temp[mask,:]



    #as can't easily figure out how to create a line plot with h2o (only hist())

    temp = temp.as_data_frame()

    #print(temp)



    print("Month", i, ": ", temp['default.payment.next.month'].sum()*100 / len(temp['default.payment.next.month']), "%", sep="")

for i in range (1, 7):

    status_var = 'PAY_' + str(i)



    temp = data[[status_var, 'default.payment.next.month']]



    #Now, we want a table of only when the PAY_0 == -1.

    mask = temp[status_var] == -1

    temp = temp[mask,:]



    #as can't easily figure out how to create a line plot with h2o (only hist())

    temp = temp.as_data_frame()

    #print(temp)



    print("Month", i, ": ", temp['default.payment.next.month'].sum()*100 / len(temp['default.payment.next.month']), "%", sep="")

    
status_var1 = 'PAY_1'

status_var6 = 'PAY_6'



temp = data[[status_var1, status_var6, 'default.payment.next.month']]



mask = temp[status_var1] == -1

temp = temp[mask,:]



mask = temp[status_var6] == -1

temp = temp[mask,:]



temp = temp.as_data_frame()



#print("Month", ": ", temp['default.payment.next.month'].sum()*100 / len(temp['default.payment.next.month']), "%", sep="")

#print("Month", ": ", temp['default.payment.next.month'].sum()*100 / len(temp['default.payment.next.month']), "%", sep="")
# If they have a payment status of -1 on their first month, they are [likely/not-likely] to default

# If they are [likely/not-likely] to get a -1 on their last month, they are [likely/not-likely] to default





first_month = 'PAY_1'

last_month = 'PAY_6'



temp1 = data[[first_month, 'default.payment.next.month']]

temp2 = data[[last_month, 'default.payment.next.month']]



#Now, we want a table of only when the PAY_0 == 0.

mask = temp1[first_month] == -1

temp1 = temp1[mask,:]



mask = temp2[last_month] == -1

temp2 = temp2[mask,:]



print("Month1 (recent month)", ": ", temp1['default.payment.next.month'].sum()*100 / len(temp1['default.payment.next.month']), "%", sep="")

print("Month6 (first month)", ": ", temp2['default.payment.next.month'].sum()*100 / len(temp2['default.payment.next.month']), "%", sep="")
import matplotlib.pyplot as plt

from numpy import arange



count = []



for i in range (1, 7):

    month = 'PAY_' + str(i)

    temp = data[[month, 'default.payment.next.month']]

    count.append(data[month].table())

    count[i-1] = count[i-1].as_data_frame()



    plt.plot(count[i-1]['PAY_' + str(i)].tolist(), count[i-1]['Count'].tolist())



plt.xticks(arange(-2, 8, step=1))

plt.legend(["Month_1", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6"])
import matplotlib.pyplot as plt

from numpy import arange



count = []



for i in range (1, 7):

    count.append(data['PAY_' + str(i)].table())

    count[i-1] = count[i-1].as_data_frame()





    plt.plot(count[i-1]['PAY_' + str(i)].tolist(), count[i-1]['Count'].tolist())



plt.xticks(arange(-2, 8, step=1))

plt.legend(["Month_1", "Month_2", "Month_3", "Month_4", "Month_5", "Month_6"])
corr_matrix = data[data.columns].cor()

corr_matrix
import matplotlib.pyplot as plt

import seaborn as sns





plt.figure(figsize=(10,10))



#converting to data frame - couldn't easily figure out how to do it with h2o

corr = corr_matrix.as_data_frame()

corr.index = corr_matrix.columns

sns.heatmap(corr, annot = True, cmap='RdYlGn', vmin=-1, vmax=1)

plt.title("Correlation Heatmap", fontsize=16)

plt.show()
#Let's look at the highly-correlated features - if any are highly correlated then it may help to drop one in the feature selection stage (we don't need linearly dependent features)



#Only makes sense for numerical features (and we actually have lots of categorical features)
for col in cols_names:

    if col in not_categorical:

        print(data[col].describe())