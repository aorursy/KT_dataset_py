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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data_application_train = pd.read_csv("../input/application_train.csv")

pd.options.display.max_columns = len(data_application_train.columns)

data_application_train.head()
import missingno as msno

msno.bar(data_application_train)
total = len(data_application_train)

sum_income = data_application_train["AMT_INCOME_TOTAL"].isnull().sum()

sum_occupation = data_application_train["OCCUPATION_TYPE"].isnull().sum()



print("AMT_INCOME_TOTAL  num of missing data={}　 missing data rate={:.2f}[%]".format(sum_income, 100*sum_income/total))

print("OCCUPATION_TYPE   num of missing data={}　 missing data rate={:.2f}[%]".format(sum_occupation, 100*sum_occupation/total))
sns.countplot(x="TARGET", data=data_application_train)

plt.title("Data volume in each class('Target')")

plt.show()



count_1 = (data_application_train["TARGET"] == 1).sum()

count_0 = (data_application_train["TARGET"] == 0).sum()

print("Non-repayment rate={:.2f}[%]".format((count_1/(count_0+count_1))*100))
#Extract require data

data_to_use = data_application_train[["TARGET","AMT_INCOME_TOTAL","OCCUPATION_TYPE"]]
sns.violinplot(y='AMT_INCOME_TOTAL', data=data_to_use)

plt.ylabel("Income")

plt.title("Income distribution")

plt.show()

data_to_use.describe()
def binning_data(data_source, col_name, binned_col_name, num_of_bin=10):

    """

    Binning the specified column data

    & add the new column that has bin label to original data

    

    parameter

    --------------

    data_source : Pandas dataframe

    col_name : string 

    binned_col_name : string 

    num_of_bin : int

    

    return

    --------------

    data_source : pandas dataframe

    

    """

    #Check quartile value

    data_info = data_source.describe()

    bin_min = data_info.loc["25%", col_name]

    bin_max = data_info.loc["75%", col_name]

    IQR = bin_max - bin_min

    #Use interquartile value to decide bin_min & max value

    bin_min = bin_min-IQR*1.5 if bin_min > IQR*1.5 else 0

    bin_max += IQR*1.5

    

    bin_width = (int)((bin_max - bin_min) / num_of_bin)

    #bins = [value for value in gen_range(bin_min, bin_max, bin_width )]

    bins = [value for value in range((int)(bin_min), (int)(bin_max), bin_width )]

    labels = [i for i in range(0, len(bins)-1)]

    binned_label = pd.cut(data_source[col_name], bins=bins, labels=labels)

    data_source[binned_col_name] = binned_label

    

    for i in range(0, len(bins)-1):

        print("range label={} : range={:.5f}~{:.5f}".format(i, bins[i], bins[i+1]) )

    

    return data_source
#Binning data

data_to_use = binning_data(data_to_use, "AMT_INCOME_TOTAL", "BINNED_AMT_INCOME_TOTAL", 8)
#Check distribution

income_destribution_w_income_range = data_to_use.groupby("BINNED_AMT_INCOME_TOTAL", as_index=False).count()

plt.bar(income_destribution_w_income_range["BINNED_AMT_INCOME_TOTAL"]

        , income_destribution_w_income_range["TARGET"])

plt.xlabel('Low<- Income range label ->High)')

plt.ylabel("count")

plt.title("Income distribution")

for x, y in zip(income_destribution_w_income_range["BINNED_AMT_INCOME_TOTAL"]

                , income_destribution_w_income_range["TARGET"]):

    plt.text(x, y, y, ha='center', va='bottom')

plt.show()
#Cal non-repayment rate

non_repayment_rate_w_income_range = data_to_use.groupby("BINNED_AMT_INCOME_TOTAL", as_index=False).mean()

non_repayment_rate_w_income_range["TARGET"] *= 100 #% notation



#Bar plot

plt.figure(figsize=(10,6))

plt.bar(non_repayment_rate_w_income_range["BINNED_AMT_INCOME_TOTAL"]

        , non_repayment_rate_w_income_range["TARGET"], color="Blue")

plt.xlabel('Low<- Income range label ->High')

plt.ylabel('non-payment rate[%]')

plt.ylim(0,10)

plt.title("non-payment rate")



for x, y in zip(non_repayment_rate_w_income_range["BINNED_AMT_INCOME_TOTAL"]

                , non_repayment_rate_w_income_range["TARGET"]):

    plt.text( x, y, str("{:.2f}").format(y), ha='center', va='bottom')

plt.show()
#Cal non-repayment rate for each occupation type

non_repayment_rate_w_occupation = data_to_use.groupby("OCCUPATION_TYPE", as_index=False).mean()

non_repayment_rate_w_occupation["TARGET"] *= 100

#non_repayment_rate_w_occupation["OCCUPATION_TYPE_LABEL"] = [i for i in range(0, len(non_repayment_rate_w_occupation))]

print(non_repayment_rate_w_occupation)



#Bar plot

plt.figure(figsize=(30,12))

plt.bar(non_repayment_rate_w_occupation.index, non_repayment_rate_w_occupation["TARGET"], color="Blue")

plt.xlabel("OCCUPATION_TYPE")

plt.ylabel("Non-payment rate[%]")

plt.title("Non-payment rate")



for x, y in zip(non_repayment_rate_w_occupation.index, non_repayment_rate_w_occupation["TARGET"]):

    plt.text(x, y, str("{:.2f}").format(y), ha='center', va='bottom')



plt.xticks(non_repayment_rate_w_occupation.index, non_repayment_rate_w_occupation["OCCUPATION_TYPE"])

plt.show()
#1.Drop high missing value data

#Set drop criteria as 40%. If missing value rate is over 40%, then drop those features.

REDUCTION_CRITERIA_FOR_MISSING_DATA = 40 #[%] 

total = len(data_application_train)

col_name = data_application_train.columns

missing_data_rate = [100*data_application_train[col_name[i]].isnull().sum() / total for i in range(0, len(col_name))]



for i in range(0, len(col_name)):

    if REDUCTION_CRITERIA_FOR_MISSING_DATA < missing_data_rate[i]:

        del data_application_train[col_name[i]]



msno.bar(data_application_train)

print("Num of Featurs: Before reduction {} => After Reduction {}".format(len(col_name), len(data_application_train.columns.values)))
object_data_application_train = data_application_train.select_dtypes(['object'])

object_data_application_train.head()
#Replace object type data to number

object_col_name = object_data_application_train.columns.values

array_object_to_int = np.array([object_data_application_train[object_col_name[i]].unique() for i in range(0,len(object_col_name))])



for i in range(0,len(object_col_name)):

    labels, uniques = pd.factorize(data_application_train[object_col_name[i]])

    data_application_train[object_col_name[i]] = labels   



#Repalce "NAN" to -1    

data_application_train = data_application_train.replace(np.nan, -1)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
#Drop SK_ID_CURR

del data_application_train["SK_ID_CURR"]

#Drop OCCUPATION_TYPE (Already done in Step2)

del data_application_train["OCCUPATION_TYPE"]
target = data_application_train['TARGET']

feature = data_application_train.iloc[:,1:len(data_application_train.columns)]

model.fit(feature, target)
rank = np.argsort(-model.feature_importances_)

f, ax = plt.subplots(figsize=(11, 11)) 

sns.barplot(x=model.feature_importances_[rank], y=data_application_train.columns.values[rank], orient='h')

ax.set_xlabel("Importance")

plt.tight_layout()

plt.show()
#Pick up top10 important feature

for i in range(0, 10):

    print("Rank{} => {} (Importance={:.3f})".format(i+1, 

                                                    data_application_train.columns.values[rank[i]], 

                                                    model.feature_importances_[rank[i]]))
train_data =  pd.read_csv("../input/application_train.csv")
#Replace object type data to number

object_train_data = train_data.select_dtypes(['object'])

object_col_name = object_train_data.columns.values

array_object_to_int = np.array([object_train_data[object_col_name[i]].unique() for i in range(0,len(object_col_name))])



for i in range(0,len(object_col_name)):

    labels, uniques = pd.factorize(train_data[object_col_name[i]])

    train_data[object_col_name[i]] = labels   



#Repalce "NAN" to -1    

train_data = train_data.replace(np.nan, -1)
#Drop SK_ID_CURR

del train_data["SK_ID_CURR"]

#Drop OCCUPATION_TYPE (Already done in Step2)

del train_data["OCCUPATION_TYPE"]
target = train_data['TARGET']

feature = train_data.iloc[:,1:len(train_data.columns)]

model.fit(feature, target)
rank = np.argsort(-model.feature_importances_)

f, ax = plt.subplots(figsize=(15, 15)) 

sns.barplot(x=model.feature_importances_[rank], y=train_data.columns.values[rank], orient='h')

ax.set_xlabel("Importance")

plt.tight_layout()

plt.show()
for i in range(0, 10):

    print("Rank{} => {} (Importance={:.3f})".format(i+1, train_data.columns.values[rank[i]], model.feature_importances_[rank[i]]))
data =  pd.read_csv("../input/application_train.csv")
def plot_non_repayment_rate(data_source, col_name, flag_rename_x_label=False):

    """

    Plot non-repayment rate. (Xaxis=Col_name, Yaxis=non-repayment rate)

    

    parameter

    --------------

    data_source : Pandas dataframe

    col_name : string 

    flag_rename_x_label : bool

    

    return

    --------------

    None

    """

    

    #Cal non-repayment rate

    non_repayment_rate = data_source.groupby(col_name, as_index=False).mean()

    non_repayment_rate["TARGET"] *= 100

    #print(non_repayment_rate)

    

    #Bar plot

    plt.figure(figsize=(25,10))

    plt.bar(non_repayment_rate.index, non_repayment_rate["TARGET"], color="Blue")

    plt.xlabel(col_name, fontsize=18)

    plt.ylabel("Non-payment rate[%]", fontsize=18)

    plt.title("Non-payment rate")

    plt.tight_layout()

    

    for x, y in zip(non_repayment_rate.index, non_repayment_rate["TARGET"]):

        plt.text(x, y, str("{:.2f}").format(y), ha='center', va='bottom')

        if flag_rename_x_label == True:

            plt.xticks(non_repayment_rate.index, non_repayment_rate[col_name])

    plt.show()
def gen_range(value_start, value_end, value_step):

    """

    Extend range() so it can handle float type data

    ----------------

    Parameter

        value_start:float

        value_end:float

        value_step:float

    ----------------

    ----------------

    Return

    　　value:float

    ----------------

    """

    value = value_start

    while value+value_step < value_end:

     yield value

     value += value_step
def binning_data2(data_source, col_name, binned_col_name, num_of_bin=10):

    """

    parameter

    --------------

    data_source : Pandas dataframe

    col_name : string 

    binned_col_name : string 

    num_of_bin : int

    

    return

    --------------

    data_source : pandas dataframe

    

    """

    #Use interquartile value to decide bin_min & max value

    data_info = data_source.describe()

    bin_min = data_info.loc["25%", col_name]

    bin_max = data_info.loc["75%", col_name]

    IQR = bin_max - bin_min

    bin_min = bin_min-IQR*1.5 if bin_min > IQR*1.5 else 0

    bin_max += IQR*1.5

    

    bin_width = (bin_max - bin_min) / num_of_bin

    bins = [value for value in gen_range(bin_min, bin_max, bin_width )]

    labels = [i for i in range(0, len(bins)-1)]

    binned_label = pd.cut(data_source[col_name], bins=bins, labels=labels)

    data_source[binned_col_name] = binned_label



    for i in range(0, len(bins)-1):

        print("range label={} : range={:.5f}~{:.5f}".format(i, bins[i], bins[i+1]) )

    

    return data_source
#ORGANIZATION_TYPE

plot_non_repayment_rate(data, 'ORGANIZATION_TYPE', False)

tmp = data["ORGANIZATION_TYPE"].unique()

for i in range(0, len(tmp)):

    print("{}:{}".format(i, tmp[i]))
#EXT_SOURCE_1

plot_non_repayment_rate(binning_data2(data, 'EXT_SOURCE_1', 'BIN_EXT_SOURCE_1', 8), 'BIN_EXT_SOURCE_1')
#EXT_SOURCE_2

plot_non_repayment_rate(binning_data2(data, 'EXT_SOURCE_2', 'BIN_EXT_SOURCE_2', 8), 'BIN_EXT_SOURCE_2')
#REGION_POPULATION_RELATIVE

plot_non_repayment_rate(binning_data2(data, 'REGION_POPULATION_RELATIVE', 'BIN_REGION_POPULATION_RELATIVE', 8), 'BIN_REGION_POPULATION_RELATIVE')
#AMT_CREDIT

plot_non_repayment_rate(binning_data(data, 'AMT_CREDIT', 'BIN_AMT_CREDIT', 8), 'BIN_AMT_CREDIT')
#DEF_60_CNT_SOCIAL_CIRCLE

plot_non_repayment_rate(data, 'DEF_60_CNT_SOCIAL_CIRCLE', True)
#CNT_CHILDREN

plot_non_repayment_rate(data, 'CNT_CHILDREN', True)
#AMT_ANNUITY

plot_non_repayment_rate(binning_data2(data, 'AMT_ANNUITY', 'BIN_AMT_ANNUITY', 8), 'BIN_AMT_ANNUITY')