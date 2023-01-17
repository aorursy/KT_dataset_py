import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time

from datetime import datetime

from collections import Counter

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')

#sample_submission_zero= pd.read_csv('../input/sample_submission_zero.csv')

members = pd.read_csv('../input/members.csv')

transactions = pd.read_csv('../input/transactions.csv')

#user_logs = pd.read_csv('../input/user_logs.csv',nrows = 2e7)



train.head()
train.info()
train.describe()
training = pd.merge(left = train,right = members,how = 'left',on=['msno'])

training.head()
training.info()
training['city'] = training.city.apply(lambda x: int(x) if pd.notnull(x) else "NAN")

training['registered_via'] = training.registered_via.apply(lambda x: int(x) if pd.notnull(x) else "NAN")

training['gender']=training['gender'].fillna("NAN")

training.info()
training['registration_init_time'] = training.registration_init_time.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN" )

training['expiration_date'] = training.expiration_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")

training.head()
# City count

plt.figure(figsize=(12,12))

plt.subplot(411)

city_order = training['city'].unique()

city_order=sorted(city_order, key=lambda x: float(x))

sns.countplot(x="city", data=training , order = city_order)

plt.ylabel('Count', fontsize=12)

plt.xlabel('City', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of City Count", fontsize=12)

plt.show()

city_count = Counter(training['city']).most_common()

print("City Count " +str(city_count))



#Registered Via Count

plt.figure(figsize=(12,12))

plt.subplot(412)

R_V_order = training['registered_via'].unique()

R_V_order = sorted(R_V_order, key=lambda x: str(x))

R_V_order = sorted(R_V_order, key=lambda x: float(x))

#above repetion of commands are very silly, but this was the only way I was able to diplay what I wanted

sns.countplot(x="registered_via", data=training,order = R_V_order)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Registered Via', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Registered Via Count", fontsize=12)

plt.show()

RV_count = Counter(training['registered_via']).most_common()

print("Registered Via Count " +str(RV_count))



#Gender count

plt.figure(figsize=(12,12))

plt.subplot(413)

sns.countplot(x="gender", data=training)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Gender', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Gender Count", fontsize=12)

plt.show()

gender_count = Counter(training['gender']).most_common()

print("Gender Count " +str(gender_count))



#registration_init_time yearly trend

training['registration_init_time_year'] = pd.DatetimeIndex(training['registration_init_time']).year

training['registration_init_time_year'] = training.registration_init_time_year.apply(lambda x: int(x) if pd.notnull(x) else "NAN" )

year_count=training['registration_init_time_year'].value_counts()

#print(year_count)

plt.figure(figsize=(12,12))

plt.subplot(311)

year_order = training['registration_init_time_year'].unique()

year_order=sorted(year_order, key=lambda x: str(x))

year_order = sorted(year_order, key=lambda x: float(x))

sns.barplot(year_count.index, year_count.values,order=year_order)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Year', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Yearly Trend of registration_init_time", fontsize=12)

plt.show()

year_count_2 = Counter(training['registration_init_time_year']).most_common()

print("Yearly Count " +str(year_count_2))



#registration_init_time monthly trend

training['registration_init_time_month'] = pd.DatetimeIndex(training['registration_init_time']).month

training['registration_init_time_month'] = training.registration_init_time_month.apply(lambda x: int(x) if pd.notnull(x) else "NAN" )

month_count=training['registration_init_time_month'].value_counts()

plt.figure(figsize=(12,12))

plt.subplot(312)

month_order = training['registration_init_time_month'].unique()

month_order = sorted(month_order, key=lambda x: str(x))

month_order = sorted(month_order, key=lambda x: float(x))

sns.barplot(month_count.index, month_count.values,order=month_order)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Month', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Monthly Trend of registration_init_time", fontsize=12)

plt.show()

month_count_2 = Counter(training['registration_init_time_month']).most_common()

print("Monthly Count " +str(month_count_2))



#registration_init_time day wise trend

training['registration_init_time_weekday'] = pd.DatetimeIndex(training['registration_init_time']).weekday_name

training['registration_init_time_weekday'] = training.registration_init_time_weekday.apply(lambda x: str(x) if pd.notnull(x) else "NAN" )

day_count=training['registration_init_time_weekday'].value_counts()

plt.figure(figsize=(12,12))

plt.subplot(313)

#day_order = training['registration_init_time_day'].unique()

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','NAN']

sns.barplot(day_count.index, day_count.values,order=day_order)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Day', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Day-wise Trend of registration_init_time", fontsize=12)

plt.show()

day_count_2 = Counter(training['registration_init_time_weekday']).most_common()

print("Day-wise Count " +str(day_count_2))
members.info()
# City count in Members Data Set

plt.figure(figsize=(12,12))

plt.subplot(311)

sns.countplot(x="city", data=members)

plt.ylabel('Count', fontsize=12)

plt.xlabel('City', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of City Count in Members Data Set", fontsize=12)

plt.show()

city_count = Counter(members['city']).most_common()

print("City Count " +str(city_count))



#Registered Via Count in Members Data Set

plt.figure(figsize=(12,12))

plt.subplot(312)

sns.countplot(x="registered_via", data=members)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Registered Via', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Registered Via Count in Members Data Set", fontsize=12)

plt.show()

RV_count = Counter(members['registered_via']).most_common()

print("Registered Via Count " +str(RV_count))





#Gender count in Members Data Set

plt.figure(figsize=(12,12))

plt.subplot(313)

sns.countplot(x="gender", data=members)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Gender', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Gender Count in Members Data Set", fontsize=12)

plt.show()

gender_count = Counter(members['gender']).most_common()

print("Gender Count " +str(gender_count))

tmp_1=training.bd.value_counts()

tmp_1.head()
training['bd'] = training.bd.apply(lambda x: int(x) if pd.notnull(x) else "NAN" )

bd_count = Counter(training['bd']).most_common()

print("BD Count " +str(bd_count))
#training.loc[(training['bd'] <= 1), 'bd'] = -99999

#training.loc[(training['bd'] >= 100), 'bd'] = -99999

training['bd'] = training.bd.apply(lambda x: -99999 if float(x)<=1 else x )

training['bd'] = training.bd.apply(lambda x: -99999 if float(x)>=100 else x )
#Birth Date count in training Data Set

plt.figure(figsize=(12,8))

bd_order = training['bd'].unique()

bd_order = sorted(bd_order, key=lambda x: str(x))

bd_order = sorted(bd_order, key=lambda x: float(x))

#above repetion of commands are very silly, but this was the only way I was able to diplay what I wanted

sns.countplot(x="bd", data=training , order = bd_order)

plt.ylabel('Count', fontsize=12)

plt.xlabel('BD', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of BD Count", fontsize=12)

plt.show()

bd_count = Counter(training['bd']).most_common()

print("BD Count " +str(bd_count))
tmp_bd = training[(training.bd != "NAN") & (training.bd != -99999)]

print("Mean of Birth Date = " +str(np.mean(tmp_bd['bd'])))

print("Median of Birth Date = " +str(np.median(tmp_bd['bd'])))

#print("Mode of Birth Date = " +str(np.mode(tmp_bd['bd'])))

plt.figure(figsize=(12,8))

plt.subplot(211)

bd_order_2 = tmp_bd['bd'].unique()

bd_order_2 = sorted(bd_order_2, key=lambda x: float(x))

sns.countplot(x="bd", data=tmp_bd , order = bd_order_2)

plt.ylabel('Count', fontsize=12)

plt.xlabel('BD', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of BD Count without ouliers and NAN values", fontsize=12)

plt.show()



plt.figure(figsize=(4,12))

plt.subplot(212)

sns.boxplot(y=tmp_bd["bd"],data=tmp_bd)

plt.xlabel('BD', fontsize=12)

plt.title("Box Plot of Birth Date without ouliers and NAN values", fontsize=12)

plt.show()
#Gender

gender_crosstab=pd.crosstab(training['gender'],training['is_churn'])

gender_crosstab.plot(kind='bar', stacked=True, grid=True)

gender_crosstab["Ratio"] =  gender_crosstab[1] / gender_crosstab[0]

gender_crosstab
#Registered Via

registered_via_crosstab=pd.crosstab(training['registered_via'],training['is_churn'])

registered_via_crosstab.plot(kind='bar', stacked=True, grid=True)

registered_via_crosstab["Ratio"] =  registered_via_crosstab[1] / registered_via_crosstab[0]

registered_via_crosstab
#city

city_crosstab=pd.crosstab(training['city'],training['is_churn'])

city_crosstab.plot(kind='bar', stacked=True, grid=True)

city_crosstab["Ratio"] =  city_crosstab[1] / city_crosstab[0]

city_crosstab
#Birth Date

sns.boxplot(x=tmp_bd["is_churn"],y=tmp_bd["bd"],data=tmp_bd);

del tmp_bd # memory cleaning
transactions.head()
transactions.info()
transactions.describe()
# payment_method_id count in transactions Data Set

plt.figure(figsize=(18,6))

#plt.subplot(311)

sns.countplot(x="payment_method_id", data=transactions)

plt.ylabel('Count', fontsize=12)

plt.xlabel('payment_method_id', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of payment_method_id Count in transactions Data Set", fontsize=12)

plt.show()

payment_method_id_count = Counter(transactions['payment_method_id']).most_common()

print("payment_method_id Count " +str(payment_method_id_count))

# payment_plan_days count in transactions Data Set

plt.figure(figsize=(18,6))

sns.countplot(x="payment_plan_days", data=transactions)

plt.ylabel('Count', fontsize=12)

plt.xlabel('payment_plan_days', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of payment_plan_days Count in transactions Data Set", fontsize=12)

plt.show()

payment_plan_days_count = Counter(transactions['payment_plan_days']).most_common()

print("payment_plan_days Count " +str(payment_plan_days_count))

# plan_list_price count in transactions Data Set

plt.figure(figsize=(18,6))

sns.countplot(x="plan_list_price", data=transactions)

plt.ylabel('Count', fontsize=12)

plt.xlabel('plan_list_price', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of plan_list_price Count in transactions Data Set", fontsize=12)

plt.show()

plan_list_price_count = Counter(transactions['plan_list_price']).most_common()

print("plan_list_price Count " +str(plan_list_price_count))

# actual_amount_paid count in transactions Data Set

plt.figure(figsize=(18,6))

sns.countplot(x="actual_amount_paid", data=transactions)

plt.ylabel('Count', fontsize=12)

plt.xlabel('actual_amount_paid', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of actual_amount_paid Count in transactions Data Set", fontsize=12)

plt.show()

actual_amount_paid_count = Counter(transactions['actual_amount_paid']).most_common()

print("actual_amount_paid Count " +str(actual_amount_paid_count))
# is_auto_renew count in transactions Data Set

plt.figure(figsize=(4,4))

sns.countplot(x="is_auto_renew", data=transactions)

plt.ylabel('Count', fontsize=12)

plt.xlabel('is_auto_renew', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of is_auto_renew Count in transactions Data Set", fontsize=6)

plt.show()

is_auto_renew_count = Counter(transactions['is_auto_renew']).most_common()

print("is_auto_renew Count " +str(is_auto_renew_count))
# is_cancel count in transactions Data Set

plt.figure(figsize=(4,4))

sns.countplot(x="is_cancel", data=transactions)

plt.ylabel('Count', fontsize=12)

plt.xlabel('is_cancel', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of is_cancel Count in transactions Data Set", fontsize=6)

plt.show()

is_cancel_count = Counter(transactions['is_cancel']).most_common()

print("is_cancel Count " +str(is_cancel_count))
#Changing the format of dates in YYYY-MM-DD in transaction data set

#transactions['transaction_date'] = transactions.transaction_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date())

#transactions['membership_expire_date'] = transactions.membership_expire_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date())

#transactions.head()
#Correlation between plan_list_price and actual_amount_paid

transactions['plan_list_price'].corr(transactions['actual_amount_paid'],method='pearson')                           
#transactions['msno'].value_counts() 

(transactions['msno'].value_counts().reset_index())['msno'].value_counts()
#tmp1=transactions['msno'].value_counts() 

#tmp2=tmp1[tmp1==8]

#print(tmp2.head(1))

#del tmp1, tmp2
tmp1=transactions[transactions.msno=="LNScSgIQZsX+hC3eVrwGFcdan0nftusOwk0jMAu7q9I="]

tmp1=tmp1.sort_values('transaction_date')

tmp1
del tmp1 # memory cleaning
#merging the training and transaction data set

training = pd.merge(left = training,right = transactions ,how = 'left',on=['msno'])



#changing the format of the dates

training['transaction_date'] = training.transaction_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN" )

training['membership_expire_date'] = training.membership_expire_date.apply(lambda x: datetime.strptime(str(int(x)), "%Y%m%d").date() if pd.notnull(x) else "NAN")

training.head()
