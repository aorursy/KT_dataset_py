#%matplotlib inline

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os


print(os.listdir("../input"))

app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
print (app_train.columns)
print(app_train.head(5))
pd.set_option('display.max_rows', 122)
nulls_data = app_train.isnull().sum().sum()
print("There are {} null data on the dataset".format(nulls_data))
print(app_train.isnull().sum())
print(app_train.info(verbose=True))
app_train.TARGET.value_counts().plot.bar()
target_true = app_train[app_train["TARGET"] == 1].shape[0]
target_false = app_train[app_train["TARGET"] == 0].shape[0]
total = target_true + target_false
print(total)
print("target true: {}".format(target_true))
print("target false: {}".format(target_false))
print("Payment difficulties {}".format(target_true/(total)*100) )
print(app_train.CODE_GENDER.head(5))
app_train.CODE_GENDER.value_counts().plot.bar()
app_train_total = app_train.shape[0]
female_total = app_train[app_train.CODE_GENDER == 'F'].shape[0]
male_total = app_train[app_train.CODE_GENDER == 'M'].shape[0]
xna_total = app_train[app_train.CODE_GENDER == 'XNA'].shape[0]
print("Gender XNA: {}?".format(app_train[app_train.CODE_GENDER == 'XNA'].shape[0])) # What is XNA gender?
print("% of Female target: {}".format(female_total/total*100))
print("% of Male target: {}".format(male_total/total*100))

tab = pd.crosstab(app_train.TARGET, app_train.CODE_GENDER)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')
fig = plt.figure(figsize=[5,5])
# Female
ax = fig.add_subplot(121)
female = app_train[app_train.CODE_GENDER == 'F']
female.TARGET.value_counts().plot.bar()
ax.set_title("Female loans")
# Male
ax = fig.add_subplot(122)
male = app_train[app_train.CODE_GENDER == 'M']
male.TARGET.value_counts().plot.bar()
ax.set_title("Male loans")
female_true = female[female.TARGET == 1].shape[0]
female_false = female[female.TARGET == 0].shape[0]

male_true = male[male.TARGET == 1].shape[0]
male_false = male[male.TARGET == 0].shape[0]

print("{:.2f}% female with difficulties".format(female_true/female_total*100))
print("{:.2f}% female with not difficulties".format(female_false/female_total*100))
print("{:.2f}% male with difficulties".format(male_true/male_total*100))
print("{:.2f}% male with not difficulties".format(male_false/male_total*100))
# print(female_true, female_false)
# print(male_true, male_false)
print(app_train.NAME_CONTRACT_TYPE.head(5))
app_train.NAME_CONTRACT_TYPE.value_counts().plot.bar()
revolving_loans = app_train[app_train.NAME_CONTRACT_TYPE == 'Revolving loans'].shape[0]
cash_loans = app_train[app_train.NAME_CONTRACT_TYPE == 'Cash loans'].shape[0]
print("Cash loans represent the {:.2f}%".format(cash_loans/app_train_total*100))
print("Revolving loans represent the {:.2f}%".format(revolving_loans/app_train_total*100))
tab = pd.crosstab(app_train.CODE_GENDER, app_train.NAME_CONTRACT_TYPE)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')
tab = pd.crosstab(app_train.TARGET, app_train.NAME_CONTRACT_TYPE)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')
print(app_train.FLAG_OWN_CAR.head(5))
app_train.FLAG_OWN_CAR.value_counts().plot.bar()
car_yes = app_train[app_train.FLAG_OWN_CAR == 'Y'].shape[0]
car_no = app_train[app_train.FLAG_OWN_CAR == 'N'].shape[0]
print("Own Car Yes {:.2f}%".format(car_yes/app_train_total*100))
print("Not Own Car {:.2f}%".format(car_no/app_train_total*100))
tab = pd.crosstab(app_train.TARGET, app_train.FLAG_OWN_CAR)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')
tab = pd.crosstab(app_train.CODE_GENDER, app_train.FLAG_OWN_CAR)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')
print(app_train.FLAG_OWN_REALTY.head(5))
app_train.FLAG_OWN_REALTY.value_counts().plot.bar()
house = app_train[app_train.FLAG_OWN_REALTY == 'Y'].shape[0]
flat = app_train[app_train.FLAG_OWN_REALTY == 'N'].shape[0]
print("Owns house {:.2f}%".format(house/app_train_total*100))
print("Owns flat {:.2f}%".format(flat/app_train_total*100))
tab = pd.crosstab(app_train.TARGET, app_train.FLAG_OWN_REALTY)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')
tab = pd.crosstab(app_train.CODE_GENDER, app_train.FLAG_OWN_REALTY)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')
tab = pd.crosstab(app_train.FLAG_OWN_CAR, app_train.FLAG_OWN_REALTY)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')
print(app_train.CNT_CHILDREN.head(5))
app_train.CNT_CHILDREN.value_counts().plot.bar()
print(app_train[app_train.CNT_CHILDREN == 19].shape)
app_train['CNT_CHILDREN'] = app_train.apply(lambda x: 3 if x.CNT_CHILDREN > 2.0 else x.CNT_CHILDREN, axis=1)

# tdr['Team1_Score'] = tdr.apply(lambda r: r.WScore if r.Pred == 1.0 else r.LScore, axis=1)
tab = pd.crosstab(app_train.TARGET, app_train.CNT_CHILDREN)
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('xlabel')
dummy = plt.ylabel('ylabel')



