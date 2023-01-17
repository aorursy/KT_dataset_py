import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set_style("white")





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
df.shape
df.columns
df.dtypes
# Converting Total Charges to a numerical data type.

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')

df.isnull().sum()
#Removing missing values 

df.dropna(inplace = True)
df.describe()
%matplotlib inline 

import matplotlib.pyplot as plt

df.hist(bins= 15, figsize=(10,10)) # fig size width & height

# bins - the higher the more refined.

plt.show() 



# majority of the users are not senior citizen

# the sample for the tenure are fairly new or have been around quite long 70+



# different scales - feature scaling resolve



# the MonthlyCharges and TotalCharges are tail heavy - shape back to a normal distribution
# we can do stratified sampling here as we can see the monthly charges is quite a spread in our ML model

# we do this by categorising the continious variable and grouping. We then can split it into different groups,

# the model will have more of a relative sample rather than random split of the train and test data which might be bias
# This is creating an monthly charge category list, showing that the test set is representative of the various changes of income in the whole dataset.



#Could be sensible and divide by 5



df["MonthlyChargesCat"] = pd.cut(df["MonthlyCharges"],

                               bins=[0., 24.0, 48.0, 72.0, 96, np.inf],

                               labels=[1, 2, 3, 4, 5])
df["MonthlyChargesCat"].value_counts()
#Changing the continuous numerical attributes to income category attribute

sns.countplot(x="MonthlyChargesCat", data=df, palette="Set2")
sns.countplot(x="gender", hue="Churn", data=df, palette="Set2")



# This is a very balance, between male and female. We could drop this column for our ML Model possibly to ensure the model runs slight quicker.
sns.countplot(x="SeniorCitizen", hue="Churn", data=df, palette="Set2")



# This is interesting, senior citizen have a high chance of churning
# let's see what contract the senior citizen have



ax = sns.countplot(x="SeniorCitizen", hue="Contract", data=df, palette="Set2")



# majority of the customer tend to have month to month contracts, especailly senior citizen
sns.countplot(x="Contract", hue='Churn', data=df, palette="Set2")

# Customers that are on month-to-month have a higher churn rate, and this make sense, because the churn rate is measure if they have left last month.

# Whereas if the customer is still on contract they are very unlikely to churn as I assume they have to pay a fine/penalty
a = df.groupby(['SeniorCitizen'])['Churn'].count().rename("count")



print(a/a.sum()*100)

print('\n')

# we can see that senior citizen is 16% of the total sample set





b = df.groupby(['Contract','Churn','SeniorCitizen'])['Churn'].count().rename("count")



print(b)



b/b.groupby(level=2).sum()



# we arrange it in a way where we can compare on a percentage term how many churn via contract level and via senior citizen vs. non senior citizen.

# we can see that majority of senior citizen churn during their month-to-month contract compared to non senior citizen. 

# we also noticed the proportion of contract types vary between the two age group as seen from the visual for the different contract types.
c = df.groupby(['Contract','Churn'])['Churn'].count().rename("count")



c/c.groupby(level=0).sum()



# 42% of the month-to-month contract has churned

# 11% of the one year contract has churned, if there was equal amount of customers throughout the year monthly that churns would be 1/12 ~ 8%

# maybe a good aim would be to have customer churn rate for 1 year contract to be below 8% per month on average as a Key Performance Measure on one year contracts



# likewise for 2 year contract 1/24 ~ is about 4% over 2 years, this is a rough ball park, suggestion of target churn rate, this will vary due to factors, 

# deals/packages, competition and economic factors. 



# This means the ML model needs to be maintained and continous training and learning as environment changes
plt.figure(figsize=(18,5))



plt.subplot(1, 2, 1)



# Let's look at more about the demographic info.

sns.countplot(x="Partner", hue='Churn', data=df, palette="Set2")



plt.subplot(1, 2, 2)



sns.countplot(x="Dependents", hue='Churn', data=df, palette="Set2")



plt.tight_layout()



# for month-to-month having no dependcies seems like a big factor in a lower churn rate 

# this makes it very interesting
d = df.groupby(['Churn','Partner','Dependents','Contract'])['Churn'].count().rename("count")



print(d)





# figures to show in tabular format

# for month-to-month having a partner does not seem to affect the factor in the churn rate
plt.figure(figsize=(10,4))

ax = sns.violinplot(x="MonthlyChargesCat", y="tenure", hue='Churn', data=df)

# if the customer is paying at the a very high price, the customer is unlikely to churn, this should be maintained if possible

# the interesting part is that customers from a spread of tenures are starting to churn this is not great.



# This is where continous numerical data shines and provides very great insight in a violin plot with categorical data for the chart

# it would be a lot difficult to visualise if the monthly charges were not categorised
plt.figure(figsize=(10,5))

sns.scatterplot(data=df, x="MonthlyCharges", y="tenure", hue="Churn", palette="Set2", alpha =0.3)



# the violin graph is using the same data and express more easier to read data as some point is too clustered
sns.countplot(x="PaperlessBilling", hue='Churn', data=df, palette="Set2")

# paperless billing feature seems to have a impact on the churn rate of customers
plt.figure(figsize=(10,4))

sns.countplot(x="PaymentMethod", hue='Churn', data=df, palette="Set2")



# this is very suprising that payment type has an effect on the churn rate - especially electric check
plt.figure(figsize=(10,4))

sns.countplot(x="InternetService", hue='Churn', data=df, palette="Set2")



# you have a very high churn rate if you are on fiber optic
plt.figure(figsize=(20,8))



plt.subplot(221)



sns.countplot(x="OnlineSecurity", hue='Churn', data=df, palette="Set2")



plt.subplot(2, 2, 2)



sns.countplot(x="OnlineBackup", hue='Churn', data=df, palette="Set2")



plt.subplot(2, 2, 3)



sns.countplot(x="DeviceProtection", hue='Churn', data=df, palette="Set2")





plt.subplot(2, 2, 4)



sns.countplot(x="TechSupport", hue='Churn', data=df, palette="Set2")



plt.tight_layout()



# all the services are very similar, if you don't have the service the churn rate is expected to be higher
plt.figure(figsize=(18,5))



plt.subplot(1, 2, 1)



sns.countplot(x="StreamingTV", hue='Churn', data=df, palette="Set2")



plt.subplot(1, 2, 2)



#plt.figure(figsize=(10,4))

sns.countplot(x="StreamingMovies", hue='Churn', data=df, palette="Set2")



plt.tight_layout()



# very similar base on streaming tv and movies

# having streamingtv or streaming movies does not actually help improve the churn rate
plt.figure(figsize=(20,20))



plt.subplot(2, 1, 1)



sns.scatterplot(data=df, x="tenure", y="MonthlyCharges", hue="PaymentMethod", palette="Set2", alpha=0.4)



plt.subplot(2, 1, 2)



sns.scatterplot(data=df, x="tenure", y="MonthlyCharges", hue="Contract", palette="Set2", alpha =0.3)

plt.tight_layout()



# payment method

# we can see that higher monthly charges tend to be electronic payment, except for the high payig and longer tenures are clustered around bank transfers





# contract type

# we can see that the two year fixed contract seems to hae stay for a two year contract for a long time

# 1 year contract are also concentrated from 20 to 60 tenures (months)