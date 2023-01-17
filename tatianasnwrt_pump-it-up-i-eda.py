import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

%matplotlib inline



from sklearn.metrics import accuracy_score



pd.set_option('display.max_columns', None)



print("Setup Complete")
pylab.rcParams["figure.figsize"] = (14,8)
# Read the file into a variable my_data

X_train = pd.read_csv("../input/X_train_raw.csv")

y_train = pd.read_csv("../input/y_train_raw.csv")

X_test = pd.read_csv("../input/X_test_raw.csv")



# Merge train X and y values

train_df = X_train.merge(y_train,how='outer',left_index=True, right_index=True)
train_df.head()
train_df.describe()
train_df.info()
label_dict = {"functional":2,"functional needs repair":1,"non functional":0}

train_df["label"] = train_df["status_group"].map(label_dict)

sns.distplot(train_df["label"],kde=False)
majority_class = train_df['status_group'].mode()[0]

print("The most frequent label is", majority_class)



y_prelim_pred = np.full(shape=train_df['status_group'].shape, fill_value=majority_class)

accuracy_score(train_df['status_group'], y_prelim_pred)
# Select numerical columns

numerical_vars = [col for col in train_df.columns if 

                train_df[col].dtype in ['int64', 'float64']]
sns.countplot(x=train_df["construction_year"],hue=train_df["status_group"])

plt.xticks(rotation=45, 

    horizontalalignment='right')

plt.title("Number of pumps constructed over the years", fontsize=14)

plt.xlabel("Construction year", fontsize=12)

plt.ylabel("Number of pumps constructed", fontsize=12)
sns.scatterplot(y=train_df["amount_tsh"],x=train_df["status_group"])
fig = plt.figure(figsize=(12,18))

sns.distributions._has_statsmodels=False

for i in range(len(numerical_vars)):

    fig.add_subplot(9,4,i+1)

    sns.distplot(train_df[numerical_vars].iloc[:,i].dropna())

    plt.xlabel(numerical_vars[i])



plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(12, 18))



for i in range(len(numerical_vars)):

    fig.add_subplot(9, 4, i+1)

    sns.boxplot(y=train_df[numerical_vars].iloc[:,i])



plt.tight_layout()

plt.show()
f = plt.figure(figsize=(14,20))



for i in range(len(numerical_vars)):

    f.add_subplot(9, 4, i+1)

    sns.scatterplot(train_df[numerical_vars].iloc[:,i], train_df["label"])

    

plt.tight_layout()

plt.show()
correlation = train_df.corr()



f, ax = plt.subplots(figsize=(8,6))

plt.title('Correlation of numerical attributes', size=12)

sns.heatmap(correlation)
correlation['label'].sort_values(ascending=False)
train_df[numerical_vars].isna().sum().sort_values(ascending=False)
len(train_df.population[train_df.population == 0])
cat_vars = train_df.select_dtypes(include='object').columns

print(cat_vars)
train_df[cat_vars].isna().sum().sort_values(ascending=False)
## Count of categories within Scheme_management attribute

sns.countplot(x='scheme_management', data=train_df)

plt.xticks(rotation=90)

plt.ylabel('Frequency')

plt.show()
train_df.to_csv("train_df_after_EDA.csv", index=False)

X_test.to_csv("X_test_after_EDA.csv", index=False)