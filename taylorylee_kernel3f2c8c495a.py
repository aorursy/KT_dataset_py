# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
telcom = pd.read_csv(r"../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

#first few rows

telcom.head()

telcom.replace({'No' : 0, 'Yes' : 1})

counts = telcom['Churn'].value_counts()

plot = counts.plot.bar()

plot.set_xlabel("Churn")

plot.set_ylabel("Amount of Customers")



total = counts[0] + counts[1]

no = counts[0] / total

yes = int((counts[1] / total) * 100)

print("Approximately " + str(yes) + "% of customers left in the last month.")
#Now we're going to visualize the associations between churn and all other variables

churn_yes = telcom[telcom['Churn'] == 'Yes']



def add_percentage(ax):

    for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y))

        

def show_churn(variable):

    plt.figure(figsize=(4, 3))

    ax = sns.countplot(y = churn_yes[variable], data = telcom, palette = "Set3")

    sns.despine()

    add_percentage(ax)



categorical_vars = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService'

                   , 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',

                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',

                   'Contract', 'PaperlessBilling', 'PaymentMethod']

for var in categorical_vars:

    show_churn(var)

plt.figure(figsize = (12,8))

sns.set()

sns.boxplot(x= 'Churn', y= "MonthlyCharges", hue= "Contract", data=telcom)

plt.title("Distribution of Monthly Charges based on Churn, separated by Contract")

plt.show()
plt.figure(figsize = (12,8))

sns.boxplot(x= 'Churn', y= "tenure", hue= "PaperlessBilling", data=telcom)

plt.title("Distribution of Tenure based on Churn, separated by BillingType")

plt.show()
#telcom.head()

pd.set_option('display.max_columns', None)

#Drop unimportant features

feature_list = ["MultipleLines", "gender", "StreamingTV", "StreamingMovies"]



telcom_copy = telcom.copy()

telcom_copy.drop(columns = feature_list, inplace = True)



dummy_df = pd.get_dummies(data = telcom_copy, columns=["InternetService", "Contract", "PaymentMethod","TechSupport","DeviceProtection","OnlineBackup","OnlineSecurity"])

dummy_df = dummy_df.replace({'No' : 0, 'Yes' : 1})

# Don't drop "TechSupport_No internet service" to keep that population

dummy_df = dummy_df.drop(["customerID","DeviceProtection_No internet service", "OnlineBackup_No internet service", "OnlineSecurity_No internet service"], axis=1)

dummy_df = dummy_df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).dropna()

dummy_df.sample(5)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()



#Train-test split

np.random.seed(1337)

shuffled_indices = np.random.permutation(len(dummy_df))

# Set train_indices to the first 80% of shuffled_indices and and test_indices to the rest.

desired_indices = int(len(dummy_df) * 0.8)

train_indices = shuffled_indices[: desired_indices]

test_indices = shuffled_indices[desired_indices :]



# Create train and test` by indexing into `full_data` using 

# `train_indices` and `test_indices`

train = dummy_df.take(train_indices)

test = dummy_df.take(test_indices)
X_train = train.drop(['Churn'], axis = 1)

Y_train= train.loc[:, 'Churn']

X_test = test.drop(['Churn'], axis = 1)

Y_test = test.loc[:, 'Churn']
model.fit(X_train, Y_train)
model.score(X_test, Y_test)
from sklearn.metrics import classification_report

print(classification_report(Y_test, model.predict(X_test)))
tree_arr = pd.Series(model.predict(X_test)).value_counts()

tree_predictions_plot = tree_arr.plot.bar()

tree_predictions_plot.set_xlabel("Churn Prediction for Decision Tree")

tree_predictions_plot.set_ylabel("Amount of Customers")



total_pred = tree_arr[0] + tree_arr[1]

no_pred = tree_arr[0] / total_pred

yes_pred = int((tree_arr[1] / total_pred) * 100)

print("Approximately " + str(yes_pred) + "% of customers left in the last month.")