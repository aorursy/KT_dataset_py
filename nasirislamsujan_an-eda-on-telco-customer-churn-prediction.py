import numpy as np 

import pandas as pd 

import eli5

import plotly

import plotly.graph_objs as go

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

from scipy.stats import chisquare

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from eli5.sklearn import PermutationImportance



# command for work offline

plotly.offline.init_notebook_mode(connected=True)
# read the dataset

dataset = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# an overview of the dataset

dataset.head()
dataset.shape
# list of columns in the dataset

dataset.columns
# only 3 feature contain numerical values, rest are categorical feature

dataset.describe()
# customer id is unnecessary

del dataset["customerID"]
gender_map = {"Female" : 0, "Male": 1}

yes_no_map = {"Yes" : 1, "No" : 0}



dataset["gender"] = dataset["gender"].map(gender_map)



def binary_encode(features):

    for feature in features:

        dataset[feature] = dataset[feature].map(yes_no_map)
binary_encode_candidate = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

binary_encode(binary_encode_candidate)
# converting series object dataset into numeric

# errors = 'coerce’ means, if invalid parsing occur then set NaN

dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors = 'coerce')
# missing values check

print(dataset.isnull().any())

print("\n# of Null values in 'TotalCharges`: ",dataset["TotalCharges"].isnull().sum())
# fill null values with the mean values of that feature

dataset["TotalCharges"].fillna(dataset["TotalCharges"].mean(), inplace=True)
dataset = pd.get_dummies(dataset)
# now take a look at our final dataset

dataset.head()
dataset.describe().T
result = pd.DataFrame(columns=["Features", "Chi2Weights"])



for i in range(len(dataset.columns)):

    chi2, p = chisquare(dataset[dataset.columns[i]])

    result = result.append([pd.Series([dataset.columns[i], chi2], index = result.columns)], ignore_index=True)
result = result.sort_values(by="Chi2Weights", ascending=False)
result.head(20)
new_df = dataset[result["Features"].head(20)]
new_df.head()
plt.figure(figsize = (15, 12))

sns.heatmap(new_df.corr(), cmap="RdYlBu", annot=True, fmt=".1f")

plt.show()
hightly_corr_feature = ["OnlineBackup_No internet service", "StreamingMovies_No internet service", "StreamingTV_No internet service", 

"TechSupport_No internet service", "DeviceProtection_No internet service", "OnlineSecurity_No internet service"]



def remove_corr_features(features):

    for feature in features:

        del new_df[feature]
remove_corr_features(hightly_corr_feature)
plt.figure(figsize = (12, 8))

sns.heatmap(new_df.corr(), cmap="RdYlBu", annot=True, fmt=".1f")

plt.show()
trace = []



def gen_boxplot(df):

    for feature in df:

        trace.append(

            go.Box(

                name = feature,

                y = df[feature]

            )

        )

        

gen_boxplot(new_df)
data = trace

plotly.offline.iplot(data)
ax = new_df["Churn"].value_counts().plot(kind='bar', figsize=(6, 8), fontsize=13)

ax.set_ylabel("Number of Customer", fontsize=14);



totals = []

for i in ax.patches:

    totals.append(i.get_height())



total = sum(totals)



for i in ax.patches:

    ax.text(i.get_x() - .01, i.get_height() + .5, \

            str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,

                color='#444444')

plt.show()
new_df.columns
new_df["tenure"].unique()
_, ax = plt.subplots(1, 2, figsize= (16, 6))

sns.scatterplot(x="TotalCharges", y = "tenure" , hue="Churn", data=new_df, ax=ax[0])

sns.scatterplot(x="MonthlyCharges", y = "tenure" , hue="Churn", data=new_df, ax=ax[1])
facet = sns.FacetGrid(new_df, hue = "Churn", aspect = 3)

facet.map(sns.kdeplot,"TotalCharges",shade= True)

facet.set(xlim=(0, new_df["TotalCharges"].max()))

facet.add_legend()



facet = sns.FacetGrid(new_df, hue = "Churn", aspect = 3)

facet.map(sns.kdeplot,"MonthlyCharges",shade= True)

facet.set(xlim=(0, new_df["MonthlyCharges"].max()))

facet.add_legend()
_, ax = plt.subplots(1, 2, figsize= (8, 6))

plt.subplots_adjust(wspace = 0.5)

sns.boxplot(x = 'Churn',  y = 'TotalCharges', data = new_df, ax=ax[0])

sns.boxplot(x = 'Churn',  y = 'MonthlyCharges', data = new_df, ax=ax[1])
_, axs = plt.subplots(1, 2, figsize=(9, 6))

plt.subplots_adjust(wspace = 0.3)

ax = sns.countplot(data = new_df, x = "SeniorCitizen", hue = "Churn", ax = axs[0])

ax1 = sns.countplot(data = new_df, x = "MultipleLines_No phone service", hue = "Churn", ax = axs[1])



for p in ax.patches:

        height = p.get_height() 

        ax.text(

                p.get_x()+p.get_width()/2,

                height + 3.4,

                "{:1.2f}%".format(height/len(new_df),0),

                ha = "center", rotation = 0

               ) 

        

for p in ax1.patches:

        height = p.get_height() 

        ax1.text(

                p.get_x()+p.get_width()/2,

                height + 3.4,

                "{:1.2f}%".format(height/len(new_df),0),

                ha = "center", rotation = 0

               ) 
plt.figure(figsize=(8, 6))

sns.swarmplot(x = 'SeniorCitizen', y = 'MonthlyCharges', hue="Churn", data = new_df)

plt.legend(loc='upper-right')
fig, ax = plt.subplots(1,3, figsize=(14, 4))

plt.subplots_adjust(wspace=0.4)

sns.countplot(x = "Contract_One year", hue="Churn" , ax=ax[0], data=new_df)

sns.countplot(data = new_df, x = "PaymentMethod_Credit card (automatic)", ax=ax[1], hue="Churn")

sns.countplot(data = new_df, x ="InternetService_No", ax=ax[2], hue="Churn")

fig.show()
fig, ax = plt.subplots(1,2, figsize=(10, 4))

plt.subplots_adjust(wspace=0.4)

sns.swarmplot(x = 'PaymentMethod_Bank transfer (automatic)', y = 'TotalCharges', hue="Churn", data = new_df, ax=ax[0])

sns.swarmplot(x = 'Contract_Two year', y = 'TotalCharges', hue="Churn", data = new_df, ax=ax[1])
fig, ax = plt.subplots(1,2, figsize=(8, 4))

plt.subplots_adjust(wspace=0.3)

sns.swarmplot(x = 'PaymentMethod_Mailed check', y = 'TotalCharges', hue="Churn", data = new_df, ax=ax[0])

sns.swarmplot(x = 'Contract_Two year', y = 'TotalCharges', hue="Churn", data = new_df, ax=ax[1])

fig.show()
cols = ["TotalCharges", "MonthlyCharges", "tenure", "Churn"] 

pairplot_feature = new_df[cols]

sns.pairplot(pairplot_feature, hue = "Churn")
X = new_df.drop("Churn", axis=1)

y = new_df["Churn"]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

clf = RandomForestClassifier(random_state=0).fit(X_train, y_train)
perm = PermutationImportance(clf, random_state = 1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())