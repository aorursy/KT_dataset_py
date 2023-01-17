import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, auc

%matplotlib inline
df = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head()
df["MonthlyCharges"] = df["MonthlyCharges"].apply(float)

df["TotalCharges"] = df["TotalCharges"].replace(' ',None) # to ensure that TotalCharges can be successfully converted to a float column

df["TotalCharges"] = df["TotalCharges"].apply(float)

df.isnull().sum()
plt.figure()

sns.countplot(x="gender", data=df)

plt.figure()

sns.countplot(x="gender", hue="Churn", data=df)
plt.figure()

sns.countplot(x="SeniorCitizen", data=df)

plt.figure()

sns.countplot(x="SeniorCitizen", hue="Churn", data=df)
plt.figure()

sns.countplot(x="Partner", data=df)

plt.figure()

sns.countplot(x="Partner", hue="Churn", data=df)
plt.figure()

sns.countplot(x="Dependents", data=df)

plt.figure()

sns.countplot(x="Dependents", hue="Churn", data=df)
plt.figure()

sns.countplot(x="PhoneService", data=df)

plt.figure()

sns.countplot(x="PhoneService", hue="Churn", data=df)
plt.figure()

sns.countplot(x="MultipleLines", data=df)

plt.figure()

sns.countplot(x="MultipleLines", hue="Churn", data=df)
plt.figure()

sns.countplot(x="InternetService", data=df)

plt.figure()

sns.countplot(x="InternetService", hue="Churn", data=df)
plt.figure()

sns.countplot(x="OnlineSecurity", data=df)

plt.figure()

sns.countplot(x="OnlineSecurity", hue="Churn", data=df)
plt.figure()

sns.countplot(x="OnlineBackup", data=df)

plt.figure()

sns.countplot(x="OnlineBackup", hue="Churn", data=df)
plt.figure()

sns.countplot(x="DeviceProtection", data=df)

plt.figure()

sns.countplot(x="DeviceProtection", hue="Churn", data=df)
plt.figure()

sns.countplot(x="TechSupport", data=df)

plt.figure()

sns.countplot(x="TechSupport", hue="Churn", data=df)
plt.figure()

sns.countplot(x="StreamingTV", data=df)

plt.figure()

sns.countplot(x="StreamingTV", hue="Churn", data=df)
plt.figure()

sns.countplot(x="StreamingMovies", data=df)

plt.figure()

sns.countplot(x="StreamingMovies", hue="Churn", data=df)
plt.figure()

sns.countplot(x="Contract", data=df)

plt.figure()

sns.countplot(x="Contract", hue="Churn", data=df)
plt.figure()

sns.countplot(x="PaperlessBilling", data=df)

plt.figure()

sns.countplot(x="PaperlessBilling", hue="Churn", data=df)
plt.figure()

sns.countplot(x="PaymentMethod", data=df)

plt.figure()

sns.countplot(x="PaymentMethod", hue="Churn", data=df)
sns.countplot(x="Churn", data=df)
sns.distplot(df["tenure"].tolist())
df["tenure"].describe()
sns.distplot(df["MonthlyCharges"].tolist())
sns.distplot(df["TotalCharges"].tolist())
sns.scatterplot(x="TotalCharges", y="tenure", hue="Churn",

                     data=df)
sns.scatterplot(x="MonthlyCharges", y="tenure", hue="Churn",

                     data=df)
sns.scatterplot(x="MonthlyCharges", y="TotalCharges", hue="Churn",

                     data=df)
label_encoding = {

                    "Partner": {

                            "Yes": 1,

                            "No": 0

                        },

                        "Dependents": {

                            "Yes": 1,

                            "No": 0

                        },

                        "PhoneService": {

                            "Yes": 1,

                            "No": 0

                        },

                        "MultipleLines": {

                            "Yes": 2,

                            "No": 1,

                            "No phone service": 0

                        },

                        "InternetService": {

                            "Fiber optic": 2,

                            "DSL": 1,

                            "No": 0

                        },

                        "OnlineSecurity": {

                            "Yes": 2,

                            "No": 1,

                            "No internet service": 0

                        },

                        "OnlineBackup": {

                            "Yes": 2,

                            "No": 1,

                            "No internet service": 0

                        },

                        "DeviceProtection": {

                            "Yes": 2,

                            "No": 1,

                            "No internet service": 0

                        },

                        "TechSupport": {

                            "Yes": 2,

                            "No": 1,

                            "No internet service": 0

                        },

                        "StreamingTV": {

                            "Yes": 2,

                            "No": 1,

                            "No internet service": 0

                        },

                        "StreamingMovies": {

                            "Yes": 2,

                            "No": 1,

                            "No internet service": 0

                        },

                        "Contract": {

                            "Two year": 2,

                            "One year": 1,

                            "Month-to-month": 0

                        },

                        "PaymentMethod": {

                            "Credit card (automatic)": 1,

                            "Bank transfer (automatic)": 1,

                            "Mailed check": 0,

                            "Electronic check": 0

                        },

                        "Churn": {

                            "Yes": 1,

                            "No": 0

                        },

                        "gender": {

                            "Male": 1,

                            "Female": 0

                        }

                    }



for column, val_mapping in label_encoding.items():

    df[column] = df[column].apply(lambda i: val_mapping[i])

df.head()
std_cols = ["MonthlyCharges", "TotalCharges"]

for column in std_cols:

    df[column] = df[column].apply(lambda i: (i-df[column].mean()) / df[column].std())

df.head()
new_tenure = []

for i in df["tenure"]:

    if i <= 12:

        new_tenure.append(0)

    elif i <= 24:

        new_tenure.append(1)

    elif i <= 36:

        new_tenure.append(2)

    elif i <= 48:

        new_tenure.append(3)

    elif i <= 60:

        new_tenure.append(4)

    else:

        new_tenure.append(5)

df["tenure"] = new_tenure

df["tenure"].head()
df["Responsibility_Score"] = (df["Partner"] + df["Dependents"] + df["SeniorCitizen"]) / 3

df["Phone_Reliance"] = (df["PhoneService"] + df["MultipleLines"]) / 2

df["Support"] = (df["OnlineSecurity"] + df["OnlineBackup"] + df["DeviceProtection"] + df["TechSupport"]) / 4

df["Online_Services"] = (df["InternetService"] + df["StreamingTV"] + df["StreamingMovies"]) / 3

df["Duration"] = (df["tenure"] + df["Contract"]) / 2
drop_cols = ["PaperlessBilling","Partner", "Dependents", "SeniorCitizen", "PhoneService", "MultipleLines",

            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",

            "InternetService", "StreamingTV", "StreamingMovies",

            "tenure", "Contract"]

df = df.drop(drop_cols, axis=1)

df.head()
# Train Test Split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=100, stratify=df["Churn"].tolist())

train_X = train_df.drop(["customerID", "Churn"], axis=1).values

train_Y = train_df["Churn"].tolist()

test_X = test_df.drop(["customerID", "Churn"], axis=1).values

test_Y = test_df["Churn"].tolist()



# Oversample using SMOTE

sampler = SMOTE(random_state=100, k_neighbors=7)

train_X, train_Y = sampler.fit_resample(train_X, train_Y)



# Train Model

# rf = RandomForestClassifier()

# model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=100, n_jobs = -1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(train_X, train_Y)
predictions = model.predict(test_X)

prediction_proba = model.predict_proba(test_X)
report_ = []

lines = classification_report(test_Y, predictions).split('\n')

for l in lines[2:-4]:

    row_ = l.split()

    if len(row_) != 0:

        row = {}

        row['precision'] = float(row_[-4])

        row['recall'] = float(row_[-3])

        row['f1_score'] = float(row_[-2])

        row['support'] = float(row_[-1])

        row['class'] = {1: "Churn", 0: "No Churn"}[int(row_[0])]

        report_.append(row)

dataframe = pd.DataFrame.from_dict(report_)

dataframe.head()
print("ROC AUC Score: {}\n\n".format(roc_auc_score(test_Y, prediction_proba[:, 1])))

print("Accuracy Score: {}\n\n".format(accuracy_score(test_Y, predictions)))



confusion_mat = confusion_matrix(test_Y, predictions, labels=[0, 1])

_row = confusion_mat.sum(axis=0)

_col = [np.nan] + list(confusion_mat.sum(axis=1)) + [sum(_row)]

con_df = pd.DataFrame({})

con_df["Predicted"] = ["Actual"] + ["No Churn", "Churn"] + ["All"]

for label, idx in {"No Churn": 0, "Churn": 1}.items():

    temp = [np.nan] + list(confusion_mat[:, idx]) + [_row[idx]]

    con_df[label] = temp



con_df["All"] = _col

print("Confusion Matrix\n")

con_df
sorted_proba = [y for _, y in sorted(zip(prediction_proba[:, -1], test_Y), reverse = True)] # sorting with predicted probabilities of churning as key

y_cum = np.append([0], np.cumsum(sorted_proba)) # cumulative sum of true labels sorted previously

total_points = len(test_Y) # total number of plot points

class_1_count = np.sum(test_Y) # Number of true churn samples

x_points = np.arange(0, total_points + 1) # generate data points for x axis

random_model_area = auc([0, total_points], [0, class_1_count]) # area under random model

perfect_model_area = auc([0, class_1_count, total_points], [0, class_1_count, class_1_count]) # area under perfect model

trained_model_area = auc(x_points, y_cum) # area under trained model

perfect_vs_random = perfect_model_area - random_model_area # area between perfect and random model

trained_vs_random = trained_model_area - random_model_area # area between trained and random model

accuracy_rate = trained_vs_random / perfect_vs_random # accuracy rate



plt.figure(figsize = (20, 12))

plt.plot([0, 100], [0, 100], c = 'r', linestyle = '--', label = 'Random Model with Area: {}'.format(random_model_area))

plt.plot([0, (class_1_count/total_points)*100, 100], 

        [0, 100, 100], 

        c = 'grey', 

        linewidth = 2, 

        label = 'Perfect Model with Area: {}'.format(perfect_model_area))

cum = (np.cumsum(sorted_proba)/class_1_count)*100

cum = [cum[i] for i in range(0, len(cum), len(cum)//100)]

y_values = np.append([0], cum[-100:])

x_values = np.arange(0, 101)

plt.plot(x_values, 

        y_values, 

        c = 'b', 

        label = "KNNClassifier with Area: {}".format(trained_model_area), 

        linewidth = 4)

plt.xlabel('Total observations (%)', fontsize = 16)

plt.ylabel('Churn observations (%)', fontsize = 16)

plt.title('Cumulative Accuracy Profile with Accuracy Rate: {}'.format(accuracy_rate), fontsize = 16)

plt.legend(loc = 'lower right', fontsize = 16)

print("Our Model Has Selected 500 Customers Out of {} Customers and Managed to Identify {} Churn Cases Out of a Total of {} Churn Cases"\

     .format(len(sorted_proba), sum(sorted_proba[:500]), sum(sorted_proba)))