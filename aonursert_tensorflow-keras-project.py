import pandas as pd
data_info = pd.read_csv('../input/lending-club/lending_club_info.csv',index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])
def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
feat_info('mort_acc')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
%matplotlib inline
df = pd.read_csv('../input/lending-club/lending_club_loan_two.csv')
df.info()
sns.countplot(df["loan_status"])
sns.set_style("whitegrid")
plt.figure(figsize=(12,4))
sns.distplot(df["loan_amnt"], kde=False)
feat_info('loan_amnt')
df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True)
feat_info('installment')
feat_info('loan_amnt')
sns.scatterplot(x="installment", y="loan_amnt", data=df)
sns.boxplot(x="loan_status", y="loan_amnt", data=df)
df.groupby("loan_status")["loan_amnt"].describe()
df.head()
np.sort(df["grade"].unique())
np.sort(df["sub_grade"].unique())
sns.countplot(x="grade", hue="loan_status", data=df)
plt.figure(figsize=(12,6))
sns.countplot(x="sub_grade", data=df, palette="coolwarm", order=np.sort(df["sub_grade"].unique()))
plt.figure(figsize=(12,6))
sns.countplot(x="sub_grade", hue="loan_status", data=df, palette="coolwarm", order=np.sort(df["sub_grade"].unique()))
df_FG = df[(df["grade"] == "F") | (df["grade"] == "G")]
plt.figure(figsize=(12,6))
sns.countplot(x="sub_grade", hue="loan_status", data=df_FG, palette="coolwarm", order=np.sort(df_FG["sub_grade"].unique()))
df["loan_repaid"] = pd.get_dummies(df["loan_status"], drop_first=True)
df.head()
df.corr()["loan_repaid"][:-1].sort_values().plot(kind="bar")
df.head()
len(df)
df.isnull().sum()
df.isnull().mean()
feat_info("emp_title")
print()
feat_info("emp_length")
df["emp_title"].nunique()
df["emp_title"].value_counts()
df.drop("emp_title", axis=1, inplace=True)
df["emp_length"].dropna().unique()
sorted(df["emp_length"].dropna().unique())
sort_emp = [
'< 1 year',
'1 year',
'2 years',
'3 years',
'4 years',
'5 years',
'6 years',
'7 years',
'8 years',
'9 years',
'10+ years'
]
plt.figure(figsize=(10,6))
sns.countplot(x="emp_length", data=df, order=sort_emp)
plt.figure(figsize=(12,4))
sns.countplot(x="emp_length", data=df, order=sort_emp, hue="loan_status")
emp_co = df[df["loan_status"]=="Charged Off"].groupby("emp_length").count()["loan_status"]
emp_fp = df[df["loan_status"]=="Fully Paid"].groupby("emp_length").count()["loan_status"]
emp_co/emp_fp
emp_len = emp_co / (emp_co + emp_fp)
emp_len.plot(kind="bar")
df.drop("emp_length", axis=1, inplace=True)
df.isnull().sum()
df["purpose"].head()
feat_info("purpose")
df["title"].head(10)
feat_info("title")
df.drop("title", axis=1, inplace=True)
feat_info("mort_acc")
df["mort_acc"].value_counts()
df.corr()["mort_acc"].sort_values()
total_acc_avg = df.groupby("total_acc")["mort_acc"].mean()
total_acc_avg
def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc
df["mort_acc"] = df.apply(lambda x: fill_mort_acc(x["total_acc"], x["mort_acc"]), axis=1)
df["mort_acc"].isnull().sum()
df.isnull().sum()
df = df.dropna()
df.isnull().sum()
df.select_dtypes(["object"]).columns
feat_info("term")
df["term"].value_counts()
df["term"] = df["term"].apply(lambda x: int(x.split()[0]))
df["term"]
df["term"].value_counts()
df.drop("grade", axis=1, inplace=True)
dummies = pd.get_dummies(df["sub_grade"], drop_first=True)
dummies
df.drop("sub_grade", axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)
df.head()
df.columns
df.select_dtypes(["object"]).columns
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose']], drop_first=True)
df.drop(['verification_status', 'application_type','initial_list_status','purpose'], axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)
df["home_ownership"].value_counts()
df["home_ownership"] = df["home_ownership"].replace(["NONE", "ANY"], "OTHER")
df["home_ownership"].value_counts()
dummies = pd.get_dummies(df["home_ownership"], drop_first=True)
df.drop("home_ownership", axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)
df["address"]
df["zipcode"] = df["address"].apply(lambda x: x[-5:])
df["zipcode"].value_counts()
dummies = pd.get_dummies(df["zipcode"], drop_first=True)
df.drop("zipcode", axis=1, inplace=True)
df = pd.concat([df, dummies], axis=1)
df.drop("address", axis=1, inplace=True)
feat_info("issue_d")
df.drop("issue_d", axis=1, inplace=True)
feat_info("earliest_cr_line")
df["earliest_cr_line"]
df["earliest_cr_year"] = df["earliest_cr_line"].apply(lambda x: int(x.split("-")[1]))
df.head()
df.drop("earliest_cr_line", axis=1, inplace=True)
df.select_dtypes(["object"]).columns
from sklearn.model_selection import train_test_split
df["loan_status"]
df["loan_repaid"]
df.drop("loan_status", axis=1, inplace=True)
df.select_dtypes(["object"]).columns
X = df.drop("loan_repaid", axis=1).values
y = df["loan_repaid"].values
# df = df.sample(frac=0.1,random_state=101)
print(len(df))
df.shape
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
# CODE HERE
model = Sequential()

# Choose whatever number of layers/neurons you want.
model.add(Dense(79, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(40, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(20, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# Remember to compile()
model.compile(loss="binary_crossentropy", optimizer="adam")
model.fit(X_train, y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test))
from tensorflow.keras.models import load_model
model.save('keras_project.h5')  
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)
import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer
model.predict_classes(new_customer.values.reshape(1,78))
df.iloc[random_ind]['loan_repaid']

# with Early Stopping
# CODE HERE
model = Sequential()

# Choose whatever number of layers/neurons you want.
model.add(Dense(79, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(40, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(20, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# Remember to compile()
model.compile(loss="binary_crossentropy", optimizer="adam")
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=0.005)
model.fit(X_train, y_train, epochs=25, batch_size=256, validation_data=(X_test, y_test), callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)