#linear algebra

import numpy as np



#dataframe

import pandas as pd



#data visualization

import matplotlib.pyplot as plt

import seaborn as sns



#regex

import re



#machine learning

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report,f1_score,accuracy_score

from xgboost import XGBClassifier



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from vecstack import stacking



#neural network

import tensorflow as tf

from tensorflow import keras
df_flight = pd.read_csv("/kaggle/input/datavidia2019/flight.csv")

df_hotel = pd.read_csv("/kaggle/input/datavidia2019/hotel.csv")

df_test = pd.read_csv("/kaggle/input/datavidia2019/test.csv")
df_flight.sample(3)
df_flight.info()
df_hotel.head()
df_hotel.info()
df_test.head(3)
df_test.info()
df_flight.drop(["order_id"],axis=1,inplace=True)

df_test.drop(["order_id"],axis=1,inplace=True)
# function for fix list in form of string into a python accessable list

def stringoflist_to_list(text,as_float=False):

    result = re.sub("[\'\,\[\]]","",text)

    result = re.split("\s",result)

    if(as_float):  

        result = [float(x) for x in result]  

    return result
#fix the df_flight dataframe

visited_city_idx = df_flight.columns.get_loc("visited_city")

log_transaction_idx = df_flight.columns.get_loc("log_transaction")

for index, row in df_flight.iterrows():

    df_flight.iat[index, visited_city_idx] = stringoflist_to_list(row["visited_city"],False)

    df_flight.iat[index, log_transaction_idx] = stringoflist_to_list(row["log_transaction"],True)
#fix the df_test dataframe

visited_city_idx = df_test.columns.get_loc("visited_city")

log_transaction_idx = df_test.columns.get_loc("log_transaction")

for index, row in df_test.iterrows():

    df_test.iat[index, visited_city_idx] = stringoflist_to_list(row["visited_city"],False)

    df_test.iat[index, log_transaction_idx] = stringoflist_to_list(row["log_transaction"],True)
df_flight.sample(3)
df_test.sample(3)
df_test["member_duration_days"] = df_test["member_duration_days"].astype("float64")

df_test["no_of_seats"] = df_test["no_of_seats"].astype("float64")
df_flight["member_duration_days"].describe()
df_test["member_duration_days"].describe()
df_flight["account_id"].value_counts(sort=True)
df_flight["gender"].value_counts()
df_test["gender"].value_counts()
df_flight["trip"].value_counts()
df_test["trip"].value_counts()
df_flight["trip"] = np.where(df_flight["trip"]=="trip","trip","roundtrip")

df_test["trip"] = np.where(df_test["trip"]=="trip","trip","roundtrip")
df_flight["trip"].value_counts()
df_test["trip"].value_counts()
df_flight["service_class"].value_counts()
df_test["service_class"].value_counts()
df_flight["price"].describe()
df_test["price"].describe()
df_flight["is_tx_promo"].value_counts()
df_test["is_tx_promo"].value_counts()
df_flight["no_of_seats"].describe()
df_test["no_of_seats"].describe()
df_flight["airlines_name"].value_counts()
df_test["airlines_name"].value_counts()
df_flight["route"].value_counts()
df_test["route"].value_counts()
df_flight.drop(["route"],axis=1,inplace=True)

df_test.drop(["route"],axis=1,inplace=True)
df_flight["hotel_id"].value_counts()
df_flight["is_cross_sell"] = np.where(df_flight["hotel_id"]=="None",0,1)
df_flight["is_cross_sell"].value_counts()
df_flight.drop(["hotel_id"],axis=1,inplace=True)
df_flight.sample(3)
Var_Corr = df_flight.corr()

sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)

plt.title("Correlation heatmap")

plt.show()
sns.countplot(x="is_cross_sell", data=df_flight)

plt.title("is_cross_sell class count")

plt.show()
sns.distplot(df_flight["member_duration_days"])

plt.title("member_duration_days data distribution")

plt.show()
sns.boxplot(x="is_cross_sell", y="member_duration_days", data=df_flight)

plt.title("is_cross_sell vs member_duration_days")

plt.show()
sns.countplot(x="gender", hue="is_cross_sell", data=df_flight)

plt.title("gender count")

plt.show()
sns.countplot(x="trip", hue="is_cross_sell", data=df_flight)

plt.title("trip count")

plt.show()
sns.distplot(df_flight["price"])

plt.title("price data distribution")

plt.show()
sns.boxplot(x="is_cross_sell", y="price", data=df_flight)

plt.title("is_cross_sell vs price")

plt.show()
sns.countplot(x="is_tx_promo", hue="is_cross_sell", data=df_flight)

plt.title("tx_with_promo count")

plt.show()
sns.boxplot(x="is_cross_sell", y="no_of_seats", data=df_flight)

plt.title("is_cross_sell vs num_of_seats")

plt.show()
sns.countplot(x="airlines_name", hue="is_cross_sell", data=df_flight)

plt.title("airlines_name count")

plt.show()
## df_flight



#initial value for the column

df_flight["number_city_visited"] = 0.0



#get the index of the column number_city_visited

the_idx = df_flight.columns.get_loc("number_city_visited")





#assign the value for every row

for index, row in df_flight.iterrows():

    df_flight.iat[index, the_idx] = len(row["visited_city"])

    

#drop the original column as we already got what we need

df_flight.drop(["visited_city"],axis=1,inplace=True)
##same process as the above cell but for df_test

df_test["number_city_visited"] = 0.0

the_idx = df_test.columns.get_loc("number_city_visited")

for index, row in df_test.iterrows():

    df_test.iat[index, the_idx] = len(row["visited_city"])

df_test.drop(["visited_city"],axis=1,inplace=True)
##df_flight

#initial column value

df_flight["amount_spent"] = 0.0

df_flight["min_spend"] = 0.0

df_flight["max_spend"] = 0.0

df_flight["average_spend"] = 0.0

df_flight["number_of_transaction"] = 0.0



#assign value for each row

for index, row in df_flight.iterrows():

    df_flight.iat[index, -5] = sum(row["log_transaction"])

    df_flight.iat[index, -4] = min(row["log_transaction"])

    df_flight.iat[index, -3] = max(row["log_transaction"])

    df_flight.iat[index, -2] = np.average(row["log_transaction"])

    df_flight.iat[index, -1] = len(row["log_transaction"])

    

# drop the original column as we already got what we need

df_flight.drop(["log_transaction"],axis=1,inplace=True)
#same process for the above cell but fr df_test

df_test["amount_spent"] = 0.0

df_test["min_spend"] = 0.0

df_test["max_spend"] = 0.0

df_test["average_spend"] = 0.0

df_test["number_of_transaction"] = 0.0

for index, row in df_test.iterrows():

    df_test.iat[index, -5] = sum(row["log_transaction"])

    df_test.iat[index, -4] = min(row["log_transaction"])

    df_test.iat[index, -3] = max(row["log_transaction"])

    df_test.iat[index, -2] = np.average(row["log_transaction"])

    df_test.iat[index, -1] = len(row["log_transaction"])

df_test.drop(["log_transaction"],axis=1,inplace=True)
df_flight.sample(3)
df_test.sample(3)
ics_history = df_flight.groupby(["account_id"]).mean()["is_cross_sell"]
df_flight = pd.merge(df_flight, ics_history, on='account_id', how="left")

old_column_name = df_flight.columns[-1]

df_flight = df_flight.rename(columns = {"is_cross_sell_x" : "is_cross_sell", old_column_name : "ics_probability"})
df_test = pd.merge(df_test, ics_history, on='account_id', how="left")

old_column_name = df_test.columns[-1]

df_test[old_column_name] = df_test[old_column_name].fillna(0)

df_test = df_test.rename(columns = {"is_cross_sell_x" : "is_cross_sell", old_column_name : "ics_probability"})
df_flight.drop(["account_id"],axis=1,inplace=True)

df_test.drop(["account_id"],axis=1,inplace=True)
df_flight.sample(3)
df_test.sample(3)
#trip

le = LabelEncoder()

df_flight['trip'] = le.fit_transform(df_flight['trip'])



#service_class

le = LabelEncoder()

df_flight['service_class'] = le.fit_transform(df_flight['service_class'])



#is_tx_promo

le = LabelEncoder()

df_flight['is_tx_promo'] = le.fit_transform(df_flight['is_tx_promo'])
#trip

le = LabelEncoder()

df_test['trip'] = le.fit_transform(df_test['trip'])



#service_class

le = LabelEncoder()

df_test['service_class'] = le.fit_transform(df_test['service_class'])



#is_tx_promo

le = LabelEncoder()

df_test['is_tx_promo'] = le.fit_transform(df_test['is_tx_promo'])
# make sure to concat the training and testing so both of the data get equal data dummies

df = pd.concat([df_flight[df_flight.columns[:-1]], df_test],sort = False)

airlines_bin_df = pd.get_dummies(df["airlines_name"], prefix="an")

gender_bin_df = pd.get_dummies(df["gender"], prefix="g")
df_flight = pd.concat([df_flight, gender_bin_df.iloc[:len(df_flight)], airlines_bin_df.iloc[:len(df_flight)]], axis = 1)

df_flight.drop(["gender"],axis=1,inplace=True)

df_test.drop(["gender"],axis=1,inplace=True)



df_test = pd.concat([df_test, gender_bin_df.iloc[len(df_flight):], airlines_bin_df.iloc[len(df_flight):]], axis = 1)

df_flight.drop(["airlines_name"],axis=1,inplace=True)

df_test.drop(["airlines_name"],axis=1,inplace=True)
df_flight.sample(3)
df_test.sample(3)
feature_to_scale = ["member_duration_days","price","no_of_seats","number_city_visited","amount_spent","min_spend","max_spend","average_spend","number_of_transaction"]
minmax_scaler = MinMaxScaler()

df_temp = pd.concat([df_flight.drop(["is_cross_sell"],axis=1), df_test],sort=False)

minmax_scaler.fit(df_temp[feature_to_scale],)
df_flight[feature_to_scale] = minmax_scaler.transform(df_flight[feature_to_scale])

df_test[feature_to_scale] = minmax_scaler.transform(df_test[feature_to_scale])
df_flight.sample(3)
df_test.sample(3)
df_flight.sample(3)
df_test.sample(3)
X = df_flight.drop(["is_cross_sell"],axis=1)

y = df_flight["is_cross_sell"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# the hyperparameter is tuned on other kernel

rf_clf = RandomForestClassifier(

    bootstrap=False,

    max_depth=50,

    max_features = "auto",

    min_samples_leaf=1,

    min_samples_split=2,

    n_estimators=1000,

    n_jobs=-1

)
rf_clf.fit(X_train,y_train)
rf_clf.score(X_test,y_test)
f1_score(rf_clf.predict(X_test),y_test)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train,y_train)
knn_clf.score(X_test,y_test)
f1_score(knn_clf.predict(X_test),y_test)
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
f1_score(lr_clf.predict(X_test),y_test)
n_cols = X_train.shape[1] 

nn_clf = keras.Sequential()

nn_clf.add(keras.layers.Dense(20, activation='relu', input_shape=(n_cols,)))

nn_clf.add(keras.layers.Dropout(rate=0.2))

nn_clf.add(keras.layers.Dense(20, activation='relu'))

nn_clf.add(keras.layers.Dropout(rate=0.2))

nn_clf.add(keras.layers.Dense(1, activation='sigmoid'))
nn_clf.compile( 

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)
nn_clf.fit(X_train, y_train, epochs=10, verbose=2)
nn_clf.evaluate(X_test,y_test)
f1_score(np.where(nn_clf.predict(X_test)> 0.5, 1, 0),y_test)
xgb_clf = XGBClassifier(random_state=0)
xgb_clf.fit(X_train,y_train)
xgb_clf.score(X_test,y_test)
f1_score(xgb_clf.predict(X_test),y_test)
rf_clf = RandomForestClassifier(

    bootstrap=False,

    max_depth=50,

    max_features = "auto",

    min_samples_leaf=1,

    min_samples_split=2,

    n_estimators=1000,

    n_jobs=-1

)
knn_clf = KNeighborsClassifier(n_neighbors=5)
lr_clf = LogisticRegression(max_iter=1000)
models = [

    rf_clf,

    knn_clf,

    lr_clf,

]
S_train, S_test = stacking(

    models,                   

    X_train, y_train, X_test,   

    regression=False, 

    mode='oof_pred_bag', 

    needs_proba=False,

    save_dir=None, 

    metric=accuracy_score, 

    n_folds=4, 

    stratified=True,

    shuffle=True,  

    random_state=0,    

    verbose=2

)
S_train = pd.DataFrame(S_train, columns = ["rf","knn","lr"])

S_test = pd.DataFrame(S_test, columns = ["rf","knn","lr"])
X_train.reset_index(inplace=True)

X_test.reset_index(inplace=True)
df_stack_train = pd.concat([X_train,S_train],axis=1)

df_stack_test = pd.concat([X_test,S_test],axis=1)
df_stack_train.drop(["index"],axis=1,inplace=True)

df_stack_train.head()
df_stack_test.drop(["index"],axis=1,inplace=True)

df_stack_test.head()
n_cols = df_stack_train.shape[1]
nn_clf = keras.Sequential()

nn_clf.add(keras.layers.Dense(16, activation='relu', input_shape=(n_cols,)))

nn_clf.add(keras.layers.Dropout(rate=0.2))

nn_clf.add(keras.layers.Dense(1, activation='sigmoid'))
nn_clf.compile( 

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)
xgb_clf = XGBClassifier(random_state=0)
nn_clf.fit(df_stack_train, y_train, epochs=10, verbose=2)
y_pred_nn = np.where(nn_clf.predict(df_stack_test)> 0.5, 1, 0)
f1_score(y_pred_nn,y_test)
xgb_clf.fit(df_stack_train,y_train)
y_pred_xgb = xgb_clf.predict(df_stack_test)
xgb_clf.score(df_stack_test,y_test)
f1_score(y_pred_xgb,y_test)
rf_clf = RandomForestClassifier(

    bootstrap=False,

    max_depth=50,

    max_features = "auto",

    min_samples_leaf=1,

    min_samples_split=2,

    n_estimators=1000,

    n_jobs=-1

)
knn_clf = KNeighborsClassifier(n_neighbors=5)
lr_clf = LogisticRegression(max_iter=1000)
models = [

    rf_clf,

    knn_clf,

    lr_clf,

]
S_train, S_test = stacking(

    models,                   

    X, y, df_test,   

    regression=False, 

    mode='oof_pred_bag', 

    needs_proba=False,

    save_dir=None, 

    metric=accuracy_score, 

    n_folds=4, 

    stratified=True,

    shuffle=True,  

    random_state=0,    

    verbose=2

)
S_train = pd.DataFrame(S_train, columns = ["rf","knn","lr"])

S_test = pd.DataFrame(S_test, columns = ["rf","knn","lr"])
df_stack_train = pd.concat([X,S_train],axis=1)

df_stack_test = pd.concat([df_test,S_test],axis=1)
xgb_clf = XGBClassifier(random_state=0)
xgb_clf.fit(df_stack_train,y)
y_pred_xgb = xgb_clf.predict(df_stack_test)
temp = pd.read_csv("/kaggle/input/datavidia2019/test.csv")

temp = temp["order_id"]



d = {'order_id': temp, 'is_cross_sell': y_pred_xgb}

output = pd.DataFrame(data = d)



output["is_cross_sell"] = np.where(output["is_cross_sell"]==0,"no","yes")

output.to_csv("stacking-meta-xgb.csv",index=False)