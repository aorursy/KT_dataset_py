import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score,f1_score
df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df
df.isnull().sum()
replace = {"children:": 0.0,"country": "Unknown", "agent": 0, "company": 0}

df1 = df.fillna(replace)
df1.drop(df1.loc[df['adults']+df['children']+df['babies']==0].index,inplace=True)
df1.shape
resort = df1.loc[(df1["hotel"] == "Resort Hotel") & (df1["is_canceled"] == 0)]

city = df1.loc[(df1["hotel"] == "City Hotel") & (df1["is_canceled"] == 0)]
ms=df1["market_segment"].value_counts()

ms
lab = ['Online TA', 'Offline TA/TO','Groups','Direct','Corporate','Complementary','Aviation','Undefined'] 
plt.figure(figsize=(15,8))

plt.pie(ms,autopct='%1.2f%%',explode=[0,0,0,0,0,0.5,0.8,1.4],labels=lab)

plt.show()
total_cancels = df1["is_canceled"].sum()

resort_cancels = df1.loc[df1["hotel"] == "Resort Hotel"]["is_canceled"].sum()

city_cancels = df1.loc[df1["hotel"] == "City Hotel"]["is_canceled"].sum()



cancel_per = (total_cancels / df1.shape[0]) * 100

rh_cancel_per = (resort_cancels / df1.loc[df1["hotel"] == "Resort Hotel"].shape[0]) * 100

ch_cancel_per = (city_cancels / df1.loc[df1["hotel"] == "City Hotel"].shape[0]) * 100



print(f"Total bookings canceled: {total_cancels:,} ({cancel_per:.0f} %)")

print(f"Resort hotel bookings canceled: {resort_cancels:,} ({rh_cancel_per:.0f} %)")

print(f"City hotel bookings canceled: {city_cancels:,} ({ch_cancel_per:.0f} %)")
rbook_per_month = df1.loc[(df1["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["hotel"].count()

rcancel_per_month = df1.loc[(df1["hotel"] == "Resort Hotel")].groupby("arrival_date_month")["is_canceled"].sum()



cbook_per_month = df1.loc[(df1["hotel"] == "City Hotel")].groupby("arrival_date_month")["hotel"].count()

ccancel_per_month = df1.loc[(df1["hotel"] == "City Hotel")].groupby("arrival_date_month")["is_canceled"].sum()



rcancel = pd.DataFrame({"Hotel": "Resort Hotel",

                                "Month": list(rbook_per_month.index),

                                "Bookings": list(rbook_per_month.values),

                                "Cancelations": list(rcancel_per_month.values)})

ccancel = pd.DataFrame({"Hotel": "City Hotel",

                                "Month": list(cbook_per_month.index),

                                "Bookings": list(cbook_per_month.values),

                                "Cancelations": list(ccancel_per_month.values)})



full_cancel_data = pd.concat([rcancel, ccancel], ignore_index=True)

full_cancel_data["cancel_percent"] = full_cancel_data["Cancelations"] / full_cancel_data["Bookings"] * 100



ordered_months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

full_cancel_data["Month"] = pd.Categorical(full_cancel_data["Month"], categories=ordered_months, ordered=True)



# show figure:

plt.figure(figsize=(12, 8))

sns.barplot(x = "Month", y = "cancel_percent" , hue="Hotel",

            hue_order = ["City Hotel", "Resort Hotel"], data=full_cancel_data)

plt.title("Cancelations per month", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("Cancelations [%]", fontsize=16)

plt.legend(loc="upper right")

plt.show()
cancel_corr = df.corr()["is_canceled"]

cancel_corr.abs().sort_values(ascending=False)[1:]
df.groupby("is_canceled")["reservation_status"].value_counts()
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

df1.isnull().sum()
df1['children'].fillna(df1['children'].median(),inplace=True)
df1.isnull().sum()
df1 = df1.drop_duplicates()
y = df1["is_canceled"]

X = df1.drop(["is_canceled"], axis=1)



x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=0)





num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",

                "stays_in_weekend_nights","stays_in_week_nights","adults","children",

                "babies","is_repeated_guest", "previous_cancellations",

                "previous_bookings_not_canceled","agent","company",

                "required_car_parking_spaces", "total_of_special_requests", "adr"]



cat_features = ["hotel","arrival_date_month","meal","market_segment",

                "distribution_channel","reserved_room_type","deposit_type","customer_type"]



x_train = x_train[num_features + cat_features].copy()

x_test = x_test[num_features + cat_features].copy()



num_transformer = SimpleImputer(strategy="constant")





cat_transformer = Pipeline(steps=[

    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),

    ("onehot", OneHotEncoder(handle_unknown='ignore'))])





preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features),

                                               ('cat', cat_transformer, cat_features)])
from xgboost import XGBClassifier



model = XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=1000, gamma=0, 

                        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005,random_state=101)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])



pipeline.fit(x_train, y_train)





predictions = pipeline.predict(x_test)



from sklearn.metrics import classification_report

from sklearn import metrics



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print(classification_report(y_test,predictions))



confusion_matrix=metrics.confusion_matrix(y_test,predictions)



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])



precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])



f1_score = (2*sensitivity*precision)/(sensitivity+precision)

print('f1_score1 :% .2f '% f1_score)
model = KNeighborsClassifier(n_neighbors=5,

    weights='uniform',

    algorithm='auto',

    leaf_size=30,

    p=2,

    metric='minkowski')

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])



pipeline.fit(x_train, y_train)





predictions = pipeline.predict(x_test)



from sklearn.metrics import classification_report

from sklearn import metrics



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print(classification_report(y_test,predictions))



confusion_matrix=metrics.confusion_matrix(y_test,predictions)



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])



precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])



f1_score = (2*sensitivity*precision)/(sensitivity+precision)

print('f1_score :% .2f '% f1_score)
model = DecisionTreeClassifier(criterion='gini',

    splitter='best',

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    min_weight_fraction_leaf=0.0,

    max_features=None,

    random_state=None,

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    class_weight=None,

    presort='deprecated',

    ccp_alpha=0.0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])



pipeline.fit(x_train, y_train)





predictions = pipeline.predict(x_test)



from sklearn.metrics import classification_report

from sklearn import metrics



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print(classification_report(y_test,predictions))



confusion_matrix=metrics.confusion_matrix(y_test,predictions)



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])



precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])



f1_score = (2*sensitivity*precision)/(sensitivity+precision)

print('f1_score :% .2f '% f1_score)
model = RandomForestClassifier(n_estimators=100,

    criterion='gini',

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    min_weight_fraction_leaf=0.0,

    max_features='auto',

    max_leaf_nodes=None,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    bootstrap=True,

    oob_score=False,

    n_jobs=None,

    random_state=None,

    verbose=0,

    warm_start=False,

    class_weight=None,

    ccp_alpha=0.0,

    max_samples=None)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])



pipeline.fit(x_train, y_train)





predictions = pipeline.predict(x_test)



from sklearn.metrics import classification_report

from sklearn import metrics



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print(classification_report(y_test,predictions))



confusion_matrix=metrics.confusion_matrix(y_test,predictions)



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])



precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])



f1_score = (2*sensitivity*precision)/(sensitivity+precision)

print('f1_score :% .2f '% f1_score)
model = BaggingClassifier(base_estimator=None,

    n_estimators=10,

    max_samples=1.0,

    max_features=1.0,

    bootstrap=True,

    bootstrap_features=False,

    oob_score=False,

    warm_start=False,

    n_jobs=None,

    random_state=None,

    verbose=0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])



pipeline.fit(x_train, y_train)





predictions = pipeline.predict(x_test)



from sklearn.metrics import classification_report

from sklearn import metrics



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print(classification_report(y_test,predictions))



confusion_matrix=metrics.confusion_matrix(y_test,predictions)



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])



precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])



f1_score = (2*sensitivity*precision)/(sensitivity+precision)

print('f1_score :% .2f '% f1_score)
model = AdaBoostClassifier(base_estimator=None,

    n_estimators=50,

    learning_rate=1.0,

    algorithm='SAMME.R',

    random_state=None)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])



pipeline.fit(x_train, y_train)





predictions = pipeline.predict(x_test)



from sklearn.metrics import classification_report

from sklearn import metrics



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print(classification_report(y_test,predictions))



confusion_matrix=metrics.confusion_matrix(y_test,predictions)



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])



precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])



f1_score = (2*sensitivity*precision)/(sensitivity+precision)

print('f1_score :% .2f '% f1_score)
model = GradientBoostingClassifier(loss='deviance',

    learning_rate=0.1,

    n_estimators=100,

    subsample=1.0,

    criterion='friedman_mse',

    min_samples_split=2,

    min_samples_leaf=1,

    min_weight_fraction_leaf=0.0,

    max_depth=3,

    min_impurity_decrease=0.0,

    min_impurity_split=None,

    init=None,

    random_state=None,

    max_features=None,

    verbose=0,

    max_leaf_nodes=None,

    warm_start=False,

    presort='deprecated',

    validation_fraction=0.1,

    n_iter_no_change=None,

    tol=0.0001,

    ccp_alpha=0.0)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)])



pipeline.fit(x_train, y_train)





predictions = pipeline.predict(x_test)



from sklearn.metrics import classification_report

from sklearn import metrics



accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print(classification_report(y_test,predictions))



confusion_matrix=metrics.confusion_matrix(y_test,predictions)



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])



specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])



precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])



f1_score = (2*sensitivity*precision)/(sensitivity+precision)

print('f1_score :% .2f '% f1_score)