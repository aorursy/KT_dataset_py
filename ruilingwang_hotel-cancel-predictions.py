# import libs
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
df.head()
df["hotel"].value_counts()
df_resort = df[df.hotel == 'Resort Hotel']
cancel_resort = round(len(df_resort[df_resort.is_canceled == 1])/ len(df_resort),3)
df_city = df[df.hotel == 'City Hotel']
cancel_city = round(len(df_city[df_city.is_canceled == 1])/ len(df_city),3)
print("cancel rate for resort is ",cancel_resort)
print("cancel rate for city is ", cancel_city)

df1 = df.copy()
month = {'January':1, "February":2, "March":3, "April":4,"May":5,"June":6, "July":7, "August":8, "September":9,"October":10, "November":11, "December":12}
df1.arrival_date_month = df1.arrival_date_month.apply(lambda x: month[x])
df1 = df1.sort_values(by = 'arrival_date_month')
for i in range(0, 3): # i is index for combine chart and count year
    s = df1[df1.arrival_date_year == i+2015].arrival_date_month.value_counts().sort_index()
    s.plot.bar(width=0.5)
    plt.show()
df1["stays_duation_total"] =  df1.stays_in_weekend_nights + df1.stays_in_week_nights
for name in ['Resort Hotel', 'City Hotel']:
    df2 = df1[df1.hotel == name].groupby(['arrival_date_year'])["stays_duation_total"].mean().to_frame()
    df2.plot.bar()
    plt.show()

cancel_corr = df.corr()["is_canceled"]
cancel_corr.abs().sort_values(ascending = False)
# preprocessing
df = df[df['is_canceled'].notna()]
features = ['lead_time', 'total_of_special_requests', 'required_car_parking_spaces', 'booking_changes','previous_cancellations']
X = df[features]
Y = df["is_canceled"]
# missing value with median
num_transformer = SimpleImputer(strategy="median")
num_transformer.fit_transform(X)
# extract training data (60%) and test data (40%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.40, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred=logreg.predict(X_test)
# get confusion matrix
cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)
# visualize confusion matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# accuracy, percision, recall
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))
# First, we can extract some features from the original features to make model more general:
# (arrival_date_year, assigned_room_type, booking_changes, reservation_status, days_in_waiting_list)
# The data can be simply divided into numerial data and categorical data:
num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
                "stays_in_weekend_nights","stays_in_week_nights","adults","children",
                "babies","is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled","agent","company",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]
cat_features = ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]
features = num_features + cat_features
X = df.drop(["is_canceled"], axis=1)[features]
y = df["is_canceled"]
# deal with num_features - fill missing value - choose 0 as the filled value for all columns except date
# however, date does not have missing value
# (PS: simpleInputer can be not only used in filling numerical data but also string type data)
num_transformer = SimpleImputer(strategy="constant", fill_value=0)
# deal with categorical data
cat_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy="constant", fill_value="unkown")), 
                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                               ("cat", cat_transformer, cat_features)])

# referred from https://www.kaggle.com/marcuswingen/eda-of-bookings-and-ml-to-predict-cancelations
# split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
model = Pipeline(steps=[('preprocessor', preprocessor),('rf', RandomForestClassifier(random_state=42,n_jobs=-1))])
model.fit(X_train, y_train)
pred = model.predict(X_test)
#score = metrics.accuracy_score(y_test, pred)
#print("the accuracy score is: ", round(score, 2))

# get confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, pred)
# visualize confusion matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
# accuracy, percision, recall
print("Accuracy:",metrics.accuracy_score(y_test, pred))
print("Precision:",metrics.precision_score(y_test, pred))
print("Recall:",metrics.recall_score(y_test, pred))