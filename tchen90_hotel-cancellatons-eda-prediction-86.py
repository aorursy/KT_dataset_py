import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load the dataset

df = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df.head()
df.describe()
# check for missing values

df.isnull().sum()
# check for the shape of dataset

df.shape
# check for data type of each column

df.dtypes
df['is_canceled'].value_counts(normalize=True)
df['reservation_status'].value_counts(normalize=True)
df.drop(columns=['agent', 'company', 'reservation_status'],inplace=True)

df.dropna(axis=0,inplace=True)

df.shape
df.isnull().sum()
df['meal'].value_counts()
# "meal" contains values "Undefined", which is equal to SC

df['meal'].replace('Undefined','SC',inplace=True)
df.hist(figsize=(20,20))

plt.show()
len(df[(df['adults']==0) & (df['children']==0) & (df['babies']==0)])
zero_guests = df[(df['adults']==0) & (df['children']==0) & (df['babies']==0)].index

df.drop(zero_guests, inplace=True)

df.shape
print('There are ' + str(len(df[(df['hotel']=='Resort Hotel') & (df['is_canceled']==1)])) + ' cancelations at Resort Hotel')

print('There are ' + str(len(df[(df['hotel']=='City Hotel') & (df['is_canceled']==1)])) + ' cancelations at City Hotel')
plt.figure(figsize=(6,6))

plt.title(label='Cancellations by Hotel Types')

sns.countplot(x='hotel',hue='is_canceled',data=df)

plt.show()
# % of cancellations in Resort Hotel

df[df['hotel']=='Resort Hotel']['is_canceled'].value_counts(normalize=True)
# % of cancellations in City Hotel

df[df['hotel']=='City Hotel']['is_canceled'].value_counts(normalize=True)
plt.figure(figsize=(12,6))

plt.title(label='Cancellation by Lead Time')

sns.barplot(x='hotel',y='lead_time',hue='is_canceled',data=df)

plt.show()
plt.figure(figsize=(6,6))

plt.title(label='Cancellation by ADR')

sns.barplot(x='is_canceled',y='adr',data=df)

plt.show()
plt.figure(figsize=(6,6))

plt.title(label='Cancellation by ADR & Hotel Type')

sns.barplot(x='hotel',y='adr',hue='is_canceled',data=df)

plt.show()
plt.figure(figsize=(6,6))

plt.title(label='Cancellation by Deposit Type')

sns.countplot(x='deposit_type',hue='is_canceled',data=df)

plt.show()
plt.figure(figsize=(6,6))

plt.title(label='Cancellation by Market Segments')

plt.xticks(rotation=45) 

sns.countplot(x='market_segment',hue='is_canceled',data=df)

plt.show()
plt.figure(figsize=(6,6))

plt.title(label='Cancellation by Market Segments & ADR')

plt.xticks(rotation=45) 

sns.barplot(x='market_segment',y='adr',hue='is_canceled',data=df)

plt.show()
plt.figure(figsize=(6,6))

plt.title(label='Cancellation by Distribution Channels')

plt.xticks(rotation=45) 

sns.countplot(x='distribution_channel',hue='is_canceled',data=df)

plt.show()
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], format='%Y-%m-%d')
plt.figure(figsize=(6,6))

plt.title(label='Cancellation by Month')

plt.xticks(rotation=45) 

sns.countplot(x=df['reservation_status_date'].dt.month,hue='is_canceled',data=df)

plt.show()
plt.figure(figsize=(19,6))

plt.title(label='Cancellation by Week Number')

plt.xticks(rotation=45) 

sns.countplot(x=df['arrival_date_week_number'],hue='is_canceled',data=df)

plt.show()
plt.figure(figsize=(16,6))

plt.title(label='Cancellation by day')

plt.xticks(rotation=45) 

sns.countplot(x=df['reservation_status_date'].dt.day,hue='is_canceled',data=df)

plt.show()
cat_cols=['is_canceled','arrival_date_month','meal','market_segment','distribution_channel','reserved_room_type',

      'is_repeated_guest','deposit_type','customer_type']

df[cat_cols] = df[cat_cols].astype('category')

num_cols = ['lead_time','arrival_date_week_number','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights',

        'adults','children','babies','previous_cancellations','previous_bookings_not_canceled','required_car_parking_spaces',

        'total_of_special_requests','adr']
model_df = df[cat_cols+num_cols]

model_df.shape
model_df.corr()
# Create dummy variables

df_dummies = pd.get_dummies(model_df.drop(columns=['is_canceled']))
df_dummies.head()
y = model_df['is_canceled']

X = df_dummies
# Load modules for machine learning

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=0)

sc = StandardScaler()



sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train_std, y_train)

y_lr_pred = lr.predict(X_test_std)



print('Accuracy: %.4f' % accuracy_score(y_test, y_lr_pred))

print(confusion_matrix(y_test, y_lr_pred))

print(classification_report(y_test,y_lr_pred))
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()

clf.fit(X_train_std, y_train)

y_clf_pred = clf.predict(X_test_std)



print('Accuracy: %.4f' % accuracy_score(y_test, y_clf_pred))

print(confusion_matrix(y_test, y_clf_pred))

print(classification_report(y_test,y_clf_pred))
from sklearn.ensemble import AdaBoostClassifier



ada = AdaBoostClassifier()

ada.fit(X_train_std, y_train)

y_ada_pred = ada.predict(X_test_std)



print('Accuracy: %.4f' % accuracy_score(y_test, y_ada_pred))

print(confusion_matrix(y_test, y_ada_pred))

print(classification_report(y_test,y_ada_pred))
from sklearn.ensemble import GradientBoostingClassifier



gbc = GradientBoostingClassifier()

gbc.fit(X_train_std, y_train)

y_gbc_pred = gbc.predict(X_test_std)



print('Accuracy: %.4f' % accuracy_score(y_test, y_gbc_pred))

print(confusion_matrix(y_test, y_gbc_pred))

print(classification_report(y_test,y_gbc_pred))
from xgboost import XGBClassifier



xgb = XGBClassifier()

xgb.fit(X_train_std, y_train)

y_xgb_pred = xgb.predict(X_test_std)



print('Accuracy: %.4f' % accuracy_score(y_test, y_xgb_pred))

print(confusion_matrix(y_test, y_xgb_pred))

print(classification_report(y_test,y_xgb_pred))
from sklearn.ensemble import RandomForestClassifier



rfl = RandomForestClassifier()

rfl.fit(X_train_std, y_train)

y_rfl_pred = rfl.predict(X_test_std)



print('Accuracy: %.4f' % accuracy_score(y_test, y_rfl_pred))

print(confusion_matrix(y_test, y_rfl_pred))

print(classification_report(y_test,y_rfl_pred))
importances = rfl.feature_importances_ 

# Sort feature importances in descending order

indices = np.argsort(importances)[::-1]



# Rearrange feature names so they match the sorted feature importances

names = [X.columns[i] for i in indices]



# Create plot

plt.figure(figsize=(30,30))



# Create plot title

plt.title("Feature Importance")



# Add bars

plt.bar(range(X.shape[1]), importances[indices])



# Add feature names as x-axis labels

plt.xticks(range(X.shape[1]), names, rotation=90)



# Show plot

plt.show()
# pring all feature importance

indices = np.argsort(importances)[::-1]

feat_labels = X.columns[:]



for f in range(X_train_std.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30, 

                            feat_labels[indices[f]], 

                            importances[indices[f]]))
from sklearn.metrics import roc_curve, roc_auc_score
y_score = rfl.predict_proba(X_test_std)[:,1]

# Create true and false positive rates

false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)

# Plot ROC curve

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate)

plt.plot([0, 1], ls="--")

plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
import scikitplot as skplt
rf = rfl.fit(X_train_std, y_train)

y_probas = rf.predict_proba(X_test_std)

skplt.metrics.plot_roc(y_test,y_probas)

plt.show()
skplt.metrics.plot_lift_curve(y_test, y_probas)

plt.show()