import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Load the data
df = pd.read_csv('../input/KaggleV2-May-2016.csv')
df.head(20)
# See descriptive summary
df.describe()
# There are some age values which are less than 0, so need to clean that
df.loc[df['Age'] < 0, 'Age'] = 0
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,1,1)
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
df['Age'].hist(grid=False, ax=ax1)

ax2 = fig.add_subplot(2,1,2)
df['Age'].plot(kind='box', ax=ax2)
# Visualize the distribution of the categorical variables

categorical_variables = [
    'Gender', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'No-show'
]

fig = plt.figure(figsize=(16, 11))
for ctr, variable in enumerate(categorical_variables):
    ax = fig.add_subplot(3, 3, ctr+1)
    ax.set_xlabel(variable)
    df[variable].value_counts().plot(kind='bar', ax=ax)
# Visualize the categorical variables grouped by No-Show

categorical_variables = [
    'Gender', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received'
]

fig = plt.figure(figsize=(16, 11))
for ctr, variable in enumerate(categorical_variables):
    ax = fig.add_subplot(3, 3, ctr+1)
    df.groupby([variable, 'No-show'])[variable].count().unstack('No-show').plot(ax=ax, kind='bar', stacked=True)
# Convert the ScheduledDay and AppointmentDay columns to type 'datetime' instead of string

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Scheduled Day')
ax.set_ylabel('Frequency')
df['ScheduledDay'].hist(grid=False, ax=ax)
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Appointment Day')
ax.set_ylabel('Frequency')
df['AppointmentDay'].hist(grid=False, ax=ax)
print(df['ScheduledDay'].min())
print(df['ScheduledDay'].max())
print(df['AppointmentDay'].min())
print(df['AppointmentDay'].max())
df['difference_between_scheduled_and_appointment'] = df['AppointmentDay'] - pd.to_datetime(df['ScheduledDay'].dt.date)
df['difference_between_scheduled_and_appointment'] = (
    df['difference_between_scheduled_and_appointment'].apply(lambda x: int(str(x).split('days')[0].strip()))
)
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,1,1)
ax1.set_xlabel('Difference between scheduled and appointment days')
ax1.set_ylabel('Frequency')
df['difference_between_scheduled_and_appointment'].hist(grid=False, ax=ax1)

ax2 = fig.add_subplot(2,1,2)
df['difference_between_scheduled_and_appointment'].plot(kind='box', ax=ax2)
# Create separate features for scheduled hour and date, appointment date

df['scheduled_hour'] = df['ScheduledDay'].apply(lambda x: x.hour)
df['scheduled_date'] = df['ScheduledDay'].apply(lambda x: x.day)
df['appointment_date'] = df['AppointmentDay'].apply(lambda x: x.day)
# Convert text columns to numeric

df['Gender'].replace({'F': 0, 'M': 1}, inplace=True)
df['No-show'].replace({'No': 0, 'Yes': 1}, inplace=True)
# Create dummy variables for Handcap variable

handicap_df = pd.get_dummies(df['Handcap'], prefix='handicap')
df = pd.concat([df, handicap_df], axis=1)
df.drop(['Handcap', 'handicap_4'], axis=1, inplace=True)
# Define features and target variable for the model
features = [
    'Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received',
    'difference_between_scheduled_and_appointment', 'scheduled_hour', 'scheduled_date',
    'appointment_date', 'handicap_0', 'handicap_1', 'handicap_2', 'handicap_3'
]
target_variable = 'No-show'
# Scale the data using min max scaling

scaler = MinMaxScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df[features]))
scaled_df.columns = features
scaled_df['No-show'] = df['No-show']
# Create testing and training data
X_train, X_test, y_train, y_test = train_test_split(scaled_df[features], scaled_df[target_variable], test_size=0.2)
# Try different models

results = {}
models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('lr', LogisticRegression()),
    ('lda', LinearDiscriminantAnalysis())
]

for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = (model, accuracy)
sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
for model_name, (model, accuracy) in sorted_results:
    print(model_name, accuracy)
# Check the important features of random forest
print(sorted(zip(results['rf'][0].feature_importances_, features), reverse=True))
# Check the important features of logistic regression
print(sorted(zip(abs(results['lr'][0].coef_[0]), features), reverse=True))
model = LogisticRegression()
cross_val_scores = cross_val_score(model, scaled_df[features], scaled_df[target_variable], cv=5)
print(cross_val_scores.mean())
