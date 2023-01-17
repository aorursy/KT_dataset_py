import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
bpi_challenge_2017_data = pd.read_csv('../input/bpichallenge2017/BPI_Challenge_2017.csv') 
bpi_challenge_2017_data.head()
bpi_challenge_2017_data.describe(include='all')
bpi_challenge_2017_data.info()
bpi_challenge_2017_data.isnull().sum().sort_values(ascending=False)
bpi_challenge_2017_data[bpi_challenge_2017_data['EventOrigin'] == 'Offer'].isnull().sum().sort_values(ascending=False)
bpi_challenge_2017_data[bpi_challenge_2017_data['EventOrigin'] == 'Workflow'].isnull().sum().sort_values(ascending=False)
bpi_challenge_2017_data['Accepted'].unique()
bpi_challenge_2017_data['Accepted'].value_counts()
plt.figure(figsize=[10,8])
accepted = bpi_challenge_2017_data[bpi_challenge_2017_data['Accepted'] == True]['Selected']
ax = sns.countplot(accepted)
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(accepted.dropna())*100), 
                 (p.get_x()+0.3, p.get_height()+1), fontsize=14))
ax.set_title("Accepted Distribution")
ax.set(xlabel="Accepted", ylabel="Frequence")
plt.show()
plt.figure(figsize=[10,8])
ax = sns.distplot(bpi_challenge_2017_data['case:RequestedAmount'])
ax.set_title("Requested Amount Distribution")
ax.set(xLabel="Requested Amount", yLabel="Density")
plt.show()
plt.figure(figsize=[10,8])
ax = sns.distplot(bpi_challenge_2017_data['OfferedAmount'])
ax.set_title("Offered Amount Distribution")
ax.set(xLabel="Offered Amount", yLabel="Density")
plt.show()
offered_less = (bpi_challenge_2017_data[(bpi_challenge_2017_data['OfferedAmount']) 
                                        < (bpi_challenge_2017_data['case:RequestedAmount'])])
offered_more = (bpi_challenge_2017_data[(bpi_challenge_2017_data['OfferedAmount']) 
                                        > (bpi_challenge_2017_data['case:RequestedAmount'])])
offered_eq = (bpi_challenge_2017_data[(bpi_challenge_2017_data['OfferedAmount']) 
                                     == (bpi_challenge_2017_data['case:RequestedAmount'])])
less_acc = offered_less[offered_less['Accepted'] == True]
more_acc = offered_more[offered_more['Accepted'] == True]
equal_acc = offered_eq[offered_eq['Accepted'] == True]

plt.figure(figsize=[10,8])
ax = sns.countplot(less_acc['Selected'])
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(less_acc)), 
                 (p.get_x()+0.3, p.get_height()+1), fontsize=14))
ax.set_title("Accepted Distribution - (OfferedAmount < RequestedAmount)")
ax.set(xLabel="Accepted", yLabel='Frequence')

plt.figure(figsize=[10,8])
ax = sns.countplot(more_acc['Selected'])
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(more_acc)), 
                 (p.get_x()+0.3, p.get_height()+1),fontsize=14))
ax.set_title("Accepted Distribution - (OfferedAmount > RequestedAmount)")
ax.set(xLabel="Accepted", yLabel='Frequence')

plt.figure(figsize=[10,8])
ax = sns.countplot(equal_acc['Selected'])
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(equal_acc)), 
                 (p.get_x()+0.3, p.get_height()+1), fontsize=14))
ax.set_title("Accepted Distribution (OfferedAmount = RequestedAmount)")
ax.set(xLabel="Accepted", yLabel='Frequence')
plt.show()

plt.figure(figsize=[10,8])
loan_goals_occ = bpi_challenge_2017_data['case:LoanGoal'].value_counts()
ax = sns.barplot( x=loan_goals_occ.index, y=loan_goals_occ)
ax.set(ylabel="Frequence", xlabel = "Loan Goal")
ax.set_title("Loan Goal Distribution")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
offer_refused = (bpi_challenge_2017_data[(bpi_challenge_2017_data['Accepted'] == True) 
                                         & (bpi_challenge_2017_data['Selected'] == False)])
grouped_by_lg = (offer_refused.groupby(['case:LoanGoal'])['Selected'].count()
                 .sort_values(ascending=False))

plt.figure(figsize=[10,8])
ax = sns.barplot(y=grouped_by_lg.values, x=grouped_by_lg.index)
ax.set(ylabel="Offer Refused", xlabel = "Loan Goal")
ax.set_title("Offer Refused by Loan Goal")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
plt.figure(figsize=[10,8])
ax = sns.boxplot(data=bpi_challenge_2017_data, x='case:LoanGoal', y='case:RequestedAmount')
ax.set(xlabel="Loan Goal", ylabel = "Requested Amount")
ax.set_title("Loan Goal Variation (Requested Amount)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.show()
plt.figure(figsize=[10,8])
ax = sns.boxplot(data=bpi_challenge_2017_data, x='case:LoanGoal', y='OfferedAmount')
ax.set(xlabel="Loan Goal", ylabel = "Offered Amount")
ax.set_title("Loan Goal Variation (Offered Amount)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.show()
mean_per_loan_goal = (bpi_challenge_2017_data.groupby(['case:LoanGoal'])['case:RequestedAmount']
                      .mean().sort_values(ascending=False))
plt.figure(figsize=[10,8])
ax = sns.barplot(y=mean_per_loan_goal.values, x=mean_per_loan_goal.index)
ax.set(ylabel="Mean of Requested Amount", xlabel = "Loan Goal")
ax.set_title("Mean of Requested Amount by Loan Goal")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
mean_per_loan_goal = (bpi_challenge_2017_data.groupby(['case:LoanGoal'])['OfferedAmount']
                      .mean().sort_values(ascending=False))
plt.figure(figsize=[10,8])
ax = sns.barplot(y=mean_per_loan_goal.values, x=mean_per_loan_goal.index)
ax.set(ylabel="Mean Offered Amount", xlabel = "Loan Goal")
ax.set_title("Mean of Offered Amount by Loan Goal")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
offer_refused = (bpi_challenge_2017_data[(bpi_challenge_2017_data['Accepted'] == True) 
                                         & (bpi_challenge_2017_data['Selected'] == False) ])
grouped_by_lg = (offer_refused.groupby(['case:LoanGoal'])['Selected'].count()
                 .sort_values(ascending=False))

plt.figure(figsize=[10,8])
ax = sns.barplot(y=grouped_by_lg.values, x=grouped_by_lg.index)
ax.set(ylabel="Offer Refused", xlabel = "Loan Goal")
ax.set_title("Mean Offer Refused by Loan Goal")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
df_model = (bpi_challenge_2017_data[['concept:name', 'case:concept:name', 'EventOrigin',
                                     'case:RequestedAmount', 'CreditScore', 'OfferedAmount']])
df_model.head()
df_model = df_model[df_model['EventOrigin'] != 'Workflow']
df_model.head()
# Aplications ok
app_ok = (df_model[(df_model['concept:name'] == 'A_Pending')]['case:concept:name']
          .drop_duplicates())
df_1 = df_model.merge(app_ok, on='case:concept:name')
df_1['label'] = df_1.apply(lambda x: 1, axis=1)
df_1.head()
# Aplications not ok
app_not_ok = (df_model[(~df_model['case:concept:name'].isin(app_ok.tolist()))]['case:concept:name']
              .drop_duplicates())
df_2 = df_model.merge(app_not_ok, on='case:concept:name')
df_2['label'] = df_2.apply(lambda x: 0, axis=1)
df_2.head()
df_model = (pd.concat([df_1, df_2])[['case:concept:name', 'case:RequestedAmount', 
                                     'CreditScore', 'OfferedAmount', 'label']]
            .drop_duplicates())
df_model.sample(10)
df_model.isnull().sum().sort_values(ascending=False)
df_model = df_model.dropna(axis=0)
df_model.isnull().sum().sort_values(ascending=False)
plt.figure(figsize=[10,8])
labels_occ = df_model['label']
ax = sns.barplot(data=df_model, x=labels_occ.value_counts().index, y=labels_occ.value_counts(), orient='v')
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(labels_occ)*100), 
                 (p.get_x()+0.3, p.get_height()+1), fontsize=14))
ax.set(xlabel="Label", ylabel = "Frequence")
ax.set_title("Label Distribution")
plt.show()
X_train, X_test = train_test_split(df_model['case:concept:name'].drop_duplicates(), test_size=0.3)
X_train = df_model.merge(X_train, on="case:concept:name")
X_test = df_model.merge(X_test, on="case:concept:name")

y_test = X_test['label']
y_train = X_train['label']
X_train = X_train.drop(['case:concept:name', 'label'], axis=1)
X_test = X_test.drop(['case:concept:name', 'label'], axis=1)
classifier_dt = DecisionTreeClassifier()
classifier_dt = classifier_dt.fit(X_train,y_train)
y_pred = classifier_dt.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)

print("Accuracy: ", accuracy)
print("Recall: ", recall)
print("Precision: ", precision)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], [ 'Loan ok', 'Loan Not ok'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], [ 'Loan ok', 'Loan Not ok'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()

features = X_train.columns

features_importances = pd.Series(classifier_dt.feature_importances_, index=features)

plt.figure(figsize=[10,8])
ax = sns.barplot(x=features_importances.values, y=features_importances)
ax.set_title("Feature importances")
plt.xticks(list(range(len(features_importances))), features)
plt.show()
df_model_reg = (bpi_challenge_2017_data[['case:LoanGoal','case:RequestedAmount', 
                                         'CreditScore', 'OfferedAmount']])
df_model_reg.head()
df_model_reg.isnull().sum().sort_values(ascending=False)
df_model_reg = df_model_reg.dropna(axis=0)
df_model_reg.isnull().sum().sort_values(ascending=False)
df_model_reg.columns.str.strip()
x = df_model_reg[['case:LoanGoal', 'case:RequestedAmount', 'CreditScore']]
y = df_model_reg[['OfferedAmount']]
x = pd.get_dummies(x, columns=['case:LoanGoal'])
x.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
linear_regressor = LinearRegression()  
linear_regressor.fit(x_train, y_train)
y_pred = linear_regressor.predict(x_test)
mae = metrics.mean_absolute_error(y_test, y_pred)
mse =  metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("Mean Absolute Error (MAE): ", mae)
print("Mean Squared Error (MSE)", mse)
print("Root Mean Squared Error (RMSE): ", rmse)