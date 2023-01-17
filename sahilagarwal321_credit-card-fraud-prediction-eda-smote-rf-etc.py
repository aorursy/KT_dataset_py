import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import gridspec





from imblearn.pipeline import make_pipeline as make_pipeline_imb # To do our transformation in a unique time

from imblearn.over_sampling import SMOTE

from sklearn.pipeline import make_pipeline

from imblearn.metrics import classification_report_imbalanced



from sklearn.model_selection import train_test_split

from collections import Counter



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, accuracy_score



import warnings

warnings.filterwarnings("ignore")
#loading the data

data = pd.read_csv("../input/creditcard.csv")
data.sample(10)
data.info()
data.describe()
data[["Time","Amount","Class"]].describe()
# Lets start looking the difference by Normal and Fraud transactions

data["Class"].value_counts()
plt.figure(figsize=(7,5))

sns.countplot(data['Class'])

plt.title("Class Histogram", fontsize=18)

plt.xlabel("Class", fontsize=15)

plt.ylabel("Count", fontsize=15)

plt.show()
#To clearly the data of frauds and no frauds

df_fraud = data[data['Class'] == 1]

df_normal = data[data['Class'] == 0]



print("Fraud transaction statistics")

print(df_fraud["Amount"].describe())

print("\nNormal transaction statistics")

print(df_normal["Amount"].describe())
#Feature engineering to a better visualization of the values

data['Amount_log'] = np.log(data.Amount + 0.01)
plt.figure(figsize=(14,6))

plt.subplot(121)

ax = sns.boxplot(x ="Class",y="Amount", data=data)

ax.set_title("Class x Amount", fontsize=20)

ax.set_xlabel("Is Fraud?", fontsize=16)

ax.set_ylabel("Amount(US)", fontsize = 16)



plt.subplot(122)

ax1 = sns.boxplot(x ="Class",y="Amount_log", data=data)

ax1.set_title("Class x Amount", fontsize=20)

ax1.set_xlabel("Is Fraud?", fontsize=16)

ax1.set_ylabel("Amount(Log)", fontsize = 16)



plt.subplots_adjust(hspace = 0.6, top = 0.8)



plt.show()
timedelta = pd.to_timedelta(data['Time'], unit='s')

data['Time_hour'] = (timedelta.dt.components.hours).astype(int)
plt.figure(figsize=(20,8))

sns.distplot(data[data['Class'] == 0]["Time_hour"], color='g')

sns.distplot(data[data['Class'] == 1]["Time_hour"],  color='r')

plt.title('Fraud x Normal Transactions by Hours', fontsize=17)

plt.xlim([-1,25])

plt.show()
ax = sns.lmplot(y="Amount", x="Time_hour", fit_reg=False,aspect=1.8, data=data, hue='Class')

plt.title("Amount x Hour of Fraud, Normal Transactions", fontsize=16)



plt.show()
#Looking the V's features

columns = data.iloc[:,1:29].columns



frauds = data.Class == 1

normals = data.Class == 0



grid = gridspec.GridSpec(10, 3)

plt.figure(figsize=(15,20*4))



for n, col in enumerate(data[columns]):

    ax = plt.subplot(grid[n])

    sns.distplot(data[col][frauds], bins = 50, color='g') #Will receive the "semi-salmon" violin

    sns.distplot(data[col][normals], bins = 50, color='r') #Will receive the "ocean" color

    ax.set_ylabel('Density')

    ax.set_title(str(col))

    ax.set_xlabel('')

plt.show()
data.head()
X = data.iloc[:, 0:30]

X.head()
y = data['Class']

y.head()
corr = X.corr().abs()

lower_right_ones = np.tril(np.ones(corr.shape, dtype='bool'), k=-1)

correlations = corr.where(lower_right_ones)

correlations
plt.figure(figsize=(30,30))

sns.heatmap(correlations, annot=True, cmap='RdBu_r', fmt= '.2f', vmax=1, vmin=-1)

plt.xticks(rotation=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, test_size=0.20)
print("Data transformation -")

print("Normal data distribution: {}".format(Counter(y)))

X_smote, y_smote = SMOTE().fit_sample(X, y)

print("SMOTE data distribution: {}".format(Counter(y_smote)))
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), RandomForestClassifier(random_state=42))
SMOTE_model = smote_pipeline.fit(X_train, y_train)

SMOTE_prediction = SMOTE_model.predict(X_test)
def print_results(headline, true_value, pred):

    print(headline)

    print("accuracy: {}".format(accuracy_score(true_value, pred)))

    print("precision: {}".format(precision_score(true_value, pred)))

    print("recall: {}".format(recall_score(true_value, pred)))

    print("f2: {}".format(fbeta_score(true_value, pred, beta=2)))

print("Confusion Matrix: ")

print(confusion_matrix(y_test, SMOTE_prediction))



print('\nSMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))



print_results("\nSMOTE + RandomForest classification", y_test, SMOTE_prediction)
# Compute predicted probabilities: y_pred_prob

y_pred_prob = smote_pipeline.predict_proba(X_test)[:,1]



# Generate precision recall curve values: precision, recall, thresholds

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)



# Plot ROC curve

plt.plot(precision, recall)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision Recall Curve')

plt.show()