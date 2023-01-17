%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix  
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv",sep=r'\s*,\s*',

                           header=0, encoding='ascii', engine='python')

df
df.plot.scatter(x='platelets', y='age', title='Platelets',color = "red")
df['ejection_fraction'].value_counts().sort_index().plot.bar()
import seaborn as sns

sns.distplot(df['serum_sodium'], bins=10, kde=True)
sns.boxplot( 'age','diabetes',data=df)
for col in df.columns:

    df[col].value_counts().plot.bar()

    plt.show()


df_dedupped = df.drop_duplicates()



# there were duplicate rows

print(df.shape)

print(df_dedupped.shape)
for col in df.columns:

    pct_missing = np.mean(df[col].isnull())

    print('{} - {}%'.format(col, round(pct_missing*100)))
for c in df.columns:

    print ("---- %s ---" % c)

    print (df[c].value_counts().sort_index())
df[['anaemia','diabetes','high_blood_pressure','sex','smoking']] = df[['anaemia','diabetes','high_blood_pressure','sex','smoking']].astype(bool)
quantile_list = [0, .25, .5, .75, 1.]

quantiles = df['age'].quantile(quantile_list)

quantiles
fig, ax = plt.subplots()

df['age'].hist(bins=30, color='#A9C5D3', 

                             edgecolor='black', grid=False)

for quantile in quantiles:

    qvl = plt.axvline(quantile, color='r')

    ax.legend([qvl], ['Quantiles'], fontsize=10)

    ax.set_title('Age', fontsize=12)

ax.set_xlabel('Age of patient', fontsize=12)

ax.set_ylabel('Frequency', fontsize=12)
quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

df['age_quantile_label'] = pd.qcut(df['age'], 

                                   q=quantile_list,       

                                   labels=quantile_labels)
encoder = LabelEncoder()

age_labels = encoder.fit_transform(df['age_quantile_label'])

age_mappings = {index: label for index, label in 

                  enumerate(encoder.classes_)}

age_mappings
df['age_quantile_label'] = age_labels

df
final_df = df[['anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'age_quantile_label', 'DEATH_EVENT']]

final_df
X=final_df.iloc[:,:-1]

y=final_df.iloc[:,-1]

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2)
sc_train = StandardScaler()

sc_train.fit_transform(X_train)

sc_test = StandardScaler()

sc_test.fit_transform(X_test)
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
accuracy_score(y_train, dt.predict(X_train)),accuracy_score(y_test, dt.predict(X_test))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
max_depths = np.linspace(1, 32, 32, endpoint=True)

train_results = []

test_results = []

for max_depth in max_depths:

   dt = DecisionTreeClassifier(max_depth=max_depth)

   dt.fit(X_train, y_train)

   train_pred = dt.predict(X_train)   

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   # Add auc score to previous train results

   train_results.append(roc_auc)   

   y_pred = dt.predict(X_test)

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   # Add auc score to previous test results

   test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, "b", label="Train AUC")

line2, = plt.plot(max_depths, test_results, "r", label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel("AUCscore")

plt.xlabel("Treedepth")

plt.show()
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []

test_results = []

for min_samples_split in min_samples_splits:

   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)

   dt.fit(X_train, y_train)   

   train_pred = dt.predict(X_train)

   false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   train_results.append(roc_auc)   

   y_pred = dt.predict(X_test)   

   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

   roc_auc = auc(false_positive_rate, true_positive_rate)

   test_results.append(roc_auc)



from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(min_samples_splits, train_results, "b", label="Train AUC")

line2, = plt.plot(min_samples_splits, test_results, "r", label="Test AUC")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel("AUC score")

plt.xlabel("min samples split")

plt.show()
classifier = DecisionTreeClassifier(criterion= 'entropy', max_depth=None,max_features= 6, min_samples_leaf= 3)

classifier.fit(X, y)
accuracy_score(y_train, classifier.predict(X_train)),accuracy_score(y_test, classifier.predict(X_test))
cf_matrix=confusion_matrix(y_test,classifier.predict(X_test))
sns.heatmap(cf_matrix, annot=True)