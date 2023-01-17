import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import matplotlib.pyplot as plt
df = pd.read_csv('../input/creditcard.csv')
labels = df[list(df)[-1:]]
plt.figure(figsize=(3,3))

print(pd.Series(df['Class']).value_counts())

pd.Series(df['Class']).value_counts().plot(kind ='pie', autopct='%1.2f%%')
df_2 = df[list(df)[1:]]
df_2['Amount'].mean()
random_seed = 5

perce = 0.1
fraud_df = df_2.loc[df_2['Class'] == 1]

len(fraud_df)
non_fraud_df = df_2.loc[df_2['Class'] == 0]

len(non_fraud_df)
non_fraud_resampled =  non_fraud_df.sample(n=int(len(fraud_df)+(len(fraud_df)*perce)), random_state=random_seed)

len(non_fraud_resampled)

dataset = pd.concat([fraud_df,non_fraud_resampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)
plt.figure(figsize=(3,3))

pd.Series(dataset['Class']).value_counts().plot(kind ='pie', autopct='%1.2f%%')
X = dataset[dataset.columns[:-1]]

y = dataset[dataset.columns[-1]]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed, test_size=0.2)
n_train_frauds = pd.Series(y_train).value_counts().get_value(1)

n_test_frauds = pd.Series(y_test).value_counts().get_value(1)



plt.figure(figsize=(8,3))

plt.subplot(121)

pie_train = pd.Series(y_train).value_counts().plot(kind='bar', title = ('Train subset ballance\n fraud ratio: {0:.2f}'.format(n_train_frauds/len(y_train))))

plt.subplot(122)

pie_test = pd.Series(y_test).value_counts().plot(kind='bar', title = 'Test subset ballance\n fraud ratio: {0:.2f}'.format(n_test_frauds/len(y_test)))
# random forests

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 15,max_depth=5,

                            random_state=0).fit(X_train, y_train)



y_score_rf = rf.predict_proba(X_test)[:,-1]
from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve

average_precision = average_precision_score(y_test, y_score_rf)



print('Average precision-recall score RF: {}'.format(average_precision))
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt



precision, recall, _ = precision_recall_curve(y_test, y_score_rf)



plt.step(recall, precision, color='b', alpha=0.2,

         where='post')

plt.fill_between(recall, precision, step='post', alpha=0.2,

                 color='b')



plt.xlabel('Recall')

plt.ylabel('Precision')

plt.ylim([0.0, 1.05])

plt.xlim([0.0, 1.0])

plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(

          average_precision))
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_score_rf)

roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8,8))

plt.xlim([-0.01, 1.00])

plt.ylim([-0.01, 1.01])

plt.plot(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))





plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC curve', fontsize=16)

plt.legend(loc='lower right', fontsize=13)

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

plt.axes().set_aspect('equal')

plt.show()