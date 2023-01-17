import pandas as pd

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_test= pd.read_csv('../input/test.csv')

print(df.head(3))
X = df.iloc[:,2:202]

y = df.iloc[:,1]
sns.countplot(x='target',data=df, palette='hls')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2)
classifier = LogisticRegression(random_state=0,solver='lbfgs',max_iter=2000,tol=0.01)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
# calculate AUC

auc = roc_auc_score(y_test, y_pred)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')

# show the plot

plt.show()
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
df_x_test = df_test.drop(columns = ['ID_code'])

y_pred = classifier.predict(df_x_test)

df_y = pd.DataFrame(y_pred)

df_submission = pd.merge(pd.DataFrame(df_test['ID_code']),df_y,left_index=True,right_index=True)

#df_submission.to_csv('../input/submit.csv', encoding='utf-8', index=False)

print(df_submission)