import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
df = pd.read_csv("../input/creditcard.csv")
df.head()
df['Class'].value_counts()
y = df['Class']
y = np.array(y).astype(np.float)
X = df.drop(['Class'], axis=1)
X = np.array(X).astype(np.float)
natural_acc = (1 - y.sum()/len(y)) * 100
print('Anything with an accuracy below %.4f would be useless' % natural_acc)
def plot_data(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5, linewidth=0.15)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5, linewidth=0.15, c='r')
    plt.legend()
    return plt.show()
plot_data(X, y)
method = SMOTE(kind='regular')
X_resampled, y_resampled = method.fit_sample(X, y)
plot_data(X_resampled, y_resampled)
new_ratio = (1 - y_resampled.sum()/len(y_resampled)) * 100
print(new_ratio)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

model = RandomForestClassifier(random_state=5, 
                               class_weight='balanced_subsample', criterion= 'entropy') 
# weights are calculated with each iteration of growing a tree in the forest

model.fit(X_train, y_train) # the resampled data are used for training only, not for testing
predicted = model.predict(X) 
print(accuracy_score(y, predicted) * 100)
print(confusion_matrix(y, predicted))
probabilities = model.predict_proba(X)

print(roc_auc_score(y, probabilities[:,1]) * 100)
print(classification_report(y, predicted))
