import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import model_selection, neighbors
df = pd.read_csv('../input/forest-cover-type-kernels-only/train.csv')
fig, ax = plt.subplots(figsize=(10, 9))
sns.stripplot('Cover_Type', 'Id', data=df, jitter=True, ax=ax, size=2)
X = np.array(df['Id']).reshape(-1, 1)
y = df['Cover_Type']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Grid search to determine best k
accuracies=[]
for k in range(1, 31):
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, p=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    accuracies.append(accuracy)
    
best_k = np.argmax(accuracies) + 1 # best k is consistently 1
print('Highest-performing k: {} (acc: {})'.format(best_k, max(accuracies)))
df = pd.read_csv('../input/forest-cover-type-eda-baseline-model/etc.csv')
fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True, figsize=(18, 9))
ax1.set_title('Unshuffled')
ax2.set_title('Shuffled (for comparison)')
sns.stripplot('Cover_Type', 'Id', data=df, jitter=True, ax=ax1, size=0.5)
np.random.shuffle(df['Id'])
sns.stripplot('Cover_Type', 'Id', data=df, jitter=True, ax=ax2, size=0.5)