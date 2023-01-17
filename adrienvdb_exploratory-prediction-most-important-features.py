import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()
df.dtypes
df = df.set_index('Serial No.')
sns.lmplot(data=df, x='GRE Score', y='TOEFL Score', height=5)
corr_coeff, p_value = pearsonr(df['GRE Score'], df['TOEFL Score'])
print('There is a correlation coefficient of', corr_coeff, 'with p-value', p_value)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
sns.distplot(df['GRE Score'], bins=25, ax=ax1)
sns.distplot(df['TOEFL Score'], bins=25, ax=ax2)
ttest_ind(df['GRE Score']/340, df['TOEFL Score']/120, equal_var=False)
pd.concat([df['GRE Score']/340, df['TOEFL Score']/120],axis =1).plot.box(figsize=(14, 5))
plt.ylabel('Normalized score')
plt.show()
X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit '] > 0.72
sns.distplot(y.astype(int))
msk = np.random.rand(len(df)) < 0.8

X_train, y_train = X.iloc[msk], y.iloc[msk]
X_test, y_test = X.iloc[~msk], y.iloc[~msk]
clf = LogisticRegression(solver='lbfgs', C=10, max_iter=1000)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.predict(np.array([327, 110, 5, 3.5, 3.5, 9.1, 0]).reshape(1, -1))
def greedy_backward_selection(x_train, x_test, k, clf):
    """Perform Greedy Backward Selection algorithm
    in order to select the subset of top-k features
    """
    
    #Number of features to remove
    n_to_remove = x_train.columns.size - k
    
    #lists to store the best accuracies and best columns
    best_accs = []
    best_cols = []
    
    #Perform greedy backward selection
    for f in range(n_to_remove):
        worst_feature = ''
        best_acc = -1
        for column in x_train:
            #We try to remove every column
            x_train_new = x_train.drop(column, axis=1)
            x_test_new = x_test.drop(column, axis=1)
            
            clf.fit(x_train_new, y_train)
            #Evaluate the model without the feature
            a = cross_val_score(clf, x_train_new, y_train, cv=4).mean()
            
            #Keep it if it gives a better accuracy
            if a > best_acc:
                best_acc = a
                worst_feature = column
        
        #Drop the feature that is the "worst"
        x_train = x_train.drop(worst_feature, axis=1)
        x_test = x_test.drop(worst_feature, axis=1)
        
        print('Accuracy when keeping k=', x_train.columns.size, ' features: ', best_acc)
        
        #Store the accuracy and columns
        best_accs.append(best_acc)
        best_cols.append(x_train.columns)
        
    return best_accs[::-1], best_cols[::-1]
best_accs, best_cols = greedy_backward_selection(X_train, X_test, 1, LogisticRegression(solver='lbfgs', C=10, max_iter=1000))
plt.figure(figsize=(10, 5))
plt.plot(range(1, 7), best_accs)
plt.ylabel('Accuracy')
plt.xlabel('Number of features kept')
plt.show()
list(best_cols[np.argmax(best_accs)])
list(best_cols[0])
