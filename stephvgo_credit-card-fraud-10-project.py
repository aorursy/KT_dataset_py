import numpy as np #library that imports linear algebra functions
import pandas as pd #library used for data processing, particularly CSV file I/O
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
sns.set()
%matplotlib inline


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/ccfraud/creditcard.csv")
print(df.shape)
df.head()
df.describe()
df.isna().sum()
sns.boxplot(x=df['V1'])
def outliers_transform(base_dataset):
    for i in df.var().sort_values(ascending=False).index[0:10]:
        x=np.array(df[i])
        qr1=np.quantile(x,0.25)
        qr3=np.quantile(x,0.75)
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        y=[]
        #"""Based on clients input(ltv,utv) run the below code """
        for p in x:
            if p <ltv or p>utv:
                y.append(np.median(x))
            else:
                y.append(p)
        df[i]=y
        
outliers_transform(df)

sns.boxplot(x=df['V1'])
sns.boxplot(x=df['V2'])
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
class_0 = df.loc[df['Class'] == 0]["Time"] # not fraud
class_1 = df.loc[df['Class'] == 1]["Time"] # fraud
hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')
fraud = df.loc[df['Class'] == 1]
valid = df.loc[df['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Cases: {}'.format(len(valid)))
# correlation matrix
corrmat = df.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
# First, we get the columns from the dataframe.
columns = df.columns.tolist()

# Then we filter the columns to remove undesired data.
columns = [c for c in columns if c not in ['Class']]

# We store the variable we will be predicting.
target = 'Class'

# X includes everything except our class column.
X = df[columns]

# Y includes all the class labels for each sample, this is also one-dimensional.
Y = df[target]

# p¡Print the shapes of X and Y.
print(X.shape)
print(Y.shape)

# We define a random state as 1.
state = 1

# Then we define the outlier detection methods.
classifiers = {
    # Contamination is the number of outliers we think there are.
    'Isolation Forest': IsolationForest(max_samples = len(X),
                                       contamination = outlier_fraction,
                                       random_state = state),
    # Number of neighbors to consider, the higher the percentage of outliers the higher you want to make this number.
    'Local Outlier Factor': LocalOutlierFactor(
    n_neighbors = 20,
    contamination = outlier_fraction)
}
n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # We fit the data and tag the outliers.
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    # Reshape the prediction values to 0 for valid and 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # Calculate the number of errors
    n_errors = (y_pred != Y).sum()
    
    # Classification matrix
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))
