# Data manipulation
import pandas as pd
import numpy as np

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Finance related operations
from pandas_datareader import data

# Import this to silence a warning when converting data column of a dataframe on the fly
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

%matplotlib inline
# Load data
df = pd.read_csv('../input/df-out1/df_out.csv', index_col=0)

df.head()
df_2018 = pd.read_csv('../input/200-financial-indicators-of-us-stocks-20142018/2018_Financial_Data.csv', index_col=0)
df_2018.head()
df_2018 = df_2018.rename(columns={'Enterprise Value': 'Enterprise Value 2018'})
df_2018x = df_2018[df_2018['Enterprise Value 2018'].notna()]
df_2018x = df_2018x['Enterprise Value 2018']
df_1418 = pd.merge(df, df_2018x, right_index =True, left_index = True, how='inner')
df_1418.head()
df_2014 = pd.read_csv('../input/200-financial-indicators-of-us-stocks-20142018/2014_Financial_Data.csv', index_col=0)
EV_2014 = df_2014['Enterprise Value']
df_1418x = pd.merge(df_1418, EV_2014, right_index =True, left_index = True, how='inner')
df_1418x.head()
df_1418x['Growth Rate'] = df_1418x['Enterprise Value 2018'] / df_1418x['Enterprise Value']
df_1418x.head()
df_1418x = df_1418x.dropna()
##category = pd.cut(df_1418x['Growth Rate'],bins=[-np.inf,0,1.461,np.inf],labels=[-1,1,10])
##category = pd.cut(df_1418x['Growth Rate'],bins=[-np.inf,0,np.inf],labels=[-1,1])
category = pd.cut(df_1418x['Growth Rate'],bins=[-np.inf,1.461,np.inf],labels=['NG','G'])
df_1418x.insert(67,'Stock Performance',category)
df_1418x['Stock Performance'].value_counts(normalize=True)
features = ['Revenue', 'EPS', 'EBITDA Margin', 'returnOnEquity', 'Operating Income Growth', 'Sector', '2015 PRICE VAR [%]']
X = df_1418x[features]
y = df_1418x.loc[:, df_1418x.columns.intersection(['Stock Performance'])]
one_hot_X = pd.get_dummies(X)
one_hot_X.replace([np.inf, -np.inf], np.nan)
one_hot_X.fillna(0)
one_hot_X.head()
y = y.values.ravel()
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(one_hot_X, y, test_size=0.25, random_state=42)
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()