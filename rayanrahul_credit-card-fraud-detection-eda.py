import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
#Loading the required dataset
df = pd.read_csv('../input/creditcard.csv', encoding ='ISO-8859-1')
df.head()
df.shape
#Performing a basic NULL Check
df.isnull().sum()
print(df.Class.unique())
sns.barplot(df.Class.value_counts().index, df.Class.value_counts())

sns.relplot('Time','Amount', data = df, hue = 'Class')
plt.title('Scatter plot showing transactions by time across various classes')
plt.show()
round(sum(df.Class == 1)/sum(df.Class == 0) * 100, 2)
print('The total of the amount of transactions done in the dataframe is :',round(df.Amount.sum(),2))
df.Amount.describe()
bins = range(0,174000,1000)
df['binned'] = pd.cut(df.Time,bins)
bins
df.groupby('binned').mean()
plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
plt.xlabel('Time interval')
plt.title('The distribution of amount transaction over various time frame')
sns.barplot(df.groupby('binned').sum().index,df.groupby('binned').sum().Amount)
plt.show()
sns.barplot(df.groupby('Class').sum().index,df.groupby('Class').sum().Amount)
plt.figure()
plt.title('Amount total as time passes by for Fraudulent transactions')
plt.plot(df[df.Class == 1].Time,np.cumsum(df[df.Class == 1].Amount), color = 'Blue')
plt.figure()
plt.title('Amount total as time passes by for Non-Fraudulent transactions')
plt.plot(df[df.Class == 0].Time,np.cumsum(df[df.Class == 0].Amount), color = 'Red')
df.drop('binned', axis = 1, inplace = True)
sns.distplot(df.Amount.values, color='r')
plt.title('The amount values ditribution')
plt.show()
sns.boxplot(df.Class,df.Amount)
df.Amount.describe(percentiles = [0,0.20,.25,.50,.75,.99])
len(df)
from scipy import stats
df[stats.zscore(df.Amount) < 5].head()
sns.boxplot(df.Amount)
len(df)
df.Amount.describe(percentiles = [0,0.20,.25,.50,.75,.99])
df.Amount -= df.Amount.min()
df.Amount /= df.Amount.max()
sns.distplot(df.Amount.values, color='r')
plt.title('The amount values ditribution')
plt.show()
#Checking if any 1s were lost in the analysis
df.Class.unique()
sns.distplot(df.Time, color='b')
plt.title('The time values ditribution')
plt.show()
sns.boxplot(df.Time)
df.Time -= df.Time.min()
df.Time /= df.Time.max()
sns.boxplot(df.Class,df.Time)
df_rem = df.drop(['Amount','Time','Class'], axis = 1)
df_rem.skew()
df_rem.kurtosis()
df_rem.head()
mod_cols = df_rem.skew()[np.abs(df_rem.skew()) < 1]
mod_cols
for col in mod_cols.index:
    print('Kurtosis '+col,df_rem[col].kurtosis())
from scipy.stats import norm
for col in mod_cols.index:
    print('The skew of '+col+' is:',df_rem[col].skew())
    plt.figure()
    sns.distplot(df_rem[col], fit = norm, kde=False, hist = False)
    plt.show()
cols1 = mod_cols.index
cols1
cols = df_rem.columns
cols


skew_cols = cols.difference(cols1)
skew_cols
from scipy.stats import kurtosis, skew
from scipy.stats import norm

for col in skew_cols:
    print('The skew of '+col+' is:',df_rem[col].skew())
    plt.figure()
    sns.distplot(df_rem[col], fit = norm, kde=False, hist = False)
    plt.show()

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(copy = False)
skewd = pd.DataFrame(pt.fit_transform(df[skew_cols]))
skewd.columns = skew_cols
df[skew_cols] = skewd
from scipy.stats import kurtosis, skew
from scipy.stats import norm

plt.figure(figsize=(12,6))
skew_cols = list(skew_cols)

for col in skew_cols:
    plt.title('Skew after the skew treatment')
    print('The skew of '+col+' is:',df[col].skew())
    plt.subplot(5,4,skew_cols.index(col)+1)    
    sns.distplot(df[col], fit = norm, kde=False, hist = False)
plt.tight_layout()
df.head()
len(df)
df[skew_cols]
df[skew_cols].describe([0.95,0.99])
len(df)
from scipy.stats import kurtosis, skew
from scipy.stats import norm

for col in skew_cols:
    print('The skew of '+col+' is:',df[col].skew())
    plt.figure()
    sns.boxplot(df[col])
    plt.show()
df[(stats.zscore(df_rem[skew_cols].drop('V1', axis = 1)) < 3).all(axis=1)]

from scipy.stats import kurtosis, skew
from scipy.stats import norm

for col in skew_cols:
    print('The skew of '+col+' is:',df_rem[col].skew())
    plt.figure()
    sns.distplot(df_rem[col], fit = norm, kde=False, hist = False)
    #print('The kurtosis of '+col+' is:',df_rem[col].kurtosis())

df.Class[df.Class == 1].sum()
for col in df_rem.columns:
    print(df_rem[col].describe(percentiles = [0.25,.5,.75,.99]))
import nltk
import sklearn

print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from imblearn.over_sampling import ADASYN 
sm = ADASYN()
X,y = sm.fit_sample(df.drop('Class', axis = 1), df.Class)
X.head()

df_adj = pd.DataFrame(X)
df_adj = pd.concat([df_adj,y], axis = 1)
sum(df_adj.Class.astype(int) == 1)/sum(df_adj.Class.astype(int) == 0)
sns.relplot('Time','Amount', data = df_adj, hue = 'Class')
plt.title('Scatter plot showing transactions by time across various classes after ADASYN')
plt.show()
sns.barplot(df_adj.Class.value_counts().index, df_adj.Class.value_counts())
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 33)
X,y = sm.fit_sample(df.drop('Class', axis = 1), df.Class)
X.head()
df_smote = pd.DataFrame(X)

sum(y.isnull())
df_smote = pd.concat([df_smote,y], axis = 1)
sum(df_smote.Class.astype(int) == 1)/sum(df_smote.Class.astype(int) == 0)
sns.relplot('Time','Amount', data = df_smote, hue = 'Class')
plt.title('Scatter plot showing transactions by time across various classes after SMOTE')
plt.show()
sns.barplot(df_smote.Class.value_counts().index, df_smote.Class.value_counts())
df_adj
df_adj.drop('Class', axis = 1).skew() 
len(df_adj)
#importing required libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#creating a function for evaluation of the model
from sklearn.metrics import confusion_matrix
def CalcMetrics(y_test,y_pred):
    confusion2 = confusion_matrix(y_test,y_pred)
    TP = confusion2[1,1] # true positive 
    TN = confusion2[0,0] # true negatives
    FP = confusion2[0,1] # false positives
    FN = confusion2[1,0] # false negatives
    # Let's see the sensitivity of our logistic regression model
    print('Sensitivity',TP / float(TP+FN))
    #The ability to predict the True positives from the false positives is very low
    print('Precision',TP/ float(TP+FP))
    print(classification_report(y_test, y_pred))
#test train split
train,test = train_test_split(df_adj, test_size = 0.3)
X_train = train.drop('Class', axis = 1 )
y_train = train.Class
X_test = test.drop('Class', axis = 1 )
y_test = test.Class
X_train.shape
y_train.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.feature_selection import RFE
rfe = RFE(logreg, 20)             # running RFE with 20 variables as output
rfe = rfe.fit(X_train,y_train)
print(rfe.support_)           # Printing the boolean results
print(rfe.ranking_)           # Printing the ranking
#selected features
X_train.columns[rfe.support_]
col = X_train.columns[rfe.support_]
# runningthe model using the selected variables
from sklearn import metrics
logsk = LogisticRegression(C=1e9)
logsk.fit(X_train[col], y_train)
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
logreg1 = sm.GLM(y_train,(sm.add_constant(X_train[col])), family = sm.families.Binomial())
model = logreg1.fit()
model.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#prediction
X_test_rfe = pd.DataFrame(data = X_test).iloc[:, rfe.support_]
y_pred = logsk.predict(X_test_rfe)
#evaluation
CalcMetrics(y_test,y_pred)
print(confusion_matrix(y_test, y_pred))
X_train.shape
y_test.shape
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
CalcMetrics(y_test,y_pred)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(min_samples_leaf=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
CalcMetrics(y_test,y_pred)
# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier
# Importing classification report and confusion matrix from sklearn metrics
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

# GridSearchCV to find optimal n_estimators
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#running random forest with default parameters 
rfc = RandomForestClassifier()
# Fit
rfc.fit(X_train, y_train)
# Making predictions
y_pred = rfc.predict(X_test)
#evaluation
CalcMetrics(y_test,y_pred)
print(confusion_matrix(y_test, y_pred))
# Plot Important Features predicted by model.
feat_importances = pd.Series(rfc.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')

# Importing gradient boosting classifier from sklearn library
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
# Fit
gbc.fit(X_train, y_train)
# Predict
predictions_gbc = gbc.predict(X_test)
CalcMetrics(y_test, predictions_gbc)
#confusion matrix for random forest
print(confusion_matrix(y_test, y_pred))
confusion2 = confusion_matrix(y_test, y_pred)