# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from scipy.stats import pearsonr, skew, boxcox, chi2
from scipy.stats.stats import pearsonr
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix,classification_report,mean_absolute_error
from sklearn.svm import SVC

from xgboost import XGBClassifier

from pprint import pprint

from imblearn.over_sampling import SMOTE
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
# Plot configurations
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'figure.figsize': (24,15)
})
def confusion_matrix(model,X,y):
  matrix = plot_confusion_matrix(model,X,y,normalize='true',cmap=plt.cm.Reds)
  matrix.ax_.set_title('Confusion Matrix')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.gcf().set_size_inches(15,8)
  plt.show()
df.head()
# Lets see if there is missing data first
is_missing = df.isnull()
sns.heatmap(is_missing)
df.info() 
# Lets see if the variable 'quality' has an even distribution of data
sns.countplot(df['quality'])
df['quality'].value_counts()
y = df['quality']
X = df.drop(['quality'],axis=1)
X_cols = X.columns


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.80,random_state=0) 
for i in range(11):
  plt.subplot(3,4,i+1)
  sns.boxplot(y=X[X.columns[i]],x=y)
plt.tight_layout()
# Since no data is missing, lets look at the summary of the dataset and see if we find something weird 
display(df.describe())
print(df.shape)
print(df.skew())
for i in range(11):
  plt.subplot(3,4,i+1)
  sns.distplot(df[df.columns[i]])
plt.tight_layout()
corrMatrix = df.corr(method='spearman') # spearman coefficient can be used on numerical and ordinal variables (quality is ordinal). Pearson only has meaning for numerical values. 
g = sns.clustermap(corrMatrix, cmap="coolwarm", linewidths=1,annot=True,vmin = -1.0 , vmax=1.0, cbar_kws={"ticks":[-1,-0.5,0,0.5,1]}) 

g.fig.set_figwidth(15)
g.fig.set_figheight(11)

# Clustermaps do hierarchical clustering and orders the rows and columns based on similarity, making it easier to see correlations
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)
pred = random_forest.predict(X_test)

cross_val = cross_val_score(random_forest,X_train,y_train,cv=2,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_train, y_train))
print('Test accuracy score:\n',random_forest.score(X_test,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test,y_test)
robust_scaler = RobustScaler(quantile_range=(25,100))
X_train_scaled = robust_scaler.fit_transform(X_train)
X_test_scaled = robust_scaler.transform(X_test)

X_train_scaled_aux = pd.DataFrame(X_train_scaled,columns=X_cols)
for i in range(11):
  plt.subplot(4,3,i+1)
  sns.boxplot(y=X_train_scaled_aux[X_train_scaled_aux.columns[i]],x=y_train)
plt.tight_layout()

# Observe the y axis values
for i in range(11):
  plt.subplot(4,3,i+1)
  sns.distplot(X_train_scaled_aux[X_train_scaled_aux.columns[i]])
plt.tight_layout()
display(X_train.skew())                 # before tranformation
display(X_train_scaled_aux.skew())      # after transformation
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train_scaled, y_train)
pred = random_forest.predict(X_test_scaled)

cross_val = cross_val_score(random_forest,X_train_scaled, y_train,cv=2,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_train_scaled, y_train))
print('Test accuracy score:\n',random_forest.score(X_test_scaled,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test_scaled,y_test)
Q1 = df.quantile(q=.25)
Q3 = df.quantile(q=.75)
IQR = df.loc[ : , df.columns != 'quality'].apply(stats.iqr)


#only keep rows in dataframe that have values within 2.5*IQR of Q1 and Q3
df_cleaned = df[~((df < (Q1-2.5*IQR)) | (df > (Q3+2.5*IQR))).any(axis=1)]

#find how many rows are left in the dataframe 
print(df_cleaned.shape)
print(df.shape)
y_cleaned = df_cleaned['quality']
X_cleaned = df_cleaned.drop(['quality'],axis=1)
X_cols = X.columns


X_train,X_test,y_train,y_test = train_test_split(X_cleaned,y_cleaned,train_size=0.80,random_state=0) 
for i in range(11):
  plt.subplot(4,3,i+1)
  sns.boxplot(y=X_train[X_train.columns[i]],x=y_train)
plt.tight_layout()
for i in range(11):
  plt.subplot(4,3,i+1)
  sns.distplot(X_train[X_train.columns[i]])
plt.tight_layout()
display(X_train.skew())  # Considerable reduced skew compared to using all the outliers
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)
pred = random_forest.predict(X_test)

cross_val = cross_val_score(random_forest,X_train, y_train,cv=2,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_train, y_train))
print('Test accuracy score:\n',random_forest.score(X_test,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test,y_test)
power = PowerTransformer(method='yeo-johnson', standardize=True)
X_train_power = power.fit_transform(X_train)
X_test_power = power.transform(X_test)

X_train_power_aux = pd.DataFrame(X_test_power,columns=X_cols)
for i in range(11):
  plt.subplot(4,3,i+1)
  sns.distplot(X_train_power_aux[X_train_power_aux.columns[i]],bins=10)
plt.tight_layout()
display(X_train.skew())                 # before tranformation
display(X_train_power_aux.skew())       # after transformation
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train_power, y_train)
pred = random_forest.predict(X_test_power)

cross_val = cross_val_score(random_forest,X_train_power, y_train,cv=2,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_train_power, y_train))
print('Test accuracy score:\n',random_forest.score(X_test_power,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test_power,y_test)
y_train.value_counts()
#sampling_strategy={3: 250,
#                   4: 300,
#                   8: 400,
#                   }
oversample = SMOTE(random_state=0,

                   k_neighbors=2
                   )
X_resampled,y_resampled = oversample.fit_resample(X_train_power,y_train)
sns.countplot(y_resampled)
pd.DataFrame(y_resampled).value_counts()
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_resampled, y_resampled)
pred = random_forest.predict(X_test_power)

cross_val = cross_val_score(random_forest,X_resampled, y_resampled,cv=5,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_resampled, y_resampled))
print('Test accuracy score:\n',random_forest.score(X_test_power,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test_power,y_test)
bins = (2, 6.5, 8) # Suggested by the main page of the problem
target_groups = ['Bad', 'Good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = target_groups)
label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])
df['quality'].value_counts()
sns.countplot(df['quality'])
# Plot configurations
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 20,
    'figure.figsize': (24,15)
})
y = df['quality']
X = df.drop(['quality'],axis=1)
X_cols = X.columns


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.80,random_state=0) 
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)
pred = random_forest.predict(X_test)

cross_val = cross_val_score(random_forest,X_train,y_train,cv=5,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_train, y_train))
print('Test accuracy score:\n',random_forest.score(X_test,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test,y_test)
Q1 = df.quantile(q=.25)
Q3 = df.quantile(q=.75)
IQR = df.loc[ : , df.columns != 'quality'].apply(stats.iqr)


#only keep rows in dataframe that have values within 2*IQR of Q1 and Q3
df_cleaned = df[~((df < (Q1-1.5*IQR)) | (df > (Q3+1.5*IQR))).any(axis=1)]

#find how many rows are left in the dataframe 
print(df_cleaned.shape)
print(df.shape)
y_cleaned = df_cleaned['quality']
X_cleaned = df_cleaned.drop(['quality'],axis=1)

X_train,X_test,y_train,y_test = train_test_split(X_cleaned,y_cleaned,train_size=0.80,random_state=24) 
for i in range(11):
  plt.subplot(4,3,i+1)
  sns.boxplot(y=X_train[X_train.columns[i]],x=y_train)
plt.tight_layout()
for i in range(11):
  plt.subplot(4,3,i+1)
  sns.distplot(X_train[X_train.columns[i]])
plt.tight_layout()
display(X_train.skew())  
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train, y_train)
pred = random_forest.predict(X_test)

cross_val = cross_val_score(random_forest,X_train, y_train,cv=5,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_train, y_train))
print('Test accuracy score:\n',random_forest.score(X_test,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test,y_test)
power = PowerTransformer(method='yeo-johnson', standardize=True)
X_train_power = power.fit_transform(X_train)
X_test_power = power.transform(X_test)

X_train_power_aux = pd.DataFrame(X_test_power,columns=X_cols)
for i in range(11):
  plt.subplot(4,3,i+1)
  sns.distplot(X_train_power_aux[X_train_power_aux.columns[i]],bins=10)
plt.tight_layout()
display(X_train.skew())                 # before tranformation
display(X_train_power_aux.skew())       # after transformation
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_train_power, y_train)
pred = random_forest.predict(X_test_power)

cross_val = cross_val_score(random_forest,X_train_power, y_train,cv=5,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_train_power, y_train))
print('Test accuracy score:\n',random_forest.score(X_test_power,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test_power,y_test)
y_train.value_counts()
sampling_strategy={1: 800
                   }
oversample = SMOTE(random_state=0,sampling_strategy=sampling_strategy,k_neighbors=5)
X_resampled,y_resampled = oversample.fit_resample(X_train_power,y_train)
sns.countplot(y_resampled)
pd.DataFrame(y_resampled).value_counts()
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)
random_forest.fit(X_resampled, y_resampled)
pred = random_forest.predict(X_test_power)

cross_val = cross_val_score(random_forest,X_resampled, y_resampled,cv=5,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_resampled, y_resampled))
print('Test accuracy score:\n',random_forest.score(X_test_power,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test_power,y_test)
print('Parameters currently in use:\n')
pprint(random_forest.get_params())
random_grid = {"max_depth": [None],
              "max_features": [3, 5, 10],
              "min_samples_split": [2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "bootstrap": [False, True],
              "n_estimators" :[100,500],
              "criterion": ["gini"]}


pprint(random_grid)
%%time 
search_param_forest = RandomizedSearchCV(random_forest,random_grid,cv=5,verbose=3,n_jobs=-1,n_iter=5)
search_param_forest.fit(X_resampled,y_resampled)
search_param_forest.best_score_
search_param_forest.best_params_
search_param_forest.best_estimator_
random_forest = RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features=3,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=500,
                       n_jobs=None, oob_score=False, random_state=0, verbose=0,
                       warm_start=False)

random_forest.fit(X_resampled, y_resampled)
pred = random_forest.predict(X_test_power)

cross_val = cross_val_score(random_forest,X_resampled, y_resampled,cv=5,scoring="accuracy")

print('Test MAE :\n', mean_absolute_error(y_test,pred))
print('Train accuracy score:\n',random_forest.score(X_resampled, y_resampled))
print('Test accuracy score:\n',random_forest.score(X_test_power,y_test))
print('Cross validation accuracy:\n',cross_val.mean())
print(classification_report(y_test,pred))
confusion_matrix(random_forest,X_test_power,y_test)
random_forest.feature_importances_
#The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.
feat_importances = pd.Series(random_forest.feature_importances_,index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()
