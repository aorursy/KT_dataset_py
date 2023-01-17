#Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# Load the Dataset
MAIN_PATH = '../input/'
full_data = pd.read_csv(MAIN_PATH + 'bank-additional-full.csv', sep=";")
full_data.info()
full_data.describe()
full_data.head()
print(full_data["job"].value_counts())
print("*"*25)
print(full_data["marital"].value_counts())
print("*"*25)
print(full_data["education"].value_counts())
print(full_data["y"].value_counts())
plt.figure(figsize=(8,6))
Y = full_data["y"]
total = len(Y)*1.
ax= sns.countplot(x="y", data=full_data)
for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x() + 0.1, p.get_height()+5))

#put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
ax.yaxis.set_ticks(np.linspace(0, total, 11))
#adjust the ticklabel to the desired format, without changing the position of the ticks.
ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
# ax.legend(labels=["no","yes"])
plt.show()
# Making function for count plot drawing
def countplot(label, dataset):
  plt.figure(figsize=(15,10))
  Y = full_data[label]
  total = len(Y)*1.
  ax=sns.countplot(x=label, data=dataset)
  for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

  #put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
  ax.yaxis.set_ticks(np.linspace(0, total, 11))
  #adjust the ticklabel to the desired format, without changing the position of the ticks.
  ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
  ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
  # ax.legend(labels=["no","yes"])
  plt.show()
%matplotlib inline
# Making function for count plot with Y drawing
def countplot_withY(label, dataset):
  plt.figure(figsize=(20,10))
  Y = full_data[label]
  total = len(Y)*1.
  ax=sns.countplot(x=label, data=dataset, hue="y")
  for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))

  #put 11 ticks (therefore 10 steps), from 0 to the total number of rows in the dataframe
  ax.yaxis.set_ticks(np.linspace(0, total, 11))
  #adjust the ticklabel to the desired format, without changing the position of the ticks.
  ax.set_yticklabels(map('{:.1f}%'.format, 100*ax.yaxis.get_majorticklocs()/total))
  ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
  # ax.legend(labels=["no","yes"])
  plt.show()
countplot("job", full_data)
countplot_withY("job", full_data)
countplot("marital", full_data)
countplot_withY("marital", full_data)
countplot("default", full_data)
countplot_withY("default", full_data)
countplot("education",full_data)
countplot_withY("education", full_data)
countplot("housing", full_data)
countplot_withY("housing", full_data)
countplot("loan", full_data)
countplot_withY("loan", full_data)
countplot("contact", full_data)
countplot_withY("contact", full_data)
countplot("month", full_data)
countplot_withY("month", full_data)
countplot("day_of_week", full_data)
countplot_withY("day_of_week", full_data)
countplot("poutcome", full_data)
countplot_withY("poutcome", full_data)
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="age")
plt.show()
plt.figure(figsize=(10,8))
sns.distplot(full_data["age"])
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="duration")
plt.show()
plt.figure(figsize=(10,8))
sns.distplot(full_data["duration"])
plt.show()
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="campaign")
plt.show()
%matplotlib inline
plt.figure(figsize=(10,8))
sns.distplot(full_data["campaign"])
plt.show()
full_data["pdays"].unique()
full_data["pdays"].value_counts()
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="pdays")
plt.show()
full_data["previous"].unique()
full_data["previous"].value_counts()
full_data[full_data["y"]=="yes"]["previous"].value_counts()
full_data[full_data["y"]=="no"]["previous"].value_counts()
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="previous")
plt.show()
countplot("previous", full_data)
countplot_withY("previous", full_data)
full_data["emp.var.rate"].value_counts()
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="emp.var.rate")
plt.show()
%matplotlib inline
plt.figure(figsize=(10,8))
sns.distplot(full_data["emp.var.rate"])
plt.show()
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="cons.price.idx")
plt.show()
%matplotlib inline
plt.figure(figsize=(10,8))
sns.distplot(full_data["cons.price.idx"])
plt.show()
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="cons.conf.idx")
plt.show()
%matplotlib inline
plt.figure(figsize=(10,8))
sns.distplot(full_data["cons.conf.idx"])
plt.show()
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="euribor3m")
plt.show()
%matplotlib inline
sns.boxplot(data=full_data, x="y", y="nr.employed")
plt.show()
%matplotlib inline
plt.figure(figsize=(10,8))
sns.distplot(full_data["nr.employed"])
plt.show()

# Idea of correlation matrix of numerical feature: https://medium.com/datadriveninvestor/introduction-to-exploratory-data-analysis-682eb64063ff
%matplotlib inline
corr = full_data.corr()

f, ax = plt.subplots(figsize=(10,12))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

_ = sns.heatmap(corr, cmap="YlGn", square=True, ax=ax, annot=True, linewidth=0.1)

plt.title("Pearson correlation of Features", y=1.05, size=15)
# Import the librariesimport os
import pandas as pd
import matplotlib
matplotlib.use(u'nbAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.manifold import TSNE
from sklearn import preprocessing
import pandas as pd

from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the Dataset
MAIN_PATH = '../input/'
data = pd.read_csv(MAIN_PATH + 'bank-additional-full.csv', sep=";")
data.info()
# Check for duplicated data
data_dup = data[data.duplicated(keep="last")]
data_dup
data_dup.shape
# Remove the 12 duplicated rows
data = data.drop_duplicates()
data.shape
# Seperate independent and target variables from one another
data_x = data.iloc[:, :-1]
print("Shape of X:", data_x.shape)
data_y = data["y"]
print("Shape of Y:", data_y.shape)
# Train test split
from sklearn.model_selection import train_test_split

X_rest, X_test, y_rest, y_test = train_test_split(data_x, data_y, test_size=0.2)
X_train, X_cv, y_train, y_cv = train_test_split(X_rest, y_rest, test_size=0.2)

print("X Train:", X_train.shape)
print("X CV:", X_cv.shape)
print("X Test:", X_test.shape)
print("Y Train:", y_train.shape)
print("Y CV:", y_cv.shape)
print("Y Test:", y_test.shape)
# Replace yes with 1's and nos with 0's

y_train.replace({"no":0, "yes":1}, inplace=True)
y_cv.replace({"no":0, "yes":1}, inplace=True)
y_test.replace({"no":0, "yes":1}, inplace=True)
# Categorical boolean mask
categorical_feature_mask = data_x.dtypes==object

# filter categorical columns using mask and turn it into a list
categorical_cols = data_x.columns[categorical_feature_mask].tolist()
categorical_cols
from sklearn.feature_extraction.text import CountVectorizer

def add_onehot_to_dataframe(sparse, df, vectorizer, name):
  '''
      This function will add the one hot encoded to the dataframe.

  '''
  for i, col in enumerate(vectorizer.get_feature_names()):
    colname = name+"_"+col
    # df[colname] = pd.SparseSeries(sparse[:, i].toarray().flatten(), fill_value=0)
    df[colname] = sparse[:, i].toarray().ravel().tolist()
  
  return df

def OneHotEncoder(categorical_cols, X_train, X_test, X_cv=None, include_cv=False):
  '''
    This function takes categorical column names as inputs. The objective
    of this function is to take the column names iteratively and encode the 
    features using One hot Encoding mechanism and also adding the encoded feature
    to the respective dataframe.

    The include_cv parameter indicates whether we should include CV dataset or not.
    This is added specifically because when using GridSearchCV or RandomizedSearchCV,
    we only split the dataset into train and test to give more data to training purposes.
    This is done because GridSearchCV splits the data internally anyway.
  '''

  for i in categorical_cols:
    Vectorizer = CountVectorizer(token_pattern="[A-Za-z0-9-.]+")
    print("Encoding for feature: ", i)
    # Encoding training dataset 
    temp_cols = Vectorizer.fit_transform(X_train[i])
    X_train = add_onehot_to_dataframe(temp_cols, X_train, Vectorizer, i)

    # Encoding Cross validation dataset
    if include_cv:
      temp_cols = Vectorizer.transform(X_cv[i])
      X_cv = add_onehot_to_dataframe(temp_cols, X_cv, Vectorizer, i)

    # Encoding Test dataset
    temp_cols = Vectorizer.transform(X_test[i])
    X_test = add_onehot_to_dataframe(temp_cols, X_test, Vectorizer, i)
OneHotEncoder(categorical_cols, X_train, X_test, X_cv, True)

# Drop the categorical features as the one hot encoded representation is present
X_train = X_train.drop(categorical_cols, axis=1)
X_cv = X_cv.drop(categorical_cols, axis=1)
X_test = X_test.drop(categorical_cols, axis=1)

print("Shape of train: ", X_train.shape)
print("Shape of CV: ", X_cv.shape)
print("Shape of test: ", X_test.shape)
X_train.info()
data_x.to_csv("encoded_data_x.csv")
data_y.to_csv("data_y.csv")
%matplotlib inline

# T-SNE plot for train dataset
model = TSNE(n_components=2, random_state=0, perplexity=30)
tsne_data = model.fit_transform(X_train) 
plt.figure(figsize=(8,8))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=y_train.values)
plt.show()
%matplotlib inline

# T-SNE plot for CV dataset
model = TSNE(n_components=2, random_state=0, perplexity=30)
tsne_data = model.fit_transform(X_cv) 
plt.figure(figsize=(8,8))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=y_cv.values)
plt.show()
%matplotlib inline

# T-SNE plot for test dataset
model = TSNE(n_components=2, random_state=0, perplexity=30)
tsne_data = model.fit_transform(X_test) 
plt.figure(figsize=(8,8))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=y_test.values)
plt.show()
