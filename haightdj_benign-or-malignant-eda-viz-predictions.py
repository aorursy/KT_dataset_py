# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Data Handling:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Other:
import warnings
warnings.filterwarnings('ignore')

# Machine Learning:
from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# First load the CSV to a Pandas dataframe
data_DF = pd.read_csv('../input/data.csv')

# Next examine the Dataframe
print(data_DF.shape)
print(data_DF.keys())
print(data_DF.dtypes)
# view the top 5 rows:
data_DF.head()
# Look at some basic stats of all numeric data:
data_DF.describe()
data_DF.isnull().sum()
data_DF = data_DF.drop(labels = ['Unnamed: 32'], axis = 1)
sns.countplot(data_DF.diagnosis);
data_DF.diagnosis.value_counts()
vars = data_DF.keys().drop(['id','diagnosis'])
plot_cols = 5
plot_rows = math.ceil(len(vars)/plot_cols)

plt.figure(figsize = (5*plot_cols,5*plot_rows))

for idx, var in enumerate(vars):
    plt.subplot(plot_rows, plot_cols, idx+1)
    sns.boxplot(x = 'diagnosis', y = var, data = data_DF)
fig, (ax) = plt.subplots(1, 1, figsize=(20,10))

hm = sns.heatmap(data_DF.corr(), 
                 ax=ax, # Axes in which to draw the plot
                 cmap="coolwarm", # color-scheme
                 annot=True, 
                 fmt='.2f',       # formatting  to use when adding annotations.
                 linewidths=.05)

fig.suptitle('Breast Cancer Correlations Heatmap', 
              fontsize=14, 
              fontweight='bold');
vars_to_drop = ['id', 'radius_mean', 'perimeter_mean', 'radius_worst', 'area_worst', 'perimeter_worst', 'radius_se', 'perimeter_se',
               'concave points_mean', 'compactness_mean', 'compactness_worst', 'concavity_worst', 'concavity_mean', 'concavity_se',
               'texture_worst', 'smoothness_worst', 'texture_se']
g = sns.pairplot(data_DF.drop(vars_to_drop, axis = 1), hue='diagnosis', height=3)
g.map_lower(sns.kdeplot)
y = data_DF.diagnosis
X = data_DF.drop(vars_to_drop, axis = 1)
X = X.drop('diagnosis', axis = 1)
print(y.shape)
print(X.shape)
X.head()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size = 0.25)
XGBC_model = XGBClassifier(random_state=1, objective = 'multi:softprob', num_class=2)
XGBC_model.fit(train_X, train_y)

# make predictions
XGBC_predictions = XGBC_model.predict(val_X)

# Print Accuracy for initial RF model
XGBC_accuracy = accuracy_score(val_y, XGBC_predictions)
print("Accuracy score for XGBoost Classifier model : " + str(XGBC_accuracy))
# Create dataframe of feature name and importance 
feature_imp_DF = pd.DataFrame({'feature': X.keys().tolist(), 'importance': XGBC_model.feature_importances_})

# Print the sorted values form the dataframe:
print("Feature Importance:\n")
print(feature_imp_DF.sort_values(by=['importance'], ascending=False))
# Plot feature importance from the dataframe..
# feature_imp_DF.sort_values(by=['importance'],ascending=False).plot(kind='bar');
g = sns.barplot(x="feature", y="importance", data=feature_imp_DF.sort_values(by=['importance'],ascending=False))
g.set_xticklabels(feature_imp_DF.sort_values(by=['importance'],ascending=False)['feature'],rotation=90);

