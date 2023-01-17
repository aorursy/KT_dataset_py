import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt # this is for visualization
import seaborn as sns # for visualization
%matplotlib inline
from scipy.stats import zscore
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib.cbook import boxplot_stats
#Preparing Data Frame
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
vehicle_df = pd.read_csv('/kaggle/input/silhouette-of-vehicles-dataset/vehicle.csv')
vehicle_df.head(10)
# We can see total 846 Rows and 19 Columns in the DataFrame
vehicle_df.shape
# Checking datatypes of the predictors and target class
vehicle_df.dtypes
# Checking for null values and data types
# From the data we can see that there are some null values present in some columns
# So there are missing values in the columns indicated below
vehicle_df.info()
vehicle_df.isnull().sum()
numeric_cols = vehicle_df.drop('class', axis=1)
# Copy the 'class' column alone into the y dataframe. This is the dependent variable
vehicle_class = pd.DataFrame(vehicle_df[['class']])

numeric_cols = numeric_cols.apply(lambda x: x.fillna(x.median()),axis=0)
vehicle_df = numeric_cols.join(vehicle_class)   # Recreating vehicle_df by combining numerical columns with vehicle_class

vehicle_df.info()
vehicle_df.isnull().sum()
vehicle_df.head(10)
# 5 point summary of the input features
vehicle_df.describe()
# renaming helps removing unwanted dot characters in the column names
vehicle_df.rename(columns = {'pr.axis_aspect_ratio':'pr_axis_aspect_ratio', 'max.length_aspect_ratio':'max_length_aspect_ratio', 
                              'pr.axis_rectangularity':'pr_axis_rectangularity', 'max.length_rectangularity':'max_length_rectangularity', 'scaled_variance.1':'scaled_variance_1', 'scaled_radius_of_gyration.1':'scaled_radius_of_gyration_1', 'skewness_about.1':'skewness_about_1', 'skewness_about.2':'skewness_about_2'}, inplace = True)
# Getting the distribution of the target class variable
pd.value_counts(vehicle_df["class"]).plot(kind="bar")
result = pd.DataFrame()
result['class'] = vehicle_df.iloc[:,18]
vehicle_df1 = vehicle_df.drop('class', axis=1)
fig = plt.figure(figsize = (20, 25))
j = 0
for i in vehicle_df1.columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(vehicle_df1[i][result['class']=="car"], color='g', label = 'car')
    sns.distplot(vehicle_df1[i][result['class']=="bus"], color='r', label = 'bus')
    sns.distplot(vehicle_df1[i][result['class']=="van"], color='b', label = 'van')
    plt.legend(loc='best')
fig.suptitle('Vehicle Features')
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.show()
sns.pairplot(vehicle_df, hue="class")
plt.figure(figsize= (20,4))

plt.subplot(1, 4, 1)
sns.boxplot(vehicle_df['compactness'])
plt.title('compactness')

plt.subplot(1, 4, 2)
sns.boxplot(vehicle_df['circularity'])
plt.title('circularity')

plt.subplot(1, 4, 3)
sns.boxplot(vehicle_df['distance_circularity'])
plt.title('distance_circularity')

plt.subplot(1, 4, 4)
sns.boxplot(vehicle_df['radius_ratio'])
plt.title('radius_ratio')

print("compactness Skewness: %f" % vehicle_df['compactness'].skew())
out = boxplot_stats(vehicle_df.compactness).pop(0)['fliers']
print("No of Outliers for compactness = ", len(out))

print("circularity Skewness: %f" % vehicle_df['circularity'].skew())
out = boxplot_stats(vehicle_df.circularity).pop(0)['fliers']
print("No of Outliers for circularity = ", len(out))

print("distance_circularity Skewness: %f" % vehicle_df['distance_circularity'].skew())
out = boxplot_stats(vehicle_df.distance_circularity).pop(0)['fliers']
print("No of Outliers for distance_circularity = ", len(out))

print("radius_ratio Skewness: %f" % vehicle_df['radius_ratio'].skew())
out = boxplot_stats(vehicle_df.radius_ratio).pop(0)['fliers']
print("No of Outliers for radius_ratio = ", len(out))
plt.figure(figsize= (20,4))

plt.subplot(1, 4, 1)
sns.boxplot(vehicle_df['pr_axis_aspect_ratio'])
plt.title('pr_axis_aspect_ratio')

plt.subplot(1, 4, 2)
sns.boxplot(vehicle_df['max_length_aspect_ratio'])
plt.title('max_length_aspect_ratio')

plt.subplot(1, 4, 3)
sns.boxplot(vehicle_df['scatter_ratio'])
plt.title('scatter_ratio')

plt.subplot(1, 4, 4)
sns.boxplot(vehicle_df['elongatedness'])
plt.title('elongatedness')

print("pr_axis_aspect_ratio Skewness: %f" % vehicle_df['pr_axis_aspect_ratio'].skew())
out = boxplot_stats(vehicle_df.pr_axis_aspect_ratio).pop(0)['fliers']
print("No of Outliers for pr_axis_aspect_ratio = ", len(out))

print("max_length_aspect_ratio Skewness: %f" % vehicle_df['max_length_aspect_ratio'].skew())
out = boxplot_stats(vehicle_df.max_length_aspect_ratio).pop(0)['fliers']
print("No of Outliers for max_length_aspect_ratio = ", len(out))

print("scatter_ratio Skewness: %f" % vehicle_df['scatter_ratio'].skew())
out = boxplot_stats(vehicle_df.scatter_ratio).pop(0)['fliers']
print("No of Outliers for scatter_ratio = ", len(out))

print("elongatedness Skewness: %f" % vehicle_df['elongatedness'].skew())
out = boxplot_stats(vehicle_df.elongatedness).pop(0)['fliers']
print("No of Outliers for elongatedness = ", len(out))
plt.figure(figsize= (20,4))

plt.subplot(1, 4, 1)
sns.boxplot(vehicle_df['pr_axis_rectangularity'])
plt.title('pr_axis_rectangularity')

plt.subplot(1, 4, 2)
sns.boxplot(vehicle_df['max_length_rectangularity'])
plt.title('max_length_rectangularity')

plt.subplot(1, 4, 3)
sns.boxplot(vehicle_df['scaled_variance'])
plt.title('scaled_variance')

plt.subplot(1, 4, 4)
sns.boxplot(vehicle_df['scaled_variance_1'])
plt.title('scaled_variance_1')

print("pr_axis_rectangularity Skewness: %f" % vehicle_df['pr_axis_rectangularity'].skew())
out = boxplot_stats(vehicle_df.pr_axis_rectangularity).pop(0)['fliers']
print("No of Outliers for pr_axis_rectangularity = ", len(out))

print("max_length_rectangularity Skewness: %f" % vehicle_df['max_length_rectangularity'].skew())
out = boxplot_stats(vehicle_df.max_length_rectangularity).pop(0)['fliers']
print("No of Outliers for max_length_rectangularity = ", len(out))

print("scaled_variance Skewness: %f" % vehicle_df['scaled_variance'].skew())
out = boxplot_stats(vehicle_df.scaled_variance).pop(0)['fliers']
print("No of Outliers for scaled_variance = ", len(out))

print("scaled_variance_1 Skewness: %f" % vehicle_df['scaled_variance_1'].skew())
out = boxplot_stats(vehicle_df.scaled_variance_1).pop(0)['fliers']
print("No of Outliers for scaled_variance_1 = ", len(out))
plt.figure(figsize= (20,4))

plt.subplot(1, 4, 1)
sns.boxplot(vehicle_df['scaled_radius_of_gyration'])
plt.title('scaled_radius_of_gyration')

plt.subplot(1, 4, 2)
sns.boxplot(vehicle_df['scaled_radius_of_gyration_1'])
plt.title('scaled_radius_of_gyration_1')

plt.subplot(1, 4, 3)
sns.boxplot(vehicle_df['skewness_about'])
plt.title('skewness_about')

plt.subplot(1, 4, 4)
sns.boxplot(vehicle_df['skewness_about_1'])
plt.title('skewness_about_1')

print("scaled_radius_of_gyration Skewness: %f" % vehicle_df['scaled_radius_of_gyration'].skew())
out = boxplot_stats(vehicle_df.scaled_radius_of_gyration).pop(0)['fliers']
print("No of Outliers for scaled_radius_of_gyration = ", len(out))

print("scaled_radius_of_gyration_1 Skewness: %f" % vehicle_df['scaled_radius_of_gyration_1'].skew())
out = boxplot_stats(vehicle_df.scaled_radius_of_gyration_1).pop(0)['fliers']
print("No of Outliers for scaled_radius_of_gyration_1 = ", len(out))

print("skewness_about Skewness: %f" % vehicle_df['skewness_about'].skew())
out = boxplot_stats(vehicle_df.skewness_about).pop(0)['fliers']
print("No of Outliers for skewness_about = ", len(out))

print("skewness_about_1 Skewness: %f" % vehicle_df['skewness_about_1'].skew())
out = boxplot_stats(vehicle_df.skewness_about_1).pop(0)['fliers']
print("No of Outliers for skewness_about_1 = ", len(out))
plt.figure(figsize= (20,4))

plt.subplot(1, 2, 1)
sns.boxplot(vehicle_df['skewness_about_2'])
plt.title('skewness_about_2')

plt.subplot(1, 2, 2)
sns.boxplot(vehicle_df['hollows_ratio'])
plt.title('hollows_ratio')

print("skewness_about_2 Skewness: %f" % vehicle_df['skewness_about_2'].skew())
out = boxplot_stats(vehicle_df.skewness_about_2).pop(0)['fliers']
print("No of Outliers for skewness_about_2 = ", len(out))

print("hollows_ratio Skewness: %f" % vehicle_df['hollows_ratio'].skew())
out = boxplot_stats(vehicle_df.hollows_ratio).pop(0)['fliers']
print("No of Outliers for hollows_ratio = ", len(out))
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(vehicle_df.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=0.5, linecolor='black', ax=ax)
corr = vehicle_df.corr()
df = vehicle_df.drop('class', axis=1)
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if (corr.iloc[i,j] >= 0.9 or corr.iloc[i,j] <= -0.9):
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df1 = df[selected_columns]
df1
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(df1.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm', linewidths=0.5, linecolor='black', ax=ax)
vehicle_attr_11=df1.iloc[:,:11]
vehicle_attr_11.head()
vehicleDataScaled_11=vehicle_attr_11.apply(zscore)
vehicleDataScaled_11.head(10)
vehicle_attr_18=vehicle_df.iloc[:,:18]
vehicleDataScaled_18=vehicle_attr_18.apply(zscore)
vehicleDataScaled_18.head(10)
replaceStruct = {
                "class":     {"car": 0, "bus": 1, "van": 2},
                }
vehicle_df1 = vehicle_df.replace(replaceStruct)
# Creating scores dataframe to store respective model scores for assessment
df_scores = pd.DataFrame({"accuracy_score":[0,0,0,0,0,0,0,0]})
data = {'accuracy_score':[0,0,0,0,0,0,0,0]}
df_scores = pd.DataFrame(data, index=['SVM_11', 'SVM_18', 'SVM_Cross_Validation_11', 'SVM_Cross_Validation_18', 'SVM_PCA_18', 'SVM_PCA_7', 'SVM_Cross_Validation_PCA_18','SVM_Cross_Validation_PCA_7'])
X_11=np.array(vehicleDataScaled_11)
Y_11=vehicle_df1['class']
# Creating train and test data on 11 dimensions as remaining 7 dimensions were reduced due to multi-colinearity
X_train_11, X_test_11, y_train_11, y_test_11 = train_test_split(X_11, Y_11, test_size=0.30, random_state = 1)
X_train_11.shape
X_18=np.array(vehicleDataScaled_18)
Y_18=vehicle_df1['class']
# Creating train and test data on entire 18 dimensions
X_train_18, X_test_18, y_train_18, y_test_18 = train_test_split(X_18, Y_18, test_size=0.30, random_state = 1)
X_train_18.shape
svc_model_11 = SVC (C = 3, kernel='rbf', gamma = 1)
svc_model_11.fit(X_train_11, y_train_11)
y_pred_11 = svc_model_11.predict(X_test_11)
print("Accuracy score before PCA  = ", accuracy_score(y_test_11, y_pred_11))
df_scores.loc['SVM_11', 'accuracy_score'] = accuracy_score(y_test_11, y_pred_11)
df_scores
svc_model_18 = SVC (C = 3, kernel='rbf', gamma = 1)
svc_model_18.fit(X_train_18, y_train_18)
y_pred_18 = svc_model_18.predict(X_test_18)
print("Accuracy score before PCA  = ", accuracy_score(y_test_18, y_pred_18))
df_scores.loc['SVM_18', 'accuracy_score'] = accuracy_score(y_test_18, y_pred_18)
df_scores
num_folds = 50
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = svc_model_11
results = cross_val_score(model, X_11, Y_11, cv=kfold)
print(results)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
df_scores.loc['SVM_Cross_Validation_11', 'accuracy_score'] = results.mean()
df_scores
model = svc_model_18
results = cross_val_score(model, X_18, Y_18, cv=kfold)
print(results)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
df_scores.loc['SVM_Cross_Validation_18', 'accuracy_score'] = results.mean()
df_scores
covMatrix = np.cov(vehicleDataScaled_18,rowvar=False)
print(covMatrix)
pca = PCA(n_components=18)
pca.fit(vehicleDataScaled_18)
print(pca.explained_variance_)
print(pca.components_)
print(pca.explained_variance_ratio_)
plt.figure(figsize= (20,4))

plt.bar(list(range(1,19)),pca.explained_variance_ratio_,alpha=0.5, align='center')
plt.ylabel('Variation explained')
plt.xlabel('eigen Value')
plt.show()
plt.step(list(range(1,19)),np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('eigen Value')
plt.show()
np.sum(pca.explained_variance_ratio_[:7])
pca2 = PCA(n_components=18)
X_train_pca_18 = pca2.fit_transform(X_train_18)
X_test_pca_18 = pca2.transform(X_test_18)
## Fitting SVC model on PCA transformed train data
svc_model_18.fit(X_train_pca_18, y_train_18)
y_pred_pca_18 = svc_model_18.predict(X_test_pca_18)
print("Accuracy score  = ", accuracy_score(y_test_18, y_pred_pca_18))
df_scores.loc['SVM_PCA_18', 'accuracy_score'] = accuracy_score(y_test_18, y_pred_pca_18)
df_scores
pca3 = PCA(n_components=7)
X_train_pca_7 = pca3.fit_transform(X_train_18)
X_test_pca_7 = pca3.transform(X_test_18)
## Fitting SVC model on PCA transformed train data
svc_model_18.fit(X_train_pca_7, y_train_18)
y_pred_pca_7 = svc_model_18.predict(X_test_pca_7)
print("Accuracy score  = ", accuracy_score(y_test_18, y_pred_pca_7))
df_scores.loc['SVM_PCA_7', 'accuracy_score'] = accuracy_score(y_test_18, y_pred_pca_7)
df_scores
sns.pairplot(pd.DataFrame(X_train_pca_7), diag_kind='kde')
X_pca = pca2.fit_transform(X_18)
model = svc_model_18
results = cross_val_score(model, X_pca, Y_18, cv=kfold)
print(results)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
df_scores.loc['SVM_Cross_Validation_PCA_18', 'accuracy_score'] = results.mean()
df_scores
X_pca = pca3.fit_transform(X_18)
model = svc_model_18
results = cross_val_score(model, X_pca, Y_18, cv=kfold)
print(results)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
df_scores.loc['SVM_Cross_Validation_PCA_7', 'accuracy_score'] = results.mean()
df_scores
s = df_scores.loc[['SVM_11','SVM_18','SVM_Cross_Validation_11','SVM_Cross_Validation_18','SVM_PCA_18','SVM_PCA_7','SVM_Cross_Validation_PCA_18','SVM_Cross_Validation_PCA_7']]

plt.figure(figsize= (15,6))

plt.subplot(1, 1, 1)
s['accuracy_score'].plot.bar()
plt.title('Accuracy Scores')

s
