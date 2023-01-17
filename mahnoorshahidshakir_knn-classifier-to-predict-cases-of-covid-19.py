# for performing mathematical operations

import numpy as np 



# for data processing, CSV file I/O 

import pandas as pd 



# for plotting and visualozing data

import matplotlib.pyplot as plt 

import seaborn as sns
# read the data from the excel file into a dataframe

dataset = pd.read_excel('../input/covid19/dataset.xlsx', index_col=0)
# checking first thirty rows of our dataset

pd.set_option("display.max_rows",500)

pd.set_option("display.max_columns",500)

dataset.head(30)
# checking the shape of my dataset

dataset.shape
# extracting information from the dataset for the predictor and target variables

dataset.info()
# Understanding the fields of the dataset on the basis of statistical variables

dataset.describe(include="all").T
# finding out the number of positive and negative SARS-Cov-2 Cases

print("\nNumber of Positive and Negative Cases of SARS-COV-2")

dataset['SARS-Cov-2 exam result'].value_counts()
# dropping the selected columns 

dataset.drop(columns=['Patient addmited to regular ward (1=yes, 0=no)',

                      'Patient addmited to semi-intensive unit (1=yes, 0=no)',

                      'Patient addmited to intensive care unit (1=yes, 0=no)'], inplace=True)
# looking for null values

total_null_values = dataset.isnull().sum().sort_values(ascending=False) 

not_null_values = dataset.notnull().sum().sort_values(ascending=False) 

null_values_percentage = (dataset.isnull().sum()/dataset.notnull().count().sort_values(ascending=False)) * 100



# concating the calculated values with the data frame of null values

dataset_missing_values = pd.concat({'Null': total_null_values, 'Not Null': not_null_values, 'Percentage': null_values_percentage}, axis=1)



# view the newly formed dataframe

dataset_missing_values
sns.set(style="whitegrid")



# initialize the matplotlib figure

fig, axs = plt.subplots(figsize=(20,8))



# plot the Total Missing Values

sns.set_color_codes("muted")

sns.barplot(x=dataset_missing_values.index, y="Percentage", data=dataset_missing_values, color="g")



# customizing Bar Graph

plt.xticks(rotation='90')

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Missing Data in our Dataset', fontsize=20)
# finding those columns that completely doesn't have any values

dataset_missing_values[dataset_missing_values['Percentage']==100]
# finding those columns that doesn't have any values more than 5 values

dataset_missing_values[dataset_missing_values['Not Null'] <= 6]
# dropping the selected columns 

dataset.drop(columns=['Mycoplasma pneumoniae','Urine - Sugar','Prothrombin time (PT), Activity','D-Dimer','Fio2 (venous blood gas analysis)','Urine - Nitrite','Vitamin B12'], inplace=True)
# replace NaNs by 0

dataset = dataset.fillna(0)
dataset.replace('not_detected', 0, inplace=True)

dataset.replace('detected', 0, inplace=True)

dataset.replace('absent', 0, inplace=True)

dataset.replace('present', 1, inplace=True)

dataset.replace('negative', 0, inplace=True)

dataset.replace('positive', 1, inplace=True)
# Our Dataset

dataset
# visualize the relationship between the features and the target using scatterplots

fig, axs = plt.subplots(3, 3, figsize=(20, 20))

dataset.plot(kind='scatter', x='SARS-Cov-2 exam result', y='Hemoglobin',ax=axs[0,0], c='red')

dataset.plot(kind='scatter',  x='SARS-Cov-2 exam result', y='Hematocrit', ax=axs[0,1], c='green')

dataset.plot(kind='scatter', x='SARS-Cov-2 exam result', y='Platelets',ax=axs[0,2], c='blue')

dataset.plot(kind='scatter',  x='SARS-Cov-2 exam result', y='Eosinophils', ax=axs[1,0], c='orange')

dataset.plot(kind='scatter', x='SARS-Cov-2 exam result', y='Red blood Cells',ax=axs[1,1], c='purple')

dataset.plot(kind='scatter',  x='SARS-Cov-2 exam result', y='Lymphocytes', ax=axs[1,2], c='pink')

dataset.plot(kind='scatter', x='SARS-Cov-2 exam result', y='Leukocytes',ax=axs[2,0], c='blue')

dataset.plot(kind='scatter',  x='SARS-Cov-2 exam result', y='Basophils', ax=axs[2,1], c='red')

dataset.plot(kind='scatter',  x='SARS-Cov-2 exam result', y='Monocytes', ax=axs[2,2], c='green')
# visualize the relationship between the data points using heatmap

corr_matrix = abs(dataset.corr())



# correlation with target variable

corr_target = corr_matrix["SARS-Cov-2 exam result"]



# selecting highly correlated features

relevant_features = ["Platelets","Leukocytes","Eosinophils","Monocytes","Hemoglobin","Segmented","ctO2 (arterial blood gas analysis)","pCO2 (arterial blood gas analysis)","HCO3 (venous blood gas analysis)"]



# plotting the heatmap

fig, axs = plt.subplots(figsize=(18, 10))

sns.heatmap(abs(dataset[relevant_features].corr()), yticklabels=relevant_features, xticklabels=relevant_features, vmin = 0.0, square=True, annot=True, vmax=1.0, cmap='OrRd')
# visualize positive cases vs negative cases

dataset_negative = dataset['SARS-Cov-2 exam result'] == 0

dataset_positive = dataset['SARS-Cov-2 exam result'] == 1



# data to plot

labels = 'Positive Cases', 'Negative Cases'

sizes = [dataset_positive.sum(), dataset_negative.sum()]

colors = ['lightcoral', 'lightskyblue']

# explode 1st slice

explode = (0.1, 0) 



fig, axs = plt.subplots(figsize=(14, 7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.show()
# import the KNeighborsClassifier module

from sklearn.neighbors import KNeighborsClassifier
# instantiating KNeighborsClassifier

knn = KNeighborsClassifier()
# for splitting data into training and testing data

from sklearn.model_selection import train_test_split
# defining target variables 

target = dataset['SARS-Cov-2 exam result']



# defining predictor variables 

features = dataset.select_dtypes(exclude=[object])



# assigning the splitting of data into respective variables

X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify = target)
print("Number of samples in train set: %d" % y_train.shape)

print("Number of positive samples in train set: %d" % (y_train == 1).sum(axis=0))

print("Number of negative samples in train set: %d" % (y_train == 0).sum(axis=0))

print()

print("Number of samples in test set: %d" % y_test.shape)

print("Number of positive samples in test set: %d" % (y_test == 1).sum(axis=0))

print("Number of negative samples in test set: %d" % (y_test == 0).sum(axis=0))
# to display the HTML representation of an object.

from IPython.display import display_html



X_train_data = X_train.describe().style.set_table_attributes("style='display:inline'").set_caption('Summary of Training Data')

X_test_data = X_test.describe().style.set_table_attributes("style='display:inline'").set_caption('Summary of Testing Data')



# to display the summary of both training and testing data, side by side for comparison 

display_html(X_train_data._repr_html_(), raw = True)

display_html(X_test_data._repr_html_(), raw = True)
# for exhaustive search over specified parameter values for an estimator

from sklearn.model_selection import GridSearchCV
# assigning the dictionary of variables whose optimium value is to be retrieved

param_grid = {'n_neighbors' : np.arange(1,50)}
# performing Grid Search CV on knn-model, using 5-cross folds for validation of each criteria

knn_cv = GridSearchCV(knn, param_grid, cv=5)
# training the model with the training data and best parameter

knn_cv.fit(X_train,y_train)
# finding out the best parameter chosen to train the model

print("The best paramter we have is: {}" .format(knn_cv.best_params_))



# finding out the best score the chosen parameter achieved

print("The best score we have achieved is: {}" .format(knn_cv.best_score_))
# predicting the values using the testing data set

y_pred = knn_cv.predict(X_test)
# the score() method allows us to calculate the mean accuracy for the test data

print("The score accuracy for training data is: {}" .format(knn_cv.score(X_train,y_train)))

print("The score accuracy for testing data is: {}" .format(knn_cv.score(X_test,y_test)))
# for performance metrics

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
# call the classification_report and print the report

print(classification_report(y_test, y_pred))
# call the confusion_matrix and print the matrix

print(confusion_matrix(y_test, y_pred))