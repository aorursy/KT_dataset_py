# To import all the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
from sklearn import model_selection
from sklearn.cross_validation import train_test_split 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
import warnings

# To ignore or hide all the warnings
warnings.filterwarnings('ignore');
# To load the training dataset into a pandas dataFrame
emp_train_df = pd.read_csv('../input/train_LZdllcl.csv')
emp_test_df = pd.read_csv('../input/test_2umaH9m.csv')

# To add is_test_set column in training dataset to identify the type of dataset
emp_train_df['is_test_set']=0

# To add is_train_test column in test dataset
emp_test_df['is_test_set']=1

# To add is_promoted column in test dataset
emp_test_df['is_promoted']=np.nan
# To see columns of train dataset
emp_train_df.columns
# To change sequence of columns to match as that from test dataset column sequence
emp_train_df = emp_train_df[['employee_id', 'department', 'region', 'education', 'gender',
       'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating',
       'length_of_service', 'KPIs_met >80%', 'awards_won?',
       'avg_training_score', 'is_test_set', 'is_promoted']]
# Let's have a look at first few records of train dataset
emp_train_df.head()
# To see columns of test dataset
emp_test_df.columns
# Let's have a look at first few records of test dataset
emp_test_df.head()
# Combining train and test data using pandas
emp_df = emp_train_df.append(emp_test_df)
emp_df.columns
# High level statistics of the dataset 

# To check the number of datapoints and number of attributes or features available in the training dataset
print(emp_train_df.shape) 
# High level statistics of the dataset 

# To check the number of datapoints and number of attributes or features available in the test dataset
print(emp_test_df.shape) 
# High level statistics of the dataset 

# To check the number of datapoints and number of attributes or features available in the merged dataset
print(emp_df.shape) 
# To see column names in the dataset
print(emp_df.columns)

# To see first few data points in the dataset
emp_df.head()
# check the data frame info
emp_df.info()
# Let's have look at the different column values
emp_df.describe().transpose()
# To check duplicate employee_id values if any
emp_df[emp_df.duplicated(['employee_id'], keep=False)] #No duplicate employee_id found
# To check distinct values for different attributes
print("Department:\n{0}\n\nRegion:\n{1}\n\nEducation:\n{2}\n\nGender:\n{3}\n\nRecruitment_channel:\n{4}\n\nno_of_trainings:\n{5}\n\nAge:\n{6}\n\nprevious_year_rating:\n{7}\n\nlength_of_service:\n{8}\n\nKPIs_met >80%:\n{9}\n\nawards_won?:\n{10}\n\navg_training_score:\n{11}\n\nis_promoted:\n{12}\n\n"\
      .format(emp_df["department"].unique(),sorted(emp_df["region"].unique()),emp_df["education"].unique(),emp_df["gender"].unique(),emp_df["recruitment_channel"].unique(),sorted(emp_df["no_of_trainings"].unique()),sorted(emp_df["age"].unique()),sorted(emp_df["previous_year_rating"].unique()),sorted(emp_df["length_of_service"].unique()),emp_df["KPIs_met >80%"].unique(),emp_df["awards_won?"].unique(),sorted(emp_df["avg_training_score"].unique()),emp_df["is_promoted"].unique())); 
# To check count of nan values in each column of trating dataset

#count_nan = len(emp_df) - emp_df.count()
#count_nan

emp_df.isnull().sum(axis = 0)
# To replace the nan values with zero for easy manupulation
emp_df['education']=emp_df['education'].fillna('Not_Known')

# To replace the previous_year_rating : nan values with 2 (taking average rating) for easy data manupulation
emp_df['previous_year_rating']=emp_df['previous_year_rating'].fillna(2)

emp_df.head()
# To check number of classes in is_promoted
emp_df["is_promoted"].value_counts()
# Let's classify the data based on is_promoted status
promoted = emp_df[emp_df["is_promoted"]==1];
not_promoted = emp_df[emp_df["is_promoted"]==0];
# To verify the above classified variables value looking into first few records
print("Employees eligible for promotion:")
promoted.head()
print("\n\nEmployees not eligible for promotion:")
not_promoted.head()
# To remove region column values prefiex as below
emp_df['region'] = emp_df['region'].str.replace('region_','')
#df['range'].str.replace(',','-')
emp_df.head()
# Creating dummy variables for categorical datatypes
emp_df_dummies = pd.get_dummies(emp_df, columns=['department','region','education','recruitment_channel'])
emp_df_dummies.head()
# To replace gender categorical variable value as 1 for 'm' and 0 for 'f'
gender_mapping = {'m': 1, 'f': 0}
emp_df_dummies['gender'] = emp_df['gender'].map(gender_mapping) 
emp_df_dummies.head()
#Probability Density Functions (PDF)
#Cumulative Distribution Function (CDF)

##Let's Plots for PDF and CDF of age for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["age"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["age"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "middle left")
plt.xlabel("Employee Age") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html
##Let's Plots for PDF and CDF of length_of_service for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["length_of_service"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["length_of_service"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "middle left")
plt.xlabel("Employee service (No. of yrs)") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html
##Let's Plots for PDF and CDF of no_of_trainings for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["no_of_trainings"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["no_of_trainings"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "middle left")
plt.xlabel("no_of_trainings") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html
##Let's Plots for PDF and CDF of KPIs_met >80% for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["KPIs_met >80%"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["KPIs_met >80%"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "upper left")
plt.xlabel("KPIs_met >80%") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html
##Let's Plots for PDF and CDF of previous_year_rating for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["previous_year_rating"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["previous_year_rating"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "upper left")
plt.xlabel("previous_year_rating") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html
##Let's Plots for PDF and CDF of awards_won for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["awards_won?"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["awards_won?"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "middle left")
plt.xlabel("awards_won?\n(0 - No, 1 - Yes)") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html
#Probability Density Functions (PDF)
#Cumulative Distribution Function (CDF)

##Let's Plots for PDF and CDF of avg_training_score for all employee.
#For employee who got promotion
counts, bin_edges = np.histogram(promoted["avg_training_score"], bins=10, density=True) #bin='auto'
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_promoted") 

#For employee who didn't get promotion
counts, bin_edges = np.histogram(not_promoted["avg_training_score"], bins=10, density=True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label="Age - PDF of employee_not_promoted")
plt.plot(bin_edges[1:],cdf, label="Age - CDF of employee_not_promoted")
plt.legend(loc = "upper left")
plt.xlabel("avg_training_score") 

plt.show()

#Ref: https://matplotlib.org/api/legend_api.html
## Box-plot for no_of_trainings
ax = sbn.boxplot(x="is_promoted", y="no_of_trainings", hue = "is_promoted", data=emp_df)  
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["Yes", "No"], loc = "upper center")
plt.xlabel("Promotion Status? (1=Yes, 0=No)") 
plt.ylabel("no_of_trainings") 
plt.show()
### A violin plot combines the benefits of Box-plot and PDF
#Let's have a look at employee age wise Violin Plot
sbn.violinplot(x="is_promoted", y="age", data=emp_df, width=0.9)
plt.xlabel("Employee Promotion Status? \n[1 = Yes, 0 = No]") 
plt.ylabel("Employee Age") 
plt.show()
#Let's have a look at employee previous_year_rating wise Violin Plot
sbn.violinplot(x="is_promoted", y="previous_year_rating", data=emp_df, width=0.9)
plt.xlabel("Employee Promotion Status? \n[1 = Yes, 0 = No]") 
plt.ylabel("previous_year_rating") 
plt.show()
#Let's have a look at employee avg_training_score wise Violin Plot
sbn.violinplot(x="is_promoted", y="avg_training_score", data=emp_df, width=0.9)
plt.xlabel("Employee Promotion Status? \n[1 = Yes, 0 = No]") 
plt.ylabel("avg_training_score") 
plt.show()
# Let's see promotion behaviour using 2D scatter plot
sbn.set_style("whitegrid");
sbn.FacetGrid(emp_df, hue="is_promoted", size=4)\
   .map(plt.scatter, "age", "length_of_service")\
   .add_legend();
plt.show(); 
# Compare b/w avg_training_score and KPIs_met >80%
sbn.set_style("whitegrid");
sbn.FacetGrid(emp_df, hue="is_promoted", size=4)\
   .map(plt.scatter, "KPIs_met >80%", "avg_training_score")\
   .add_legend();
plt.show(); 
# Compare b/w previous_year_rating and no_of_trainings
sbn.set_style("whitegrid");
sbn.FacetGrid(emp_df, hue="is_promoted", size=4)\
   .map(plt.scatter, "previous_year_rating", "no_of_trainings")\
   .add_legend();
plt.show();
# Let's see all the possible combination using pair plot
plt.close();
sbn.set_style("whitegrid"); #white, dark, whitegrid, darkgrid, ticks
sbn.pairplot(emp_df, hue="is_promoted", vars=['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', \
                                              'KPIs_met >80%', 'awards_won?', 'avg_training_score'], size=5);
plt.show();
#Employee Age wise Promotion status
sbn.FacetGrid(emp_df, hue="is_promoted", size=5)\
   .map(sbn.distplot, "age")\
   .add_legend();

plt.show();
#Employee previous_year_rating wise Promotion status
sbn.FacetGrid(emp_df, hue="is_promoted", size=5)\
   .map(sbn.distplot, "previous_year_rating")\
   .add_legend();

plt.show();
#Employee avg_training_score wise Promotion status
sbn.FacetGrid(emp_df, hue="is_promoted", size=5)\
   .map(sbn.distplot, "avg_training_score")\
   .add_legend();

plt.show();
# To change the data type of previous_year_rating as int
emp_df_dummies[['previous_year_rating']] = emp_df_dummies[['previous_year_rating']].astype(int) 
# To validate the data types of all the variables
emp_df_dummies.info()
#To see the total number of columns in final dataframe
len(emp_df_dummies.columns)
# To check the all column names
emp_df_dummies.columns
# To change the sequence of columns and store the data points into final_df

final_df = emp_df_dummies[['gender', 'no_of_trainings', 'age', \
       'previous_year_rating', 'length_of_service', 'KPIs_met >80%', \
       'awards_won?', 'avg_training_score', \
       'department_Analytics', 'department_Finance', 'department_HR', \
       'department_Legal', 'department_Operations', 'department_Procurement', \
       'department_R&D', 'department_Sales & Marketing', 'department_Technology', \
       'region_1', 'region_2', 'region_3', 'region_4', 'region_5', \
       'region_6', 'region_7', 'region_8', 'region_9', 'region_10', 'region_11', \
       'region_12', 'region_13', 'region_14', 'region_15', 'region_16', \
       'region_17', 'region_18', 'region_19', 'region_20', \
       'region_21', 'region_22', 'region_23', 'region_24', 'region_25', \
       'region_26', 'region_27', 'region_28', 'region_29', 'region_30', \
       'region_31', 'region_32', 'region_33', 'region_34', \
       'education_Below Secondary', 'education_Bachelor\'s', \
       'education_Master\'s & above', 'education_Not_Known', \
       'recruitment_channel_referred', 'recruitment_channel_sourcing', \
       'recruitment_channel_other', 'is_test_set', 'is_promoted']];
final_df.columns
len(final_df.columns)
# Let's divide final_df into train and test dataset
train = final_df[final_df["is_test_set"] == 0]
test = final_df[final_df["is_test_set"] == 1]

# Remove is_test_set column from both train and test
del train['is_test_set']
del test['is_test_set']

# To check location of dependent variable/column
train.columns.get_loc("is_promoted")
# Assigning default value as Zero for now to test dataset for is_promoted column
test['is_promoted'] = 0.0

array = train.values
X_train = array[:,0:58] 
Y_train = array[:,58]

test_array = test.values
X_test = test_array[:,0:58] 
Y_test = test_array[:,58]
# To set test options and evaluation metric
seed = 7
scoring = 'accuracy'
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# To Compare Algorithms
# To create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. 

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# Assigning default value as Zero for now to test dataset is_promoted
test['is_promoted'] = 0.0

# Make predictions on test dataset
test_array = test.values
X_test = test_array[:,0:58]
Y_test = test_array[:,58] 

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_test)

print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
#To store the final predicted result for test dataset into sample_submission.csv

sample_submission_lda = emp_df_dummies.loc[(emp_df_dummies.is_test_set == 1), ['employee_id']]
sample_submission_lda['is_promoted'] = list(predictions)

sample_submission_lda.to_csv('sample_submission_lda.csv',index=False)
# To check the number of eligible employees in the predicted test dataset
sample_submission_lda[sample_submission_lda['is_promoted']==1].count()
