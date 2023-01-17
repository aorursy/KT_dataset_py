# mounting GDrive to our Colab Notebook
from google.colab import drive
drive.mount('/content/drive', force_remount = True)
# filtering out the warnings after cell execution
import warnings
warnings.filterwarnings('ignore')
# Installing the Dependencies
!pip install -r '/content/drive/My Drive/Internship/Day 6/HR Analytics/Checkpoints/requirements.txt'
# Importing Libraries

# General Commonly Used Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing Libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm_notebook

# EDA
import scipy.stats as stats 
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from pandas.plotting import scatter_matrix
from sklearn import base
from sklearn.model_selection import KFold

# Feature Engineering and Selection
from sklearn.feature_selection import SelectKBest, chi2, RFECV
from sklearn.preprocessing import KBinsDiscretizer
from feature_engine.discretisers import EqualFrequencyDiscretiser
from mlxtend.feature_selection import SequentialFeatureSelector as SFS, ExhaustiveFeatureSelector
from sklearn.decomposition import PCA, KernelPCA as kp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight

# Modeling & Accuracy Metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.ensemble import RandomForestClassifier as rfc, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgbm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Validation and Hyperparameter Tuning
from sklearn.model_selection import KFold, cross_val_score as cvs, RandomizedSearchCV, GridSearchCV

# Saving the Model - Pickling
import pickle

# Utility Libraries
from collections import Counter
# Reading the Train Dataset into a pandas dataframe
train_set = pd.read_csv('/content/drive/My Drive/Internship/Day 6/HR Analytics/Datasets/train_LZdllcl.csv', encoding = 'ISO-8859-1', error_bad_lines = False)
# Reading the Test Dataset into a pandas dataframe
test_set = pd.read_csv('/content/drive/My Drive/Internship/Day 6/HR Analytics/Datasets/test_2umaH9m.csv', encoding = 'ISO-8859-1', error_bad_lines = False)
# to find out the shape of the dataframes
train_set.shape, test_set.shape
# Shape of the Given Train Dataset
train_set.shape
# Spliting Train Dataset into Train and Train-Remain (For further Validation and Test Spliting)
train, train_remain = train_test_split(train_set, test_size = 0.3, random_state = 0, shuffle = True)
# Spliting Train_Remain into Validation and Test 
val, test = train_test_split(train_remain, test_size = 0.5, random_state = 0)
# Shapes of Train, Validation and Test
train.shape, val.shape, test.shape
# Figuring out the % of Split

print(f"Split % of Train Set : {round(train.shape[0]/train_set.shape[0] * 100)} %")
print(f"Split % of Validation Set : {round(val.shape[0]/train_set.shape[0] * 100)} %")
print(f"Split % of Test Set : {round(test.shape[0]/train_set.shape[0] * 100)} %")
# resetting the index of the newly formed dataframes
train.reset_index(inplace = True)
val.reset_index(inplace = True)
test.reset_index(inplace = True)
# dropping the extra redundant feature - 'index'
train.drop(['index'], axis = 1, inplace = True)
val.drop(['index'], axis = 1, inplace = True)
test.drop(['index'], axis = 1, inplace = True)
# displaying the first 10 records of the dataframe
train.head(10)
# displaying the last 10 records of the dataframe
train.tail(10)
# to know about the datatypes of each feature
train.info()
# to derive statistical distributions of the features
train.describe()
# count for unique elements per feature
unique_Counts = train.nunique(dropna = False) # dropna = False - makes nunique treat NaNs as a distinct value
unique_Counts.sort_values()
const_Features = unique_Counts.loc[unique_Counts == 1].index.tolist()
print(const_Features)
# To Find out Repeatative Features (Duplicate)
train_enc =  pd.DataFrame(index = train.index)

for col in tqdm_notebook(train.columns):
    train_enc[col] = train[col].factorize()[0]

dup_cols = {}

for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
            dup_cols[c2] = c1
dup_cols
# to see how many values are missing in each column.
train.isnull().sum()
# visualizing and observing the null elements in the dataset
nullPlot = sns.heatmap(train.isnull(), cbar = False, cmap = 'YlGnBu')   # ploting missing data && # cbar, cmap = colour bar, colour map
nullPlot.set_xticklabels(labels = train.columns, rotation = 30)
plt.gcf().set_size_inches(20, 5)
# to find out the percentage of missing values for each feature
train.isnull().mean()
# max of nans for a particular record
train.isnull().sum(axis = 1).sort_values()[-1:]
# percentage of null values in their respective feature
percentage_Of_Null_Education = train.education.isnull().mean()
percentage_Of_Null_Prev_Year_Rating = train.previous_year_rating.isnull().mean()

print(f"% of 'Education' Records having 'NULL' values : {round(percentage_Of_Null_Education * 100, 2)} %")
print(f"% of 'Previous Year Rating' Records having 'NULL' values : {round(percentage_Of_Null_Prev_Year_Rating * 100, 2)} %")
# finding out insights about Education
print(f"No. of Categories of Education : {train.education.nunique()} \nThe Slabs are : {list(train.education.unique())}")
# Imputing NULL Values of 'Education' as 'Others'
edu_impute = pd.DataFrame(train["education"].fillna("Others"))
edu_impute.education.unique()
# Before Imputation
train.education.unique(), val.education.unique(), test.education.unique()
# imputing the same in all the dataframes as it's independent of the central limit tendency 
train["education"].fillna("Others", inplace = True)
val["education"].fillna("Others", inplace = True)
test["education"].fillna("Others", inplace = True)
# After Imputation
train.education.unique(), val.education.unique(), test.education.unique()
# deleting redundant dataframe
del edu_impute
# finding out insights about previous year's rating
print(f"No. of categories of employees' previous years' rating : {train.previous_year_rating.nunique()} \nThe different ratings are : {list(train.previous_year_rating.unique())}")
# Finding out the Mean, Median, Mode of Previous Year Ratings
train.previous_year_rating.mean(), train.previous_year_rating.median(), train.previous_year_rating.mode().iloc[0]
# trying out different imputations to find out best fit
prev_Year_Ratings_impu = pd.DataFrame()
prev_Year_Ratings_impu['previous_year_rating'+'_mean'] = train['previous_year_rating'].fillna(train.previous_year_rating.mean())
prev_Year_Ratings_impu['previous_year_rating'+'_median'] = train['previous_year_rating'].fillna(train.previous_year_rating.median())
prev_Year_Ratings_impu['previous_year_rating'+'_mode'] = train['previous_year_rating'].fillna(train.previous_year_rating.mode().iloc[0])
prev_Year_Ratings_impu['previous_year_rating'+'_zero'] = train['previous_year_rating'].fillna(0)

# KNN Imputer
imputer = KNNImputer(n_neighbors = 3)
x = np.array(train['previous_year_rating'])
x = x.reshape(-1,1)
prev_Year_Ratings_impu['previous_year_rating'+'_knn'] = imputer.fit_transform(x)
prev_Year_Ratings_impu.head(10)
# Distribution Plot of all the Imputation Types
fig = plt.figure(figsize = (20,8))
ax = fig.add_subplot(111)
train['previous_year_rating'].plot(kind = 'kde', ax = ax, color = 'red')
prev_Year_Ratings_impu.previous_year_rating_zero.plot(kind = 'kde', ax = ax, color = 'green')
prev_Year_Ratings_impu.previous_year_rating_mean.plot(kind = 'kde', ax = ax, color = 'blue')
prev_Year_Ratings_impu.previous_year_rating_median.plot(kind = 'kde', ax = ax, color = 'yellow')
prev_Year_Ratings_impu.previous_year_rating_mode.plot(kind = 'kde', ax = ax, color = 'black')
prev_Year_Ratings_impu.previous_year_rating_knn.plot(kind = 'kde', ax = ax, color = 'pink')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc = 'best')
# we can see that the distribution has changed slightly with now more values accumulating towards the median
fig = plt.figure(figsize = (20,8))
ax = fig.add_subplot(111)
train['previous_year_rating'].plot(kind = 'kde', ax = ax, color = 'red')
prev_Year_Ratings_impu.previous_year_rating_zero.plot(kind = 'kde', ax = ax, color = 'green')
prev_Year_Ratings_impu.previous_year_rating_mean.plot(kind = 'kde', ax = ax, color = 'blue')
# prev_Year_Ratings_impu.previous_year_rating_median.plot(kind = 'kde', ax = ax, color = 'yellow')
# prev_Year_Ratings_impu.previous_year_rating_mode.plot(kind = 'kde', ax = ax, color = 'black')
prev_Year_Ratings_impu.previous_year_rating_knn.plot(kind = 'kde', ax = ax, color = 'pink')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc = 'best')
# Relating Length of Service with Ratings to find out relations
temp = train.copy()
temp.previous_year_rating.fillna(-1, inplace = True)
temp.groupby(['previous_year_rating']).length_of_service.value_counts()
# No. of Employee who had no Rating
train.previous_year_rating.isnull().sum()
# Before Imputation
train.previous_year_rating.unique(), val.previous_year_rating.unique(), test.previous_year_rating.unique()
# imputing the same in all the dataframes as it's independent of the central limit tendency 
# Imputing NULL Values of 'Previous Year's' as '0'
train["previous_year_rating"].fillna(0, inplace = True)
val["previous_year_rating"].fillna(0, inplace = True)
test["previous_year_rating"].fillna(0, inplace = True)
# After Imputation
train.previous_year_rating.unique(), val.previous_year_rating.unique(), test.previous_year_rating.unique()
# deleting redundant dataframes
del prev_Year_Ratings_impu, temp
# Checking out if any NULL value is still present in the dataframe
train.isnull().sum(), val.isnull().sum(), test.isnull().sum()
# temporarily segregating into independent vector to take out quasi constant features
x = train.drop(['is_promoted'], axis = 1)
x.head(10)
# Multi-Feature Label Encoder Class

class multiFeatureLabelEncoder:

    def __init__(self, cols = None):
        self.cols = cols

    def fit(self, x, y = None):
        return self

    def transform(self, x):
        output = x.copy()
        if self.cols is not None:
            for col in self.cols:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, x, y = None):
        z = self.fit(x,y).transform(x)
        print(z.info())
        return z
# displaying the head aiding to get string features
x.head(1)
# Label Encoding Multiple Feature
x = multiFeatureLabelEncoder(['department', 'region', 'education', 'gender', 'recruitment_channel']).fit_transform(x)
# Finding out Quasi Constant Features
quasiDetect = VarianceThreshold(threshold = 0.01)  
quasiDetect.fit(x)
# No. of Columns Retained ie non-quasi features
len(x.columns[quasiDetect.get_support()])
# Displaying the Quasi Constant Features
print(f"No. of Quasi-Constant Features : {len([y for y in x.columns if y not in x.columns[quasiDetect.get_support()]])}")
print(f"Quasi-Constant Features : {[y for y in x.columns if y not in x.columns[quasiDetect.get_support()]]}")
# Deleting off redundant objects/variables - Memory Efficient Techniques
del x, unique_Counts, const_Features, train_enc, dup_cols, nullPlot, percentage_Of_Null_Education, percentage_Of_Null_Prev_Year_Rating, imputer, fig, ax, quasiDetect
# Displaying out the columns of train
train.columns
# Statistical Info about train
train.describe()
# checking out if there's any duplicacy in employee ID, ie it should be 38365
train.employee_id.nunique()
# finding out insights about departments present in the company
print(f"No. of Departments : {train.department.nunique()} \nThe Departments are : {list(train.department.unique())}")
# count plot to visualise the employee counts of the department
plot = sns.catplot(data = train, kind = 'count', x = 'department')
# plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(20, 5)
# count of no of people in each department

# train['department'].value_counts().sort_values(ascending = False)
x = []
dept = train['department'].unique()
for i in dept:
  x.append(train.loc[train.department == i, 'department'].count())

i=0
while i < len(dept):
  #print(i)
  print("No of Employees in the Department of ", train['department'].unique()[i]," are : ", x[i])
  i+=1
# % of the departments having highest and lowest employees
sales = train.loc[train.department == 'Sales & Marketing', 'department'].count()
rnd = train.loc[train.department == 'R&D', 'department'].count()

print(f"% of Employees in 'Sales & Marketing' : {round(sales/train.shape[0]*100, 2)} %")
print(f"% of Employees in 'R&D' : {round(rnd/train.shape[0]*100, 2)} %")
# finding out insights about regions
print(f"No. of Regions : {train.region.nunique()} \nThe Regions are : {list(train.region.unique())}")
# sort the regions in ascending order
pd.Series(train.region.unique()).sort_values()
# count plot to visualise the counts of the regions
plot = sns.catplot(data = train, kind = 'count', x = 'region')
plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(20, 5)
# No. of Employees per Region
train['region'].value_counts().sort_values(ascending = False)
# % of the regions having highest and lowest employees
region_2 = train.loc[train.region == 'region_2', 'region'].count()
region_18 = train.loc[train.region == 'region_18', 'region'].count()

print(f"% of Employees from 'region_2' : {round(region_2/train.shape[0]*100, 2)} %")
print(f"% of Employees from 'region_18' : {round(region_18/train.shape[0]*100, 2)} %")
# finding out insights about Education
print(f"No. of Categories of Education : {train.education.nunique()} \nThe Slabs are : {list(train.education.unique())}")
# count plot to visualise the counts of the slabs of education
plot = sns.catplot(data = train, kind = 'count', x = 'education')
#plot.set_xticklabels(rotation = 30)
plt.gcf().set_size_inches(20, 5)
train['education'].value_counts().sort_values(ascending = False)
# % of the slabs of education having highest and lowest employees
bachelors = train.loc[train.education == "Bachelor's", 'education'].count()
below_Secondary = train.loc[train.education == 'Below Secondary', 'education'].count()

print(f"% of Employees from 'Bachelors' : {round(bachelors/train.shape[0]*100, 2)} %")
print(f"% of Employees from 'Below Secondary' : {round(below_Secondary/train.shape[0]*100, 2)} %")
# finding out insights about Gender
print(f"No. of Categories for Gender : {train.gender.nunique()} \nThe Categories are : {list(train.gender.unique())}")
# count plot to visualise the counts of Gender
plot = sns.catplot(data = train, kind = 'count', x = 'gender')
# plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(15, 5)
train['gender'].value_counts()
# Gender % of Employees
male = train.loc[train.gender == 'm', 'gender'].count()
female = train.loc[train.gender == 'f', 'gender'].count()

print(f"% of 'Male' Employees : {round(male/train.shape[0]*100, 2)} %")
print(f"% of 'Female' Employees : {round(female/train.shape[0]*100, 2)} %")
# finding out insights about Recruitment Channel
print(f"No. of Categories for Recruitment Channel : {train.recruitment_channel.nunique()} \nThe Recruitment Channels are : {list(train.recruitment_channel.unique())}")
# count plot to visualise the counts of Gender
plot = sns.catplot(data = train, kind = 'count', x = 'recruitment_channel')
# plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(15, 5)
train['recruitment_channel'].value_counts().sort_values(ascending = False)
# % of the Employee Recruitment_Channel having highest and lowest employees
other = train.loc[train.recruitment_channel == 'other', 'recruitment_channel'].count()
reff = train.loc[train.recruitment_channel == 'referred', 'recruitment_channel'].count()

print(f"% of Employees recruited through 'other' sources : {round(other/train.shape[0]*100, 2)} %")
print(f"% of Employees recruited through 'Reference' : {round(reff/train.shape[0]*100, 2)} %")
# finding out insights about no of trainings employees have
print(f"No. of Unique No. of Training : {train.no_of_trainings.nunique()} \nNo. of Trainings : {list(train.no_of_trainings.unique())}")
# count plot to visualise the counts of Gender
plot = sns.catplot(data = train, kind = 'count', y = 'no_of_trainings')
# plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(15, 5)
# Scatter Plot to Visualise the Points in detail
plt.figure(figsize = (18, 7))
plt.scatter(range(train.shape[0]), train['no_of_trainings'])
train['no_of_trainings'].value_counts().sort_values(ascending = False)
# % of the Employees having 1 training
one_Train = train.loc[train.no_of_trainings == 1, 'no_of_trainings'].count()

print(f"% of Employees having 1 training : {round(one_Train/train.shape[0]*100, 2)} %")
# Box Plot to find more information about it's Quartiles and also detect Outliers
plt.figure(figsize = (8, 8))
plt.title('No of Trainings Distribution Spread')
sns.boxplot(y = train.no_of_trainings)
# trying out and observing the distribution plot of ‘no_of_trainings’ after boxcox transformation.
boxc = stats.boxcox(train['no_of_trainings'])[0]
print(f"Skewness after Transformation : {pd.Series(boxc).skew()}")
plt.title('No of Trainings Distribution Spread')
sns.boxplot(y = boxc)
# finding out insights about age of employees working
print(f"No. of unique value of age amonsgt the employees : {train.age.nunique()}")
# Binnig
range = ['<25', '26-30', '30-35', '36-40', '41-45', '46-50', '51-55', '56-60']
bins = [20, 26, 31, 36, 41, 46, 51, 56, 61] # [20, 26, 31, 36, 50] 
age_EDA = pd.DataFrame(pd.cut(x = train['age'], bins = bins, labels = range, include_lowest = True))
#X_train['age'] = pd.cut(x=X_train['age'], bins=[20, 30, 39, 49], labels=['20', '30', '40'] )
#X_test['age']  = pd.cut(x=X_test['age'], bins=[20, 30, 39, 49], labels=['20', '30', '40'] )
# No. of Unique Age Groups
age_EDA.nunique()
# displaying the 1st 10 records after binning
train.age.head(10)
# displaying the 1st 10 records after binning
age_EDA.head(10)
# displaying no. of employees in each age category
age_EDA.age.value_counts().sort_values(ascending = False)
# count plot to visualise the counts of Gender
plot = sns.catplot(data = age_EDA, kind = 'count', y = 'age')
# plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(15, 5)
# % of the Slabs of Age Group having highest and lowest employees
cat26To30 = age_EDA.loc[age_EDA.age == '26-30', 'age'].count()
cat55To60 = age_EDA.loc[age_EDA.age == '56-60', 'age'].count()

print(f"% of Employees under the age-group '26-30' : {round(cat26To30/train.shape[0]*100, 2)} %")
print(f"% of Employees under the age-group '55-60' : {round(cat55To60/train.shape[0]*100, 2)} %")
# deleting redundant dataframe
del age_EDA
# Box Plot to find more information about it's Quartiles and also detect Outliers
plt.figure(figsize = (8, 8))
plt.title('Age Distribution Spread')
sns.boxplot(y = train.age)
# For IQR Analysis, we need to sort our feature vector first
arr = np.array(sorted(train.age))
# Defining the Quartile Ranges
quantile_1, quantile_3 = np.percentile(arr, [25, 75])
print(f"1st Quartile (25th Percentile) of Age Feature : {quantile_1} \n3rd Quartile (75th Percentile) of Age Feature : {quantile_3}")
# Defining the IQR
iqr = quantile_3 - quantile_1
print(iqr)
# Finding out the Lower Bound Value and the Higher Bound Value
lower_Bound = quantile_1 - (1.5 * iqr) 
upper_Bound = quantile_3 + (1.5 * iqr)
print(f"Lower Bound Value : {lower_Bound} \nUpper Bound Value : {upper_Bound}")
# Any number below the Lower Bound or above the Upper Bound would be considered as an Outlier
outliers = train.age[(train.age < lower_Bound) | (train.age > upper_Bound)]
print(f"Number of Outliers : {len(outliers)} \nThe Outliers are : {outliers.unique()}")
# % of entries which are outliers
print(f"% of Records which are been considered Outliers : {round(len(outliers)/train.shape[0] * 100, 2)} %")
# Distribution Plot to visualise the distribution of age of employees
sns.distplot(train.age)
# plot.set_xticklabels(rotation = 30)
plt.gcf().set_size_inches(20, 5)
# Finding out age's distribution statistics
plt.hist(train.age, bins = 'auto')

print("mean : ", np.mean(train.age))
print("var  : ", np.var(train.age))
print("skew : ", stats.skew(train.age))
print("kurt : ", stats.kurtosis(train.age), "\n")
# trying out square-root and log transformations to reduce skewness
log = np.log(train['age'])
sqr = np.sqrt(train['age'])
print(log.skew(), sqr.skew())
# trying out and observing the distribution plot of ‘age’ after boxcox transformation.
boxc = stats.boxcox(train['age'])[0]
print(f"Skewness after Transformation : {pd.Series(boxc).skew()}")
sns.distplot(boxc)
# count plot to visualise the counts of the slabs of education
plot = sns.catplot(data = train, kind = 'count', x = 'previous_year_rating')
#plot.set_xticklabels(rotation = 30)
plt.gcf().set_size_inches(20, 5)
# Showing the counts of employee per ratings
train['previous_year_rating'].value_counts().sort_values(ascending = False)
# % of the previous years' ratings of employees having highest and lowest ratings along with the employees having 5 rating
high_Per_Rating = train.loc[train.previous_year_rating == 3.0, 'previous_year_rating'].count()
low_Per_Rating = train.loc[train.previous_year_rating == 0.0, 'previous_year_rating'].count()
_5_Rating = train.loc[train.previous_year_rating == 5.0, 'previous_year_rating'].count()

print(f"% of Employees having Rating '3.0' : {round(high_Per_Rating/train.shape[0]*100, 2)} %")
print(f"% of Employees having Rating '0.0' or 'NO Rating' : {round(low_Per_Rating/train.shape[0]*100, 2)} %")
print(f"% of Employees having Rating '5.0' : {round(_5_Rating/train.shape[0]*100, 2)} %")
train.previous_year_rating.unique()
# finding out insights about length of service
print(f"No. of unique value for employees' length of service : {train.length_of_service.nunique()}")
print(f"Max Service Length : {max(train.length_of_service)} \nMin Service Length : {min(train.length_of_service)}")
# count plot to visualise the distribution of length of service
sns.distplot(train.length_of_service)
# plot.set_xticklabels(rotation = 30)
plt.gcf().set_size_inches(20, 5)
# Finding out LOS's distribution statistics
plt.hist(train.length_of_service, bins = 'auto')

print("mean : ", np.mean(train.length_of_service))
print("var  : ", np.var(train.length_of_service))
print("skew : ", stats.skew(train.length_of_service))
print("kurt : ", stats.kurtosis(train.length_of_service), "\n")
# trying out square-root and log transformations to reduce skewness
log = np.log(train.length_of_service)
sqr = np.sqrt(train.length_of_service)
print(log.skew(), sqr.skew())
# trying out and observing the distribution plot of ‘LOS’ after boxcox transformation.
boxc = stats.boxcox(train.length_of_service)[0]
print(f"Skewness after Transformation : {pd.Series(boxc).skew()}")
sns.distplot(boxc)
# Box Plot to find more information about it's Quartiles and also detect Outliers
plt.figure(figsize = (8, 8))
plt.title('Length of Service Distribution Spread')
sns.boxplot(y = train.length_of_service)
# For IQR Analysis, we need to sort our feature vector first
arr = np.array(sorted(train.length_of_service))
# Defining the Quartile Ranges
quantile_1, quantile_3= np.percentile(arr, [25, 75])
print(f"1st Quartile (25th Percentile) of Age Feature : {quantile_1} \n3rd Quartile (75th Percentile) of Age Feature : {quantile_3}")
# Defining the IQR
iqr = quantile_3 - quantile_1
print(iqr)
# Finding out the Lower Bound Value and the Higher Bound Value
lower_Bound = quantile_1 - (1.5 * iqr) 
upper_Bound = quantile_3 + (1.5 * iqr)
print(f"Lower Bound Value : {lower_Bound} \nUpper Bound Value : {upper_Bound}")
# Any number below the Lower Bound or above the Upper Bound would be considered as an Outlier
outliers = train.length_of_service[(train.length_of_service < lower_Bound) | (train.length_of_service > upper_Bound)]
print(f"Number of Outliers : {len(outliers)} \nThe Outliers are : {sorted(outliers.unique())}")
# % of entries which are outliers
print(f"% of Records which are been considered Outliers : {round(len(outliers)/train.shape[0] * 100, 2)} %")
# finding out insights about KPIs met
print(f"No. of unique value for KPIs met by employees : {train['KPIs_met >80%'].nunique()}")
# count plot to visualise the counts of 'KPIs_met >80%'
plot = sns.catplot(data = train, kind = 'count', x = 'KPIs_met >80%')
# plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(15, 5)
# count the occurence for each category
train['KPIs_met >80%'].value_counts()
# 'KPIs_met >80%' % of Employees
kpis_Met = train.loc[train['KPIs_met >80%'] == 0, 'KPIs_met >80%'].count()
kpis_Not_Met = train.loc[train['KPIs_met >80%'] != 0, 'KPIs_met >80%'].count()

print(f"% of Employees who satisfied KPIs : {round(kpis_Met/train.shape[0]*100, 2)} %")
print(f"% of Employees who didn't satisfied KPIs : {round(kpis_Not_Met/train.shape[0]*100, 2)} %")
# finding out insights about awards won by employees
print(f"No. of unique value for awards won by employees : {train['awards_won?'].nunique()}")
# count plot to visualise the counts of 'awards_won?'
plot = sns.catplot(data = train, kind = 'count', x = 'awards_won?')
# plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(15, 5)
# count the occurence for each category
train['awards_won?'].value_counts()
# 'awards_won?' % of Employees
emp_Win = train.loc[train['awards_won?'] == 0, 'awards_won?'].count()
emp_Not_Win = train.loc[train['awards_won?'] != 0, 'awards_won?'].count()

print(f"% of Employees who have won atleast an award : {round(emp_Win/train.shape[0]*100, 2)} %")
print(f"% of Employees who haven't won any award : {round(emp_Not_Win/train.shape[0]*100, 2)} %")
# finding out insights about average training scores
print(f"No. of unique value for employees' average training scores : {train.avg_training_score.nunique()}")
print(f"Max Average Training Score : {max(train.avg_training_score)} \nMin Average Training Score : {min(train.avg_training_score)}")
# count plot to visualise the distribution of average training score
sns.distplot(train.avg_training_score)
# plot.set_xticklabels(rotation = 30)
plt.gcf().set_size_inches(20, 5)
# Finding out Avg Training Score distribution statistics
plt.hist(train.avg_training_score, bins = 100)

print("mean : ", np.mean(train.avg_training_score))
print("var  : ", np.var(train.avg_training_score))
print("skew : ", stats.skew(train.avg_training_score))
print("kurt : ", stats.kurtosis(train.avg_training_score), "\n")
# Box Plot to find more information about it's Quartiles and also detect Outliers
plt.figure(figsize = (8, 8))
plt.title('Length of Average Training Score Distribution Spread')
sns.boxplot(y = train.avg_training_score)
# Binnig to find out more information about the same.
range = ['<40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
bins = [31, 41, 51, 61, 71, 81, 91, 101] # [20, 26, 31, 36, 50] 
avg_Tr_Score_EDA = pd.DataFrame(pd.cut(x = train['avg_training_score'], bins = bins, labels = range, include_lowest = True))
#X_train['age'] = pd.cut(x=X_train['age'], bins=[20, 30, 39, 49], labels=['20', '30', '40'] )
#X_test['age']  = pd.cut(x=X_test['age'], bins=[20, 30, 39, 49], labels=['20', '30', '40'] )
avg_Tr_Score_EDA.nunique()
avg_Tr_Score_EDA.avg_training_score.value_counts().sort_values(ascending = False)
# count plot to visualise the counts of Gender
plot = sns.catplot(data = avg_Tr_Score_EDA, kind = 'count', y = 'avg_training_score')
# plot.set_xticklabels(rotation = 75)
plt.gcf().set_size_inches(15, 5)
# % of the Slabs of Employees having Highest and Lowest Average Test Scores
cat51To60 = avg_Tr_Score_EDA.loc[avg_Tr_Score_EDA.avg_training_score == '51-60', 'avg_training_score'].count()
catless40 = avg_Tr_Score_EDA.loc[avg_Tr_Score_EDA.avg_training_score == '<40', 'avg_training_score'].count()

print(f"% of Average Test Score of  Employees under the group '51-60' : {round(cat51To60/avg_Tr_Score_EDA.shape[0]*100, 2)} %")
print(f"% of Average Test Score of  Employees under the group '<40' : {round(catless40/avg_Tr_Score_EDA.shape[0]*100, 2)} %")
# No. of Employees having Max Score
train.loc[train.avg_training_score == max(train.avg_training_score.values), 'avg_training_score'].count()
# deleting redundant dataframe
del avg_Tr_Score_EDA
# finding out insights about average training scores
print(f"Unique values for target variable is_promoted : {train.is_promoted.nunique()}")
# count plot to visualise the counts of the categories
sns.catplot(data = train, kind = 'count', x = 'is_promoted')
# to have a count for how many people got promoted or hot
gotPromoted = train.loc[train.is_promoted == 1, 'is_promoted'].count()
didNotGetPromoted = train.loc[train.is_promoted != 1, 'is_promoted'].count()

print(f"No. of Employees that got Promoted : {gotPromoted} and it's % : {round(gotPromoted/train.shape[0]*100, 2)}")
print(f"No. of Employees that didn't get Promoted : {didNotGetPromoted} and it's % : {round(didNotGetPromoted/train.shape[0]*100, 2)}")
# Deleting off redundant objects/variables - Memory Efficient Techniques
del plot, x, dept, region_2, region_18, bachelors, below_Secondary, male, female, other, reff, one_Train, boxc, range, bins, cat26To30, cat55To60, arr, quantile_1, quantile_3, iqr, lower_Bound , upper_Bound, outliers, log, sqr, high_Per_Rating, low_Per_Rating, _5_Rating, kpis_Met, kpis_Not_Met, emp_Win, emp_Not_Win, cat51To60, catless40, gotPromoted, didNotGetPromoted
# Displaying the Columns
train.columns
# Displaying the 'Object' Columns for encoding (for EDA)
list(train.select_dtypes(include = ['object']).columns)
# Encoding 'Education'
dummy_Train = train.copy()  # to keep the integrity of main train intact.

# Mannualy Assigning Weight and Encoding 'ordinal' Feature 'Education'
edu_enc = {"Below Secondary" : 0, "Others" : 1, "Bachelor's": 2, "Master's & above": 3}
dummy_Train['education'] = train['education'].map(edu_enc)
dummy_Train.education.unique()
# Label Encoding 'Gender'
l = LabelEncoder()
dummy_Train.loc[:, 'gender'] = l.fit_transform(train.loc[:, 'gender'])
dummy_Train.gender.unique()
# Defining K-Fold Target Encoding Class for Train (K-Fold as for Regularization)

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colname, targetName, n_fold = 5):

        self.colnames = colname
        self.targetName = targetName
        self.n_fold = n_fold

    def fit(self, x, y = None):
        return self

    def transform(self, x):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in x.columns)
        assert(self.targetName in x.columns)

        mean_of_target = x[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold, shuffle = False, random_state=0)

        col_mean_name = 'tgt_' + self.colnames
        x[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(x):
            x_tr, x_val = x.iloc[tr_ind], x.iloc[val_ind]
            x.loc[x.index[val_ind], col_mean_name] = x_val[self.colnames].map(x_tr.groupby(self.colnames)[self.targetName].mean())

        x[col_mean_name].fillna(mean_of_target, inplace = True)

        return x
# K-Fold Target Encoding 'recruitment_channel'
targetc = KFoldTargetEncoderTrain('recruitment_channel', 'is_promoted', n_fold = 5)
dummy_Train = targetc.fit_transform(dummy_Train)
dummy_Train.drop(['recruitment_channel'], axis = 1, inplace = True)
# K-Fold Target Encoding 'region'
targetc = KFoldTargetEncoderTrain('region', 'is_promoted', n_fold = 5)
dummy_Train = targetc.fit_transform(dummy_Train)
dummy_Train.drop(['region'], axis = 1, inplace = True)
# K-Fold Target Encoding 'department'
targetc = KFoldTargetEncoderTrain('department', 'is_promoted', n_fold = 5)
dummy_Train = targetc.fit_transform(dummy_Train)
dummy_Train.drop(['department'], axis = 1, inplace = True)
# The Modified Dummy Dataset for Multivariate EDA
dummy_Train.head(10)
# Plotty Interactive Visualisation to find out the inter-relations amongst all the features to dependent vector
fig = px.parallel_categories(train[train.columns], color = "is_promoted", color_continuous_scale = px.colors.sequential.Aggrnyl)
fig.show()
# plotting pairwise relationships in train with respect to the dependent vector
sns.pairplot(dummy_Train.drop(['employee_id'], axis = 1), hue = 'is_promoted')
# Correlation using heatmap
hm = dummy_Train.corr().where(np.tril(np.ones(dummy_Train.corr().shape)).astype(np.bool)) # to delete the upper triangle
plot = sns.heatmap(hm, annot = True, cmap="YlGnBu")
plt.setp(plot.get_xticklabels(), rotation=45)
plt.gcf().set_size_inches(15, 8)
# Covariance using Heatmap
dummy_Train = dummy_Train.round(decimals = 2)
covMatrix = pd.DataFrame.cov(dummy_Train.drop(['employee_id'], axis = 1))
sns.heatmap(covMatrix, annot = True, fmt = 'g')
plt.gcf().set_size_inches(30, 15)
# Scatter Plots of the whole train set with respect to each other.
scatter_matrix(dummy_Train.drop(['employee_id'], axis = 1), alpha = 0.2, figsize = (25, 25))
# displaying column names for reference
dummy_Train.columns
# Bi-Variate Analysis on 'Age' and 'Average Training Score' which are inversely related
sns.jointplot(x = 'age', y = 'avg_training_score', data = train, kind = 'reg')
# Multivariate Analysis of - 'is_promoted', 'length_of_service', 'awards_won?'
sns.FacetGrid(dummy_Train, hue = "is_promoted", size = 5).map(plt.scatter, "length_of_service", "awards_won?").add_legend()
plt.show()
# Multivariate Analysis of - 'is_promoted', 'no_of_trainings', 'previous_year_rating'
sns.FacetGrid(train, hue = "is_promoted", size = 5).map(plt.scatter, "no_of_trainings", "previous_year_rating").add_legend()
plt.show()
# Multivariate Analysis of - 'is_promoted', 'average training score', 'age'
sns.FacetGrid(train, hue = "is_promoted", size = 5).map(plt.scatter, "avg_training_score", "age").add_legend()
plt.show()
# Checking out the distribution of 'is_promoted' across the newly Encoded Variables
plt.figure(figsize=(25, 6))

df = pd.DataFrame(train.groupby(['department'])['is_promoted'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Dept. vs Promotion')
plt.show()

df = pd.DataFrame(train.groupby(['region'])['is_promoted'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Region vs Promotion')
plt.show()

df = pd.DataFrame(train.groupby(['gender'])['is_promoted'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Gender vs Promotion')
plt.show()

df = pd.DataFrame(train.groupby(['education'])['is_promoted'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Education vs Promotion')
plt.show()

df = pd.DataFrame(train.groupby(['recruitment_channel'])['is_promoted'].mean().sort_values(ascending = False))
df.plot.bar()
plt.title('Recruitment Channel vs Promotion')
plt.show()

# Deleting redundant dataframe
del df
# Deleting off redundant objects/variables - Memory Efficient Techniques
del l, targetc, fig, hm, covMatrix
# Dividing Train into Independent and Dependent Features
x = train.drop(['is_promoted'], axis = 1)
y = train['is_promoted'].values.reshape(-1, 1)
x.shape, y.shape
# Print out the Names of Object Columns
objectColumns = []
for i in x.columns[x.dtypes == 'object']:   objectColumns.append(i)
print(f"Object Column Names : {objectColumns}")

# Preserve the Name of the Columns
colNames = x.columns
print(f"Column Names : {list(colNames)}")
# Temporarily Encoding Department and Preserving Information about each Variable.
l_dept = LabelEncoder()
x['department'] = l_dept.fit_transform(x['department'])
# Temporarily Encoding Region
l_region = LabelEncoder()
x['region'] = l_region.fit_transform(x['region'])
# Temporarily Encoding Education
l_edu = LabelEncoder()
x['education'] = l_edu.fit_transform(x['education'])
# Temporarily Encoding Gender
l_gender = LabelEncoder()
x['gender'] = l_gender.fit_transform(x['gender'])
# Temporarily Encoding Recruitment
l_rc = LabelEncoder()
x['recruitment_channel'] = l_rc.fit_transform(x['recruitment_channel'])
# Listing the Different Classes
print(list(l_dept.classes_))
print(list(l_region.classes_))
print(list(l_edu.classes_))
print(list(l_gender.classes_))
print(list(l_rc.classes_))
# Visualising the 1st two Records of the Dataframe
x.head(2)
# Decoding and Testing the Intregity
'''
# Temporarily Decoding Department
x['department'] = l_dept.inverse_transform(x['department'])
# Temporarily Decoding Region
x['region'] = l_region.inverse_transform(x['region'])
# Temporarily Decoding Education
x['education'] = l_edu.inverse_transform(x['education'])
# Temporarily Decoding Gender
x['gender'] = l_gender.inverse_transform(x['gender'])
# Temporarily Decoding Recruitment
x['recruitment_channel'] = l_rc.inverse_transform(x['recruitment_channel'])
x.head(2)
'''
# Over-Sampling to Increase the Number of '1' Samples

sm = SMOTE(sampling_strategy = 'minority', random_state = 0, n_jobs = -1)
x_os_train, y_os_train = sm.fit_sample(x, y)
# Converting the Over-Sampled Array into a Dataframe for further Operations
train_OS = pd.DataFrame(x_os_train, columns = colNames)
train_OS.rename(index = str).index
train_OS.columns
train_OS.info()
# Maintaining the Types as Old DataFrame
train_OS = train_OS.astype(int)
train_OS['previous_year_rating'] = train_OS['previous_year_rating'].astype(float)
train_OS.head(1)
# Decoding the Features as Old DataFrame

train_OS['department'] = l_dept.inverse_transform(train_OS['department'])
# Temporarily Decoding Region
train_OS['region'] = l_region.inverse_transform(train_OS['region'])
# Temporarily Decoding Education
train_OS['education'] = l_edu.inverse_transform(train_OS['education'])
'''
# Not Required as it'll be label encoded to the same later.
## Temporarily Decoding Gender
train_OS['gender'] = l_gender.inverse_transform(train_OS['gender'])
'''
# Temporarily Decoding Recruitment
train_OS['recruitment_channel'] = l_rc.inverse_transform(train_OS['recruitment_channel'])

# Printing the 1st 2 records of the transformed Dataframe
train_OS.head(2)
# After Sampling Size
train_OS['is_promoted'] = y_os_train
print(f"Shape of the Over-Sampled DataFrame : {train_OS.shape}")
print(f"No. of Records Simulated as Compared to Old : {train_OS.shape[0] - x.shape[0]}")
# count plot to visualise the counts of the categories
sns.catplot(data = train_OS, kind = 'count', x = 'is_promoted')
print(f"No. of People Not Promoted vs Promoted : {Counter(train_OS.is_promoted)}")
# Visualising Items for Occurence per Categorical Feature
for i in objectColumns:
  print("No. of Records per Category of {} is {}".format(i, Counter(train_OS[i])))
# visualising to know the features types and names
train_OS.head(1)
# creating another dummy dataframe to maintain the integrity of the main dataframe
trans_Train_OS = train_OS.copy()
trans_Train_OS.head(1)
# Box Plot to find more information about it's Quartiles and also detect Outliers
'''
plt.figure(figsize = (15, 5))
plt.title('Age Distribution Spread')
sns.boxplot(x = trans_Train.age)
print("1st Quartile : {}".format(trans_Train_OS.age.quantile(0.25)))
print("3rd Quartile : {}".format(trans_Train.age.quantile(0.75)))
'''
# For Age
fig = px.box(trans_Train_OS, x = "is_promoted", y = "age", points = "outliers")
fig.show()

# For Length of Service
fig = px.box(trans_Train_OS, x = "is_promoted", y = "length_of_service", points = "outliers")
fig.show()

# Avg Training Score
fig = px.box(trans_Train_OS, x = "is_promoted", y = "avg_training_score", points = "outliers")
fig.show()
# Applying Boxcox Transformation and Checking for Result
boxc_Age = stats.boxcox(trans_Train_OS['age'])[0]
boxc_LOS = stats.boxcox(trans_Train_OS['length_of_service'])[0]

# Creating a Copy for Testing
xyz = trans_Train_OS.copy()

# Loading the Results
xyz['age'] = boxc_Age
xyz['length_of_service'] = boxc_LOS

# Printing out the Results
print(f"After Transformation, \nSkewness of Age : {xyz['age'].skew()} \nSkewness of LOS : {xyz['length_of_service'].skew()}")
# For Age
fig = px.box(xyz, x = "is_promoted", y = "age", points = "outliers")
fig.show()

# For Length of Service
fig = px.box(xyz, x = "is_promoted", y = "length_of_service", points = "outliers")
fig.show()
print("***** For LOS *****")
# For IQR Analysis, we need to sort our feature vector first
arr = np.array(sorted(xyz.length_of_service))

# Defining the Quartile Ranges
quantile_1, quantile_3 = np.percentile(arr, [25, 75])
print(f"1st Quartile (25th Percentile) of Age Feature : {quantile_1} \n3rd Quartile (75th Percentile) of Age Feature : {quantile_3}")

# Defining the IQR
iqr = quantile_3 - quantile_1
print(f"IQR = {iqr}")

# Finding out the Lower Bound Value and the Higher Bound Value
lower_Bound = quantile_1 - (1.5 * iqr) 
upper_Bound = quantile_3 + (1.5 * iqr)
print(f"Lower Bound Value : {lower_Bound} \nUpper Bound Value : {upper_Bound}")

# Any number below the Lower Bound or above the Upper Bound would be considered as an Outlier
outliers = xyz.length_of_service[(xyz.length_of_service < lower_Bound) | (xyz.length_of_service > upper_Bound)]
print(f"Number of Outliers : {len(outliers)}")
## print(f"The Outliers are : {sorted(outliers.unique())}")

# % of entries which are outliers
print(f"% of Records which are been considered Outliers : {round(len(outliers)/xyz.shape[0] * 100, 2)} %")

print("*" * 10)

print("***** For Age *****")
# For IQR Analysis, we need to sort our feature vector first
arr = np.array(sorted(xyz.age))

# Defining the Quartile Ranges
quantile_1, quantile_3 = np.percentile(arr, [25, 75])
print(f"1st Quartile (25th Percentile) of Age Feature : {quantile_1} \n3rd Quartile (75th Percentile) of Age Feature : {quantile_3}")

# Defining the IQR
iqr = quantile_3 - quantile_1
print(f"IQR = {iqr}")

# Finding out the Lower Bound Value and the Higher Bound Value
lower_Bound = quantile_1 - (1.5 * iqr) 
upper_Bound = quantile_3 + (1.5 * iqr)
print(f"Lower Bound Value : {lower_Bound} \nUpper Bound Value : {upper_Bound}")

# Any number below the Lower Bound or above the Upper Bound would be considered as an Outlier
outliers = xyz.age[(xyz.age < lower_Bound) | (xyz.age > upper_Bound)]
print(f"Number of Outliers : {len(outliers)}")
## print(f"The Outliers are : {sorted(outliers.unique())}")

# % of entries which are outliers
print(f"% of Records which are been considered Outliers : {round(len(outliers)/xyz.shape[0] * 100, 2)} %")
# Adding these Transformed Features in the Dataset
trans_Train_OS['age_boxcox'] = boxc_Age
trans_Train_OS['length_of_service_boxcox'] = boxc_LOS

trans_Train_OS.head(2)
# Equal Frequency Discretization
discretizer = EqualFrequencyDiscretiser(q = 10, variables = ['age', 'length_of_service'], return_object = True, return_boundaries = True)
discretizer
# fit the discretization transformer
discretizer.fit(trans_Train_OS)

# transform the data
train_t = discretizer.transform(trans_Train_OS)

# Visualising the Bins
print(f"Age : {list(train_t.age.unique())} \nLOS : {list(train_t.length_of_service.unique())}")
# Adding Binned Categories to the Dataframe
trans_Train_OS['age_bins'] = train_t.age
trans_Train_OS['length_of_service_bins'] = train_t.length_of_service

# Deleting Redundant Dataframe
del train_t

# Visualising the 1st 5 records of transformed dataframe
trans_Train_OS.head(5)
# IGNORE -> Will do Data Leakage and not insightful.
# Checked at Last

##  Probability of Promotion Per Dept.
'''
temp = pd.DataFrame()

# trans_Train_OS['prob_per_dept'] = train.groupby(['department'])['is_promoted'].mean()
# trans_Train_OS.prob_per_dept.unique()

temp['prob_per_dept'] = train.groupby(['department'])['is_promoted'].apply(lambda x : x.mean())
trans_Train_OS = pd.merge(trans_Train_OS, temp, on = ['department'], how = 'left')
trans_Train_OS['prob_per_dept'].fillna(np.median(temp['prob_per_dept']), inplace = True)
trans_Train_OS['prob_per_dept'].unique()
'''
# IGNORE -> Will do Data Leakage and not insightful.
# Checked at Last

##  # Probability of Promotion Per Dept Per Region.
'''
temp = pd.DataFrame()

# temp['prob_per_dept_per_region'] = trans_Train_OS.groupby(['region', 'department'])['is_promoted'].mean()
# temp.head(10)

temp['prob_per_dept_per_region'] = trans_Train_OS.groupby(['region', 'department'])['is_promoted'].mean()
trans_Train_OS = pd.merge(trans_Train_OS, temp, on = ['region', 'department'], how = 'left')
trans_Train_OS['prob_per_dept_per_region'].fillna(np.median(temp['prob_per_dept_per_region']), inplace = True)
# trans_Train_OS['prob_per_dept_per_region'].unique()
temp.head(10)
'''
'''
# K-Fold Cross Validified Promotion Per Dept Per Region.
temp = pd.DataFrame()
temp['prob_per_dept_per_region'] = np.nan

kf = KFold(n_splits = 5, shuffle = False, random_state = 0)
for train_in, val_in in kf.split(trans_Train_OS):
  x_tr, x_val = trans_Train_OS.iloc[train_in], trans_Train_OS.iloc[val_in]
  trans_Train_OS.loc[trans_Train_OS.index[val_in], 'prob_per_dept_per_region'] = x_val['prob_per_dept_per_region'].map(x_tr.groupby(['region', 'department'])['is_promoted'].mean())

trans_Train_OS.prob_per_dept_per_region.fillna(0, inplace = True)
temp.head(10)
'''
# Cummulative Training Score
temp = pd.DataFrame()

# temp = trans_Train_OS.groupby(['recruitment_channel'])['no_of_trainings'].mean()
trans_Train_OS["cummulative_train'_score"] = trans_Train_OS.no_of_trainings * trans_Train_OS.avg_training_score

# Visualising the 1st 5 records after transformation
trans_Train_OS.head(5)
# Checking out for the Region Counts
trans_Train_OS.region.value_counts()
# Checking out where % of employees in a particular region to country is more than 1%
643/trans_Train_OS.shape[0] * 100   # Region_12 & Below -> regions = [10,1,8,9]
# Groupby Function to Club Regions and Promotions
temp = trans_Train_OS.groupby(['region', 'is_promoted'])['is_promoted'].count().sort_values(ascending = False).unstack()    #.apply(lambda r : r/r.sum())
# temp.tail(10)
pd.crosstab(trans_Train_OS['region'], trans_Train_OS.is_promoted).apply(lambda r: r/r.sum(), axis = 1)
# IGNORE -> Will do Data Leakage and not insightful.
# Checked at Last

## Promotion Ratio Per Region
'''
# K-Fold Cross Validified Ratio
kf = KFold(n_splits = 5, shuffle = False, random_state = 0)
temp['prom_ratio_per_region_XXX'] = np.nan

for train_in, val_in in kf.split(temp):
  x_tr, x_val = temp.iloc[train_in], temp.iloc[val_in]
  temp.loc[temp.index[val_in], 'prom_ratio_per_region'] = temp.iloc[:,1] / ((temp.iloc[:,0] + temp.iloc[:,1]) * 100)

temp.fillna(0, axis = 1, inplace = True)
#temp.sort_values(by=['prom_ratio_per_region'], inplace = True, ascending = True)
'''
'''
# Assigning Values
temp['prom_ratio_per_region'] = temp.iloc[:,1] / ((temp.iloc[:,0] + temp.iloc[:,1]) * 100)
temp.fillna(0, axis = 1, inplace = True)
temp.sort_values(by=['prom_ratio_per_region'], inplace = True, ascending = True)
temp.tail(5)
'''
# KPI and Award Concatenation
trans_Train_OS["KPI_n_Award"] = np.where(((trans_Train_OS["KPIs_met >80%"] == 1) & (trans_Train_OS["awards_won?"] == 1)), 1, 0)
trans_Train_OS.head(3)
# Gender - No of Training - Promotion Relation
trans_Train_OS.groupby(['gender', 'no_of_trainings'])['is_promoted'].sum()
# Seggregating no_of_trainings > 4
trans_Train_OS['trainings>4?'] = np.where(trans_Train_OS.no_of_trainings > 4, 1, 0)  # 1 if True else 0
# IGNORE -> Got Reversed after SMOTE
'''
# Reject Region_18 - as noone get's promoted from there
trans_Train_OS['is_Region_18?'] = np.where(trans_Train_OS.region == 'region_18', 1, 0)     # 1 if True else 0 
'''
# IGNORE - Not a insightful feature
'''
# KPI Per Dept.

temp = pd.DataFrame()
temp['KPI_per_dept'] = trans_Train_OS.groupby(['department', 'KPIs_met >80%'])['KPIs_met >80%'].count()
#trans_Train_OS = pd.merge(trans_Train_OS, temp, on = ['department'], how = 'left')
#trans_Train_OS['prob_per_dept'].fillna(np.median(temp['prob_per_dept']), inplace = True)
#trans_Train_OS['prob_per_dept'].unique()
#temp
pd.crosstab([trans_Train_OS['KPIs_met >80%'], trans_Train_OS.is_promoted], trans_Train_OS.department, margins = True).style.background_gradient(cmap = 'summer_r')
'''
# Categorise Employees having good overall performance
trans_Train_OS['good_overall_performance?'] = np.where((trans_Train_OS.previous_year_rating >= 3) & (trans_Train_OS['awards_won?'] == 1) & 
                                                   (trans_Train_OS.avg_training_score >= trans_Train_OS.avg_training_score.quantile(0.25)), 1, 0)
# IGNORE - Not a insightful feature
'''
temp['KPI_per_dept'] = train.groupby(['department', 'KPIs_met >80%'])['KPIs_met >80%'].agg(['count', 'mean'])
temp
'''
'''
train.groupby(['department', 'KPIs_met >80%']).count().unstack()
# train.groupby(['department', 'KPIs_met >80%'])['KPIs_met >80%'].agg(['mean', 'count'])
train.groupby(['department', 'KPIs_met >80%']).size().reset_index(name='counts')
'''
# Mean KPI by Department
trans_Train_OS['mean_kpi_by_dept'] = trans_Train_OS['department'].map(trans_Train_OS.groupby('department')['KPIs_met >80%'].mean())
# Mean Training by Department
trans_Train_OS['mean_training_by_dept'] = trans_Train_OS['department'].map(trans_Train_OS.groupby('department')['avg_training_score'].mean())
# Mean Rating by Department
trans_Train_OS['mean_rating_by_dept'] = trans_Train_OS['department'].map(trans_Train_OS.groupby('department')['previous_year_rating'].mean())
# Prev Years' Rating by Department
trans_Train_OS['dept_rating_mean_ratio'] = trans_Train_OS['previous_year_rating'] / trans_Train_OS['mean_rating_by_dept']
# Visualizing the First Record of Transformed Train
trans_Train_OS.head(1)
# K-Fold Target Encoding 'recruitment_channel', 'region' and 'department'

# 'recruitment_channel'
targetc = KFoldTargetEncoderTrain('recruitment_channel', 'is_promoted', n_fold = 5)
trans_Train_OS = targetc.fit_transform(trans_Train_OS)
## trans_Train_OS.drop(['recruitment_channel'], axis = 1, inplace = True)

# 'region'
targetc = KFoldTargetEncoderTrain('region', 'is_promoted', n_fold = 5)
trans_Train_OS = targetc.fit_transform(trans_Train_OS)
## trans_Train_OS.drop(['region'], axis = 1, inplace = True)

# 'department'
targetc = KFoldTargetEncoderTrain('department', 'is_promoted', n_fold = 5)
trans_Train_OS = targetc.fit_transform(trans_Train_OS)
## trans_Train_OS.drop(['department'], axis = 1, inplace = True)

trans_Train_OS.head(1)
# IGNORE -> Already Done at First
'''
# Label Encoding 'Gender'
l = LabelEncoder()
trans_Train_OS.loc[:, 'gender'] = l.fit_transform(trans_Train_OS.loc[:, 'gender'])
'''
# Encoding 'Education'

# Mannualy Assigning Weight and Encoding 'ordinal' Feature 'Education'
edu_enc = {"Below Secondary" : 0, "Others" : 1, "Bachelor's": 2, "Master's & above": 3}
trans_Train_OS['education'] = trans_Train_OS['education'].map(edu_enc)
trans_Train_OS.education.unique()
trans_Train_OS.head(1)
# K-Fold Target Encoding 'age_bin', 'length_of_service_bins'

# 'age_bin'
targetc = KFoldTargetEncoderTrain('age_bins', 'is_promoted', n_fold = 5)
trans_Train_OS = targetc.fit_transform(trans_Train_OS)
## trans_Train_OS.drop(['age_bins'], axis = 1, inplace = True)

# 'length_of_service_bins'
targetc = KFoldTargetEncoderTrain('length_of_service_bins', 'is_promoted', n_fold = 5)
trans_Train_OS = targetc.fit_transform(trans_Train_OS)
## trans_Train_OS.drop(['length_of_service_bins'], axis = 1, inplace = True)

trans_Train_OS.head(1)
# Before and After Feature Engineering Comparison
print("Shape of Train Before FE : {}".format(train_OS.shape))
print(f"Shape of Train Before FE : {trans_Train_OS.shape}")
print("No of Features added during FE : ", (trans_Train_OS.shape[1] - train_OS.shape[1]))
# Removing Redundant Feature 'Employee ID' and other un-encoded categorical features from Trans-Train-OS
trans_Train_OS.drop(['employee_id',                                  # Unique Identifier for Every Feature
                     'department', 'region', 'recruitment_channel',  # Un-Encoded Categorical Feature
                     'age_bins', 'length_of_service_bins'], 
                    axis = 1, inplace = True)
# Visualising the Columns of Trans-Train-OS [Feature Engineered DataFrame]
trans_Train_OS.columns
# Spliting into Dependent and Independent Features Vector
x = trans_Train_OS.drop(['is_promoted'], axis = 1)
y = trans_Train_OS['is_promoted'].values.reshape(-1, 1)
# Visualising Independent Vector Information
x.info()
# Segregating into Categorical and Continuous Features ->
cat = ["education", "gender", "no_of_trainings", "previous_year_rating", "KPIs_met >80%", "awards_won?", 
       "is_promoted", "KPI_n_Award", "trainings>4?", "good_overall_performance?",
       "tgt_recruitment_channel", "tgt_region", "tgt_department", "tgt_age_bins", "tgt_length_of_service_bins",
       "mean_kpi_by_dept", "mean_training_by_dept", "mean_rating_by_dept"]

cont = ["age", "length_of_service", "avg_training_score", "age_boxcox", "length_of_service_boxcox", "cummulative_train'_score", 
        "dept_rating_mean_ratio"]

print(f"{len(cat) + len(cont)} & {trans_Train_OS.shape[1]}")
# Fitting Variance Threshold to our Dataframe
const_thres = VarianceThreshold(threshold = 0).fit(trans_Train_OS)
len(trans_Train_OS[trans_Train_OS.select_dtypes([np.number]).columns].columns[const_thres.get_support()])
# Printing out Constant Columns if any
constant_columns = [col for col in trans_Train_OS.columns if col not in trans_Train_OS.columns[const_thres.get_support()]]
constant_columns
# Finding out Quasi Constant Features
quasiDetect = VarianceThreshold(threshold = 0.01)  
quasiDetect.fit(trans_Train_OS)
print(f"No. of Non Quasi Constant Features : {len(trans_Train_OS.columns[quasiDetect.get_support()])}")
print(f"Quasi-Constant Features : {[col for col in trans_Train_OS.columns if col not in trans_Train_OS.columns[quasiDetect.get_support()]]}")
# The features we would be dropping are 'trainings>4?' and 'mean_kpi_by_dept' as 'age_boxcox' is a multilabel target encoded class. 
trans_Train_OS.drop(['trainings>4?', 'mean_kpi_by_dept'], axis = 1, inplace = True)
# Updating Categorical Columns List
cat = [e for e in cat if e not in ['trainings>4?', 'mean_kpi_by_dept']]
cat
# To Find out Repeatative Features (Duplicate)
train_enc =  pd.DataFrame(index = trans_Train_OS.index)

dup_cols = {}

for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
            dup_cols[c2] = c1
dup_cols
# Correlation using heatmap - Continuous Variable & Target Variable - Pearson Correlation
hm = trans_Train_OS[cont + ["is_promoted"]].corr().where(np.tril(np.ones(trans_Train_OS[cont + ["is_promoted"]].corr().shape)).astype(np.bool)) # to delete the upper triangle
plot = sns.heatmap(hm, annot = True, cmap = "YlGnBu")
plt.setp(plot.get_xticklabels(), rotation = 90)
plt.gcf().set_size_inches(15, 8)
# Covariance using Heatmap - Continuous Variable & Target Variable
trans_Train_x = trans_Train_OS[cont + ["is_promoted"]].round(decimals = 2)
covMatrix = pd.DataFrame.cov(trans_Train_x[cont + ["is_promoted"]])
sns.heatmap(covMatrix, annot = True, fmt = 'g')
plt.gcf().set_size_inches(15, 8)
del trans_Train_x  # Deleting Redundant Object
# Correlation using heatmap - Categorical Variable & Target Variable - Cramer's V Correlation

def cramers_V(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames = None, colnames = None)) 
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape) - 1 
    return (stat / (obs * mini))

rows = []
data_encoded = trans_Train_OS.copy()
data_encoded = data_encoded[cat]
for var1 in data_encoded:
  col = []
  for var2 in data_encoded :
    cramers = cramers_V(data_encoded[var1], data_encoded[var2]) # Cramer's V test
    col.append(round(cramers, 2))  
  rows.append(col)
  
cramers_results = np.array(rows)      # Results of Cramer's V Test

# Cramer's V Test Transformed Dataframe
df = pd.DataFrame(cramers_results, columns = data_encoded.columns, index = data_encoded.columns)

# HeatMap  Visualisation
mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plot = sns.heatmap(df, mask = mask, vmin = 0., vmax = 1, 
                   annot = True, cmap = "YlGnBu", square = True)
plt.setp(plot.get_xticklabels(), rotation = 90)
plt.gcf().set_size_inches(15, 8)
# Correlation using heatmap - Continuous Variable & Target Variable - Overall Correlation.
hm = trans_Train_OS.corr().where(np.tril(np.ones(trans_Train_OS.corr().shape)).astype(np.bool)) # to delete the upper triangle
plot = sns.heatmap(hm, annot = True, cmap = "YlGnBu")
plt.setp(plot.get_xticklabels(), rotation = 90)
plt.gcf().set_size_inches(15, 8)
# Removing Highly Correlated Independent Features [corr > 0.6]
# Also, Correlation has been checked between target variable in case they're similar
trans_Train_OS.drop(["KPI_n_Award", "tgt_length_of_service_bins", "mean_training_by_dept",                  # Categorical Features
                     "mean_rating_by_dept", "tgt_department", "good_overall_performance?",
                     "age", "length_of_service", "length_of_service_boxcox", "cummulative_train'_score",    # Continuous Features
                     "dept_rating_mean_ratio"       
                     ], axis = 1, inplace = True)
trans_Train_OS.head(1)
# Dropping Features that Mean [Target Encoded and Bin, both present] the Same -> 'tgt_age_bins' as it is having the lowest correlation wrt target variable
trans_Train_OS.drop(['tgt_age_bins'], axis = 1, inplace = True)
# Updating Features List
cat = [e for e in cat if e not in ["KPI_n_Award", "tgt_length_of_service_bins", "mean_training_by_dept",
                                   "mean_rating_by_dept", "tgt_department", "good_overall_performance?", 
                                   "tgt_age_bins"]]
cont = [e for e in cont if e not in ["age", "length_of_service", "length_of_service_boxcox", "cummulative_train'_score",
                                     "dept_rating_mean_ratio"]]
# Printing the Remaining Column Names
print(f"Categorical : {cat} \nContinuous : {cont}")
print(f"No. of Categorical Features Left : {len(cat)} \nNo. of Continuous Features Left : {len(cont)}")
print("No. of Features in Total Now : {}".format(trans_Train_OS.shape[1]))
# Not Done as Drop Done Manually
# Automated Drop - Correlated Features
'''
col_corr = set() # Set of all the names of deleted columns
corr_matrix = trans_Train[cont + ["is_promoted"]].corr()
for i in range(len(corr_matrix.columns)):
  for j in range(i):
    if corr_matrix.iloc[i, j] >= threshold and (corr_matrix.columns[j] not in col_corr):
      colname = corr_matrix.columns[i] # getting the name of column
      col_corr.add(colname)
      #if colname in dataset.columns:
      #del dataset[colname] # deleting the column from the dataset
'''
# Visualising Head of the Remaining DataFrame
trans_Train_OS.head(1)
# Not Done
## Anova
'''
import statsmodels.api as sm
from statsmodels.formula.api import ols

anova = ols('SepalLengthCm ~ C(Species) + SepalWidthCm + PetalLengthCm + PetalWidthCm', data = trans_Train).fit()
sm.stats.anova_lm(anova, typ = 2)
anova.summary()
'''
# Dividing Trans-Train-OS into Independent and Dependent Features
x = trans_Train_OS.drop(['is_promoted'], axis = 1)
y = trans_Train_OS['is_promoted'].values.reshape(-1, 1)
x.shape, y.shape
# IGNORE -> Not Done as it takes a huge amount of time.
'''
efs = ExhaustiveFeatureSelector(rfc(), 
           min_features = 4,
           max_features = 10, 
           scoring = 'f1',
           cv = 5)

# fit the object to the training data.
efs = efs.fit(x, y)

# print the selected features.
selected_features = x.columns[list(efs.k_feature_idx_)]
print(selected_features)
'''
# Sequential Feature Selector Object and Configuring the Parameters -> Forward Elimination
sfs_fw = SFS(rfc(random_state = 0),
          k_features = 10,
          forward = True, 
          floating = False,
          verbose = 2,
          scoring = 'f1',
          cv = 5,
          n_jobs = -1)

# Fit the object to the Training Data.
sfs_fw.fit(x, y)
# Print the Selected Features.
selected_features = x.columns[list(sfs_fw.k_feature_idx_)]
print(selected_features)

# Print the Final Prediction Score.
print(sfs_fw.k_score_)
# IGNORE -> Not now, after all the methods are been evaluated
'''
# Transform to the newly Selected Features.
x_sfs = sfs.transform(x)
'''
# Sequential Feature Selector Object and Configuring the Parameters -> Backward Elimination
sfs_bw = SFS(rfc(random_state = 0, n_jobs = -1),
          k_features = 10,
          forward = False, 
          floating = False,
          verbose = 2,
          scoring = 'f1',
          cv = 5,
          n_jobs = -1)

# Fit the object to the Training Data.
sfs_bw.fit(x.values, y)
# Print the Selected Features.
selected_features_BW = x.columns[list(sfs_bw.k_feature_idx_)]
print(selected_features_BW)

# Print the Final Prediction Score.
print(sfs_bw.k_score_)
# IGNORE -> Not now, after all the methods are been evaluated
'''
# Transform to the newly Selected Features.
x_sfs = sfs.transform(x)
'''
# Fitting Random Forest Algorithm into our dataset
rfc_wr = rfc(random_state = 0, n_jobs = -1)
rfc_wr.fit(x, y)

# Visualising the Importance of Features by Plot Graph - for Dummy Train
features_Imp_wr = pd.Series(rfc_wr.feature_importances_, index = x.columns)
features_Imp_wr.nlargest(10).plot(kind = 'barh')
plt.show()
# Show the Whole List
features_Imp_wr * 100
# The "f1" scoring is proportional to the number of correct classifications per class
rf_r = rfc(random_state = 0, n_jobs = -1) 
rfecv = RFECV(estimator = rf_r, step = 1, cv = 5, scoring = 'f1')   # 5-fold cross-validation
rfecv = rfecv.fit(x, y)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x.columns[rfecv.support_])
# Using SelectkBest Method
  
# Applying SelectKBest to extract top 10 best features
best_Features = SelectKBest(score_func = chi2, k = 10)    # using mectric of chi-square
fit = best_Features.fit(x, y)
df_Scores = pd.DataFrame(fit.scores_)
df_Columns = pd.DataFrame(x.columns)

# Concat two dataframes for better visualization 
feature_Scores_skb = pd.concat([df_Columns, df_Scores], axis = 1)
feature_Scores_skb.columns = ['Columns', 'Score']
print(feature_Scores_skb)
# Displaying the Score of Best 15 Features
print(feature_Scores_skb.nlargest(15, 'Score'))
# Fiting Dummy set into Extra Trees
model = ExtraTreesClassifier()
model.fit(x, y)
# Visualising the Importance of Features by Plot Graph
feat_importances = pd.Series(model.feature_importances_, index = x.columns)
feat_importances.nlargest(10).plot(kind ='barh')
plt.show()
feat_importances = pd.Series(model.feature_importances_, index = x.columns) * 100
feat_importances
# Important Features after Feature Selection according to the Algorithms
print(f"*** Results from Filter Methods *** \n{x.columns}\n")                         # Filter Method
print(f"*** Results from Wrapper Methods *** \n{selected_features_BW}\n")             # Backward Selection Method
print(f"*** Results from Embedded Methods*** \n{x.columns[rfecv.support_]}\n")        # Random Forest CV Method
# Common Features in both Wrapper and Embedded Methods
set_A = set(selected_features_BW)        # Wrapper 
set_B = set(x.columns[rfecv.support_])   # Embedded
set_A & set_B                            # Common Entries
# Creating a list for the Final List of Features
f_features_OS = ['KPIs_met >80%', 'age_boxcox', 'avg_training_score', 'awards_won?', 'education',
                 'gender', 'no_of_trainings', 'previous_year_rating', 'tgt_recruitment_channel', 'tgt_region']

# Printing the Final List of Features
print(f"Printing the Final List of Features : {f_features_OS}")
# Taking in Only the Important Independent Features
x_dim = x.copy()         # Saving the Dataframe for Dimensionality Reducing Techniques.
x_OS = x[f_features_OS]
x_OS.head(1)
# Shape of the Independent Features Set
x_OS.shape
# Shape of the Dependent Feature Set
y_OS = y
y_OS.shape
# Pickling Over-Sampled Plain DataFrame
x_OS['is_promoted'] = y_OS
x_OS.to_pickle("OS_Plain.pkl")
# creating a copy
x_scale = x_dim.copy()
# Standard Scaling all the Features for Dimensionality Reduction
sc_x = StandardScaler()
x_scale = sc_x.fit_transform(x_scale)
x_scale
# Normal - PCA
pca = PCA(n_components = 5, random_state = 0)
pca_OS = pca.fit_transform(x_scale)
explained_variance = pca.explained_variance_ratio_
print(f"Variance Explained per Principal Component : {explained_variance}")
# Visualising PCA Segregation
plt.figure(figsize = (10, 5))
plt.scatter(pca_OS[:,0], pca_OS[:,1], c = y, s = 0.5)
# IGNORE -> Takes up More than 10GB+ of RAM and crashes the kernel
'''
# Kernel - PCA
kpca = kp(n_components = 5, kernel = 'rbf')
x_kpca = kpca.fit_transform(x_scale)
explained_variance = kpca.explained_variance_ratio_
'''
# t-SNE
tsne_OS = TSNE(random_state = 0).fit_transform(pca_OS)
# tsne = TSNE(n_components = 5, init = 'pca', random_state = 0, learning_rate = 150, perplexity)
# Visualising t-SNE Segregation
plt.figure(figsize = (10, 5))
plt.scatter(tsne_OS[:,0], tsne_OS[:,1], c = y, s = 0.5)
# Pickling Over-Sampled Plain DataFrame
tsne_OS = pd.DataFrame(tsne_OS)
tsne_OS['target'] = y
tsne_OS.to_pickle("OS_tsne.pkl")
# Deleting off redundant objects/variables - Memory Efficient Techniques
del objectColumns, colNames, sm, x_os_train, y_os_train, train_OS, trans_Train_OS, fig, boxc_Age, boxc_LOS, xyz, arr, quantile_1, quantile_3, iqr, lower_Bound, upper_Bound, outliers, discretizer, temp, targetc, cat, cont, dup_cols, train_enc, hm, plot, covMatrix, mask, sfs_fw, selected_features, sfs_bw, selected_features_BW, rfc_wr, rf_r, rfecv, best_Features, fit, df_Scores, df_Columns, feature_Scores_skb, model, feat_importances, set_A, set_B, x_scale, sc_x, pca, explained_variance, x_dim
# visualising to know the features types and names
train.head(1)
# visualising to know the features types and names
dummy_Train.head(1)
# creating another dummy dataframe to maintain the integrity of the main dataframe
trans_Train = train.copy()
trans_Train.head(1)
# Box Plot to find more information about it's Quartiles and also detect Outliers
'''
plt.figure(figsize = (15, 5))
plt.title('Age Distribution Spread')
sns.boxplot(x = trans_Train.age)
print("1st Quartile : {}".format(trans_Train.age.quantile(0.25)))
print("3rd Quartile : {}".format(trans_Train.age.quantile(0.75)))
'''
# For Age
fig = px.box(train, x = "is_promoted", y = "age", points = "outliers")
fig.show()

# For Length of Service
fig = px.box(train, x = "is_promoted", y = "length_of_service", points = "outliers")
fig.show()
# Applying Boxcox Transformation and Checking for Result
boxc_Age = stats.boxcox(trans_Train['age'])[0]
boxc_LOS = stats.boxcox(trans_Train['length_of_service'])[0]

# Creating a Copy for Testing
xyz = trans_Train.copy()

# Loading the Results
xyz['age'] = boxc_Age
xyz['length_of_service'] = boxc_LOS
# For Age
fig = px.box(xyz, x = "is_promoted", y = "age", points = "outliers")
fig.show()

# For Length of Service
fig = px.box(xyz, x = "is_promoted", y = "length_of_service", points = "outliers")
fig.show()
print("***** For LOS *****")
# For IQR Analysis, we need to sort our feature vector first
arr = np.array(sorted(xyz.length_of_service))

# Defining the Quartile Ranges
quantile_1, quantile_3 = np.percentile(arr, [25, 75])
print(f"1st Quartile (25th Percentile) of Age Feature : {quantile_1} \n3rd Quartile (75th Percentile) of Age Feature : {quantile_3}")

# Defining the IQR
iqr = quantile_3 - quantile_1
print(f"IQR = {iqr}")

# Finding out the Lower Bound Value and the Higher Bound Value
lower_Bound = quantile_1 - (1.5 * iqr) 
upper_Bound = quantile_3 + (1.5 * iqr)
print(f"Lower Bound Value : {lower_Bound} \nUpper Bound Value : {upper_Bound}")

# Any number below the Lower Bound or above the Upper Bound would be considered as an Outlier
outliers = xyz.length_of_service[(xyz.length_of_service < lower_Bound) | (xyz.length_of_service > upper_Bound)]
print(f"Number of Outliers : {len(outliers)}")
## print(f"The Outliers are : {sorted(outliers.unique())}")

# % of entries which are outliers
print(f"% of Records which are been considered Outliers : {round(len(outliers)/xyz.shape[0] * 100, 2)} %")

print("*" * 10)

print("***** For Age *****")
# For IQR Analysis, we need to sort our feature vector first
arr = np.array(sorted(xyz.age))

# Defining the Quartile Ranges
quantile_1, quantile_3 = np.percentile(arr, [25, 75])
print(f"1st Quartile (25th Percentile) of Age Feature : {quantile_1} \n3rd Quartile (75th Percentile) of Age Feature : {quantile_3}")

# Defining the IQR
iqr = quantile_3 - quantile_1
print(f"IQR = {iqr}")

# Finding out the Lower Bound Value and the Higher Bound Value
lower_Bound = quantile_1 - (1.5 * iqr) 
upper_Bound = quantile_3 + (1.5 * iqr)
print(f"Lower Bound Value : {lower_Bound} \nUpper Bound Value : {upper_Bound}")

# Any number below the Lower Bound or above the Upper Bound would be considered as an Outlier
outliers = xyz.age[(xyz.age < lower_Bound) | (xyz.age > upper_Bound)]
print(f"Number of Outliers : {len(outliers)}")
## print(f"The Outliers are : {sorted(outliers.unique())}")

# % of entries which are outliers
print(f"% of Records which are been considered Outliers : {round(len(outliers)/xyz.shape[0] * 100, 2)} %")
# Adding these Transformed Features in the Dataset
trans_Train['age_boxcox'] = boxc_Age
trans_Train['length_of_service_boxcox'] = boxc_LOS

trans_Train.head(2)
# Equal Frequency Discretization
discretizer = EqualFrequencyDiscretiser(q = 10, variables = ['age', 'length_of_service'], return_object = True, return_boundaries = True)
discretizer
# fit the discretization transformer
discretizer.fit(trans_Train)

# transform the data
train_t = discretizer.transform(trans_Train)

# Visualising the Bins
print(f"Age : {list(train_t.age.unique())} \nLOS : {list(train_t.length_of_service.unique())}")
# Adding Binned Categories to the Dataframe
trans_Train['age_bins'] = train_t.age
trans_Train['length_of_service_bins'] = train_t.length_of_service

# Deleting Redundant Dataframe
del train_t

# Visualising the 1st 5 records of transformed dataframe
trans_Train.head(5)
# IGNORE -> Will do Data Leakage and not insightful.
# Checked at Last

##  Probability of Promotion Per Dept.
'''
temp = pd.DataFrame()

# trans_Train['prob_per_dept'] = train.groupby(['department'])['is_promoted'].mean()
# trans_Train.prob_per_dept.unique()

temp['prob_per_dept'] = train.groupby(['department'])['is_promoted'].apply(lambda x : x.mean())
trans_Train = pd.merge(trans_Train, temp, on = ['department'], how = 'left')
trans_Train['prob_per_dept'].fillna(np.median(temp['prob_per_dept']), inplace = True)
trans_Train['prob_per_dept'].unique()
'''
# IGNORE -> Will do Data Leakage and not insightful.
# Checked at Last

##  # Probability of Promotion Per Dept Per Region.
'''
temp = pd.DataFrame()

# temp['prob_per_dept_per_region'] = train.groupby(['region', 'department'])['is_promoted'].mean()
# temp.head(10)

temp['prob_per_dept_per_region'] = train.groupby(['region', 'department'])['is_promoted'].mean()
trans_Train = pd.merge(trans_Train, temp, on = ['region', 'department'], how = 'left')
trans_Train['prob_per_dept_per_region'].fillna(np.median(temp['prob_per_dept_per_region']), inplace = True)
# trans_Train['prob_per_dept_per_region'].unique()
temp.head(10)
'''
'''
# K-Fold Cross Validified Promotion Per Dept Per Region.
temp = pd.DataFrame()
temp['prob_per_dept_per_region'] = np.nan

kf = KFold(n_splits = 5, shuffle = False, random_state = 0)
for train_in, val_in in kf.split(trans_Train):
  x_tr, x_val = trans_Train.iloc[train_in], trans_Train.iloc[val_in]
  trans_Train.loc[trans_Train.index[val_in], 'prob_per_dept_per_region'] = x_val['prob_per_dept_per_region'].map(x_tr.groupby(['region', 'department'])['is_promoted'].mean())

trans_Train.prob_per_dept_per_region.fillna(0, inplace = True)
temp.head(10)
'''
# Cummulative Training Score
temp = pd.DataFrame()

# temp = train.groupby(['recruitment_channel'])['no_of_trainings'].mean()
trans_Train["cummulative_train'_score"] = trans_Train.no_of_trainings * trans_Train.avg_training_score

# Visualising the 1st 5 records after transformation
trans_Train.head(5)
# Checking out for the Region Counts
trans_Train.region.value_counts()
# Checking out where % of employees in a particular region to country is more than 1%
372/train.shape[0] * 100        # Region_24 & Below -> regions = [24,12,9,21,3,34,33]
# Groupby Function to Club Regions and Promotions
temp = trans_Train.groupby(['region', 'is_promoted'])['is_promoted'].count().sort_values(ascending = False).unstack()    #.apply(lambda r : r/r.sum())
# temp.tail(10)
pd.crosstab(trans_Train['region'], trans_Train.is_promoted).apply(lambda r: r/r.sum(), axis=1)
# IGNORE -> Will do Data Leakage and not insightful.
# Checked at Last

## Promotion Ratio Per Region
'''
# K-Fold Cross Validified Ratio
kf = KFold(n_splits = 5, shuffle = False, random_state = 0)
temp['prom_ratio_per_region_XXX'] = np.nan

for train_in, val_in in kf.split(temp):
  x_tr, x_val = temp.iloc[train_in], temp.iloc[val_in]
  temp.loc[temp.index[val_in], 'prom_ratio_per_region'] = temp.iloc[:,1] / ((temp.iloc[:,0] + temp.iloc[:,1]) * 100)

temp.fillna(0, axis = 1, inplace = True)
#temp.sort_values(by=['prom_ratio_per_region'], inplace = True, ascending = True)
'''
'''
# Assigning Values
temp['prom_ratio_per_region'] = temp.iloc[:,1] / ((temp.iloc[:,0] + temp.iloc[:,1]) * 100)
temp.fillna(0, axis = 1, inplace = True)
temp.sort_values(by=['prom_ratio_per_region'], inplace = True, ascending = True)
temp.tail(5)
'''
# KPI and Award Concatenation
trans_Train["KPI_n_Award"] = np.where(((trans_Train["KPIs_met >80%"] == 1) & (trans_Train["awards_won?"] == 1)), 1, 0)
trans_Train.head(3)
# Gender - No of Training - Promotion Relation
trans_Train.groupby(['gender', 'no_of_trainings'])['is_promoted'].sum()
# Seggregating no_of_trainings > 4
trans_Train['trainings>4?'] = np.where(trans_Train.no_of_trainings > 4, 1, 0)
# IGNORE - Got changed when reshuffled the dataset
'''
# Reject Region_18 - as noone get's promoted from there
trans_Train['is_Region_18?'] = np.where(trans_Train.region == 'region_18', 1, 0)
'''
# IGNORE - Not a insightful feature
'''
# KPI Per Dept.

temp = pd.DataFrame()
temp['KPI_per_dept'] = train.groupby(['department', 'KPIs_met >80%'])['KPIs_met >80%'].count()
#trans_Train = pd.merge(trans_Train, temp, on = ['department'], how = 'left')
#trans_Train['prob_per_dept'].fillna(np.median(temp['prob_per_dept']), inplace = True)
#trans_Train['prob_per_dept'].unique()
#temp
pd.crosstab([train['KPIs_met >80%'], train.is_promoted], train.department, margins = True).style.background_gradient(cmap = 'summer_r')
'''
# Categorise Employees having good overall performance
trans_Train['good_overall_performance?'] = np.where((trans_Train.previous_year_rating >= 3) & (trans_Train['awards_won?'] == 1) & 
                                                   (trans_Train.avg_training_score >= trans_Train.avg_training_score.quantile(0.25)), 1, 0)
# IGNORE - Not a insightful feature
'''
temp['KPI_per_dept'] = train.groupby(['department', 'KPIs_met >80%'])['KPIs_met >80%'].agg(['count', 'mean'])
temp
'''
'''
train.groupby(['department', 'KPIs_met >80%']).count().unstack()
# train.groupby(['department', 'KPIs_met >80%'])['KPIs_met >80%'].agg(['mean', 'count'])
train.groupby(['department', 'KPIs_met >80%']).size().reset_index(name='counts')
'''
# Mean KPI by Department
trans_Train['mean_kpi_by_dept'] = trans_Train['department'].map(trans_Train.groupby('department')['KPIs_met >80%'].mean())
# Mean Training Score by Department
trans_Train['mean_training_by_dept'] = trans_Train['department'].map(trans_Train.groupby('department')['avg_training_score'].mean())
# Mean Rating by Department
trans_Train['mean_rating_by_dept'] = trans_Train['department'].map(trans_Train.groupby('department')['previous_year_rating'].mean())
# Prev Years' Rating by Department
trans_Train['dept_rating_mean_ratio'] = trans_Train['previous_year_rating'] / trans_Train['mean_rating_by_dept']
# Visualizing the First Record of Transformed Train
trans_Train.head(1)
# K-Fold Target Encoding 'recruitment_channel', 'region' and 'department'

# 'recruitment_channel'
targetc = KFoldTargetEncoderTrain('recruitment_channel', 'is_promoted', n_fold = 5)
trans_Train = targetc.fit_transform(trans_Train)
# trans_Train.drop(['recruitment_channel'], axis = 1, inplace = True)

# 'region'
targetc = KFoldTargetEncoderTrain('region', 'is_promoted', n_fold = 5)
trans_Train = targetc.fit_transform(trans_Train)
# trans_Train.drop(['region'], axis = 1, inplace = True)

# 'department'
targetc = KFoldTargetEncoderTrain('department', 'is_promoted', n_fold = 5)
trans_Train = targetc.fit_transform(trans_Train)
# trans_Train.drop(['department'], axis = 1, inplace = True)

trans_Train.head(1)
# Label Encoding 'Gender'
l = LabelEncoder()
trans_Train.loc[:, 'gender'] = l.fit_transform(train.loc[:, 'gender'])
# Encoding 'Education'

# Mannualy Assigning Weight and Encoding 'ordinal' Feature 'Education'
edu_enc = {"Below Secondary" : 0, "Others" : 1, "Bachelor's": 2, "Master's & above": 3}
trans_Train['education'] = trans_Train['education'].map(edu_enc)
trans_Train.education.unique()
trans_Train.head(1)
# K-Fold Target Encoding 'age_bin', 'length_of_service_bins'

# 'age_bin'
targetc = KFoldTargetEncoderTrain('age_bins', 'is_promoted', n_fold = 5)
trans_Train = targetc.fit_transform(trans_Train)
# trans_Train.drop(['age_bin'], axis = 1, inplace = True)

# 'length_of_service_bins'
targetc = KFoldTargetEncoderTrain('length_of_service_bins', 'is_promoted', n_fold = 5)
trans_Train = targetc.fit_transform(trans_Train)
# trans_Train.drop(['length_of_service_bins'], axis = 1, inplace = True)

trans_Train.head(1)
# Before and After Feature Engineering Comparison
print("Shape of Train Before FE : {}".format(train.shape))
print(f"Shape of Train Before FE : {trans_Train.shape}")
print("No of features added during FE : ", (trans_Train.shape[1] - train.shape[1]))
# Removing Redundant Feature 'Employee ID' and other un-encoded categorical features - Trans-Train
trans_Train.drop(['employee_id',                           # Unique Identifier for Every Feature
                  'department', 'region', 'recruitment_channel',  # Un-Encoded Categorical Feature
                  'age_bins', 'length_of_service_bins'], 
                 axis = 1, inplace = True)
# Visualising the Columns of Trans-Train [Feature Engineered DataFrame]
trans_Train.columns
# Visualising the Columns of Dummy-Train [Plain Raw Encoded DataFrame]
dummy_Train.drop(['employee_id'],           # Unique Identifier for Every Feature
                 axis = 1, inplace = True)   
dummy_Train.columns
# Spliting into Dependent and Independent Features Vector
x = trans_Train.drop(['is_promoted'], axis = 1)
y = trans_Train['is_promoted'].values.reshape(-1, 1)

x1 = dummy_Train.drop(['is_promoted'], axis = 1)
y1 = dummy_Train['is_promoted'].values.reshape(-1, 1)
trans_Train.info()
# Segregating into Categorical and Continuous Features ->
cat = ["education", "gender", "no_of_trainings", "previous_year_rating", "KPIs_met >80%", "awards_won?", 
       "is_promoted", "KPI_n_Award", "trainings>4?", "good_overall_performance?",
       "tgt_recruitment_channel", "tgt_region", "tgt_department", "tgt_age_bins", "tgt_length_of_service_bins",
       "mean_kpi_by_dept", "mean_training_by_dept", "mean_rating_by_dept"]

cont = ["age", "length_of_service", "avg_training_score", "age_boxcox", "length_of_service_boxcox", "cummulative_train'_score", 
        "dept_rating_mean_ratio"]

print(f"{len(cat) + len(cont)} & {trans_Train.shape[1]}")
# Fitting Variance Threshold to our Dataframe
const_thres = VarianceThreshold(threshold = 0).fit(trans_Train)
len(trans_Train[trans_Train.select_dtypes([np.number]).columns].columns[const_thres.get_support()])
# Printing out Constant Columns if any
constant_columns = [col for col in trans_Train.columns if col not in trans_Train.columns[const_thres.get_support()]]
constant_columns
# Finding out Quasi Constant Features
quasiDetect = VarianceThreshold(threshold = 0.01)  
quasiDetect.fit(trans_Train)
print(f"No. of Non Quasi Constant Features : {len(trans_Train.columns[quasiDetect.get_support()])}")
print(f"Quasi-Constant Features : {[col for col in trans_Train.columns if col not in trans_Train.columns[quasiDetect.get_support()]]}")
# The features we would be dropping are only the binary ones -> 'trainings>4?', 'mean_kpi_by_dept', 'age_boxcox' as others are multilabel encoded classes 
trans_Train.drop(['trainings>4?', 'mean_kpi_by_dept', 'age_boxcox'], axis = 1, inplace = True)
# Updating Categorical Features List
cat = [e for e in cat if e not in ['trainings>4?', 'mean_kpi_by_dept']]
cat
# Updating Continuous Features List
cont = [e for e in cont if e not in ['age_boxcox']]
cont
# To Find out Repeatative Features (Duplicate)
train_enc =  pd.DataFrame(index = trans_Train.index)

dup_cols = {}

for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
            dup_cols[c2] = c1
dup_cols
# Correlation using heatmap - Continuous Variable & Target Variable - Pearson Correlation
hm = trans_Train[cont + ["is_promoted"]].corr().where(np.tril(np.ones(trans_Train[cont + ["is_promoted"]].corr().shape)).astype(np.bool)) # to delete the upper triangle
plot = sns.heatmap(hm, annot = True, cmap = "YlGnBu")
plt.setp(plot.get_xticklabels(), rotation = 90)
plt.gcf().set_size_inches(15, 8)
# Covariance using Heatmap - Continuous Variable & Target Variable
trans_Train_x = trans_Train[cont + ["is_promoted"]].round(decimals = 2)
covMatrix = pd.DataFrame.cov(trans_Train_x[cont + ["is_promoted"]])
sns.heatmap(covMatrix, annot = True, fmt = 'g')
plt.gcf().set_size_inches(15, 8)
# Correlation using heatmap - Categorical Variable & Target Variable - Cramer's V Correlation

def cramers_V(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames = None, colnames = None)) 
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape) - 1 
    return (stat / (obs * mini))

rows = []
data_encoded = trans_Train.copy()
data_encoded = data_encoded[cat]
for var1 in data_encoded:
  col = []
  for var2 in data_encoded :
    cramers = cramers_V(data_encoded[var1], data_encoded[var2]) # Cramer's V test
    col.append(round(cramers, 2))  
  rows.append(col)
  
cramers_results = np.array(rows)

# Cramer's V Test Transformed Dataframe
df = pd.DataFrame(cramers_results, columns = data_encoded.columns, index =data_encoded.columns)

# HeatMap  Visualisation
mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plot = sns.heatmap(df, mask = mask, vmin = 0., vmax = 1, 
                   annot = True, cmap = "YlGnBu", square = True)
plt.setp(plot.get_xticklabels(), rotation = 90)
plt.gcf().set_size_inches(15, 8)
# Correlation using heatmap - Continuous Variable & Target Variable - Overall Correlation.
hm = trans_Train.corr().where(np.tril(np.ones(trans_Train.corr().shape)).astype(np.bool)) # to delete the upper triangle
plot = sns.heatmap(hm, annot = True, cmap = "YlGnBu")
plt.setp(plot.get_xticklabels(), rotation = 90)
plt.gcf().set_size_inches(15, 8)
trans_Train.columns
# Removing Highly Correlated Independent Features [corr > 0.6]
# Also, Correlation has been checked between target variable in case they're similar
trans_Train.drop(["awards_won?", "tgt_department", "KPI_n_Award", "mean_training_by_dept", "mean_rating_by_dept",     # Categorical Features
                  "age", "length_of_service", "no_of_trainings",                                                     # Continuous Features
                  ], axis = 1, inplace = True)

# Updating Features List
cat = [e for e in cat if e not in ["awards_won?", "tgt_department", "KPI_n_Award", "mean_training_by_dept", "mean_rating_by_dept", "no_of_trainings"]]
cont = [e for e in cont if e not in ["age", "length_of_service"]]
# Dropping Features that Mean [Target Encoded and Bin, both present] the Same -> 'length_of_service_boxcox' as it is having the lowest correlation wrt target variable
trans_Train.drop(["length_of_service_boxcox"], axis = 1, inplace = True)
# Updating Features List
cont = [e for e in cont if e not in ["length_of_service_boxcox"]]
# Printing the Remaining Column Names
print(f"Categorical : {cat} \nContinuous : {cont}")
print(f"No. of Categorical Features Left : {len(cat)} \nNo. of Continuous Features Left : {len(cont)}")
print("No. of Features in Total Now : {}".format(trans_Train.shape[1]))
# Not Done as Drop Done Manually
# Automated Drop - Correlated Features
'''
col_corr = set() # Set of all the names of deleted columns
corr_matrix = trans_Train[cont + ["is_promoted"]].corr()
for i in range(len(corr_matrix.columns)):
  for j in range(i):
    if corr_matrix.iloc[i, j] >= threshold and (corr_matrix.columns[j] not in col_corr):
      colname = corr_matrix.columns[i] # getting the name of column
      col_corr.add(colname)
      #if colname in dataset.columns:
      #del dataset[colname] # deleting the column from the dataset
'''
# Visualising Head of the Remaining DataFrame
trans_Train.head(1)
# Not Done
## Anova
'''
import statsmodels.api as sm
from statsmodels.formula.api import ols

anova = ols('SepalLengthCm ~ C(Species) + SepalWidthCm + PetalLengthCm + PetalWidthCm', data = trans_Train).fit()
sm.stats.anova_lm(anova, typ = 2)
anova.summary()
'''
# Dividing Trans-Train-OS into Independent and Dependent Features
x = trans_Train.drop(['is_promoted'], axis = 1)
y = trans_Train['is_promoted']
x.shape, y.shape
class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
im_weight = dict(enumerate(class_weights))
im_weight
# IGNORE -> Not Done as it takes a huge amount of time.
'''
efs = ExhaustiveFeatureSelector(rfc(), 
           min_features = 4,
           max_features = 10, 
           scoring = 'f1',
           cv = 5)

# fit the object to the training data.
efs = efs.fit(x, y)

# print the selected features.
selected_features = x.columns[list(efs.k_feature_idx_)]
print(selected_features)
'''
# Sequential Feature Selector Object and Configuring the Parameters -> Forward Elimination
sfs_fw = SFS(rfc(class_weight = im_weight, random_state = 0, n_jobs = -1),
          k_features = 10,
          forward = True, 
          floating = False,
          verbose = 2,
          scoring = 'f1',
          cv = 5,
          n_jobs = -1)

# Fit the object to the Training Data.
sfs_fw.fit(x, y)
# Print the Selected Features.
selected_features = x.columns[list(sfs_fw.k_feature_idx_)]
print(selected_features)

# Print the Final Prediction Score.
print(sfs_fw.k_score_)
# IGNORE -> Not now, after all the methods are been evaluated
'''
# Transform to the newly Selected Features.
x_sfs = sfs.transform(x)
'''
# Sequential Feature Selector Object and Configuring the Parameters -> Backward Elimination
sfs_bw = SFS(rfc(class_weight = im_weight, random_state = 0, n_jobs = -1),
          k_features = 10,
          forward = False, 
          floating = False,
          verbose = 2,
          scoring = 'f1',
          cv = 5,
          n_jobs = -1)

# Fit the object to the Training Data.
sfs_bw.fit(x.values, y)
# Print the Selected Features.
selected_features_BW = x.columns[list(sfs_bw.k_feature_idx_)]
print(selected_features_BW)

# Print the Final Prediction Score.
print(sfs_bw.k_score_)
# IGNORE -> Not now, after all the methods are been evaluated
'''
# Transform to the newly Selected Features.
x_sfs = sfs.transform(x)
'''
# On Feature Engineered Trans-Train

# Spliting into Independent and Dependent Feature Vectors
y = trans_Train['is_promoted']                                          # dependant feature vector
x = trans_Train.drop(['is_promoted'], axis = 1)                         # independant feature vector

y = y.values.reshape(-1,1)

'''
for i in trans_Train.columns[train.dtypes == 'object']:
    x[i] = x[i].factorize()[0]
'''

# Fitting Random Forest Algorithm into our dataset
rfc_wr_1 = rfc(class_weight = im_weight, random_state = 0, n_jobs = -1)
rfc_wr_1.fit(x, y)
# Visualising the Importance of Features by Plot Graph
features_Imp = pd.Series(rfc_wr_1.feature_importances_, index = x.columns)
features_Imp.nlargest(10).plot(kind = 'barh')
plt.show()
# Show the Whole List
features_Imp * 100
# deleting redundant dataframe
del dummy_Train
# The "f1" scoring is proportional to the number of correct classifications per class
rf_r = rfc(class_weight = im_weight, random_state = 0, n_jobs = -1) 
rfecv = RFECV(estimator = rf_r, step = 1, cv = 5, scoring = 'f1')   # 5-fold cross-validation
rfecv = rfecv.fit(x, y)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x.columns[rfecv.support_])
# Using SelectkBest Method

'''
for i in train.columns[train.dtypes == 'object']:
    x1[i] = x1[i].factorize()[0]
'''
'''
x = trans_Train.drop(["is_promoted"], axis = 1) 
y = trans_Train.loc[:, "is_promoted"] 
'''
  
# Applying SelectKBest to extract top 10 best features
best_Features = SelectKBest(score_func = chi2, k = 10)    # using mectric of chi-square
fit = best_Features.fit(x, y)
df_Scores = pd.DataFrame(fit.scores_)
df_Columns = pd.DataFrame(x.columns)

# Concat two dataframes for better visualization 
feature_Scores = pd.concat([df_Columns, df_Scores], axis = 1)
feature_Scores.columns = ['Columns', 'Score']
print(feature_Scores)
# Displaying the Score of Best 15 Features
print(feature_Scores.nlargest(15, 'Score'))
# Fiting Dummy set into Extra Trees
model = ExtraTreesClassifier(class_weight = im_weight, random_state = 0, n_jobs = -1)
model.fit(x, y)
# Visualising the Importance of Features by Plot Graph
feat_importances = pd.Series(model.feature_importances_, index = x.columns)
feat_importances.nlargest(10).plot(kind ='barh')
plt.show()
feat_importances = pd.Series(model.feature_importances_, index = x.columns) * 100
feat_importances
# Important Features after Feature Selection according to the Algorithms
print(f"*** Results from Filter Methods *** \n{x.columns}\n")                         # Filter Method
print(f"*** Results from Wrapper Methods *** \n{selected_features_BW}\n")             # Backward Selection Method
print(f"*** Results from Embedded Methods*** \n{x.columns[rfecv.support_]}\n")        # Random Forest CV Method
# Common Features in both Wrapper and Embedded Methods
set_A = set(selected_features_BW)        # Wrapper 
set_B = set(x.columns[rfecv.support_])   # Embedded
set_A & set_B                            # Common Entries
# Creating a list for the Final List of Features
f_features = ["avg_training_score", "dept_rating_mean_ratio", "tgt_age_bins", "KPIs_met >80%", "education",
              "previous_year_rating", "tgt_recruitment_channel",
              "cummulative_train'_score"]

# Printing the Final List of Features
print(f"Printing the Final List of Features : {f_features}")
# Taking in Only the Important Independent Features
x_dim = x.copy()          # Saving the Dataframe for Dimensionality Reducing Techniques.
x_wt = x[f_features]
x_wt.head(1)
# Shape of the Independent Features Set
x_wt.shape
# Shape of the Dependent Feature Set
y_wt = y
y_wt.shape
# Pickling Over-Sampled Plain DataFrame
x_wt['is_promoted'] = y_wt
x_wt.to_pickle("WT_Plain.pkl")
# creating a copy
x_scale = x_wt.copy()
# Standard Scaling all the Features for Dimensionality Reduction
sc_x = StandardScaler()
x_scale = sc_x.fit_transform(x_scale)
x_scale
# Normal - PCA
pca = PCA(n_components = 5, random_state = 0)
pca_wt = pca.fit_transform(x_scale)
explained_variance = pca.explained_variance_ratio_
print(f"Variance Explained per Principal Component : {explained_variance}")
# Visualising PCA Segregation
plt.figure(figsize = (10, 5))
plt.scatter(pca_wt[:,0], pca_wt[:,1], c = y, s = 0.5)
# IGNORE -> Takes up More than 10GB+ of RAM and crashes the kernel
'''
# Kernel - PCA
kpca = kp(n_components = 5, kernel = 'rbf')
x_kpca = kpca.fit_transform(x_scale)
explained_variance = kpca.explained_variance_ratio_
'''
# t-SNE
tsne_wt = TSNE(random_state = 0).fit_transform(pca_wt)
# tsne = TSNE(n_components = 5, init = 'pca', random_state = 0, learning_rate = 150, perplexity)
# Visualising t-SNE Segregation
plt.figure(figsize = (10, 5))
plt.scatter(tsne_wt[:,0], tsne_wt[:,1], c = y, s = 0.5)
# Pickling Over-Sampled Plain DataFrame
tsne_wt = pd.DataFrame(tsne_wt)
tsne_wt['target'] = y
tsne_wt.to_pickle("WT_tsne.pkl")
# Deleting off redundant objects/variables - Memory Efficient Techniques
del fig, boxc_Age, boxc_LOS, xyz, arr, quantile_1, quantile_3, iqr, lower_Bound, upper_Bound, outliers, discretizer, temp, targetc, cat, cont, dup_cols, train_enc, hm, plot, covMatrix, mask, sfs_fw, selected_features, sfs_bw, selected_features_BW, rfc_wr, rf_r, rfecv, best_Features, fit, df_Scores, df_Columns, feature_Scores_skb, model, feat_importances, set_A, set_B, x_scale, sc_x, pca, explained_variance, x_dim
# Importing Oversampled Pickled DataFrames
df_os_plain = pd.read_pickle('/content/drive/My Drive/Internship/Day 6/HR Analytics/Checkpoints/OS_Plain.pkl')
df_os_tsne = pd.read_pickle('/content/drive/My Drive/Internship/Day 6/HR Analytics/Checkpoints/WT_tsne.pkl')
# Segregating the Features into Independent and Dependent Vectors
x_os_plain = df_os_plain.drop(['is_promoted'], axis = 1)
y_os_plain = df_os_plain['is_promoted'].values.reshape(-1, 1)

# Visualising and Storing the Feature Names for Oversampled Plain DataFrame
os_columns = x_os_plain.columns
os_columns
# Creating a Copy of Validation Set
val_os = val.copy()
# Data Cleaning - Handling NULL Values

## KPIs_met >80% - Filling it up with 0
val_os["KPIs_met >80%"].fillna(0, inplace = True)

## Age - Filling it up with Train's Median Value
val_os["age"].fillna(round(train['age'].median()), inplace = True)

## Average Training Score - Filling it up with Train's Median Value
val_os["avg_training_score"].fillna(round(train['avg_training_score'].median()), inplace = True) 

## Awards Won? - Filling it up with 0
val_os["awards_won?"].fillna(0, inplace = True)

## Education - Filling it up with 'Others'
val_os["education"].fillna("Others", inplace = True)

## Gender - Filling it up with 'm' as maximum of the employee will be males.
val_os["gender"].fillna("m", inplace = True)

## No. of Trainings - Filling it up with Train's Mode Value
val_os["no_of_trainings"].fillna((train['no_of_trainings'].mode()), inplace = True)

## Previous Year Rating - Filling it up with Train's Median Value
val_os["previous_year_rating"].fillna(round(train['previous_year_rating'].median()), inplace = True)

## Recruitment Channel - Filling it up with Train's Mode Value
val_os["recruitment_channel"].fillna((train['recruitment_channel'].mode()), inplace = True)

## Region - Filling it up with Train's Mode Value
val_os["region"].fillna((train['region'].mode()), inplace = True)
# Data Handling - Handling Corner Cases

## KPIs_met > 80% - Filling it up with Train's Median Value
val_os['age'] = val_os.apply(lambda x: round(train.age.median()) if (x['age'] > 100 or x['age'] < 15) else x['age'], axis = 1)

## Age - Filling it up with 0
val_os['KPIs_met >80%'] = val_os.apply(lambda x: 0 if (x['KPIs_met >80%'] not in [0, 1]) else x['KPIs_met >80%'], axis = 1)

## Average Training Score - Filling it up with Train's Median Value
val_os['avg_training_score'] = val_os.apply(lambda x: round(train.avg_training_score.median()) if (x['avg_training_score'] > 100) else x['avg_training_score'], axis = 1)

## Awards Won? - Filling it up with 0
val_os['awards_won?'] = val_os.apply(lambda x: 0 if (x['awards_won?'] not in [0, 1]) else x['awards_won?'], axis = 1)

## Education - Changing it up with Train's Mode Value
lis = list(jum.education.unique())
val_os['education'] = val_os.apply(lambda x: train.education.mode() if (x['education'] not in lis) else x['education'], axis = 1)

## Gender - Filling it up with 'm' as maximum of the employee will be males.
val_os["gender"] = val_os.apply(lambda x: m if (x['gender'] not in ['m', 'f']) else x['gender'], axis = 1)

## No. of Trainings - Filling it up with Train's Mode Value
val_os['no_of_trainings'] = val_os.apply(lambda x: train['no_of_trainings'].mode() if (x['no_of_trainings'] < 0 or x['no_of_trainings'] > 20) else x['no_of_trainings'], axis = 1)

'''
## Department - Changing it up with Train's Mode Value
lis = list(train.department.unique())
val_os['department'] = val_os.apply(lambda x: train.department.mode() if (x['department'] not in lis) else x['department'], axis = 1)
'''

## Previous Year Rating - Filling it up with Train's Median Value
val_os["previous_year_rating"] = val_os.apply(lambda x: round(train.previous_year_rating.median()) if (x['previous_year_rating'] < 0 or x['previous_year_rating'] > 5) else x['previous_year_rating'], axis = 1)

## Recruitment Channel - Changing it up with Train's Mode Value
lis = list(train.recruitment_channel.unique())
val_os['recruitment_channel'] = val_os.apply(lambda x: train.recruitment_channel.mode() if (x['recruitment_channel'] not in lis) else x['recruitment_channel'], axis = 1)

## Region - Changing it up with Train's Mode Value
lis = list(train.region.unique())
val_os['region'] = val_os.apply(lambda x: train.region.mode() if (x['region'] not in lis) else x['region'], axis = 1)
# Defining K-Fold Target Encoding Class for Validation (K-Fold as for Regularization) [Mapping from Train]

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, train, colNames, encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
         
    def fit(self, X, y = None):
        return self

    def transform(self, X):

        mean = self.train[[self.colNames, self.encodedName]].groupby(self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]

        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})

        return X
## Displaying the Top 5 Records of Validation Set
val_os.head(5)
# Firstly Target Encoding in Train

## Creating a Copy of Train
dum = train.copy()

## Target Encoding Region
targetc = KFoldTargetEncoderTrain('region', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

## Target Encoding Recruitment Channel
targetc = KFoldTargetEncoderTrain('recruitment_channel', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

# Got Removed with New Set of Feature Engineered Set
'''
## Target Encoding Department
targetc = KFoldTargetEncoderTrain('department', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)
'''

## Displaying the 1st 5 records of Dummy Train
dum.head(5)
# Target Encoding in Validation by Mapping from Train

## Target Encoding Region
targetc = KFoldTargetEncoderTest(dum, 'region', 'tgt_region')
val_os = targetc.fit_transform(val_os)

## Target Encoding Recruitment Channel
targetc = KFoldTargetEncoderTest(dum, 'recruitment_channel', 'tgt_recruitment_channel')
val_os = targetc.fit_transform(val_os)

# Got Removed with New Set of Feature Engineered Set
'''
## Target Encoding Department
targetc = KFoldTargetEncoderTest(dum, 'department', 'tgt_department')
'''

# Fit Transforming the Encoding
val_os = targetc.fit_transform(val_os)
# Performing Neccessary Actions on the Features to get our final dataframe for Validation for Model Evaluation and Training

## Boxcox Transformation on 'Age'
val_os['age_boxcox'] = stats.boxcox(val_os['age'])[0]

# Got Removed with New Set of Feature Engineered Set
'''
## Cummalative Training Score for Each Employee
val_os["cummulative_train'_score"] = val_os.no_of_trainings * val_os.avg_training_score

## Categorise Employees if they're having Good Overall Performance
val_os['good_overall_performance?'] = np.where((val_os.previous_year_rating >= 3) & (val_os['awards_won?'] == 1) & 
                                                   (val_os.avg_training_score >= val_os.avg_training_score.quantile(0.25)), 1, 0)
'''

## Encoding Education
edu_enc = {"Below Secondary" : 0, "Others" : 1, "Bachelor's": 2, "Master's & above": 3}
val_os['education'] = val_os['education'].map(edu_enc)

## Label Encoding Gender
l = LabelEncoder()
val_os.loc[:, 'gender'] = l.fit_transform(val_os.loc[:, 'gender']) 
# Mapping Features from Train -> Validation

y_val_os = val_os['is_promoted'].values.reshape(-1, 1)
x_val_os = val_os[os_columns]

## Displaying the 1st 5 records of Validation
x_val_os.head(5)
# Visualising the Shape of Validation Sets
x_val_os.shape, y_val_os.shape
# Scaling so as to apply Linear Algorithms and it won't affect Tree Based Algorithms Much - Validation
sc_x = StandardScaler()
x_val_os_scale = sc_x.fit_transform(x_val_os)
# Scaling so as to apply Linear Algorithms and it won't affect Tree Based Algorithms Much - Train
sc_x = StandardScaler()
x_train_os_scale = sc_x.fit_transform(x_os_plain)
# Importing Target Weighted Pickled DataFrames
df_wt_plain = pd.read_pickle('/content/drive/My Drive/Internship/Day 6/HR Analytics/Checkpoints/WT_Plain.pkl')
df_wt_tsne = pd.read_pickle('/content/drive/My Drive/Internship/Day 6/HR Analytics/Checkpoints/WT_tsne.pkl')
# Segregating the Features into Independent and Dependent Vectors
x_wt_plain = df_wt_plain.drop(['is_promoted'], axis = 1)
y_wt_plain = df_wt_plain['is_promoted'].values.reshape(-1, 1)

# Visualising and Storing the Feature Names for Target Weighted Plain DataFrame
wt_columns = x_wt_plain.columns
wt_columns
# Creating a Copy of Validation Set
val_wt = val.copy()
# Data Cleaning - Handling NULL Values

## Average Training Score - Filling it up with Train's Median Value
val_wt["avg_training_score"].fillna(round(train['avg_training_score'].median()), inplace = True)

## Department - Filling it up with Train's Mode Value
val_wt['department'].fillna(train.department.mode(), inplace = True)

## Age - Filling it up with Train's Median Value
val_wt["age"].fillna(round(train['age'].median()), inplace = True)

## KPIs_met >80% - Filling it up with 0
val_wt["KPIs_met >80%"].fillna(0, inplace = True)

## Education - Filling it up with 'Others'
val_wt["education"].fillna("Others", inplace = True)

## Previous Year Rating - Filling it up with 0
val_wt["previous_year_rating"].fillna(0, inplace = True)

## Recruitment Channel - Filling it up with Train's Mode Value
val_wt["recruitment_channel"].fillna((train['recruitment_channel'].mode()), inplace = True)

## No of Trainings - Filling it up with Train's Median Value
val_wt["no_of_trainings"].fillna(round(train['no_of_trainings'].median()), inplace = True)
# Data Handling - Handling Corner Cases

## Average Training Score - Filling it up with Train's Median Value
val_wt['avg_training_score'] = val_wt.apply(lambda x: round(train.avg_training_score.median()) if (x['avg_training_score'] > 100) else x['avg_training_score'], axis = 1)

## Department - Changing it up with Train's Mode Value
lis = list(train.department.unique())
val_wt['department'] = val_wt.apply(lambda x: train.department.mode() if (x['department'] not in lis) else x['department'], axis = 1)

## Previous Year Rating - Filling it up with Train's Median Value
val_wt["previous_year_rating"] = val_wt.apply(lambda x: round(train.previous_year_rating.median()) if (x['previous_year_rating'] < 0 or x['previous_year_rating'] > 5) else x['previous_year_rating'], axis = 1)

## Age - Filling it up with 0
val_wt['KPIs_met >80%'] = val_wt.apply(lambda x: 0 if (x['KPIs_met >80%'] not in [0, 1]) else x['KPIs_met >80%'], axis = 1)

## KPIs_met > 80% - Filling it up with Train's Median Value
val_wt['age'] = val_wt.apply(lambda x: round(train.age.median()) if (x['age'] > 100 or x['age'] < 15) else x['age'], axis = 1)

## Education - Changing it up with Train's Mode Value
lis = list(train.education.unique())
val_wt['education'] = val_wt.apply(lambda x: train.education.mode() if (x['education'] not in lis) else x['education'], axis = 1)

## Recruitment Channel - Changing it up with Train's Mode Value
lis = list(train.recruitment_channel.unique())
val_wt['recruitment_channel'] = val_wt.apply(lambda x: train.recruitment_channel.mode() if (x['recruitment_channel'] not in lis) else x['recruitment_channel'], axis = 1)

## Awards Won? - Filling it up with 0
val_wt['awards_won?'] = val_wt.apply(lambda x: 0 if (x['awards_won?'] not in [0, 1]) else x['awards_won?'], axis = 1)

## No. of Trainings - Filling it up with Train's Mode Value
val_wt['no_of_trainings'] = val_wt.apply(lambda x: train['no_of_trainings'].mode() if (x['no_of_trainings'] < 0 or x['no_of_trainings'] > 20) else x['no_of_trainings'], axis = 1)

'''
## Gender - Filling it up with 'm' as maximum of the employee will be males.
val_wt["gender"] = val_wt.apply(lambda x: m if (x['gender'] not in ['m', 'f']) else x['gender'], axis = 1)

## Region - Changing it up with Train's Mode Value
lis = list(train.region.unique())
val_wt['region'] = val_wt.apply(lambda x: train.region.mode() if (x['region'] not in lis) else x['region'], axis = 1)
'''
## Displaying the Top 5 Records of Validation Set
val_wt.head(5)
# Binning Age and Length of Service

## Creating a Copy of Train
dum = train.copy()

## Equal Frequency Discretization
discretizer = EqualFrequencyDiscretiser(q = 10, variables = ['age'], return_object = True, return_boundaries = True)
discretizer

# Fit the Discretization Transformer
discretizer.fit(train)

# Transform the data
trans_tr = discretizer.transform(train)
trans_f = discretizer.transform(val_wt)

# Adding Binned Categories to the Dataframe - Validation
dum['age_bins'] = trans_tr.age
## dum['length_of_service_bins'] = trans_tr.length_of_service

# Adding Binned Categories to the Dataframe - Validation
val_wt['age_bins'] = trans_f.age
## val_wt['length_of_service_bins'] = trans_f.length_of_service

## Displaying the Top 5 Records of Validation Set
val_wt.head(5)
# Firstly Target Encoding in Train

## Target Encoding Age Bins
targetc = KFoldTargetEncoderTrain('age_bins', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

## Target Encoding Recruitment Channel
targetc = KFoldTargetEncoderTrain('recruitment_channel', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

# Got Removed with New Set of Feature Engineered Set

'''
## Target Encoding Region
targetc = KFoldTargetEncoderTrain('region', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

## Target Encoding Length of Service Bins
targetc = KFoldTargetEncoderTrain('length_of_service_bins', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)
'''

## Displaying the 1st Record of Dummy Train
dum.head(1)
# Target Encoding in Validation by Mapping from Train

## Target Encoding Age Bins
targetc = KFoldTargetEncoderTest(dum, 'age_bins', 'tgt_age_bins')
val_wt = targetc.fit_transform(val_wt)

## Target Encoding Recruitment Channel
targetc = KFoldTargetEncoderTest(dum, 'recruitment_channel', 'tgt_recruitment_channel')
val_wt = targetc.fit_transform(val_wt)

# Got Removed with New Set of Feature Engineered Set

'''
## Target Encoding Region
targetc = KFoldTargetEncoderTest(dum, 'region', 'tgt_region')
val_wt = targetc.fit_transform(val_wt)

## Target Encoding Length of Service Bins
targetc = KFoldTargetEncoderTest(dum, 'length_of_service_bins', 'tgt_length_of_service_bins')
val_wt = targetc.fit_transform(val_wt)
'''

## Displaying the 1st 3 records of Validation
val_wt.head(3)
# Performing Neccessary Actions on the Features to get our final dataframe for Validation for Model Evaluation and Training

# Mean Rating by Department -> To Generate 'dept_rating_mean_ratio'
val_wt['mean_rating_by_dept'] = val_wt['department'].map(val_wt.groupby('department')['previous_year_rating'].mean())

# Prev Years' Rating by Department
val_wt['dept_rating_mean_ratio'] = val_wt['previous_year_rating'] / val_wt['mean_rating_by_dept']

# Cummulative Training Score
val_wt["cummulative_train'_score"] = val_wt.no_of_trainings * val_wt.avg_training_score

## Encoding Education
edu_enc = {"Below Secondary" : 0, "Others" : 1, "Bachelor's": 2, "Master's & above": 3}
val_wt['education'] = val_wt['education'].map(edu_enc)
# Mapping Features from Train -> Validation

y_val_wt = val_wt['is_promoted'].values.reshape(-1, 1)
x_val_wt = val_wt[wt_columns]

## Displaying the 1st 5 records of Validation
x_val_wt.head(5)
# Visualising the Shape of Validation Sets
x_val_wt.shape, y_val_wt.shape
# Scaling so as to apply Linear Algorithms and it won't affect Tree Based Algorithms Much - Validation
sc_x = StandardScaler()
x_val_wt_scale = sc_x.fit_transform(x_val_wt)
# Scaling so as to apply Linear Algorithms and it won't affect Tree Based Algorithms Much - Train
sc_x = StandardScaler()
x_train_wt_scale = sc_x.fit_transform(x_wt_plain)
# Deleting off redundant objects/variables - Memory Efficient Techniques
del df_os_tsne, dum, targetc, edu_enc, sc_x, df_wt_tsne
# Shapes of the DataFrames we're going to do operations on
print(f"X-Train = {x_train_os_scale.shape}, Y-Train = {y_os_plain.shape}")
print(f"X-Validation = {x_val_os_scale.shape}, Y-Validation = {y_val_os.shape}")
# Fitting Simple Linear Regression to the Training Set
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Random Forest Classifier to the Training Set
classifier = rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Kernel SVM to the Training Set
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Kernel SVM to the Training Set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting KNN to the Training Set
classifier = knc(n_neighbors = 10, n_jobs = -1)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score 
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Decision Tree Classifier to the Training Set
classifier = dtc(random_state = 0)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Naive Bayes to the Training Set
classifier = GaussianNB()
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting XGBoost to the Training Set
classifier = XGBClassifier(n_estimators = 1000, n_jobs = -1, random_state = 0)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score (Total)
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Gradient Boosting Classifier to the Training Set
classifier = GradientBoostingClassifier(random_state = 0)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting AdaBoost Classifier to the Training Set
classifier = AdaBoostClassifier(random_state = 0)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting CatBoost Classifier to the Training Set
classifier = CatBoostClassifier(random_state = 0, eval_metric = 'F1')
classifier.fit(x_train_os_scale, y_os_plain, eval_set = (x_val_os_scale, y_val_os))
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Light GBM Classifier to the Training Set
model_lgb = lgbm.LGBMClassifier(random_state = 0, n_jobs = -1)
model_lgb.fit(x_train_os_scale, y_os_plain)    # model_lgb.score(x_val_wt_scale, y_val_wt)
# Predicting the Validation Set Results
y_pred = model_lgb.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, model_lgb.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = model_lgb, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting MLP Classifier to the Training Set
classifier = MLPClassifier(activation = "relu", random_state = 0)
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Initialising the Stacking Algorithms
estimators = [
        ('decision-tree', dtc(random_state = 0)),
        ('random-forest', rfc(n_estimators = 1000, random_state = 0, n_jobs = -1)),
        ('mlp', MLPClassifier(activation = "relu", random_state = 0))
        ]
# Setting up the Meta-Classifier [Randomly Choosed]
classifier = StackingClassifier(
        estimators = estimators, 
        final_estimator = LogisticRegression(random_state = 0),
        # final_estimator = XGBClassifier(n_estimators = 1000, n_jobs = -1, random_state = 0)
        n_jobs = -1, cv = 5
        )
# Fitting my Model
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Determining Class Weights
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_wt_plain.reshape(-1, )), y_wt_plain.reshape(-1, ))
im_weight = dict(enumerate(class_weights))
im_weight
# Shapes of the DataFrames we're going to do operations on
print(f"X-Train = {x_train_wt_scale.shape}, Y-Train = {y_wt_plain.shape}")
print(f"X-Validation = {x_val_wt_scale.shape}, Y-Validation = {y_val_wt.shape}")
# Fitting Simple Linear Regression to the Training Set
classifier = LogisticRegression(class_weight = im_weight, random_state = 0)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Random Forest Classifier to the Training Set
classifier = rfc(random_state = 0, n_jobs = -1, class_weight = im_weight, n_estimators = 1000)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Kernel SVM to the Training Set
classifier = SVC(kernel = 'rbf', random_state = 0, class_weight = im_weight)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Kernel SVM to the Training Set
classifier = SVC(kernel = 'linear', random_state = 0, class_weight = im_weight)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting KNN to the Training Set
classifier = knc(n_neighbors = 10, n_jobs = -1)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Decision Tree Classifier to the Training Set
classifier = dtc(random_state = 0, class_weight = im_weight)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Naive Bayes to the Training Set
classifier = GaussianNB()
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale) 
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Count Occurences in Each Class
counter = Counter(y_wt_plain.reshape(-1, ))

# Estimate 'scale_pos_weight' value
estimate = counter[0] / counter[1]
print('Estimate : %.3f' % estimate)
# Fitting XGBoost to the Training Set
classifier = XGBClassifier(n_jobs = -1, random_state = 0, scale_pos_weight = estimate, n_estimators = 1000)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Gradient Boosting Classifier to the Training Set
classifier = GradientBoostingClassifier(random_state = 0, n_estimators = 1000)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting AdaBoost Classifier to the Training Set
classifier = AdaBoostClassifier(random_state = 0, n_estimators = 1000)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting CatBoost Classifier to the Training Set
classifier = CatBoostClassifier(random_state = 0, eval_metric = 'F1', class_weights = im_weight)
classifier.fit(x_train_wt_scale, y_wt_plain, eval_set = (x_val_wt_scale, y_val_wt))
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting Light GBM Classifier to the Training Set
model_lgb = lgbm.LGBMClassifier(random_state = 0, n_jobs = -1, class_weight = im_weight, n_estimators = 1000)
model_lgb.fit(x_train_wt_scale, y_wt_plain)    # model_lgb.score(x_val_wt_scale, y_val_wt)
# Predicting the Validation Set Results
y_pred = model_lgb.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, model_lgb.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_wt, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = model_lgb, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting MLP Classifier to the Training Set
classifier = MLPClassifier(activation = "relu", random_state = 0)
classifier.fit(x_train_wt_scale, y_wt_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_wt_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_wt, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_wt_plain, classifier.predict(x_train_wt_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_wt_scale, y = y_val_wt, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Initialising the Stacking Algorithms
estimators = [
        ('decision-tree', dtc(random_state = 0, class_weight = im_weight)),
        ('random-forest', rfc(n_estimators = 1000, random_state = 0, n_jobs = -1, class_weight = im_weight)),
        ('kernel-svm', SVC(kernel = 'rbf', random_state = 0, class_weight = im_weight))
        ]
# Setting up the Meta-Classifier [Randomly Choosed]
classifier = StackingClassifier(
        estimators = estimators, 
        final_estimator = LogisticRegression(random_state = 0, class_weight = im_weight)
        # final_estimator = XGBClassifier(n_estimators = 1000, n_jobs = -1, random_state = 0, scale_pos_weight = estimate)
        )
# Fitting my Model
classifier.fit(x_train_os_scale, y_os_plain)
# Predicting the Validation Set Results
y_pred = classifier.predict(x_val_os_scale)
# Confusion Matrix
cm = confusion_matrix(y_val_os, y_pred)
plt.figure(figsize = (5, 5))
sns.heatmap(cm, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
# Classification Report
print(classification_report(y_val_os, y_pred))
# Checking out the F1-Score
print(f"F1 for Train : {f1_score(y_os_plain, classifier.predict(x_train_os_scale))}")
print(f"F1 for Validation : {f1_score(y_val_os, y_pred)}")
# Applying k-fold Cross Validation
f1_s = cvs(estimator = classifier, X = x_val_os_scale, y = y_val_os, cv = 10, scoring = 'f1')
print(f"Max     : {f1_s.max()}")
print(f"Mean    : {f1_s.mean()}")
print(f"Std Dev : {f1_s.std()}")
# Fitting CatBoost Classifier to the Training Set -> Sample
classifier = CatBoostClassifier(random_state = 0, eval_metric = 'F1')
classifier.fit(x_train_wt_scale, y_wt_plain, silent = True)
# Get all the Paramters of CatBoost Classifier
classifier.get_all_params()               # get_all_params() -> to get all the parameters
# Setting up the Dictionary of Hyper-Paramters
hyperparams = {
    "eval_metric" : ["F1"],                    # Evaluation Metric
    #"random_state" : [0],                     # Random State to retain the same configuration
    "iterations" : [100, 200, 500, 1000],      # Iterations is an alis for 'n_estimators' -> Maximum no of Trees
    "loss_function" : ["Logloss"],             # Loss Function for our Model
    "learning_rate" : [0.03, 0.1, 0.001],      # Learning Rate of our Model
    "l2_leaf_reg" : [3.0, 1.0, 5.0],           # L2 Regularization Parameter of our Cost Function to reduce overfitting
    # "subsample" : [1, 0.66, 0.8],            # Sample rate for bagging. -> Cannot be done with GPU Bayersian Type
    "depth" : [6, 7, 8],                       # Depth of our Trees
    "class_weights" : [im_weight],             # Class Weights
    "od_type" : ["Iter"],                      # Type of overfitting detector
    "od_wait" : [50, 100],                     # The No. of Iterations to continue the training after the iteration with the optimal metric value
    # "thread_count" : [-1],                   # No. of threads to use during the training
    "task_type": ["GPU"],                      # Processing Unit
    "bootstrap_type" : ["Poisson"]             # Method for sampling the weights of Objects, Poisson for GPU [Faster Processing]
}
# Using Grid Search CV Method to find out the Best Set of Hyper-Parameters
classifier = CatBoostClassifier()
class_cv = GridSearchCV(classifier, hyperparams, verbose = 1, scoring = 'f1', n_jobs = -1, cv = 5)
class_cv.fit(x_train_wt_scale, y_wt_plain, eval_set = (x_val_wt_scale, y_val_wt))
# Dictionary of the Best Parameters
class_cv.best_params_
# Training Data using Best Paramters
classifier = CatBoostClassifier(**class_cv.best_params_)
classifier.fit(x_train_wt_scale, y_wt_plain, eval_set = (x_val_wt_scale, y_val_wt), silent = True)
# Predicting the Results
y_pred = classifier.predict(x_train_wt_scale)
y_pred
# Classification Report
print(classification_report(y_wt_plain, y_pred))
# Predicting the Results
y_pred = classifier.predict(x_val_wt_scale)  
y_pred
print("***Classification Report After Hyperparameter Tuning***")
print("\n")
print(classification_report(y_val_wt, y_pred))
# Checking out the F1-Score
print(f"F1 Hyperparameter Tuning : {f1_score(y_val_wt, y_pred)}")
filename = '11263_HR_Analytics.pkl'
pickle.dump(classifier, open(filename, 'wb'))
# Visualising the Shape of Test Set
test.shape
# Creating a copy of Test
test_c = test.copy()
# Data Cleaning - Handling NULL Values

## Age - Filling it up with Train's Median Value
test_c["age"].fillna(round(train['age'].median()), inplace = True)

## Average Training Score - Filling it up with Train's Median Value
test_c["avg_training_score"].fillna(round(train['avg_training_score'].median()), inplace = True)

## Education - Filling it up with 'Others'
test_c["education"].fillna("Others", inplace = True)

## Previous Year Rating - Filling it up with 0
test_c["previous_year_rating"].fillna(0, inplace = True)

## Department - Filling it up with Train's Mode Value
test_c["department"].fillna((train['department'].mode()), inplace = True)

## Region - Filling it up with Train's Mode Value
test_c["region"].fillna((train['region'].mode()), inplace = True)

## KPIs_met >80% - Filling it up with 0
test_c["KPIs_met >80%"].fillna(0, inplace = True)

## Recruitment Channel - Filling it up with Train's Mode Value
test_c["recruitment_channel"].fillna((train['recruitment_channel'].mode()), inplace = True)
# Firstly Target Encoding in Train

## Creating a Copy of Train
dum = train.copy()

## Target Encoding Region
targetc = KFoldTargetEncoderTrain('region', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

## Target Encoding Recruitment Channel
targetc = KFoldTargetEncoderTrain('recruitment_channel', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

## Target Encoding Department
targetc = KFoldTargetEncoderTrain('department', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)
# Target Encoding in Validation by Mapping from Train

## Target Encoding Region
targetc = KFoldTargetEncoderTest(dum, 'region', 'tgt_region')
test_c = targetc.fit_transform(test_c)

## Target Encoding Recruitment Channel
targetc = KFoldTargetEncoderTest(dum, 'recruitment_channel', 'tgt_recruitment_channel')
test_c = targetc.fit_transform(test_c)

## Target Encoding Department
targetc = KFoldTargetEncoderTest(dum, 'department', 'tgt_department')
test_c = targetc.fit_transform(test_c)
# Performing Neccessary on the Features to get our final dataframe for Validation for Model Evaluation and Training

## Boxcox Transformation on 'Age'
test_c['age_boxcox'] = stats.boxcox(test_c['age'])[0]

## Cummalative Training Score for Each Employee
test_c["cummulative_train'_score"] = test_c.no_of_trainings * test_c.avg_training_score

## Categorise Employees if they're having Good Overall Performance
test_c['good_overall_performance?'] = np.where((test_c.previous_year_rating >= 3) & (test_c['awards_won?'] == 1) & 
                                                   (test_c.avg_training_score >= test_c.avg_training_score.quantile(0.25)), 1, 0)

## Encoding Education
edu_enc = {"Below Secondary" : 0, "Others" : 1, "Bachelor's": 2, "Master's & above": 3}
test_c['education'] = test_c['education'].map(edu_enc)
# Mapping Features from Train -> Test

x = test_c[os_columns]
y = test_c['is_promoted'].values.reshape(-1, 1)
# Scaling so as to apply Linear Algorithms and it won't affect Tree Based Algorithms Much - Validation
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
# Loading the Model from Pickle File
path = '/content/drive/My Drive/Internship/Day 6/HR Analytics/Checkpoints/11263_HR_Analytics.pkl'
with open(path, 'rb') as file:  
    classifier = pickle.load(file)
# Predicting the Results
y_pred = classifier.predict(x)
# Classification Report
print("*** Test Classification Report ***")
print("\n")
print(classification_report(y, y_pred))
# Checking out the F1-Score (Total)
print(f"F1 Hyperparameter Tuning : {f1_score(y, y_pred)}")
# Visualising the Shape of Test Set
test_set.shape
# Creating a copy of Test
test_c = test_set.copy()
# Data Cleaning - Handling NULL Values

## Age - Filling it up with Train's Median Value
test_c["age"].fillna(round(train['age'].median()), inplace = True)

## Average Training Score - Filling it up with Train's Median Value
test_c["avg_training_score"].fillna(round(train['avg_training_score'].median()), inplace = True)

## Education - Filling it up with 'Others'
test_c["education"].fillna("Others", inplace = True)

## Previous Year Rating - Filling it up with 0
test_c["previous_year_rating"].fillna(0, inplace = True)

## Department - Filling it up with Train's Mode Value
test_c["department"].fillna((train['department'].mode()), inplace = True)

## Region - Filling it up with Train's Mode Value
test_c["region"].fillna((train['region'].mode()), inplace = True)

## KPIs_met >80% - Filling it up with 0
test_c["KPIs_met >80%"].fillna(0, inplace = True)

## Recruitment Channel - Filling it up with Train's Mode Value
test_c["recruitment_channel"].fillna((train['recruitment_channel'].mode()), inplace = True)
# Firstly Target Encoding in Train

## Creating a Copy of Train
dum = train.copy()

## Target Encoding Region
targetc = KFoldTargetEncoderTrain('region', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

## Target Encoding Recruitment Channel
targetc = KFoldTargetEncoderTrain('recruitment_channel', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)

## Target Encoding Department
targetc = KFoldTargetEncoderTrain('department', 'is_promoted', n_fold = 5)
dum = targetc.fit_transform(dum)
# Target Encoding in Validation by Mapping from Train

## Target Encoding Region
targetc = KFoldTargetEncoderTest(dum, 'region', 'tgt_region')
test_c = targetc.fit_transform(test_c)

## Target Encoding Recruitment Channel
targetc = KFoldTargetEncoderTest(dum, 'recruitment_channel', 'tgt_recruitment_channel')
test_c = targetc.fit_transform(test_c)

## Target Encoding Department
targetc = KFoldTargetEncoderTest(dum, 'department', 'tgt_department')
test_c = targetc.fit_transform(test_c)
# Performing Neccessary on the Features to get our final dataframe for Validation for Model Evaluation and Training

## Boxcox Transformation on 'Age'
test_c['age_boxcox'] = stats.boxcox(test_c['age'])[0]

## Cummalative Training Score for Each Employee
test_c["cummulative_train'_score"] = test_c.no_of_trainings * test_c.avg_training_score

## Categorise Employees if they're having Good Overall Performance
test_c['good_overall_performance?'] = np.where((test_c.previous_year_rating >= 3) & (test_c['awards_won?'] == 1) & 
                                                   (test_c.avg_training_score >= test_c.avg_training_score.quantile(0.25)), 1, 0)

## Encoding Education
edu_enc = {"Below Secondary" : 0, "Others" : 1, "Bachelor's": 2, "Master's & above": 3}
test_c['education'] = test_c['education'].map(edu_enc)
# Mapping Features from Train -> Test
x = test_c[os_columns]
# Scaling so as to apply Linear Algorithms and it won't affect Tree Based Algorithms Much - Validation
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
# Loading the Model from Pickle File
path = '/content/drive/My Drive/Internship/Day 6/HR Analytics/Checkpoints/11263_HR_Analytics.pkl'
with open(path, 'rb') as file:  
    classifier = pickle.load(file)
# Predicting Results
y_pred = classifier.predict(x)
# Creating Submission DataFrame
Submission_Test_11263 = pd.DataFrame(test['employee_id'])
Submission_Test_11263['is_promoted'] = y_pred
# Converting the DataFrame to csv format
Submission_Test_11263.to_csv('Submission_Test_11263.csv', index = False)
!pip freeze > requirements.txt