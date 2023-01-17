#importing libraries for numpy and dataframe
import pandas as pd
import numpy as np

#importing libraries for data visualization
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import seaborn as sns
%matplotlib inline

#to ensure complete rows and columns are shown for a dataframe
pd.set_option('display.max_columns',150)
pd.set_option('display.max_rows',9500)

#importing library for data scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

#importing library to suppress warnings
import warnings
warnings.filterwarnings('ignore')

#importing libraries for Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm

#Checking VIF values for the feature variables
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Creation of confusion matrix
from sklearn import metrics

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score
#Reading the file from the local
leads = pd.read_csv('../Leads.csv') 

#Checking the first 5 records of the dataframe
leads.head()
#Reading the data dictionary from the local
data_dict = pd.read_excel('../Leads_Data_Dictionary.xlsx')
data_dict.head(5)

#To determine the number of rows and columns present in the dataset.
leads.shape
#Checking the statistical values of the numerical columns of the dataset
leads.describe()
#Checking the datatype of the dataset
leads.info()
#To check if there exists any duplicate records in the dataframe specially for Prospect ID

print('The number of duplicate records in the column , Prospect ID are :',leads['Prospect ID'].duplicated().sum())
print('The number of duplicate records in the dataset are :',leads.duplicated().sum())
# Converting 'Select' values to NaN.
# check for Select Specialization
leads = leads.replace('Select', np.nan)
leads = leads.replace('Select', np.nan)
#To check the percentage of missing values in each column

total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])
pd.concat([total,percentage],axis = 1)
# we will drop the columns having more than 70% NA values.

leads = leads.drop(leads.loc[:,list(round(100*(leads.isnull().sum()/len(leads.index)), 2)>50)].columns, 1)
leads.head()
#Analayzing the Asymmetric Score and Index for Profile and Activity

plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
sns.countplot(leads['Asymmetrique Activity Index'])
plt.subplot(2,2,2)
sns.countplot(leads['Asymmetrique Profile Index'])
leads.columns

#To check the percentage of missing values in these 4 colums 

temp_df =leads[['Asymmetrique Activity Index','Asymmetrique Profile Index']]

total = pd.DataFrame(temp_df.isna().sum().sort_values(ascending=False),columns=['Total Missing'])

percentage = pd.DataFrame(round(100*(temp_df.isna().sum()/len(temp_df)),2).sort_values(ascending=False),columns=['Percentage'])
pd.concat([total,percentage],axis = 1)
#Dropping the columns Asymmetrique Activity Index,Asymmetrique Profile Index,Asymmetrique Activity Score,Asymmetrique Profile Score
leads = leads.drop(['Asymmetrique Activity Index','Asymmetrique Profile Index'],axis=1)
#Rechecking the null values and percentage values
total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])
percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])
pd.concat([total,percentage],axis = 1)
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Asymmetrique Profile Score'].unique())
# Plotting a histogram for visualizing the data
leads['Asymmetrique Profile Score'].plot.hist(bins = num_unique_values)
# checking the mean of the column
print("Mean is ",leads['Asymmetrique Profile Score'].mean())
# checking the median of the column
print("Median is ",leads['Asymmetrique Profile Score'].median())
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Asymmetrique Activity Score'].unique())
# Plotting a histogram for visualizing the data
leads['Asymmetrique Activity Score'].plot.hist(bins = num_unique_values)
# checking the mean of the column
print("Mean is ",leads['Asymmetrique Activity Score'].mean())
# checking the median of the column
print("Median is ",leads['Asymmetrique Activity Score'].median())
# imputing the value of median to the null values
leads.loc[pd.isnull(leads['Asymmetrique Profile Score']),['Asymmetrique Profile Score']]=16

# imputing the value of median to the null value
leads.loc[pd.isnull(leads['Asymmetrique Activity Score']),['Asymmetrique Activity Score']]=14
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Asymmetrique Profile Score'].unique())
# Plotting a histogram for visualizing the data
leads['Asymmetrique Profile Score'].plot.hist(bins = num_unique_values)
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Asymmetrique Activity Score'].unique())
# Plotting a histogram for visualizing the data
leads['Asymmetrique Activity Score'].plot.hist(bins = num_unique_values)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
#Analysing City
plt.figure(figsize=(15,5))
sns.countplot(leads['City'])
plt.show()
#Replacing nan value to Mumbai
leads['City']=leads['City'].replace(np.nan,'Mumbai')
#Analysing the columns again based on the replacing
plt.figure(figsize=(15,5))
sns.countplot(leads['City'])
plt.show()
#Analyzing Specialization
plt.figure(figsize=(10,5))
sns.countplot(leads['Specialization'])
xticks(rotation = 90)
plt.show()
#Replacing nan with Others in Specilization
leads['Specialization'] = leads['Specialization'].replace(np.nan,'Others')
#Plotting countplot again to check the plot of the different values in Specialization
#Analyzing Specialization
plt.figure(figsize=(10,5))
sns.countplot(leads['Specialization'])
xticks(rotation = 90)
plt.show()
#Checking the count and percentage of missing values
total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])
percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])
pd.concat([total,percentage],axis = 1)
#Analyzing Tags
plt.figure(figsize=(10,5))
sns.countplot(leads['Tags'])
xticks(rotation = 90)
plt.show()
#Replacing nan with Will revert after reading the email
leads['Tags'] = leads['Tags'].replace(np.nan,'Will revert after reading the email')
#Analyzing Tags
plt.figure(figsize=(10,5))
sns.countplot(leads['Tags'])
xticks(rotation = 90)
plt.show()
#Checking the count and percentage of missing values
total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])
percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])
pd.concat([total,percentage],axis = 1)
#Analyzing What matters most to you in choosing a course
sns.countplot(leads['What matters most to you in choosing a course'])
xticks(rotation = 90)
# checking the count of different values within the column
leads['What matters most to you in choosing a course'].value_counts()
# dropping the aforementioned column
leads.drop(['What matters most to you in choosing a course'],axis=1,inplace=True)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# checking the count of different values within the column
leads['What is your current occupation'].value_counts()
#Analyzing What is your current occupation

sns.countplot(leads['What is your current occupation'])
xticks(rotation = 90)
#Replacing nan with Unemployed
leads['What is your current occupation'] = leads['What is your current occupation'].replace(np.nan,'Unemployed')
#Analyzing What is your current occupation
sns.countplot(leads['What is your current occupation'])
xticks(rotation = 90)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
#Analyzing Country
plt.figure(figsize=(10,5))
sns.countplot(leads['Country'])
xticks(rotation = 90)
plt.show()
# dropping the aforementioned column
leads.drop(['Country'],axis=1,inplace=True)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
num_unique_values =  len(leads['TotalVisits'].unique())
# Plotting a histogram for visualizing the data
leads['TotalVisits'].plot.hist(bins = num_unique_values)

leads['TotalVisits'].fillna(leads['TotalVisits'].mean(),inplace=True)
num_unique_values =  len(leads['Page Views Per Visit'].unique())
# Plotting a histogram for visualizing the data
leads['Page Views Per Visit'].plot.hist(bins = num_unique_values)
leads['Page Views Per Visit'].fillna(leads['Page Views Per Visit'].mean(),inplace=True)

##Checking the count and percentage of missing values
total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])
percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])
pd.concat([total,percentage],axis = 1)
#Analyzing Lead Activity
plt.figure(figsize=(10,5))
sns.countplot(leads['Last Activity'])
xticks(rotation = 90)
plt.show()
#Analyzing Lead Source
plt.figure(figsize=(10,5))
sns.countplot(leads['Lead Source'])
xticks(rotation = 90)
plt.show()
#Dropping NA values
leads.dropna(axis=0,inplace=True)
##Checking the count and percentage of missing values
total = pd.DataFrame(leads.isna().sum().sort_values(ascending=False),columns=['Total Missing'])
percentage = pd.DataFrame(round(100*(leads.isna().sum()/len(leads)),2).sort_values(ascending=False),columns=['Percentage'])
pd.concat([total,percentage],axis = 1)

#Dropping Prospect ID from the dataframe as it has no role to play in the modelling
leads = leads.drop(['Prospect ID','Lead Number'],axis=1)
#Let us check how many columns and rows we have now in the dataframe
leads.shape
#To check which all columns are present, looking into the first 5 records of the dataframe
leads.head()
#To analyze the statistics of the numerical columns
leads.describe()
# Initializing the figure
fig = plt.figure(figsize = (12,8))
# prining the boxplot
sns.boxplot(data=leads)
# setting the title of the figure
plt.title("PC Distribution", fontsize = 12)
# setting the y-label
plt.ylabel("Range")
# setting the x-label
plt.xlabel("Columns")
plt.xticks(rotation=90)

# printing the plot
plt.show()
leads.describe(percentiles=[.05,.25, .5, .75, .90, .95, .99])
# before we move forward, lets create a copy of the existing df
leads1=leads.copy()
# setting the lower whisker
Q1 = leads1['TotalVisits'].quantile(0.05)
    # setting the upper whisker
Q3 = leads1['TotalVisits'].quantile(0.95)
    # setting the IQR by dividing the upper with lower quantile
IQR = Q3 - Q1
    # performing the outlier analysis
leads1 = leads1[(leads1['TotalVisits'] >= Q1) & (leads1['TotalVisits'] <= Q3)]

# setting the lower whisker
Q1 = leads1['Page Views Per Visit'].quantile(0.05)
    # setting the upper whisker
Q3 = leads1['Page Views Per Visit'].quantile(0.95)
    # setting the IQR by dividing the upper with lower quantile
IQR = Q3 - Q1
    # performing the outlier analysis
leads = leads1[(leads1['Page Views Per Visit'] >= Q1) & (leads1['Page Views Per Visit'] <= Q3)]
# Checking the shape of the df now
leads1.shape
# Initializing the figure
fig = plt.figure(figsize = (12,8))
# prining the boxplot
sns.boxplot(data=leads1)
# setting the title of the figure
plt.title("PC Distribution", fontsize = 12)
# setting the y-label
plt.ylabel("Range")
# setting the x-label
plt.xlabel("Columns")
plt.xticks(rotation=90)

# printing the plot
plt.show()
# Let's look at the scarifice
print("Shape before outlier treatment: ",leads1.shape)
print("Shape after outlier treatment: ",leads.shape)

print("Percentage data removal is around {}%".format(round(100*(leads1.shape[0]-leads.shape[0])/leads1.shape[0]),2))
leads=leads1
#Analyzing Lead Source
plt.figure(figsize=(10,5))
sns.countplot(leads['Converted'])
xticks(rotation = 90)
plt.show()
# Finding all the rows where the lead is converted and moving them to a new df
df_convert = leads[leads['Converted'] == 1]
# Finding all the rows  where the lead is not converted and moving them to a new df
df_not_convert = leads[leads['Converted'] == 0]


# Finding the categorical columns and printing the same.
categorical = leads.select_dtypes(exclude=['int64','float64'])
categCols = categorical.columns

for i,col in enumerate(categCols):
    plt.figure(1)
    plt.figure(figsize=(15,5))
    for j,df in enumerate([df_convert,df_not_convert]):
        dfTemp = df[col].value_counts()
        dfTemp = dfTemp.to_frame()
        dfTemp.index.name=col
        dfTemp.rename(columns={col:'Count'},inplace=True)
        plt.subplots_adjust(wspace = 1)
        plt.subplot(1,2,j+1)
        plt.xticks(rotation=90)
        sns.barplot(x= dfTemp.index, y = dfTemp.Count)
        if j==0:
            plt.title(col + " for Customers who converted")
        else:
            plt.title(col + " for Customers who did not convert")
#Dropping columns from the dataframe
leads = leads.drop(['Search','Magazine','Newspaper Article','A free copy of Mastering The Interview',
                    'X Education Forums','Newspaper','Digital Advertisement',
                    'Through Recommendations','Receive More Updates About Our Courses','Update me on Supply Chain Content',
                   'Get updates on DM Content','I agree to pay the amount through cheque'],axis=1)
leads = leads.drop(['Tags'],axis=1)
#To assess the number of rows and columns present in the dataframe
leads.shape
# Pairplot of all numeric columns

sns.pairplot(leads)
# Finding the numrical columns and printing the same.
numCols = leads.select_dtypes(include=['int64','float64'])
# Sorting the columns
numCols = numCols[sorted(numCols.columns)]
# printing the columns
print(numCols.columns)
# Explicitly controlling the SettingWithCopy settings
pd.set_option('mode.chained_assignment', None)
# deleting the Converted column since we do not want to print the same with itself
numCols = numCols.drop(['Converted'],axis=1)

# running a for loop to print the plots 
for i,col in enumerate(numCols):
    plt.figure(1)
    plt.figure(figsize=(15,5))
    for j,df in enumerate([df_convert,df_not_convert]):
        plt.subplots_adjust(wspace = 1)
        plt.subplot(1,2,j+1)
        plt.xticks(rotation=90)
        df[col] = df[col].fillna(0).astype(int)
        sns.distplot(df[col])
        if j==0:
            plt.title(col + " for Customers who converted")
        else:
            plt.title(col + " for Customers who did not convert")

# Heatmap to understand the attributes dependency

plt.figure(figsize = (15,10))        
ax = sns.heatmap(leads.corr(),annot = True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
# List of variables to map
varlist =  ['Do Not Email', 'Do Not Call']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function
leads[varlist] = leads[varlist].apply(binary_map)
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
sns.countplot(leads['Do Not Email'])
plt.subplot(2,2,2)
sns.countplot(leads['Do Not Call'])
leads = leads.drop(['Do Not Email','Do Not Call'],axis=1)

#Checking the first 5 rows to see Yes/No converted to 1/0
leads.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy_var = pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                            'City','Last Notable Activity']], drop_first=True)
dummy_var.head()
# Adding the results to the master dataframe
leads = pd.concat([leads, dummy_var], axis=1)
leads.head()
# dropping the original columns since they are not required
leads = leads.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','City','Last Notable Activity'], axis = 1)
leads
leads.shape
#Checking the Converted Rate
converted = (sum(leads['Converted'])/len(leads['Converted'].index))*100
converted
# Putting feature variable to X
X = leads.drop(['Converted'],axis=1)

# Putting response variable to y
y = leads['Converted']

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
X_train.head()

scaler = StandardScaler()

X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', 'Asymmetrique Activity Score','Asymmetrique Profile Score']] = scaler.fit_transform(X_train[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', 'Asymmetrique Activity Score','Asymmetrique Profile Score']])
#Creating Logistic Regression Model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)),family = sm.families.Binomial())
logm1.fit().summary()
# Running RFE with the output number of the variable equal to 15
logreg = LogisticRegression()
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_
# RFE function will now determine the ranking of all the variables and rank them for our use.
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
X_train1 = X_train[col]
# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_sm = sm.add_constant(X_train1)
# Running the linear model
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
#Let's see the summary of our linear model
res.summary()
col1 = col.drop('What is your current occupation_Housewife',1)
X_train1 = X_train[col1]
# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_sm = sm.add_constant(X_train1)
# Running the linear model
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
#Let's see the summary of our linear model
res.summary()
col2 = col1.drop('Last Notable Activity_Had a Phone Conversation',1)
X_train1 = X_train[col2]
# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_sm = sm.add_constant(X_train1)
# Running the linear model
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
#Let's see the summary of our linear model
res.summary()
#Dropping Tags_Not doing further education from the col
col3 = col2.drop('Last Activity_Had a Phone Conversation',1)
X_train1 = X_train[col3]
# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_sm = sm.add_constant(X_train1)
# Running the linear model
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
#Let's see the summary of our linear model
res.summary()
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train1.columns
vif['VIF'] = [variance_inflation_factor(X_train1.values, i) for i in range(X_train1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
# Reshape
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()
#Creating a confusion matrix to check the correctness of prediction
ConfusionMatrix = metrics.confusion_matrix(y_train_pred_final.Converted,y_train_pred_final.predicted)
print(ConfusionMatrix)
# Actual/Predicted     converted    not_converted
        # converted        3532       293
        # not_converted    789       1511  
#Calculating the overall accuracy
print(metrics.accuracy_score(y_train_pred_final.Converted,y_train_pred_final.predicted))
TP = ConfusionMatrix[1,1] # true positive 
TN = ConfusionMatrix[0,0] # true negatives
FP = ConfusionMatrix[0,1] # false positives
FN = ConfusionMatrix[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false predictive rate - predicting converted when lead is not converted
print(FP/ float(TN+FP))
# Calculate postive predictive rate - predicting converted when lead is actually converted
print (TP / float(TP+FP))
# Calculate negative predictive rate - predicting not converted when lead is actually not converted 
print (TN / float(TN+ FN))
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve(y_train_pred_final.Converted,y_train_pred_final.predicted, drop_intermediate = False )
#Plotting the ROC Curve
draw_roc(y_train_pred_final.Converted,y_train_pred_final.predicted)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['Probability','Accuracy','Sensitivity','Specificity'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])
plt.show()
#Creating a new column Final_Prediction and its value with 1 if Predicted_Conversion_Probability is greater or equal to 0.38 else 0
y_train_pred_final['Final_Prediction'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.36 else 0)
y_train_pred_final.head(10)
#Plotting the ROC Curve
draw_roc(y_train_pred_final.Converted,y_train_pred_final.Final_Prediction)
# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Final_Prediction)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Final_Prediction)
confusion2
TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)
# Let us calculate specificity
TN / float(TN+FP)
# Calculate false postive rate
print(FP/ float(TN+FP))
# Positive predictive value 
print (TP / float(TP+FP))
# Negative predictive value
print (TN / float(TN+ FN))
y_train_pred_final.Converted, y_train_pred_final.predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)
#Plotting the precision recall curve
plt.title('Precision Recall curve')
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()



X_test[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', 'Asymmetrique Activity Score','Asymmetrique Profile Score']] = scaler.transform(X_test[['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', 'Asymmetrique Activity Score','Asymmetrique Profile Score']])

X_test = X_test[col3]
X_test.head()
X_test_sm = sm.add_constant(X_test)
X_test_sm
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Prediction Probability'})
# Let's see the head of y_pred_final
y_pred_final.head()
y_pred_final['Final Prediction'] = y_pred_final['Prediction Probability'].map(lambda x:1 if x>0.36 else 0)
y_pred_final['Lead Score'] = round(y_pred_final['Prediction Probability']*100,2)
y_pred_final.head()
# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final['Final Prediction'])
def scores(actuals,predictions):
    confusion=metrics.confusion_matrix(actuals,predictions)
    TP = confusion[1,1] # true positives 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] #False Positives
    FN = confusion[1,0] #False negatives
    accuracy_score = metrics.accuracy_score(actuals,predictions)
    sensitivity= TP / float(TP+FN)
    specificity= TN/ float(TN+FP)
    precision_score = metrics.precision_score(actuals,predictions)
    recall_score = metrics.recall_score(actuals,predictions)
    final_scores=pd.Series({'Accuracy':accuracy_score,'Sensitivity':sensitivity,'Specificity':specificity,'Precision':precision_score,'Recall':recall_score})
    return(final_scores)
test_scores= scores(y_pred_final['Converted'], y_pred_final['Final Prediction'])
test_scores
