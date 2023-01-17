# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
# Importing Pandas and NumPy
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 50000)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
# Importing the dataset
# Please note that the csv file needs to be in the same directory as the python file
leads = pd.read_csv("../input/Leads.csv")
leads.head()
# Let's check the dimensions of the dataframe
leads.shape
# let's look at the statistical aspects of the dataframe
leads.describe()
# Let's see the type of each column
leads.info()
#Checking for any duplicates in the data frame
leads.loc[leads.duplicated()]
leads.head()
leads.drop(['Magazine', 'Receive More Updates About Our Courses', 'Update me on Supply Chain Content'
           ,'Get updates on DM Content', 'I agree to pay the amount through cheque'],axis=1,inplace=True)
leads.drop(['Country', 'City'],axis=1,inplace=True)
leads.drop(['Lead Number','Prospect ID'],axis=1,inplace=True)
# finding the count of Select string in all columns and storing the same in a new df
df = leads.eq('Select').sum().to_frame().T
# creating a new column in the dataframe and storing the count of Select of each column
df.loc['Count_of_Select'] = leads.eq('Select').sum()
# storing the total values of each column in a new column
df.loc['Total'] = leads.count()
# renaming the index name
df = df.drop(index={0})
# finding the percentage of the Select values
df.loc['Percent'] = df.loc['Count_of_Select']/df.loc['Total'] * 100
# transposing the dataframe for better viewing
df = df.T
# soring the values based on the Percentage
df = df.sort_values(by="Percent",ascending = False)
# removing the unnecessary columns
df = df.iloc[:,[0,2]]
# fining all columns where the Percentage of Select is more than 0
df = df.loc[df['Percent'] > 0, :]
# Printing the df
df
# replacing Select with NaN for Specialization column
leads.Specialization = leads.Specialization.str.strip().replace('Select', np.nan)
# replacing Select with NaN for How did you hear about X Education column
leads['How did you hear about X Education'] = leads['How did you hear about X Education'].str.strip().replace('Select', np.nan)
# replacing Select with NaN for Lead Profile column
leads['Lead Profile'] = leads['Lead Profile'].str.strip().replace('Select', np.nan)
leads.head()
# Checking the percentage of missing values
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# removing any column which has more than 50% null values
leads = leads.loc[:,leads.isnull().sum()/leads.shape[0]*100<50]
print(leads.shape)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
varlist =  ['Asymmetrique Activity Index', 'Asymmetrique Profile Index']

def new_map(x):
    return x.map({'01.High': 1, "02.Medium": 2, "03.Low": 3})

leads[varlist] = leads[varlist].apply(new_map)
# Creating a new df with these 4 columns
corr = leads.loc[:,['Asymmetrique Profile Score', 'Asymmetrique Activity Score', 'Asymmetrique Profile Index','Asymmetrique Activity Index']].corr()
# Plotting a heat map for these 4 columns
sns.heatmap(corr, annot = True,cmap="Blues")
# dropping the aforementioned columns
leads.drop(['Asymmetrique Activity Index', 'Asymmetrique Profile Index'],axis=1,inplace=True)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# checking the count of different values within the column
leads['Asymmetrique Profile Score'].value_counts()
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Asymmetrique Profile Score'].unique())
# Plotting a histogram for visualizing the data
leads['Asymmetrique Profile Score'].plot.hist(bins = num_unique_values)
# checking the mean of the column
print("Mean is ",leads['Asymmetrique Profile Score'].mean())
# checking the mode of the column
print("Mode is ",leads['Asymmetrique Profile Score'].mode())
# checking the median of the column
print("Median is ",leads['Asymmetrique Profile Score'].median())
# imputing the value of median to the null values
leads.loc[pd.isnull(leads['Asymmetrique Profile Score']),['Asymmetrique Profile Score']]=16
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Asymmetrique Profile Score'].unique())
# Plotting a histogram for visualizing the data
leads['Asymmetrique Profile Score'].plot.hist(bins = num_unique_values)
# checking the count of different values within the column
leads['Asymmetrique Activity Score'].value_counts()
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Asymmetrique Activity Score'].unique())
# Plotting a histogram for visualizing the data
leads['Asymmetrique Activity Score'].plot.hist(bins = num_unique_values)
# checking the mean of the column
print("Mean is",leads['Asymmetrique Activity Score'].mean())
# checking the mode of the column
print("Mode is",leads['Asymmetrique Activity Score'].mode())
# checking the median of the column
print("Median is",leads['Asymmetrique Activity Score'].median())
# imputing the value of median to the null values
leads.loc[pd.isnull(leads['Asymmetrique Activity Score']),['Asymmetrique Activity Score']]=14
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Asymmetrique Activity Score'].unique())
# Plotting a histogram for visualizing the data
leads['Asymmetrique Activity Score'].plot.hist(bins = num_unique_values)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# checking the count of different values within the column
leads['Specialization'].value_counts()
# plotting a count plot for the column
sns.countplot(x= 'Specialization', data = leads)
plt.xticks(rotation=90)
# Creating a new value Others and replacing it with null values
leads['Specialization'] = leads['Specialization'].replace(np.nan, 'Others')
# plotting a count plot for the column
sns.countplot(x= 'Specialization', data = leads)
plt.xticks(rotation=90)
# checking the count of different values within the column
leads['What matters most to you in choosing a course'].value_counts()
# plotting a count plot for the column
sns.countplot(x= 'What matters most to you in choosing a course', data = leads)
plt.xticks(rotation=90)
# dropping the aforementioned column
leads.drop(['What matters most to you in choosing a course'],axis=1,inplace=True)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# checking the count of different values within the column
leads['What is your current occupation'].value_counts()
# plotting a count plot for the column
sns.countplot(x= 'What is your current occupation', data = leads)
plt.xticks(rotation=90)
# imputing the value Unemployed to the null values
leads.loc[pd.isnull(leads['What is your current occupation']),['What is your current occupation']]='Unemployed'
# plotting a count plot for the column
sns.countplot(x= 'What is your current occupation', data = leads)
plt.xticks(rotation=90)
# checking the count of different values within the column
leads['TotalVisits'].value_counts().head(10)
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['TotalVisits'].unique())
# Plotting a histogram for visualizing the data
leads['TotalVisits'].plot.hist(bins = num_unique_values)
# imputing the value of mode to the null values
leads.loc[pd.isnull(leads['TotalVisits']),['TotalVisits']]=0.0
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['TotalVisits'].unique())
# Plotting a histogram for visualizing the data
leads['TotalVisits'].plot.hist(bins = num_unique_values)
# Checking the percentage of missing values
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# checking the count of different values within the column
leads['Page Views Per Visit'].value_counts()
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Page Views Per Visit'].unique())
# Plotting a histogram for visualizing the data
leads['Page Views Per Visit'].plot.hist(bins = num_unique_values)
# imputing the value of mode to the null values
leads.loc[pd.isnull(leads['Page Views Per Visit']),['Page Views Per Visit']]=0.0
# pulling the length of number of unique values in the column
num_unique_values =  len(leads['Page Views Per Visit'].unique())
# Plotting a histogram for visualizing the data
leads['Page Views Per Visit'].plot.hist(bins = num_unique_values)
# Checking the percentage of missing values
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# checking the count of different values within the column
leads['Tags'].value_counts()
# plotting a count plot for the column
sns.countplot(x= 'Tags', data = leads)
plt.xticks(rotation=90)
# imputing the value of mode to the null values
leads.loc[pd.isnull(leads['Tags']),['Tags']]='Will revert after reading the email'
# plotting a count plot for the column
sns.countplot(x= 'Tags', data = leads)
plt.xticks(rotation=90)
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# checking the count of different values within the column
leads['Last Activity'].value_counts()
# plotting a count plot for the column
sns.countplot(x= 'Last Activity', data = leads)
plt.xticks(rotation=90)
# imputing the value of mode to the null values
leads.loc[pd.isnull(leads['Last Activity']),['Last Activity']]='Email Opened'
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
# checking the count of different values within the column
leads['Lead Source'].value_counts()
# plotting a count plot for the column
sns.countplot(x= 'Lead Source', data = leads)
plt.xticks(rotation=90)
# correcting the values
leads=leads.replace({'Lead Source': {'google': 'Google'}})
# plotting a count plot for the column
sns.countplot(x= 'Lead Source', data = leads)
plt.xticks(rotation=90)
# imputing the value of Google in place of null values.
leads.loc[pd.isnull(leads['Lead Source']),['Lead Source']]='Google'
# Checking the percentage of missing values now
missingValPercent = leads.isnull().sum()/leads.shape[0]*100
print(missingValPercent.sort_values(ascending = False))
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
Q1 = leads['TotalVisits'].quantile(0.05)
    # setting the upper whisker
Q3 = leads['TotalVisits'].quantile(0.95)
    # setting the IQR by dividing the upper with lower quantile
IQR = Q3 - Q1
    # performing the outlier analysis
leads = leads[(leads['TotalVisits'] >= Q1) & (leads['TotalVisits'] <= Q3)]
# setting the lower whisker
Q1 = leads['Page Views Per Visit'].quantile(0.05)
    # setting the upper whisker
Q3 = leads['Page Views Per Visit'].quantile(0.95)
    # setting the IQR by dividing the upper with lower quantile
IQR = Q3 - Q1
    # performing the outlier analysis
leads = leads[(leads['Page Views Per Visit'] >= Q1) & (leads['Page Views Per Visit'] <= Q3)]
# Checking the shape of the df now
leads.shape
# checking the different percentiles now
leads.describe(percentiles=[.05,.25, .5, .75, .90, .95, .99])
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
# Let's look at the scarifice
print("Shape before outlier treatment: ",leads1.shape)
print("Shape after outlier treatment: ",leads.shape)

print("Percentage data removal is around {}%".format(round(100*(leads1.shape[0]-leads.shape[0])/leads1.shape[0]),2))
# Finding all the rows where the loan was approved and moving them to a new df
df_convert = leads[leads['Converted'] == 1]
# Finding all the rows where the loan was cancelled and moving them to a new df
df_not_convert = leads[leads['Converted'] == 0]
# Finding the categorical columns and printing the same.
categorical = leads.select_dtypes(exclude=['int64','float64'])
categCols = categorical.columns
categCols
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

sns.countplot(x = "Lead Origin", hue = "Converted", data = leads)
plt.xticks(rotation = 90)
# List of variables to map

varlist =  ['Do Not Email', 'Do Not Call', 'Search', 'Newspaper Article', 'X Education Forums', 'Newspaper'
           , 'Digital Advertisement', 'Through Recommendations'
           , 'A free copy of Mastering The Interview']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function
leads[varlist] = leads[varlist].apply(binary_map)
leads.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy_var = pd.get_dummies(leads[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                              'Tags','Last Notable Activity']], drop_first=True)
dummy_var.head()
# Adding the results to the master dataframe
leads = pd.concat([leads, dummy_var], axis=1)
leads.head()
# dropping the original columns since they are not required
leads = leads.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','Tags','Last Notable Activity'], axis = 1)
# checking the df
leads.head()
leads.shape
# Normalising continuous features
df = leads[['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Asymmetrique Activity Score',
             'Asymmetrique Profile Score']]
normalized_df=(df-df.mean())/df.std()
leads = leads.drop(['TotalVisits','Total Time Spent on Website','Page Views Per Visit','Asymmetrique Activity Score',
             'Asymmetrique Profile Score'], 1)
leads = pd.concat([leads,normalized_df],axis=1)
leads.head()
converted = (sum(leads['Converted'])/len(leads['Converted'].index))*100
converted
from sklearn.model_selection import train_test_split

# Putting feature variable to X
X = leads.drop(['Converted'],axis=1)

# Putting response variable to y
y = leads['Converted']

y.head()
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
import statsmodels.api as sm
# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()
# Importing RFE and LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

# Running RFE with the output number of the variable equal to 15
logreg = LogisticRegression()
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_
# RFE function will now determine the ranking of all the variables and rank them for our use.
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
# Here we are taking the top 15 columns which are recommended by the RFE function
col = X_train.columns[rfe.support_]
col
# Printing the columns which are being discarded from further analysis
X_train.columns[~rfe.support_]
X_train1 = X_train[col]
# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_sm1 = sm.add_constant(X_train1)
# Running the linear model
logm2 = sm.GLM(y_train,X_train_sm1, family = sm.families.Binomial())
res = logm2.fit()
#Let's see the summary of our linear model
res.summary()
# dropping the column Tags_invalid number
col1 = col.drop('Tags_Lateral student',1)
X_train1 = X_train[col1]
# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_sm1 = sm.add_constant(X_train1)
# Running the linear model
logm2 = sm.GLM(y_train,X_train_sm1, family = sm.families.Binomial())
res = logm2.fit()
#Let's see the summary of our linear model
res.summary()
# Dropping the column `Last Notable Activity_Had a Phone Conversation`
col2 = col1.drop('Last Notable Activity_Had a Phone Conversation',1)
X_train1 = X_train[col2]
# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_sm1 = sm.add_constant(X_train1)
# Running the linear model
logm2 = sm.GLM(y_train,X_train_sm1, family = sm.families.Binomial())
res = logm2.fit()
#Let's see the summary of our linear model
res.summary()
# Dropping the column Tags_Lateral student
col3 = col2.drop('Lead Source_Welingak Website',1)
X_train1 = X_train[col3]
# Adding a constant variable since the statsmodels library does not come with a constant variable built-in
X_train_sm1 = sm.add_constant(X_train1)
# Running the linear model
logm2 = sm.GLM(y_train,X_train_sm1, family = sm.families.Binomial())
res = logm2.fit()
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
y_train_pred = res.predict(X_train_sm1)
y_train_pred[:10]
# lets reshape the values.
y_train_pred = y_train_pred.values.reshape(-1)
# checking some values
y_train_pred[:10]
# creating a new df with the predicted values.
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
# adding the custID to the dataframe
y_train_pred_final['CustID'] = y_train.index
# checking the new df now
y_train_pred_final.head()
# creating the new column
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()
# importing the necessary libraries
from sklearn import metrics
# Creating the Confusion matrix to calculate the Metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)
# Let's check the overall accuracy.
print("The overall Accuracy score for train set is",metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
# Pulling out all the necessary values from the confusion matrix
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print("The sensitivity for train set is",round(TP / float(TP+FN),4))
# Let us calculate specificity
print("The specificity for train set is",round(TN / float(TN+FP),4))
# Calculate false postive rate - predicting converted when customer does not have converted
print("The false postive rate for train set is",FP/ float(TN+FP))
# Positive predictive value 
print ("The Positive predictive value for train set is",TP / float(TP+FP))
# Negative predictive value
print ("The Negative predictive value for train set is",TN / float(TN+ FN))
from sklearn.metrics import precision_score, recall_score
print ("The Precision Score for train set is",precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
print ("The Recall Score for train set is",recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
# importing the libraries
from sklearn.metrics import precision_recall_curve
# checking the Original and Predicted values
y_train_pred_final.Converted, y_train_pred_final.predicted
# settin the precision and recall values along with the threshold
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.predicted)
# printing a graph to check the intersection point.
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()
# initializing a function to plot the ROC curve
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
# setting the FPR and TPR and the threshold
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_Prob, drop_intermediate = False )
# plotting the ROC curve with the Converted and Converted_Prob
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)
# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()
# Now let's calculate accuracy, sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
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
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()
# creating a new column and multiplying the probabilities with 100 
y_train_pred_final['Lead_Score'] = y_train_pred_final['Converted_Prob'].apply(lambda x:x*100)
# creating a new column based on the above mentioned column using the ROC cutoff
y_train_pred_final['final_predicted'] = y_train_pred_final.Lead_Score.map( lambda x: 1 if x > 42 else 0)

y_train_pred_final.head()
# Creating the Confusion matrix to calculate the Metrics
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
print(confusion)
# Let's check the overall accuracy.
print("The overall Accuracy score for train set is",metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print("The sensitivity for train set is",round(TP / float(TP+FN),4))
# Let us calculate specificity
print("The specificity for train set is",round(TN / float(TN+FP),4))
# Calculate false postive rate - predicting converted when customer does not have converted
print("The false postive rate for train set is",FP/ float(TN+FP))
# Positive predictive value 
print ("The Positive predictive value for train set is",TP / float(TP+FP))
# Negative predictive value
print ("The Negative predictive value for train set is",TN / float(TN+ FN))
from sklearn.metrics import precision_score, recall_score
print ("The Precision Score for train set is",precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
print ("The Recall Score for train set is",recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted))
"{:2.2f}".format(metrics.roc_auc_score(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob))
# using the columns in train dataset for our test set analysis
X_test = X_test[X_train1.columns]
X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
# Let's see the head
y_pred_1.head()
# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)
# Putting CustID to index
y_test_df['CustID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})
# Let's see the head of y_pred_final
y_pred_final.head()
# creating a new column and multiplying the probabilities with 100 
y_pred_final['Lead_Score'] = y_pred_final['Converted_Prob'].apply(lambda x:x*100)
# creating a new column based on the above mentioned column using the ROC cutoff
y_pred_final['final_predicted'] = y_pred_final.Lead_Score.map(lambda x: 1 if x > 42 else 0)
y_pred_final.head()
# Let's check the overall accuracy.
print("The overall Accuracy score for test set is",metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted))
confusion = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion
TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
print("The sensitivity for test set is",round(TP / float(TP+FN),4))
# Let us calculate specificity
print("The specificity for test set is",round(TN / float(TN+FP),4))
# Calculate false postive rate - predicting converted when customer does not have converted
print("The false postive rate for test set is",FP/ float(TN+FP))
# Positive predictive value 
print ("The Positive predictive value for test set is",TP / float(TP+FP))
# Negative predictive value
print ("The Negative predictive value for test set is",TN / float(TN+ FN))
from sklearn.metrics import precision_score, recall_score
print ("The Precision Score for test set is",precision_score(y_pred_final.Converted, y_pred_final.final_predicted))
print ("The recall Score for test set is",recall_score(y_pred_final.Converted, y_pred_final.final_predicted))
"{:2.2f}".format(metrics.roc_auc_score(y_pred_final.Converted, y_pred_final.Converted_Prob))
