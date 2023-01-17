#import all the libraries and modules

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

from scipy import stats 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

# Importing RFE and LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm  

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve



# Supress Warnings

#Enable autocomplete in Jupyter Notebook.

%config IPCompleter.greedy=True



import warnings

warnings.filterwarnings('ignore')

import os



### Set seaborn style

sns.set(style="darkgrid")



## Set the max display columns to None so that pandas doesn't sandwich the output 

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 40)
### Let us create a utility function to generate a list of null values in different dataframes

### We will utilize this function extensively througout the notebook. 

def generateNullValuesPercentageTable(dataframe):

    totalNullValues = dataframe.isnull().sum().sort_values(ascending=False)

    percentageOfNullValues = round((dataframe.isnull().sum()*100/len(dataframe)).sort_values(ascending=False),2)

    columnNamesWithPrcntgOfNullValues = pd.concat([totalNullValues, percentageOfNullValues], axis=1, keys=['Total Null Values', 'Percentage of Null Values'])

    return columnNamesWithPrcntgOfNullValues
### let us create a reuseable function that will help us in ploting our barplots for analysis



def generateBarPlot(dataframe, keyVariable, plotSize):

    fig, axs = plt.subplots(figsize = plotSize)

    plt.xticks(rotation = 90)

    ax = sns.countplot(x=keyVariable, data=dataframe)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}'.format(height/len(dataframe) * 100),

                ha="center") 
### Let us define a reusable function to carry out Bivariate analysis as well.



def generateBiplot(df,col,title,figsize,hue=None):

    

    sns.set_context('talk')

    plt.rcParams["axes.labelsize"] = 20

    plt.rcParams['axes.titlesize'] = 22

    plt.rcParams['axes.titlepad'] = 30

    plt.figure(figsize=figsize)

    

    

    temp = pd.Series(data = hue)

    fig, ax = plt.subplots()

    width = len(df[col].unique()) + 7 + 4*len(temp.unique())

    fig.set_size_inches(width , 8)

    plt.xticks(rotation=45)

    plt.title(title)

    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,

                       palette='magma')

    

    ### Also print the conversion accuracy of every field

    convertcount=df.pivot_table(values='Lead Number',index=col,columns='Converted', aggfunc='count').fillna(0)

    convertcount["Conversion(%)"] =round(convertcount[1]/(convertcount[0]+convertcount[1]),2)*100

    return print(convertcount.sort_values(ascending=False,by="Conversion(%)"),plt.show())

        

    plt.show()
### Function to generate heatmaps



def generateHeatmaps(df, figsize):

    plt.figure(figsize = figsize)        # Size of the figure

    sns.heatmap(df.corr(),annot = True, annot_kws={"fontsize":7})

### Function to generate ROC curves

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
### Let us create a reusable function to calculate VIF values for our models



def vifCalculator(inputModel):

    vif= pd.DataFrame()

    X = inputModel

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return vif

### Checking if the data has been correctly loaded or not.

leadScoreDataset = pd.read_csv('../input/leads-dataset/Leads.csv')

leadScoreDataset.head()
### It is important to know the original conversion rate of the company before we proceed. Let us calculate that

originalConversionRate = round((sum(leadScoreDataset['Converted'])/len(leadScoreDataset['Converted'].index))*100, 2)

print("The conversion rate of leads is: ",originalConversionRate)
leadScoreDataset.shape
leadScoreDataset.info()
leadScoreDataset.describe()
leadScoreDataset = leadScoreDataset.replace('Select', np.nan)

leadScoreDataset.head()
### Dropping rows with duplicate values based on unique 'Prospect ID' & for 'Lead Number' for each candidate

print('Are there NO duplicates present in Prospect Id column? ', sum(leadScoreDataset.duplicated('Prospect ID')) == 0)

print('Are there NO duplicates present in Lead Number Column? ', sum(leadScoreDataset.duplicated('Lead Number')) == 0)

generateNullValuesPercentageTable(leadScoreDataset)
### Dropping columns with null values over 70%

droppedColumns = leadScoreDataset.columns[leadScoreDataset.isnull().mean() > 0.70]

leadScoreDatasetAfterDroppedColumns = leadScoreDataset.drop(droppedColumns, axis = 1)



print('The new shape of the dataset after dropping the columns is: ', leadScoreDatasetAfterDroppedColumns.shape)



### analysing the dataframe is correct after dropping the columns

leadScoreDatasetAfterDroppedColumns.head()
### Checking the number of unique values per column

leadScoreDatasetAfterDroppedColumns.nunique()
leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.loc[:, leadScoreDatasetAfterDroppedColumns.nunique()!=1]

leadScoreDatasetAfterDroppedColumns.shape
leadScoreDatasetAfterDroppedColumns.head()
### Let us see the frequency of the different values present in the 'Lead Quality' column

leadScoreDatasetAfterDroppedColumns['Lead Quality'].value_counts()
### Since 'Lead Quality' is based on an employees intuition, let us inpute any NAN values with 'Not Sure' and take counts again

leadScoreDatasetAfterDroppedColumns['Lead Quality'] = leadScoreDatasetAfterDroppedColumns['Lead Quality'].replace(np.nan, 'Not Sure')

leadScoreDatasetAfterDroppedColumns['Lead Quality'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Lead Quality', (10,10))
### We will plot box plots and count plots repectively



fig, axs = plt.subplots(2,2, figsize = (10,7.5))

plt1 = sns.countplot(leadScoreDatasetAfterDroppedColumns['Asymmetrique Activity Index'], ax = axs[0,0])

plt2 = sns.boxplot(leadScoreDatasetAfterDroppedColumns['Asymmetrique Activity Score'], ax = axs[0,1])

plt3 = sns.countplot(leadScoreDatasetAfterDroppedColumns['Asymmetrique Profile Index'], ax = axs[1,0])

plt4 = sns.boxplot(leadScoreDatasetAfterDroppedColumns['Asymmetrique Profile Score'], ax = axs[1,1])

plt.tight_layout()
colsToDrop = ['Lead Quality', 'Asymmetrique Activity Index','Asymmetrique Activity Score','Asymmetrique Profile Index','Asymmetrique Profile Score']

leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.drop(colsToDrop,axis =1)



leadScoreDatasetAfterDroppedColumns.head()
leadScoreDatasetAfterDroppedColumns.shape
### Let us now assess the percentage of missing values in the remaining dataframe

generateNullValuesPercentageTable(leadScoreDatasetAfterDroppedColumns)
### Exploring 'City' column



leadScoreDatasetAfterDroppedColumns.City.describe()
leadScoreDatasetAfterDroppedColumns.City.value_counts(normalize=True)
## From the above we can see that the value 'Mumbai' has the most number of enteries

## Let us plot the same



generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'City', (10,5))
leadScoreDatasetAfterDroppedColumns.City = leadScoreDatasetAfterDroppedColumns.City.fillna('Mumbai')
leadScoreDatasetAfterDroppedColumns.City.value_counts(normalize=True)
### Exploring 'Specialization' column which hs 36.58% null values

leadScoreDatasetAfterDroppedColumns.Specialization.describe()
generateBarPlot(leadScoreDatasetAfterDroppedColumns,'Specialization', (20,20))
### Replacing missing values with 'Others'



leadScoreDatasetAfterDroppedColumns.Specialization = leadScoreDatasetAfterDroppedColumns.Specialization.fillna('Others')



leadScoreDatasetAfterDroppedColumns.Specialization.value_counts(normalize=True)
generateBarPlot(leadScoreDatasetAfterDroppedColumns,'Specialization', (20,20))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Specialization', 'Conversion based on Specialization',(40,30), 'Converted')
### The last column with a high percentage of null values is Tags. Let us explore this column



leadScoreDatasetAfterDroppedColumns.Tags.describe()
generateBarPlot(leadScoreDatasetAfterDroppedColumns,'Tags', (20,20))
leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.drop('Tags', axis=1)



leadScoreDatasetAfterDroppedColumns.head()
leadScoreDatasetAfterDroppedColumns.shape
### Let us check the null percentage of the dataframe now

generateNullValuesPercentageTable(leadScoreDatasetAfterDroppedColumns)
leadScoreDatasetAfterDroppedColumns['What matters most to you in choosing a course'].value_counts(normalize=True)
leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.drop('What matters most to you in choosing a course', axis=1)

leadScoreDatasetAfterDroppedColumns.head()
leadScoreDatasetAfterDroppedColumns.shape
leadScoreDatasetAfterDroppedColumns['What is your current occupation'].value_counts(normalize=True)
leadScoreDatasetAfterDroppedColumns['What is your current occupation'] = leadScoreDatasetAfterDroppedColumns['What is your current occupation'].fillna('Unemployed')

leadScoreDatasetAfterDroppedColumns['What is your current occupation'].value_counts(normalize=True)
leadScoreDatasetAfterDroppedColumns.Country.value_counts().head(5)
leadScoreDatasetAfterDroppedColumns.Country = leadScoreDatasetAfterDroppedColumns.Country.fillna('India')

leadScoreDatasetAfterDroppedColumns.Country.value_counts().head(5)
leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.drop('Country', axis=1)

leadScoreDatasetAfterDroppedColumns.head()
leadScoreDatasetAfterDroppedColumns.shape
generateNullValuesPercentageTable(leadScoreDatasetAfterDroppedColumns)
### Imputing missing values in Lead Source column

leadScoreDatasetAfterDroppedColumns['Lead Source'].value_counts()
### Imputing missing values with 'Google'



leadScoreDatasetAfterDroppedColumns['Lead Source'] = leadScoreDatasetAfterDroppedColumns['Lead Source'].fillna('Google')

leadScoreDatasetAfterDroppedColumns['Lead Source'].value_counts()
leadScoreDatasetAfterDroppedColumns['Lead Source'] =  leadScoreDatasetAfterDroppedColumns['Lead Source'].apply(lambda x:x.capitalize())



leadScoreDatasetAfterDroppedColumns['Lead Source'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns,'Lead Source', (15,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Lead Source', 'Conversion based on Lead Source', (50,30),'Converted')
colsToReplace=['Click2call', 'Live chat', 'Nc_edm', 'Pay per click ads', 'Press_release',

  'Social media', 'Welearn', 'Bing', 'Blog', 'Testone', 'Welearnblog_home', 'Youtubechannel']

leadScoreDatasetAfterDroppedColumns['Lead Source'] = leadScoreDatasetAfterDroppedColumns['Lead Source'].replace(colsToReplace, 'Others')
generateBarPlot(leadScoreDatasetAfterDroppedColumns,'Lead Source', (15,10))
leadScoreDatasetAfterDroppedColumns['Lead Source'].value_counts()
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Lead Source', 'Conversion based on Lead Source', (40,20),'Converted')
### Imputing values in Last Activity column

leadScoreDatasetAfterDroppedColumns['Last Activity'].value_counts()
#### Imputing the missing values with 'Email Opened'

leadScoreDatasetAfterDroppedColumns['Last Activity'] = leadScoreDatasetAfterDroppedColumns['Last Activity'].fillna('Email Opened')

leadScoreDatasetAfterDroppedColumns['Last Activity'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns,'Last Activity', (15,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Last Activity', 'Conversion based on Last Activity', (40,30),'Converted')
### Imputing values in Page Views Per Visit column

leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'].value_counts().head(15)
leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'].describe()
#### Imputing the missing values with '2.0' which is the median value

leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'] = leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'].replace(np.nan,'2.0')

leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'].value_counts().head(15)
### Looks like 0.0 was incorrectly imputed. Let us correct the imputation

leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'] =  leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'].astype(float)



leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'].value_counts()
### It looks like there are a lot of outliers in the data, let us verify this using a boxplot

fig, axs = plt.subplots(figsize = (10,10))

sns.boxplot(leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'])
### We will cap our data at the 1% & 95% mark so as to not lose any values or drop rows. 

capValue = leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'].quantile([0.01,0.95]).values

leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'][leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'] <= capValue[0]] = capValue[0]

leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'][leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'] >= capValue[1]] = capValue[1]



leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'].describe(percentiles=[0.01,.25, .5, .75, .90, .95, .99])
fig, axs = plt.subplots(figsize = (10,10))

sns.boxplot(leadScoreDatasetAfterDroppedColumns['Page Views Per Visit'])
generateBarPlot(leadScoreDatasetAfterDroppedColumns,'Page Views Per Visit', (25,20))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Page Views Per Visit', 'Page Views Per Visit vs Conversion', (40,40), 'Converted')
leadScoreDatasetAfterDroppedColumns.TotalVisits.describe()
### We will impute this value with the meadian value since the 

### mean and the median values are relatively close to each other

leadScoreDatasetAfterDroppedColumns.TotalVisits = leadScoreDatasetAfterDroppedColumns.TotalVisits.fillna('3.0')

leadScoreDatasetAfterDroppedColumns.TotalVisits.value_counts()
### Looks like 3.0 was incorrectly imputed. Let us correct the imputation

leadScoreDatasetAfterDroppedColumns['TotalVisits'] =  leadScoreDatasetAfterDroppedColumns['TotalVisits'].astype(float)



leadScoreDatasetAfterDroppedColumns['TotalVisits'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'TotalVisits',(30,10))
leadScoreDatasetAfterDroppedColumns['TotalVisits'].describe(percentiles=[0.01,.25, .5, .75, .90, .95, .99])
### There seem to be a large number of outliers. Let us check these using a boxplot and decide what to do next

fig, axs = plt.subplots(figsize = (10,10))



sns.boxplot(data=leadScoreDatasetAfterDroppedColumns.TotalVisits)

### We will cap our data at the 1% & 95% mark so as to not lose any values or drop rows. 

capValue = leadScoreDatasetAfterDroppedColumns.TotalVisits.quantile([0.01,0.95]).values

leadScoreDatasetAfterDroppedColumns.TotalVisits[leadScoreDatasetAfterDroppedColumns.TotalVisits <= capValue[0]] = capValue[0]

leadScoreDatasetAfterDroppedColumns.TotalVisits[leadScoreDatasetAfterDroppedColumns.TotalVisits >= capValue[1]] = capValue[1]



leadScoreDatasetAfterDroppedColumns.TotalVisits.describe(percentiles=[0.01,.25, .5, .75, .90, .95, .99])
fig, axs = plt.subplots(figsize = (10,10))



sns.boxplot(data=leadScoreDatasetAfterDroppedColumns.TotalVisits)

generateBiplot(leadScoreDatasetAfterDroppedColumns, 'TotalVisits', 'Total Visits vs Conversion', (40,20), 'Converted')
### Assessing if there are any more missing values in the data

generateNullValuesPercentageTable(leadScoreDatasetAfterDroppedColumns)
### We can also drop the column 'Prospect ID' as we already have an identifying column with unique values: 'Lead Number'

leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.drop('Prospect ID', axis=1)

leadScoreDatasetAfterDroppedColumns.head()
### Checking the shape of the dataset before beginning any further analysis

leadScoreDatasetAfterDroppedColumns.shape
### Identifying the remaining columns and their datatypes before proceeding

leadScoreDatasetAfterDroppedColumns.info()
leadScoreDatasetAfterDroppedColumns.describe()
leadScoreDatasetAfterDroppedColumns['Lead Origin'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns,'Lead Origin', (10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns,'Lead Origin', 'Conversion Based on Lead Origin', (40,20),'Converted')
leadScoreDatasetAfterDroppedColumns['Do Not Email'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Do Not Email',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Do Not Email', 'Do Not Email vs Conversion', (40,20),'Converted')
leadScoreDatasetAfterDroppedColumns['Do Not Call'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Do Not Call',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Do Not Call', 'Conversions vs Do Not Call', (40,20),'Converted')
leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.drop(['Do Not Call', 'Do Not Email'], axis=1)

leadScoreDatasetAfterDroppedColumns.shape
leadScoreDatasetAfterDroppedColumns.info()
leadScoreDatasetAfterDroppedColumns.Search.describe()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Search',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Search', 'Search vs Conversion',(40,20), 'Converted')
leadScoreDatasetAfterDroppedColumns['Newspaper Article'].describe()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Newspaper Article',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Newspaper Article', 'Newspaper Article vs Conversion',(40,20), 'Converted')
leadScoreDatasetAfterDroppedColumns['X Education Forums'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'X Education Forums',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'X Education Forums', 'X Education Forums vs Conversion', (40,20),'Converted')
leadScoreDatasetAfterDroppedColumns['Newspaper'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Newspaper',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Newspaper', 'Newspaper vs Conversion', (40,20),'Converted')
leadScoreDatasetAfterDroppedColumns['Digital Advertisement'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Digital Advertisement',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Digital Advertisement', 'Digital Advertisement vs Conversion', (40,20),'Converted')
leadScoreDatasetAfterDroppedColumns['Through Recommendations'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Through Recommendations',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Through Recommendations', 'Through Recommendations vs Conversion', (40,20),'Converted')
dropCols = ['Search', 'Newspaper', 'X Education Forums', 'Newspaper Article' , 'Digital Advertisement','Through Recommendations']

leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.drop(dropCols, axis=1)

leadScoreDatasetAfterDroppedColumns.head()
leadScoreDatasetAfterDroppedColumns.shape
leadScoreDatasetAfterDroppedColumns.info()
leadScoreDatasetAfterDroppedColumns['A free copy of Mastering The Interview'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'A free copy of Mastering The Interview',(10,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'A free copy of Mastering The Interview', 'A free copy of Mastering The Interview vs Conversion', (40,20),'Converted')
leadScoreDatasetAfterDroppedColumns.shape
leadScoreDatasetAfterDroppedColumns['Last Notable Activity'].describe()
leadScoreDatasetAfterDroppedColumns['Last Notable Activity'].value_counts()
generateBarPlot(leadScoreDatasetAfterDroppedColumns, 'Last Notable Activity',(20,10))
generateBiplot(leadScoreDatasetAfterDroppedColumns, 'Last Notable Activity', 'Last Notable Activity vs Conversion', (40,20),'Converted')
leadScoreDatasetAfterDroppedColumns = leadScoreDatasetAfterDroppedColumns.drop('Last Notable Activity', axis=1)

leadScoreDatasetAfterDroppedColumns.head()
leadScoreDatasetAfterDroppedColumns.shape
leadScoreDatasetAfterDroppedColumns['Total Time Spent on Website'].describe()
leadScoreDatasetAfterDroppedColumns['Total Time Spent on Website'].value_counts()
### Let us generate a distplot to view the split of this data

sns.distplot(leadScoreDatasetAfterDroppedColumns['Total Time Spent on Website'])

plt.show()
leadScoreDatasetAfterDroppedColumns['Total Time Spent on Website'] = leadScoreDatasetAfterDroppedColumns['Total Time Spent on Website'].apply(lambda x: round((x/60), 2))

sns.distplot(leadScoreDatasetAfterDroppedColumns['Total Time Spent on Website'])

plt.show()
leadScoreDatasetAfterDroppedColumns.head()
# Let us split our dataframe to perform better analysis

timeSpentMoreThan1HourDF=leadScoreDatasetAfterDroppedColumns[leadScoreDatasetAfterDroppedColumns['Total Time Spent on Website']>=1.0]

timeSpentMoreThan1HourDF["Hours Spent"]= timeSpentMoreThan1HourDF["Total Time Spent on Website"].astype(int)



timeSpentMoreThan1HourDF.head()

### Let us generate a bivariate analysis bar plot to better understand our conversions



generateBiplot(timeSpentMoreThan1HourDF, 'Hours Spent', 'Last Notable Activity vs Conversion', (40,40),'Converted')
plt.figure(figsize=(40,40))

plt.xticks(rotation=45)

plt.yscale('log')

sns.boxplot(data =timeSpentMoreThan1HourDF, x='TotalVisits',y='Total Time Spent on Website', hue ='Converted',orient='v')

plt.title('Total Time Spent Vs Total Visits based on Conversion')

plt.show()
generateNullValuesPercentageTable(leadScoreDatasetAfterDroppedColumns)
leadScoreDatasetAfterDroppedColumns.head()
### Assessing current dataframe

leadScoreDatasetAfterDroppedColumns.info()
### Let us assess the correlation between the existing variables to rule out any colinearity 

generateHeatmaps(leadScoreDatasetAfterDroppedColumns, (20,20))
### First we will convert the Yes/No values in the 'A free copy of Mastering The Interview' column to 1/0



leadScoreDatasetAfterDroppedColumns['A free copy of Mastering The Interview'] = leadScoreDatasetAfterDroppedColumns['A free copy of Mastering The Interview'].map(dict(Yes=1, No=0))

leadScoreDatasetAfterDroppedColumns.head()
leadScoreDatasetAfterDroppedColumns.shape
### Creating dummies

dummyCols = ['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','City']

dummyDataset = pd.get_dummies(leadScoreDatasetAfterDroppedColumns[dummyCols],drop_first=True)

dummyDataset.head()
dummyDataset.shape
### Combining dummies with the original dataset into a new dataset



combinedDummyDataset = leadScoreDatasetAfterDroppedColumns.copy()

combinedDummyDataset.head()
combinedDummyDataset.shape
### combining datasets

combinedDummyDataset = pd.concat([combinedDummyDataset, dummyDataset], axis=1)

combinedDummyDataset.head()
combinedDummyDataset.shape
### We will now drop the original columns and the columns that have 'Others' as a sub heading since we had 

### combined various values to create those columns



dummiesColsToDrop = ['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation','City',

                     'Lead Source_Others','Specialization_Others']

combinedDummyDataset = combinedDummyDataset.drop(dummiesColsToDrop, axis=1)

combinedDummyDataset.head()
combinedDummyDataset.shape
combinedDummyDataset.info()
### First we will drop the Converted & Lead Number columns 

### We will create another copy of our model and use that for this



X = combinedDummyDataset.drop(['Converted','Lead Number'], axis=1)

X.head()
X.shape
### Adding the target variable 'Converted' to y

y = combinedDummyDataset['Converted']



y.head()
# Splitting the data into train and test



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
### Now let us begin scaling features. First let us assess our training dataset

X_train.head()
### Scaling 

scaler = StandardScaler()



X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])





X_train.head()
X_train.shape
generateHeatmaps(combinedDummyDataset,(30,30))
### Dropping highly correlated variables

X_train = X_train.drop(['Lead Origin_Lead Add Form', 'Lead Source_Facebook'], axis=1)

X_test = X_test.drop(['Lead Origin_Lead Add Form', 'Lead Source_Facebook'], axis=1)
X_train.head()
X_train.shape
X_test.head()
X_test.shape
generateHeatmaps(combinedDummyDataset[X_train.columns],(30,30))
## Creating Logistic Regression Model



logisticRegressionModel = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logisticRegressionModel.fit().summary()
### RFE with 20 variables

logreg = LogisticRegression()



rfe20 = RFE(logreg, 20)

rfe20= rfe20.fit(X_train,y_train)

rfe20.support_
list(zip(X_train.columns, rfe20.support_, rfe20.ranking_))
col = X_train.columns[rfe20.support_]
X_train.columns[~rfe20.support_]
X_train_cols = X_train[col]

X_train_cols
X_train_sm = sm.add_constant(X_train_cols)

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()

### Checking VIF values



vifCalculator(X_train_cols)
X_train_cols = X_train_cols.drop('What is your current occupation_Housewife', axis=1)

X_train_cols.columns
### Rerun the model with the selected variables

X_train_sm = sm.add_constant(X_train_cols)

logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm3.fit()

res.summary()
vifCalculator(X_train_cols)
X_train_cols = X_train_cols.drop('Last Activity_Resubscribed to emails', axis=1)

X_train_cols.columns
X_train_sm = sm.add_constant(X_train_cols)

logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm4.fit()

res.summary()
vifCalculator(X_train_cols)
X_train_cols = X_train_cols.drop('Lead Origin_Lead Import', axis=1)

X_train_cols.columns
X_train_sm = sm.add_constant(X_train_cols)

logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm5.fit()

res.summary()
vifCalculator(X_train_cols)
X_train_cols = X_train_cols.drop('What is your current occupation_Working Professional', axis=1)

X_train_cols.columns 

X_train_sm = sm.add_constant(X_train_cols)

logm6 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm6.fit()

res.summary()
vifCalculator(X_train_cols)
X_train_cols = X_train_cols.drop('Specialization_Rural and Agribusiness', axis=1)

X_train_cols.columns 
X_train_sm = sm.add_constant(X_train_cols)

logm7 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm7.fit()

res.summary()
vifCalculator(X_train_cols)
generateHeatmaps(X_train_sm, (20,20))
y_train_pred = res.predict(X_train_sm)

y_train_pred.head()
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Lead_Score_Prob':y_train_pred})

y_train_pred_final['CustID'] = y_train.index

y_train_pred_final.head()
y_train_pred_final['Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head





y_train_pred_final.head()
y_train_pred_final['Lead_Score'] = round((y_train_pred_final['Lead_Score_Prob'] * 100),0)



y_train_pred_final['Lead_Score'] = y_train_pred_final['Lead_Score'].astype(int)



# Let's see the head

y_train_pred_final.head()
# Confusion matrix 



from sklearn import metrics

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted_Hot_Lead )

print(confusion)

# Let's check the overall accuracy.

print(round(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted_Hot_Lead),2))
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

round((TP / float(TP+FN)),2)
# Let us calculate specificity

round((TN / float(TN+FP)),2)
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Lead_Score_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Lead_Score_Prob)
# Let's create columns with different probability cutoffs 



numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['Probability','Accuracy','Sensitivity','Specificty'])





num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities



sns.set_style('whitegrid')

sns.set_context('paper')



cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificty'])

plt.xticks(np.arange(0,1,step=.05), size=8)

plt.yticks(size=12)

plt.show()
y_train_pred_final['Predicted_Hot_Lead'] = y_train_pred_final.Lead_Score_Prob.map( lambda x: 1 if x > 0.33 else 0)



y_train_pred_final.head()
round(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted_Hot_Lead),2)
confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted_Hot_Lead )

confusion2
TP = confusion2[1,1] # true positive 

TN = confusion2[0,0] # true negatives

FP = confusion2[0,1] # false positives

FN = confusion2[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

round(TP / float(TP+FN),2)
# Let us calculate specificity

round(TN / float(TN+FP),2)
### Calculating Precision

precision =round(TP/float(TP+FP),2)

precision
### Calculating Recall

recall = round(TP/float(TP+FN),2)

recall
### Let us generate the Precision vs Recall tradeoff curve 

p ,r, thresholds=precision_recall_curve(y_train_pred_final.Converted,y_train_pred_final['Lead_Score_Prob'])

plt.title('Precision vs Recall tradeoff')

plt.plot(thresholds, p[:-1], "g-")    # Plotting precision

plt.plot(thresholds, r[:-1], "r-")    # Plotting Recall

plt.show()
### The F statistic is given by 2 * (precision * recall) / (precision + recall)

## The F score is used to measure a test's accuracy, and it balances the use of precision and recall to do it.

### The F score can provide a more realistic measure of a test's performance by using both precision and recall

F1 =2 * (precision * recall) / (precision + recall)

round(F1,2)
X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_train_cols.shape
X_test = X_test[X_train_cols.columns]



X_test.shape
X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res.predict(X_test_sm)
y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)

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



y_pred_final= y_pred_final.rename(columns={ 0 : 'Lead_Score_Prob'})
# Rearranging the columns



y_pred_final = y_pred_final.reindex(['CustID','Converted','Lead_Score_Prob'], axis=1)
# Adding Lead_Score column



y_pred_final['Lead_Score'] = round((y_pred_final['Lead_Score_Prob'] * 100),0)



y_pred_final['Lead_Score'] = y_pred_final['Lead_Score'].astype(int)
# Let's see the head of y_pred_final

y_pred_final.head()
y_pred_final['Final_Predicted_Hot_Lead'] = y_pred_final.Lead_Score_Prob.map(lambda x: 1 if x > 0.33 else 0)
y_pred_final.head()
# Let's check the overall accuracy.

round(metrics.accuracy_score(y_pred_final.Converted, y_pred_final.Final_Predicted_Hot_Lead),2)
confusion4 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.Final_Predicted_Hot_Lead )

confusion4
TP = confusion4[1,1] # true positive 

TN = confusion4[0,0] # true negatives

FP = confusion4[0,1] # false positives

FN = confusion4[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

round((TP / float(TP+FN)),2)
# Let us calculate specificity

round(TN / float(TN+FP),2)
### Generating table

resultingTable = pd.merge(y_pred_final,leadScoreDataset,how='inner',left_on='CustID',right_index=True)

resultingTable[['Lead Number','Lead_Score']].head()
resultingTable.shape