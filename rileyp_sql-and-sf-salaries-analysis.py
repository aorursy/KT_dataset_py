import numpy as np

import pandas as pd 

import sqlite3

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#define the database (in the current working directory)

db = "database.sqlite"
#connect to the db

conn = sqlite3.connect(db)



#display the tables in the dataset

tables = pd.read_sql("""SELECT *

                        FROM sqlite_master

                        WHERE type='table'

                     """, conn)

tables
#define the salaries dataframe from the db now that we know the tables in the database

salaries = pd.read_sql('''SELECT * FROM Salaries''', conn)



#check out the head of salaries df to see what we're working with

salaries.head()
#thats a good chunk of data! And we have 13 features, ok cool

salaries.shape
#This is the feature space, agency is either irrelevant (everyone is in SF) or the dataset is of government employees

#damn with that salary? May drop that column later.

salaries.columns
#the data is from 2011 to 2014 so these salaries are low compared to 2019! Yikes

salaries['Year'].unique()
#as suspsected 'Agency' is useless. Bye.

salaries['Agency'].unique()
salaries = salaries.drop(columns='Agency')
#look at us, no null values...that's not likely, but thanks Kaggle.

salaries.isnull().sum()
#will need to convert those objects to ints for sure

salaries.dtypes
#the error thrown here told me that while I thought there were no null values before, the null values 

#are actually the string "Not Provided" so I will remove that now.

#details: returned a ValueError: ('Unable to parse string "Not Provided" at position 148646', 'occurred at index BasePay')

salaries = salaries[['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']].apply(pd.to_numeric)
#removes the rows that contain "Not Provided"

salaries = salaries[~salaries['BasePay'].isin(['Not Provided'])]
salaries[['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']] = salaries[['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']].apply(pd.to_numeric)
#double checked that it only removed 4 instances :)

salaries.shape
#I realized that pd.to_numeric applied to the df introduced NaNs. Check it out:

#Actually, for some reason these NaNs are behaving weird. Here they are:

sns.heatmap(salaries.isnull(), yticklabels=False, cbar=False, cmap='viridis');
#Now I can make a new df with noNaN and compare how much volume was lost

salaries_noNaN = salaries.dropna()



#doing a new df to compare how many rows we lost

sns.heatmap(salaries_noNaN.isnull(), yticklabels=False, cbar=False, cmap='viridis');
class format:

    BOLD = '\033[1m'

    UNBOLD = '\033[0m'



print("The new dataframe size is: " + format.BOLD + str(salaries_noNaN.shape[0]) + format.UNBOLD)

print("The old dataframe size is: " + format.BOLD + str(salaries.shape[0]) + format.UNBOLD)



print( "We lost " + format.BOLD + "{0:.0f}%".format(1/4 * 100) + format.UNBOLD + " of the observations.")
salaries.describe().transpose()
#another useless column. dropping

salaries['Notes'].unique()
salaries_noNaN = salaries_noNaN.drop(columns=['Notes'])

salaries_histogrammable = salaries_noNaN.drop(columns=['EmployeeName', 'JobTitle', 'Status'])



#salaries_histogrammable is my quantitative dataframe that I can run algorithms on so I'm going to clean that one up now too

salaries_histogrammable[['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']] = salaries_histogrammable[['BasePay', 'OvertimePay', 'OtherPay', 'Benefits']].apply(pd.to_numeric)



salaries_histogrammable = salaries_histogrammable.dropna()
#Looking at the histograms below we can see that the sampling method behind this study is a bit biased.

#It looks like only extreme values are present.

def histogram_grid(df, variables, n_rows, n_cols):

    fig=plt.figure(figsize=(14,10))

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        df[var_name].hist(bins=10,ax=ax)

        ax.set_title(var_name+" Distribution")

    fig.tight_layout()  # Improves appearance a bit.

    plt.show()



histogram_grid(salaries_histogrammable, salaries_histogrammable.columns, 3, 3)
#checking out the average pay before benefits by year, #we lost 2011 because they didn't have benefits data

salaries_noNaN[['Year', 'TotalPay']].groupby('Year').mean()
#This data looks much more reasonable. Let's look at the change with a dot chart. It looks like there was a spike in 2013

#in average pay with benefits.

year_pay_benefits = salaries_noNaN[['Year', 'TotalPayBenefits']].groupby('Year').mean().plot(style='.-', xticks=[2012, 2013, 2014])

year_pay_benefits
#Now I'm going to use status PT or FT as my binary classifier, my dependent variable

#and use different algorithms to predict employment status by BasePay and such.

pt = salaries_noNaN[salaries_noNaN['Status'] == 'PT']



ft = salaries_noNaN[salaries_noNaN['Status'] == 'FT']



#We can see by the difference in shapes that there are status values other than FT and PT. What are they you may ask?



print(ft.shape)

print(pt.shape)

print(salaries_noNaN.shape)
#They are empty strings! Bye bye!

salaries_noNaN[salaries_noNaN['Status'] == ''].head()
salaries_noNaN = salaries_noNaN[(salaries_noNaN.Status == 'FT') | (salaries_noNaN.Status == 'PT')]
#check and make sure that it worked, that ft and pt are the same length as the new salaries_noNaN

pt.shape[0] + ft.shape[0] == salaries_noNaN.shape[0]
#quantiative variables that make sense for a heatmap of correlations

corr_matrix_quants = salaries_noNaN.iloc[:,3:-2]



sns.heatmap(corr_matrix_quants.corr(), annot=True);
sns.countplot(x='Status', data=salaries_noNaN)
#I was hoping I could disaggregate the data by job title or at least cluster them into a industry variable

#but that would be a lot of work. I may come back to this column with NLP

len(salaries_noNaN['JobTitle'].unique())
salaries_noNaN.head()
#made the status FT = 1 and PT = 0 to be able to work with it with regression models

salaries_binary = salaries_noNaN.replace(to_replace=['FT', 'PT'],

                             value=[1,0])



salaries_binary = salaries_binary.iloc[:,4:]

salaries_binary.head()
def linear_regression(df, x_columns, y_column):

    '''Takes a dataframe, the x variables (columns of the 

    dataframe) to be trained on as an array and the 

    y column to use in predictions and testing variable 

    and returns the coefficients for each x variable 

    on y (the dependent variable).

    

    Requires a dataframe with only quantitative values, 

    no strings.

    

    Example of the required data cleaning and then function call:

    df = pd.read_csv("loan_data.csv")

    df = df.drop(['string columns'], axis=1)

    x_columns = df['x_a', 'x_b', ...'x_n']

    y_column = df['y_variable']

    

    linear_regression(df, x_columns, y_column)

    

    

                Coeff

    credit.policy	 11.359122

    inq.last.6mths	 0.236195

    installment	     0.040879

    days.with.cr.li  0.002172

    revol.bal	     0.000043

    dti	-0.224324

    log.annual.inc	-0.238352

    revol.util	-0.361837

    not.fully.paid	-2.132126

    delinq.2yrs	-9.282614

    pub.rec	-10.164147

    int.rate	-786.193923

    '''

    X = x_columns

    y = y_column

    

    from sklearn.model_selection import train_test_split

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    

    from sklearn.linear_model import LinearRegression

    

    lm = LinearRegression()

    

    lm.fit(X_train, y_train)

    

    cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])

    

    return cdf.sort_values('Coeff', ascending=False)
x_columns = salaries_binary.iloc[:,:-1]

y_column = salaries_binary['Status']



linear_regression(salaries_binary, x_columns, y_column)
import statsmodels.api as sm

from scipy import stats



X = x_columns

y = y_column



X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
#The only statistically insignificant variable is year, so I will drop that here to not overfit my model

#It also didn't have a large coefficient in the table earlier.

x_columns = salaries_binary.iloc[:,:-1].drop(columns='Year')

y_column = salaries_binary['Status']



linear_regression(salaries_binary, x_columns, y_column)
from sklearn.model_selection import train_test_split



X = x_columns

y = salaries_binary['Status']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)



from sklearn.linear_model import LogisticRegression



#instantiate the model

logmodel = LogisticRegression()



logmodel.fit(X_train, y_train)
#predict values based on X_test data

predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix



print(format.BOLD + "Classification Report: " + format.UNBOLD)

print(classification_report(y_test, predictions))



print(format.BOLD + "Confusion Matrix: " + format.UNBOLD)

print(confusion_matrix(y_test, predictions))
# The average of each of the X variables leads to a prediction of FT work status.

Xnew = [[5402, 3505, 24790, 75472, 100261]]



#Prediction method for logmodel

ynew = logmodel.predict(Xnew)

ynew
#train a single decision tree to start

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
print(format.BOLD + "Classification Report: " + format.UNBOLD)

print(classification_report(y_test, predictions))



print(format.BOLD + "Confusion Matrix: " + format.UNBOLD)

print(confusion_matrix(y_test, predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(format.BOLD + "Classification Report: " + format.UNBOLD)

print(classification_report(y_test, rfc_pred))



print(format.BOLD + "Confusion Matrix: " + format.UNBOLD)

print(confusion_matrix(y_test, rfc_pred))
# Get numerical feature importances

importances = list(rfc.feature_importances_)



# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]



# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)



# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(salaries_binary.drop('Status', axis=1))
kmeans.cluster_centers_
from sklearn.metrics import confusion_matrix, classification_report



print(format.BOLD + "Classification Report: " + format.UNBOLD)

print(classification_report(salaries_binary['Status'], kmeans.labels_))



print(format.BOLD + "Confusion Matrix: " + format.UNBOLD)

print(confusion_matrix(salaries_binary['Status'], kmeans.labels_))