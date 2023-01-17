#1. Develop a statistical model for predicting bad loans

#2. Use the model to identify the most important drivers of bad loans

#3. With these new insights, make recommendations to avoid funding bad loans
#importing libraries

import os

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



#import libraries for logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



#import libraries for model evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score



# importing libraries

import os

import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



#import libraries for logistic regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



#import libraries for model evaluation

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.metrics import average_precision_score



#import libraries for KNN

from sklearn.neighbors import KNeighborsClassifier



#import library for ignoring warnings in the output

import warnings

warnings.filterwarnings('ignore')



#import libraries for KNN

from sklearn.neighbors import KNeighborsClassifier



#import library for ignoring warnings in the output

import warnings

warnings.filterwarnings('ignore')
#defining working directory

#os.chdir('C:\\Users\\pallavvi\\Desktop\\ProjMat\\Data Science\\Data_Science_Course\\Piyush\\Projects\\Upload_Online\\Loan_status_prediction')
#importing dataset

target_csv = "../input/Loan_Training_data.csv"

loan_df = pd.read_csv(target_csv,sep=',')
#first five observations

loan_df.head()
#data info

loan_df.info()
#listing all the variables/fields

loan_df.columns
#shape of the data(number of rows and number of columns)

loan_df.shape
#Finding how many features are categories/numerical in data set

print(len(loan_df._get_numeric_data().columns))                  #numerical variables

len(loan_df.columns)-len(loan_df._get_numeric_data().columns)    #categorical variables
#finding How many columns have got some missing values

print(loan_df.isnull().any().sum())
#finding How many rows have got some missing values

print(loan_df.isnull().any(axis=1).sum())
#Finding How many column are having all missing values

print(loan_df.isnull().all().sum())
#Finding How many rows are having all missing values

print(loan_df.isnull().all(axis=1).sum())
#drop duplicate rows- if any

loan_df = loan_df.drop_duplicates(keep='first')
#dropping unimportant variables

#drop_cols = ['Education']

#loan_df = loan_df.drop(drop_cols,axis=1)
#Finding How many/what percentage of applications got approved /rejected? This is basically to check class imbalance.



print(pd.value_counts(loan_df['Loan_Status'].values, sort=False))                                #Gives count

round(100*((pd.value_counts(loan_df['Loan_Status'].values, sort=False))/len(loan_df.index)),2)   #Gives %
#Identifying if there are outliers in any feature using describe()

#summary statistics (gives mean , median, std, min, max for continuous numerical data only)



loan_df.describe()



#Explanation:

#1. if 25%,50%,75% values are far away from mean value that means there are outliers in that feature
#Identifying if there are outliers in any feature using quantile()

Q1 = loan_df.quantile(0.25)

Q3 = loan_df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)



lower_limit  = Q1-1.5*IQR

print(lower_limit)



upper_limit = Q3+1.5*IQR

print(upper_limit)



print(((loan_df['ApplicantIncome'] > 10171.250) | (loan_df['ApplicantIncome'] < 1498.750)).sum())      #gives count

print(((loan_df['CoapplicantIncome'] > 5743.125) | (loan_df['CoapplicantIncome'] < 3445.875)).sum())

print(((loan_df['LoanAmount'] > 270.000) | (loan_df['LoanAmount'] < 2.000)).sum())

print(((loan_df['Loan_Amount_Term'] > 360.000) | (loan_df['Loan_Amount_Term'] < 360.000)).sum())

print(((loan_df['Credit_History'] > 1) | (loan_df['Credit_History'] < 1)).sum())
#finding the % of nulls in each column

round(100*(loan_df.isnull().sum()/len(loan_df.index)), 2)
#removing nulls from Loan Amount, Credit History, Loan amount term 

loan_df = loan_df[~np.isnan(loan_df['LoanAmount'])]

round(100*(loan_df.isnull().sum()/len(loan_df.index)), 2)



loan_df = loan_df[~np.isnan(loan_df['Credit_History'])]

round(100*(loan_df.isnull().sum()/len(loan_df.index)), 2)



loan_df = loan_df[~np.isnan(loan_df['Loan_Amount_Term'])]

round(100*(loan_df.isnull().sum()/len(loan_df.index)), 2)



#alternate way to remove NA's from columns

#loan_df.dropna(subset = ["Married", "Gender","Self_Employed","Loan_Amount_Term","Dependents","LoanAmount","Credit_History"] , inplace=True)
#removing nulls from self_employed, Gender, Dependents, Married(categorical variables)

print(loan_df['Self_Employed'].value_counts())

loan_df.loc[pd.isnull(loan_df['Self_Employed']), ['Self_Employed']] = 'No'

round(100*(loan_df.isnull().sum()/len(loan_df.index)), 2)



print(loan_df['Gender'].value_counts())

loan_df.loc[pd.isnull(loan_df['Gender']), ['Gender']] = 'Male'

round(100*(loan_df.isnull().sum()/len(loan_df.index)), 2)



print(loan_df['Dependents'].value_counts())

loan_df.loc[pd.isnull(loan_df['Dependents']), ['Dependents']] = '0'

round(100*(loan_df.isnull().sum()/len(loan_df.index)), 2)



print(loan_df['Married'].value_counts())

loan_df.loc[pd.isnull(loan_df['Married']), ['Married']] = 'Yes'

round(100*(loan_df.isnull().sum()/len(loan_df.index)), 2)



#alternate way to find the count per feature

#loan_df.groupby(["Dependents"]).size()
#gives unique values of a feature/column

set(loan_df['Married'])



#alternate way to find the unique values

#loan_df.Married.unique()
#Finding the fraction of rows cost

print('data left(%):',(len(loan_df.index)/614)*100)
#calculating the combined income and inserting a new column to loan_df

loan_df['comb_income'] = pd.DataFrame(loan_df['ApplicantIncome']+loan_df['CoapplicantIncome'])
loan_df.head()
#dropping columns applicant and coapplicant income 

drop_cols = ['ApplicantIncome','CoapplicantIncome']

loan_df = loan_df.drop(drop_cols,axis=1)
#sorting with respect to a variable in descending order

loan_df = loan_df.sort_values(by = 'comb_income',ascending = False)
loan_df.shape
#converting the categorical target variable into dummies (0,1's)

loan_df['Status'] = pd.get_dummies(loan_df['Loan_Status'], drop_first=True)

loan_df.drop(['Loan_Status'], axis=1, inplace=True)
#if there is a space in columnnames then replace it

loan_df.columns = loan_df.columns.str.replace(' ','')
'''The functions takes column name as an argument and returns the top and bottom observation in that dataframe'''

def min_max_values(col):

    

    top = loan_df[col].idxmax()

    top_obs = pd.DataFrame(loan_df.loc[top])

    

    bottom = loan_df[col].idxmin()

    bottom_obs = pd.DataFrame(loan_df.loc[bottom])

    

    min_max_obs = pd.concat([top_obs, bottom_obs], axis=1)

    

    return min_max_obs
#applicable only for numerical variable

min_max_values('comb_income')

min_max_values('LoanAmount')
#histogram of continuous numerical variables

plt.hist(loan_df['comb_income'],color='Magenta')



#observation:

#1. majority of people have combined income between a range of approx.1000 to 10000

#2. data spread is very high on right hand side

#3. people with income 40000 to 80000 are far away from the other set of people 

#4. data on tail side can be removed for analysis
#probability Distribution Function

sns.distplot(loan_df['comb_income'],color='green')



#observation:

#1. In histogram we get the count/actual value for every data on x axis, But here we get % instead of actual value

#2. Area under the curve here is equal to '1'

#3. % value always lie between '0' and '1'
#histogram of continuous numerical variables

plt.hist(loan_df['LoanAmount'],color='Magenta')



#observation:

#1. majority of people applied for loan amount between a range of approx.90 to 130

#2. data spread is high on right hand side

#3. people having loan_amount applied=500-700 are far away from the other set of people 

#4. data on tail side can be removed for analysis
#probability Distribution Function

sns.distplot(loan_df['LoanAmount'],color='green')



#observation:

#1. In histogram we get the count/actual value for every data on x axis, But here we get % instead of actual value

#2. Area under the curve here is equal to '1'

#3. % value always lie between '0' and '1'
#histogram of continuous numerical variables

plt.hist(loan_df['Loan_Amount_Term'],color='Magenta')



#observation:

#1. the data is highly distributed 

#2. maximum people have term amount between 350 to 390 approx.
#probability Distribution Function

sns.distplot(loan_df['Loan_Amount_Term'],color='green')



#observation:

#1. In histogram we get the count/actual value for every data on x axis, But here we get % instead of actual value

#2. Area under the curve here is equal to '1'

#3. % value always lie between '0' and '1'
#histogram of continuous numerical variables

plt.hist(loan_df['Credit_History'],color='Magenta')



#observation:

#1. can be better represented by categopry

#2. maximum people have credit history between 0.9 to 1.0 approx.
#probability Distribution Function

sns.distplot(loan_df['Credit_History'],color='green')



#observation:

#1. In histogram we get the count/actual value for every data on x axis, But here we get % instead of actual value

#2. Area under the curve here is equal to '1'

#3. % value always lie between '0' and '1'
#select all numerical variables and analyzing them

loan_num = loan_df.select_dtypes(include=['float64','int64'])

loan_num.head()
loan_num.hist(color='orange',histtype='bar',stacked=True,fill=True,figsize=(10,10))



#observation:

#1. Credit history can be better represented by making it as 'Category'

#2. For Loan amount Term a) the data is highly distributed 

#                        b) maximum people have term amount between 350 to 390 approx.

#3. For Loan Amount a) majority of people applied for loan amount between a range of approx.90 to 130

#                   b) data spread is high on right hand side

#                   c) people having loan_amount applied=500 are far away from the other set of people 

#                   d) data on tail side can be removed for analysis

#4. For Comb Income a) majority of people have combined income between a range of approx.1000 to 10000

#                   b) data spread is very high on right hand side

#                   c) people with income approx.38000 are far away from the other set of people 

#                   d) data on tail side can be removed for analysis
#Correlation with the variable of interest

comb_income_corr = loan_num.corr()['comb_income'][:-1]  #[:-1] is for not taking the correlation with variable itself

comb_income_corr



#observation:

#1. LoanAmount is directly proportional to combined income, and other numerical variables are inversely proportional

#2. LoanAmount has +ve relation and other two have -ve relation

#3. loan amount has medium relation with comb_income, so as the combined income increases the loan amount will also increase

#4. There is no relation between comb_income and (loan_amoun_term and credit history), since correlation value is < 0.1
#correlation plots using 'pairplots'

for i in range(0,len(loan_num.columns),3):

    sns.pairplot(loan_num,y_vars=['comb_income'],

                 x_vars=loan_num.columns[i:i+3],

                 kind='reg',

                 plot_kws={'line_kws':{'color':'red'}},

                 height=3

                )



#observation:

#1. If the dots are downward sloping then negative relation, upward sloping positive relation

#2. comb_income and loan_amount has medium positive relationship as the variance is medium

#3. Useful for detecting the outliers/extreme values
sns.pairplot(loan_df, 

             hue='Status', 

             palette='husl', 

             height=3, 

             markers=["o", "s"], 

             diag_kind="kde")
#plotting significant correlation in one map ('heatmap')

corr = loan_num.corr()

sns.heatmap(corr[(corr >= 0.5)], 

            cmap='viridis', 

            annot=True, 

            vmax=1.0, 

            vmin=1.0, 

            linewidths=0.1, 

            annot_kws={"size":8}, 

            square=True

           )



#observation:

#1. there is no relation between the loan_amount, loan_amount term anmd credit history

#2. squares which have no color indicates that no relationship

#3. there is a high relation between loan amount and comb_income
#regression Plot

sns.regplot(loan_df['LoanAmount'],loan_df['comb_income'],scatter_kws={"color": "red"}, line_kws={"color": "blue"})



#observation:

#1. there is strong positive relationship between loan_amount and comb_income
#regression Plot

sns.regplot(loan_df['Loan_Amount_Term'],loan_df['comb_income'],scatter_kws={"color": "green"}, line_kws={"color": "blue"})



#observation:

#1. there is no relationship between loan_amount_term and comb_income
#regression Plot

sns.regplot(loan_df['Credit_History'],loan_df['comb_income'],scatter_kws={"color": "black"}, line_kws={"color": "blue"})



#observation:

#1. there is no relation between credit history and com_income
#linear plot

sns.lmplot(x="comb_income" , y = "LoanAmount" , hue = "Status" , markers=["o", "D"], data = loan_df, palette='Set1')



#observation:

#1. As the comb_income increase loan amount also increases
#linear plot

sns.lmplot(x="comb_income" , y = "LoanAmount" ,hue='Status', col = "Status", data=loan_df)



#observation:

#1. base on status impact of comb_income on loan amount is visualized
#linear plot

sns.lmplot(x="comb_income" , y = "LoanAmount" ,row='Status', col = "Credit_History", data=loan_df)



#observation:

#1. condition of two variables status and credit history on comb_income affecting loan_amount

#2. credit history plays a role in approval and rejection of loan application
#joint plot with kind=reg

sns.jointplot(x='LoanAmount', y='comb_income', data=loan_df, kind='reg', color='yellow',joint_kws={'line_kws':{'color':'cyan'}})



#Observation:

#1. Curve on the top shows distribution of Loan Amount

#2. Curve on the right shows distribution of comb_income

#3. Plot shows as loan amount increases com_income requriement also increase i.e loan_amount is directly proportional to comb_income
#joint plot with kind=scatter

sns.jointplot(x='LoanAmount', y='comb_income', data=loan_df, kind='scatter', joint_kws={'color':'red'})
#Joint Plot with kind=kde

sns.jointplot(x='LoanAmount', y='comb_income', data=loan_df, kind='kde', color='green')
#Joint Plot with kind=resid

sns.jointplot(x='LoanAmount', y='comb_income', data=loan_df, kind='resid', color='blue')
loan_df.head()
#Count by category- cross tabulation  (simply gives count of each categopry)

gender_dist = loan_df.groupby('Gender').size()

gender_dist
#distribution of categorical variables

#count plot

sns.countplot(loan_df.Gender)



#observation:

#1. Number of male applicants are much greater than female applicants
#Count by category- cross tabulation  (simply gives count of each categopry)

gender_dist = loan_df.groupby('Married').size()

gender_dist
#distribution of categorical variables

#count plot

sns.countplot(data=loan_df,x= 'Married', hue='Status')



#observation:

#1. Married applicants are getting the loan approved compared to unmarried 
#Count by category- cross tabulation  (simply gives count of each categopry)

gender_dist = loan_df.groupby('Education').size()

gender_dist
#distribution of categorical variables

#count plot

sns.countplot(data=loan_df,x= 'Education', hue='Status')



#observation:

#1. graduate applicants are getting loans approved compared to undergraduates
#Count by category- cross tabulation  (simply gives count of each categopry)

gender_dist = loan_df.groupby('Dependents').size()

gender_dist
#distribution of categorical variables

#count plot

sns.countplot(data=loan_df,x= 'Dependents', hue='Status')



#observation:

#1. applicants with 0 dependents got the loan approved

#2. applicants with 3+ have very less number whose loan got approved

#3. applicants with 1 and 2 dependents are average compared to other two
#Count by category- cross tabulation  (simply gives count of each categopry)

gender_dist = loan_df.groupby('Self_Employed').size()

gender_dist
#distribution of categorical variables

#count plot

sns.countplot(data=loan_df,x= 'Self_Employed', hue='Status')



#observation:

#1. those who are not self employed, their loans approved as compared to who are self employed
#Count by category- cross tabulation  (simply gives count of each categopry)

gender_dist = loan_df.groupby('Property_Area').size()

gender_dist
#distribution of categorical variables

#count plot

sns.countplot(data=loan_df,x= 'Property_Area', hue='Status')



#observation:

#1. numnber of Applicants having property area in semiurban > urban > rural

#2. applicants having semiurban property area have loans approved
#Count by category- cross tabulation  (simply gives count of each categopry)

gender_dist = loan_df.groupby('Status').size()

gender_dist
#distribution of categorical variables

#count plot

sns.countplot(loan_df.Status)



#observation:

#1. Loan is approved for more number of applicants(almost double than rejected)
#Swarmplot

sns.swarmplot(x='Property_Area', y ='comb_income', hue ='Status', data=loan_df, split=True)



#Observation:

#1. We have few outliers in Urban and semiurban area

#2. Applicants whose property area is urban are more likely to get loan approved

#3. there is no effect on comb_income for property area
#Swarmplot

sns.swarmplot(x='Property_Area', y ='comb_income', hue ='Credit_History', data=loan_df, split=True)
#boxplot for categorical variables

#box_gender = sns.boxplot(x='Gender', y='comb_income', data=loan_df, palette='Set1', linewidth=0.8, width=0.4, whis=5)

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_gender = sns.boxplot(x='Gender', y='comb_income', data=loan_df, ax=ax)



#observation:

#1. there is more variation of combined income in female category compared to male

#2. Males having comb_income of approx. >12000 are outliers

#3. Females having comb_income of approx. >9500 are outliers
#gives total count, unique values of,top category,and no of records of top category

loan_df['Gender'].describe()
#boxplot for categorical variables

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_married = sns.boxplot(x='Married', y='comb_income', data=loan_df, palette='Set1', ax=ax)



#observation:

#1. there is more variation of combined income in married category compared to not married

#2. Males having comb_income of approx. >12000 are outliers

#3. Females having comb_income of approx. >10000 are outliers
#boxplot for categorical variables

#box_dependents = sns.boxplot(x='Dependents', y='comb_income', data=loan_df, palette='Set2', linewidth=1.5, width=0.8)

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_dependents = sns.boxplot(x='Dependents', y='comb_income', data=loan_df, palette='Set2', ax=ax)



#observation:

#1. there is more variation of combined income 3+ dependents

#2. For 1 dependent, data is equally centered
#boxplot for categorical variables

#box_self_employed = sns.boxplot(x='Self_Employed', y='comb_income', data=loan_df, palette='Set2', linewidth=1.5, width=0.8)

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_self_employed = sns.boxplot(x='Self_Employed', y='comb_income', data=loan_df, palette='Set3', ax=ax)



#observation:

#1. Combined income of people who are not self_employed is higher than self_employed people

#2. variation of combined income is high in not self_employed category
#boxplot for categorical variables

#box_prop_area = sns.boxplot(x='Property_Area', y='comb_income', data=loan_df, palette='Set2', linewidth=1.5, width=0.8)

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_prop_area = sns.boxplot(x='Property_Area', y='comb_income', data=loan_df, palette='Set2', ax=ax)



#observation:

#1. Urban property area has more variation in comb_income and has more outliers than rural/semiurban 

#2. Rural and semiurban has approximately equal variance in comb_income
#boxplot for categorical variables

#box_com_inc = sns.boxplot(x='Status', y='comb_income', data=loan_df, palette='Set2', linewidth=1.5, width=0.8)

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_com_inc = sns.boxplot(x='Status', y='comb_income', data=loan_df, palette='Set3', ax=ax)



#observation:

#1. Variation is almost equal in combined income for approved and rejected loans

#2. Outliers are high in comb_income for approved loans
#boxplot for categorical variables

#box_Loan_Amt = sns.boxplot(x='Status', y='LoanAmount', data=loan_df, palette='Set2', linewidth=1.5, width=0.8)

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_Loan_Amt = sns.boxplot(x='Status', y='LoanAmount', data=loan_df, palette='Set1', ax=ax)



#observation:

#1. Loan Amount varies according to status and the variation is more in rejected loans

#2. Outliers are more in approved loans
#boxplot for categorical variables

#box_Loan_amt_hue = sns.boxplot(x='Status', y='LoanAmount', hue='Self_Employed', data=loan_df, palette='Set2', linewidth=1.5, width=0.8)

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_Loan_amt_hue = sns.boxplot(x='Status', y='LoanAmount', hue='Self_Employed', data=loan_df, palette='Set3', ax=ax)



#observation:

#1. Loan amount varies at some extent for self employed and not self employed people

#2. Doesnt really has any effect of   self employed and not self employed on loan approval or rejection
#boxplot for categorical variables

#box_comb_inc_hue = sns.boxplot(x='Status', y='comb_income', hue='Self_Employed', data=loan_df, palette='Set2', linewidth=1.5, width=0.8)

fig_size = (7, 7)

fig, ax = plt.subplots(figsize=fig_size)

box_comb_inc_hue = sns.boxplot(x='Status', y='comb_income', hue='Self_Employed', data=loan_df, palette='Set1', ax=ax)



#observation:

#1. comb_income varies alot of self employed and non self employed people

#2. no such effect of  self employed and not self employed on loan status 
sns.barplot(x='comb_income', y='Status', data=loan_df)
sns.barplot(x='LoanAmount', y='Status', hue='Gender', data=loan_df, color='violet')
sns.barplot(x='comb_income', y='Status', hue='Gender' ,  data=loan_df, color='purple')
sns.barplot(x='comb_income', y='Self_Employed', hue='Gender' , data=loan_df, color='yellow')
sns.barplot(x='comb_income', y='Self_Employed', hue='Married' , data=loan_df, color='green')
sns.barplot(x='LoanAmount', y='Gender', hue='Married' , data=loan_df, color='brown', palette='Blues_d')
sns.barplot(x='LoanAmount', y='Gender', hue='Status' , data=loan_df, color='red')
loan_df.head()
loan_df.describe()
loan_df.columns
cols = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 

       'Property_Area', 'Status']



x = loan_df.drop(cols,axis=1)

y = loan_df['Status']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=20)



logmodel = LogisticRegression()

logmodel.fit(x_train, y_train)



y_pred = logmodel.predict(x_test)

y_pred_prob = logmodel.predict_proba(x_test)

y_pred1 = logmodel.predict(x_train)



print(logmodel.intercept_ )

print(logmodel.coef_ )



print(classification_report(y_test , y_pred))

print(classification_report(y_train , y_pred1))



print(confusion_matrix(y_test,y_pred))

print(confusion_matrix(y_train,y_pred1))



print("test accuracy:",accuracy_score(y_test,y_pred )*100)

print("train accuracy:",accuracy_score(y_train,y_pred1)*100)

#new_loan= loan_df

#new_loan["Prediction"] = logmodel.predict(x)



#new_loan.head()
#new_loan.to_csv('out.csv',index=False)
#checking if the test data before and after data cleansing is consistent

y_test.value_counts(normalize=True)
#checking if the train data before and after data cleansing is consistent

y_train.value_counts(normalize=True)
loan_df.Status.value_counts(normalize=True)
Sensitivity = []

Specificity = []

Accuracy = []

prob_range = np.linspace(0.6,0.8,50)

for pc in prob_range:

   y_pred_res = [1 if np.any(prob>=pc) else 0 for prob in y_pred_prob ]

   conf_mat = confusion_matrix(y_test,y_pred_res)

   Accuracy.append(accuracy_score(y_test,y_pred_res))

   Sensitivity.append(conf_mat[1,1]/(conf_mat[1,1] + conf_mat[1,0]))

   Specificity.append(conf_mat[0,0]/(conf_mat[0,0] + conf_mat[0,1]))



plt.plot(prob_range,Accuracy)

plt.plot(prob_range,Sensitivity)

plt.plot(prob_range,Specificity)

plt.show()
#the probability of being y=1 >= 0.78

y_pred = (logmodel.predict_proba(x_test)[:,1] >= 0.78).astype(bool)

y_pred
cols = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 

       'Property_Area', 'Status']



x = loan_df.drop(cols,axis=1)

y = loan_df['Status']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=20)



#instantiate the classifier

logmodel = LogisticRegression()



#fit a model

logmodel.fit(x_train, y_train)



#predict probabilities and keep probabilities for the positive outcome only

probs = logmodel.predict_proba(x_test)[:,1]



auc = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc)



#calculating the ROC curve

fpr, tpr, thresholds = roc_curve(y_test, probs)



#plot NO skill

plt.plot([0, 1], [0, 1], linestyle='--')



# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')



# show the plot

plt.show()

cols = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 

       'Property_Area', 'Status']



x = loan_df.drop(cols,axis=1)

y = loan_df['Status']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=20)



#instantiate the classifier

logmodel = LogisticRegression()



#fit a model

logmodel.fit(x_train, y_train)



#predict probabilities and keep probabilities for the positive outcome only

probs = logmodel.predict_proba(x_test)[:,1]



#predict class values

y_pred = logmodel.predict(x_test)



# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, probs)



# calculate F1 score

f1 = f1_score(y_test, y_pred)



# calculate precision-recall AUC

#auc = auc(recall, precision)



# calculate average precision score

ap = average_precision_score(y_test, probs)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))



# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')



# plot the precision-recall curve for the model

plt.plot(recall, precision, marker='.')



# show the plot

plt.show()
#help(roc_auc_score)
cols = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 

       'Property_Area', 'Status',]



x = loan_df.drop(cols,axis=1)

y = loan_df['Status']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=20)



neighbors = np.arange(1, 17)

train_accuracy = np.empty(len(neighbors))   # creates an empty array of size 9

test_accuracy = np.empty(len(neighbors))



for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train) 

    train_accuracy[i] = knn.score(x_train,y_train)

    test_accuracy[i] = knn.score(x_test, y_test)



plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()





#took the appropriate value of k

knn = KNeighborsClassifier(n_neighbors=16)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

print(y_pred)

score = knn.score(x_test, y_test)

score1 = knn.score(x_train, y_train)

print("test score:",score, "train score:",score1)
print(confusion_matrix(y_test , y_pred))



pd.crosstab(y_test , y_pred , rownames=["True"], colnames=["Predicted"] ,margins=True)
print(classification_report(y_test , y_pred))
y_pred_proba = knn.predict_proba(x_test)[:,1]



from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Knn')

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.title('Knn(n_neighbors=16) ROC curve')

plt.show()
#import GridSearchCV



from sklearn.model_selection import GridSearchCV



#In case of classifier like knn the parameter to be tuned is n_neighbors

param_grid = {'n_neighbors': np.arange(1,150)}



knn = KNeighborsClassifier()

knn_cv= GridSearchCV(knn,param_grid)

knn_cv.fit(x,y)



print("KNN Best Score : " , knn_cv.best_score_)



print("KNN Best K Fit : " , knn_cv.best_params_)

knnf = KNeighborsClassifier()



knnf.fit(x,y)



y_pred = knnf.predict(x)



y_pred.shape
loan_df["Prediction"] = y_pred
pd.pivot_table (  data  = loan_df , index="Status" ,  aggfunc='count')
#This concludes that logistic regression model best suits this loan status prediction problem.



#for any queries reach me @pallavivibhute31@gmail.com.