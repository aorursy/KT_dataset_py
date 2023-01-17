#Import the necessary libraries

import numpy as np

import pandas as pd

import seaborn as sns

import scipy.stats as stats 

from scipy.stats import f_oneway

import matplotlib.pyplot as plt

from statsmodels.stats.proportion import proportions_ztest

from sklearn.preprocessing import LabelEncoder   # import label encoder
#Read the data as a data frame

ins_data = pd.read_csv('../input/insurance/insurance.csv')

ins_data.head()

# 3a. Shape of the data 

ins_data.shape
# 3b. Data type of each attribute 

ins_data.info()   # it gives information about the data and data types of each attribute
# 3c. Checking the presence of missing values

null_counts = ins_data.isnull().sum()  # This prints the columns with the number of null values they have

print (null_counts)
# 3d. 5 point summary of numerical attributes

ins_data.describe()
# 3e. Distribution of ‘bmi’, ‘age’ and ‘charges’ columns. 

print('Distribution of BMI')

print('Mean of BMI:',ins_data['bmi'].mean())

print('Median of BMI:',ins_data['bmi'].median())

print('Mode of BMI:',ins_data['bmi'].mode())



# Visualize the distribution of ‘bmi’ with the plot

sns.distplot(ins_data['bmi'])    # Distribution of ‘bmi’

plt.show()





print('\n\nDistribution of Age')

print('Mean of Age:',ins_data['age'].mean())

print('Median of Age:',ins_data['age'].median())

print('Mode of Age:',ins_data['age'].mode())



# Visualize the distribution of ‘age’ with the plot

sns.distplot(ins_data['age'])     # Distribution of ‘age’

plt.show()



print('\n\nDistribution of Charges')

print('Mean of charges:',ins_data['charges'].mean())

print('Median of charges:',ins_data['charges'].median())

print('Mode of charges:',ins_data['charges'].mode())



# Visualize the distribution of ‘charges’ with the plot

sns.distplot(ins_data['charges'])   # Distribution of ‘charges’

plt.show()
# 3f. Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns



#Skewness of BMI

print('Skewness of BMI is:',ins_data['bmi'].skew())

h = np.asarray(ins_data['bmi'])  #convert pandas DataFrame object to numpy array and sort

h = sorted(h)

#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(h, np.mean(h), np.std(h))



# Visualize the skewness of BMI with the plot

plt.plot(h,fit,'-',linewidth = 2)  

plt.xlabel('Normal distribution of BMI with same mean and var')

plt.show()



#Skewness of Age

print('Skewness of Age is:',ins_data['age'].skew())

h1 = np.asarray(ins_data['age'])  #convert DataFrame object to numpy array and sort

h1 = sorted(h1)

#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(h1, np.mean(h1), np.std(h1))  



# Visualize the skewness of Age with the plot

plt.plot(h1,fit,'-',linewidth = 2)

plt.xlabel('Normal distribution of Age with same mean and var')

plt.show()



#Skewness of Charges

print('Skewness of Charges is:',ins_data['charges'].skew())

h2 = np.asarray(ins_data['charges'])  #convert DataFrame object to numpy array and sort

h2 = sorted(h2)

#use the scipy stats module to fit a normal distirbution with same mean and standard deviation

fit = stats.norm.pdf(h2, np.mean(h2), np.std(h2))  



# Visualize the skewness of Charges with the plot

plt.plot(h2,fit,'-',linewidth = 2)

plt.xlabel('Normal distribution of Charges with same mean and var')

plt.show()
#3g. Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns



#calculating the outiers in 'bmi' 

Q1_bmi = ins_data['bmi'].quantile(0.25)

Q3_bmi = ins_data['bmi'].quantile(0.75)

IQR_bmi= Q3_bmi - Q1_bmi

print('IQR of bmi is:',IQR_bmi)

bool_bmi= (ins_data['bmi'] < (Q1_bmi - 1.5 *IQR_bmi)) |(ins_data['bmi'] > (Q3_bmi + 1.5 * IQR_bmi))

print('Number of outliers in bmi are:',bool_bmi.sum())   #calculating the number of outliers

# visualizing the presence of outier in 'bmi' using graph

sns.boxplot(x= ins_data['bmi'], color='cyan')

plt.xlabel('BMI')

plt.show()



#calculating the outiers in 'age' 

Q1_age = ins_data['age'].quantile(0.25)

Q3_age = ins_data['age'].quantile(0.75)

IQR_age= Q3_age - Q1_age

print('IQR of age:',IQR_age)

bool_age = (ins_data['age'] < (Q1_age - 1.5 *IQR_age)) |(ins_data['age'] > (Q3_age + 1.5 * IQR_age))

print('Number of outliers in age are:',bool_age.sum())  #calculating the number of outliers

# visualizing the presence of outier in 'age' using graph

sns.boxplot(x=ins_data['age'], color='cyan')

plt.xlabel('AGE')

plt.show()



#calculating the outiers in 'charges' 

Q1_charges = ins_data['charges'].quantile(0.25)

Q3_charges = ins_data['charges'].quantile(0.75)

IQR_charges= Q3_charges - Q1_charges

print('IQR of charges:',IQR_charges)

bool_charges= (ins_data['charges'] < (Q1_charges - 1.5 *IQR_charges)) |(ins_data['charges'] > (Q3_charges + 1.5 * IQR_charges))

print('Number of outliers in charges are:',bool_charges.sum()) #calculating the number of outliers

# visualizing the presence of outier in 'charges' using graph

sns.boxplot(x= ins_data['charges'], color= 'cyan')

plt.xlabel('CHARGES')

plt.show()
#3h. Distribution of categorical columns (include children) 

             

sns.countplot(ins_data['sex'])    # Distibution of the column 'sex'

plt.show() 

sns.countplot(ins_data['children']) # Distibution of the column 'children'

plt.show() 

sns.countplot(ins_data['smoker'])  # Distibution of the column 'smoker'

plt.show() 

sns.countplot(ins_data['region'])  # Distibution of the column 'region'

plt.show() 
#3i. Pair plot that includes all the columns of the data frame



#Label Encoding the variables since pair plots ignores strings.



encoded = ins_data.copy()

encoded.loc[:,['sex','smoker','region']] = encoded.loc[:,['sex','smoker','region']].apply(LabelEncoder().fit_transform) # returns label encoded variable(s)

sns.pairplot(encoded)

plt.show()
#Using t test to test the hypothesis 

smokers = ins_data[ins_data['smoker'] == 'yes']['charges']  # array of charges of smokers

nonsmokers = ins_data[ins_data['smoker'] == 'no']['charges']  # array of charges of non smokers

t_stat, pval = stats.ttest_ind(smokers,nonsmokers, axis = 0)

print('\ntstat value is:',t_stat)

print('pvalue value is:',pval,'\n')

if pval < 0.05:

    print(f'With a p-value of {round(pval,4)} the difference is significant. So, we reject the null Hypothesis')

    print('Thus,Charges of people who smoke differs significantly from the people who dont')

else:

    print(f'With a p-value of {round(pval,4)} the difference is not significant. So, We fail to reject the null Hypothesis')

    print('Thus,Charges of people who smoke does not differ significantly from the people who dont ')

    

# visualising the plot of smokers and nonsmokers against their charges



sns.barplot(data= ins_data,x='smoker',y='charges')

plt.xlabel('SMOKERS')

plt.ylabel('CHARGES')

plt.show()
#Using t test to test the hypothesis 

bmi_males = ins_data[ins_data['sex'] == 'male']['bmi']  # array of bmi of males

bmi_females = ins_data[ins_data['sex'] == 'female']['bmi']  # array of bmi of females

t_stat, pval = stats.ttest_ind(bmi_males,bmi_females, axis = 0)

print('\ntstat value is:',t_stat)

print('pvalue value is:',pval,'\n')

if pval < 0.05:

    print(f'With a p-value of {round(pval,4)} the difference is significant. So, we reject the null Hypothesis')

    print('Thus,Bmi of males differ significantly from that of females')

else:

    print(f'With a p-value of {round(pval,4)} the difference is not significant. So, We fail to reject the null Hypothesis')

    print('Thus,Bmi of males does not differ significantly from females')



# visualising the male and female bmi with the plots

sns.barplot(data= ins_data,x='sex',y='bmi')

plt.xlabel('GENDER')

plt.ylabel('BMI')

plt.show()
#using proportions z test to test the hypothesis



# creating a dummy dataset with a column prefix smoke and suffix 'yes' and 'no'

df_dummies = pd.get_dummies(ins_data, prefix='smoke',columns=['smoker'])



#calculating the proportion of female smokers to male smokers 

n_females = ins_data[ins_data['sex'] == 'female']['smoker'].count()         # number of  females

n_males = ins_data[ins_data['sex'] == 'male']['smoker'].count()           # number of  males 

f_smokers = df_dummies[ins_data['sex'] == 'female']['smoke_yes'].sum()     # number of  female smokers

m_smokers = df_dummies[ins_data['sex'] == 'male']['smoke_yes'].sum()       # number of  male smokers

#prop_fe_smokers = f_smokers/n_females

#prop_m_smokers  = m_smokers/n_males

#print('\nproportion of female smoker : male smoker is = ',prop_fe_smokers,':',prop_m_smokers )



#proportions z test

stat, pval = proportions_ztest([f_smokers, m_smokers] , [n_females, n_males])

print('\nZstat value is:',stat)

print('pvalue value is:',pval,'\n')

if pval < 0.05:

    print(f'With a p-value of {round(pval,4)} the difference is significant. So, we reject the null Hypothesis')

    print('Thus,the proportion of female and male smokers are significantly different')

else:

    print(f'With a p-value of {round(pval,4)} the difference is not significant. So, We fail to reject the null Hypothesis')

    print('Thus, the proportion of female and male smokers are almost same')

    

# Visualizing the smokers against their gender

sns.barplot(data= df_dummies,x='sex',y='smoke_yes')

plt.xlabel('GENDER')

plt.ylabel('SMOKERS')

plt.show()
# conducting an one way Anova f test



# extracting bmi of females with 0,1,2 children in 3 groups

f_0c = ins_data[(ins_data['children'].isin(['0'])) & (ins_data['sex'] == 'female')]['bmi'] #has bmi of females with 0 children

f_1c = ins_data[(ins_data['children'].isin(['1'])) & (ins_data['sex'] == 'female')]['bmi'] #has bmi of females with 1 child

f_2c = ins_data[(ins_data['children'].isin(['2'])) & (ins_data['sex'] == 'female')]['bmi'] #has bmi of females with 2 children



# conducting an one way Anova f test

f_stats,pval = f_oneway(f_0c, f_1c, f_2c)



if pval < 0.05:

    print(f'With a p-value of {round(pval,4)} the difference in bmi is significant. So, we reject the null Hypothesis')

else:

    print(f'With a p-value of {round(pval,4)} the difference in bmi is not significant. So, We fail to reject the null Hypothesis')

    

# Visualising the bmi of females with 0,1 and 2 children using a plot



#df1 has data of females with 0,1 or 2 children

df1 = ins_data[(ins_data['children'].isin(['0', '1','2'])) & (ins_data['sex'] == 'female')] 



sns.stripplot(df1['children'], df1['bmi'])

plt.xlabel('Women with the number of children')

plt.ylabel('BMI')

plt.show()
