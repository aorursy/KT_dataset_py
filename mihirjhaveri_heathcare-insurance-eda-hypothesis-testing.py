#Import the necessary libraries 

import numpy             as np  #adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

import pandas            as pd  #to deal with data analysis and manipulation

import scipy.stats       as stats  #contains a large number of probability distributions as well as a growing library of statistical functions

import matplotlib.pyplot as plt  #is a collection of command style functions that make matplotlib work like MATLAB.

import seaborn as sns        #data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics

#directly below the code cell that produced it.

%matplotlib inline      

import statsmodels.api as sm #allows users to explore data, estimate statistical models, and perform statistical tests.

from sklearn.preprocessing import LabelEncoder #can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels

import copy #to use the deep copy , it copies all fields, and makes copies of dynamically allocated memory pointed to by the fields
#Read the data as a data frame

healthcare_insurance_df = pd.read_csv('../input/insurance.csv')
#Display the first ten dataset



healthcare_insurance_df.head(10)
#Display the last ten dataset



healthcare_insurance_df.tail(10)
#Shape of the data

print(healthcare_insurance_df.shape)
#Data type of each attribute

healthcare_insurance_df.info()
#another approach for the data type

#Data type of each attribute

healthcare_insurance_df.dtypes

#Checking the presence of missing values

# gives non-null number of records

healthcare_insurance_df.count()
#Check for the null values - another approach



healthcare_insurance_df.isna().apply(pd.value_counts) 
#Is any of the values in the df null ?  ( # Useful in writing validation scripts on large number of files )

healthcare_insurance_df.isnull().any().any()
#Is any of the values in columns of the df null ? ( # Useful in writing validation scripts on large number of files )

healthcare_insurance_df.isnull().any() 
#Get the columns into a list and do use it to do some operations 

healthcare_insurance_df_null_cols = healthcare_insurance_df.columns[healthcare_insurance_df.isnull().any()]

healthcare_insurance_df_null_cols = list(healthcare_insurance_df_null_cols)

healthcare_insurance_df_null_cols 

#Shows  the column wise values of missing data - another approach

healthcare_insurance_df.isnull().sum()  
#Shows  the column wise values of missing data - another approach

healthcare_insurance_df.isna().sum()  
#35 point summary of numerical attributes

healthcare_insurance_df.describe()
#5 point summary of numerical attributes

# tranpose 

healthcare_insurance_df.describe().T
#Plots to see the distribution of the continuous features individually

#Distribution of ‘bmi’, ‘age’ and ‘charges’ columns



plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

plt.hist(healthcare_insurance_df.bmi, color='lightgreen', edgecolor = 'black', alpha = 0.7)

plt.xlabel('bmi')



plt.subplot(3,3,2)

plt.hist(healthcare_insurance_df.age, color='lightgreen', edgecolor = 'black', alpha = 0.7)

plt.xlabel('age')



plt.subplot(3,3,3)

plt.hist(healthcare_insurance_df.charges, color='lightgreen', edgecolor = 'black', alpha = 0.7)

plt.xlabel('charges')



plt.show()
#different approach, using simple hist command.... but we also get 'children' column

# Distribution of ‘bmi’, ‘age’ and ‘charges’ columns



healthcare_insurance_df.hist(figsize=(20,30))
# Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns

Skewness = pd.DataFrame({'Skewness' : [stats.skew(healthcare_insurance_df.bmi),stats.skew(healthcare_insurance_df.age),stats.skew(healthcare_insurance_df.charges)]},

                        index=['bmi','age','charges'])  # Measure the skeweness of the required columns

Skewness
#another apporach , by using the skew function directly.

# Measure of skewness of ‘bmi’, ‘age’ and ‘charges’ columns

skewValue_bmi = healthcare_insurance_df['bmi'].skew()

print('skew of bmi is     ' + str(skewValue_bmi))



skewValue_age = healthcare_insurance_df['age'].skew()

print('skew of age is     ' + str(skewValue_age))



skewValue_charges = healthcare_insurance_df['charges'].skew()

print('skew of charge is  ' + str(skewValue_charges))
# Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns

plt.figure(figsize= (20,15))

plt.subplot(3,1,1)

sns.boxplot(x= healthcare_insurance_df.bmi, color='tan')



plt.subplot(3,1,2)

sns.boxplot(x= healthcare_insurance_df.age, color='tan')



plt.subplot(3,1,3)

sns.boxplot(x= healthcare_insurance_df.charges, color='tan')



plt.show()
# another approach, directly using boxplot, without a subplot

# Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns

sns.boxplot(x=healthcare_insurance_df['bmi'])

# Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns

sns.boxplot(x=healthcare_insurance_df['age'])
# Checking the presence of outliers in ‘bmi’, ‘age’ and ‘charges columns

sns.boxplot(x=healthcare_insurance_df['charges'])
#Distribution of categorical columns - 'smoker','sex','region' & 'children'



plt.figure(figsize=(20,25))





x = healthcare_insurance_df.smoker.value_counts().index    #Values for x-axis

y = [healthcare_insurance_df['smoker'].value_counts()[i] for i in x]   # Count of each class on y-axis



plt.subplot(4,2,1)

plt.bar(x,y, align='center',color = 'purple',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Smoker')

plt.ylabel('Count ')

plt.title('Smoker distribution')



x1 = healthcare_insurance_df.sex.value_counts().index    #Values for x-axis

y1 = [healthcare_insurance_df['sex'].value_counts()[j] for j in x1]   # Count of each class on y-axis



plt.subplot(4,2,2)

plt.bar(x1,y1, align='center',color = 'purple',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Gender')

plt.ylabel('Count')

plt.title('Gender distribution')



x2 = healthcare_insurance_df.region.value_counts().index    #Values for x-axis

y2 = [healthcare_insurance_df['region'].value_counts()[k] for k in x2]   # Count of each class on y-axis



plt.subplot(4,2,3)

plt.bar(x2,y2, align='center',color = 'purple',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('Region')

plt.ylabel('Count ')

plt.title("Regions' distribution")



x3 = healthcare_insurance_df.children.value_counts().index    #Values for x-axis

y3 = [healthcare_insurance_df['children'].value_counts()[l] for l in x3]   # Count of each class on y-axis



plt.subplot(4,2,4)

plt.bar(x3,y3, align='center',color = 'purple',edgecolor = 'black',alpha = 0.7)  #plot a bar chart

plt.xlabel('No. of children')

plt.ylabel('Count ')

plt.title("Children distribution")



plt.show()
#another approach for the same above- without subplot

# Distribution of categorical columns -'sex'

sns.countplot(x='sex',data=healthcare_insurance_df)
# Distribution of categorical columns -'smoker'

sns.countplot(x='smoker',data=healthcare_insurance_df)
#Distribution of categorical columns - 'region'

sns.countplot(x='region',data=healthcare_insurance_df)
# Distribution of categorical columns -'children'

sns.countplot(x='children',data=healthcare_insurance_df)
#Label encoding the variables before doing a pairplot because pairplot ignores strings

healthcare_insurance_df_encoded = copy.deepcopy(healthcare_insurance_df)

healthcare_insurance_df_encoded.loc[:,['sex', 'smoker', 'region']] = healthcare_insurance_df_encoded.loc[:,['sex', 'smoker', 'region']].apply(LabelEncoder().fit_transform) 



sns.pairplot(healthcare_insurance_df_encoded)  #pairplot

plt.show()
# another approach without label encoder, then do get the categorial columsn/non numeric columns(strings)

#so it does not plot all the columns



sns.pairplot(healthcare_insurance_df)

healthcare_insurance_df.smoker.value_counts()
#Scatter plot to look for visual evidence of dependency between attributes smoker and charges accross different ages

plt.figure(figsize=(8,6))

sns.scatterplot(healthcare_insurance_df.age, healthcare_insurance_df.charges,hue=healthcare_insurance_df.smoker,palette= ['red','blue'] ,alpha=0.6)

plt.show()
# T-test to check dependency of smoking on charges

Ho = "Charges of smoker and non-smoker are same"   # Stating the Null Hypothesis

Ha = "Charges of smoker and non-smoker are not the same"   # Stating the Alternate Hypothesis



x = np.array(healthcare_insurance_df[healthcare_insurance_df.smoker == 'yes'].charges)  # Selecting charges corresponding to smokers as an array

y = np.array(healthcare_insurance_df[healthcare_insurance_df.smoker == 'no'].charges) # Selecting charges corresponding to non-smokers as an array



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
healthcare_insurance_df.sex.value_counts()   #Checking the distribution of males and females


plt.figure(figsize=(8,6))

sns.scatterplot(healthcare_insurance_df.age, healthcare_insurance_df.charges,hue=healthcare_insurance_df.sex,palette= ['pink','lightblue'] )

plt.show()



# T-test to check dependency of bmi on gender

Ho = "Gender has no effect on bmi"   # Stating the Null Hypothesis

Ha = "Gender has an effect on bmi"   # Stating the Alternate Hypothesis



x = np.array(healthcare_insurance_df[healthcare_insurance_df.sex == 'male'].bmi)  # Selecting bmi values corresponding to males as an array

y = np.array(healthcare_insurance_df[healthcare_insurance_df.sex == 'female'].bmi) # Selecting bmi values corresponding to females as an array



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round()}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')
# Chi_square test to check if smoking habits are different for different genders

Ho = "Gender has no effect on smoking habits"   # Stating the Null Hypothesis

Ha = "Gender has an effect on smoking habits"   # Stating the Alternate Hypothesis



crosstab = pd.crosstab(healthcare_insurance_df['sex'],healthcare_insurance_df['smoker'])  # Contingency table of sex and smoker attributes



chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')

crosstab
# Chi_square test to check if smoking habits are different for people of different regions

Ho = "Region has no effect on smoking habits"   # Stating the Null Hypothesis

Ha = "Region has an effect on smoking habits"   # Stating the Alternate Hypothesis



crosstab = pd.crosstab(healthcare_insurance_df['smoker'], healthcare_insurance_df['region'])  # Contingency table of sex and smoker attributes



chi, p_value, dof, expected =  stats.chi2_contingency(crosstab)



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')

crosstab
# Test to see if the distributions of bmi values for females having different number of children, are significantly different



Ho = "No. of children has no effect on bmi"   # Stating the Null Hypothesis

Ha = "No. of children has an effect on bmi"   # Stating the Alternate Hypothesis





female_df = copy.deepcopy(healthcare_insurance_df[healthcare_insurance_df['sex'] == 'female'])



zero = female_df[female_df.children == 0]['bmi']

one = female_df[female_df.children == 1]['bmi']

two = female_df[female_df.children == 2]['bmi']





f_stat, p_value = stats.f_oneway(zero,one,two)





if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')