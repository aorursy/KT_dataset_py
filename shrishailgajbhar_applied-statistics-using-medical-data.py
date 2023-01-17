import pandas as pd  # To read the dataset as dataframe

import seaborn as sns # For Data Visualization 

import matplotlib.pyplot as plt # Necessary module for plotting purpose

%matplotlib inline 

sns.set(color_codes=True) # In order to have uniformity in output plots

from scipy.stats import * # import all the methods of scipy.stats module

from statsmodels.stats.proportion import proportions_ztest # for proportion test
df = pd.read_csv("../input/insurance/insurance.csv")
df.shape
df.dtypes
df.info()
df.isnull().sum()
import numpy as np

df.select_dtypes('number')[~df.select_dtypes('number').applymap(np.isreal).all(1)]



# One can see that all the mumeric data is valid and we are good to go..!
df[df.select_dtypes('number').columns].describe()[-5:]
sns.boxplot(data = df['age'])

plt.title("Graphical 5 pioint summary for age attribute");
sns.boxplot(data = df['bmi'])

plt.title("Graphical 5 pioint summary for bmi attribute");
sns.boxplot(data = df['children'])

plt.title("Graphical 5 pioint summary for children attribute");
sns.boxplot(data = df['charges'])

plt.title("Graphical 5 pioint summary for charges attribute");
# plotting univariate distribution for bmi attribute

sns.distplot(df['bmi'],rug=True)

plt.title('distribution for bmi attribute');



# The distribution for bmi attribute follows a normal distribution
# plotting univariate distribution for age attribute

sns.distplot(df['age'])

plt.title('distribution for age attribute');



# One can oberve that the distribution for age attribute in this case seems to follow uniform-like distribution
# plotting univariate distribution for charges attribute

sns.distplot(df['charges'])

plt.title('distribution for charges attribute');



# Distribution for charges atrribute is right-skewed normal-like distribution
df[['bmi','age','charges']].skew()
# Let us check whether the distribution types using formulaes



x = df[df.select_dtypes('number').columns].describe()[-5:].transpose()

# since x is a dataframe let us calculate values for below quantities

# m1 = Median – Xsmallest 

# m2 = Xlargest – Median

# m3 = Q1 – Xsmallest

# m4 = Xlargest – Q3

# m5 = Median – Q1

# m6 = Q3 – Median



m1 = x['50%']-x['min']

m2 = x['max']-x['50%']

m3 = x['25%'] - x['min']

m4 = x['max'] - x['75%']

m5 = x['50%'] - x['25%']

m6 = x['75%'] - x['50%']
d = {'m1':m1,'m2':m2,'m3':m3,'m4':m4,'m5':m5,'m6':m6}

xd = pd.DataFrame(data=d)

xd.head()
from scipy.stats import zscore





df_bac = df[['bmi','age','charges']]



df_bacz = df_bac.apply(zscore)



df_bacz.head()
sns.boxplot(data=df_bacz);



# We can see that age column do not show any outliers, whereas bmi and charges columns show outliers over third

# quartile i.e., Q3.
# Let us plot the boxplot for bmi attribute

sns.boxplot(data = df['bmi'])

plt.title("Boxplot for bmi attribute");



# One can see from the box plot, there are outliers present in this plot. Any value greater than Q3+ 1.5 * IQR 

# is considered as an outlier
# Let us plot the boxplot for age attribute

sns.boxplot(data = df['age'])

plt.title("Boxplot for age attribute");



# One can see from the box plot, there are no outliers present in this plot. AAny value greater than Q3+ 1.5 * IQR 

# is considered as an outlier
# Let us plot the boxplot for charges attribute

sns.boxplot(data = df['charges'])

plt.title("Boxplot for charges attribute");



# One can see from the box plot, there are outliers present in this plot. Any value greater than Q3+ 1.5 * IQR 

# is considered as an outlier
# let us calculate number of outliers present in these columns and print them



df_bac = df[['bmi','age','charges']].describe()[-5:].transpose()
df_bac.head()
# Calculate the IQR as Q3-Q1

IQR = df_bac['75%']-df_bac['25%']

print(IQR)



# Let us find the cutoff values above which they will be considered as an outlier..

print("Any column entry above following values will be an outlier..")

print(df_bac['75%']+ 1.5*IQR)
# From boxplot we know that outliers are present in bmi and charges columns that to the right of Q3, 

# we only calculate number of outliers on the right hand side i.e, those greater than Q3+ 1.5 * IQR



no_bmi = df['bmi'][df['bmi']>df_bac['75%']['bmi']+1.5*IQR.bmi].count()

print("Number of outliers in bmi coumn are: ",no_bmi, "and the values are: ")

print(df['bmi'][df['bmi']>df_bac['75%']['bmi']+1.5*IQR.bmi])
# Similarly, number of outliers in age column are: (should be zero as per the box plot)

no_age = df['age'][df['age']>df_bac['75%']['age']+1.5*IQR.age].count()

print("Number of outliers in age coumn are: ",no_age)
# Similarly, number of outliers in charges column are: (should be a large number as per the box plot)

no_charges = df['charges'][df['charges']>df_bac['75%']['charges']+1.5*IQR.charges].count()

print("Number of outliers in charges coumn are: ",no_charges, "and the values are: ")

print(df['charges'][df['charges']>df_bac['75%']['charges']+1.5*IQR.charges])
# Create a copy of df dataframe

df_copy = df.copy()

# We will modify the charges column entries in this to see new distribution

# Let us caluclate the median of charges column

charges_med = df_copy['charges'].median()

df_copy['charges'].replace(to_replace=df['charges'][df['charges']>df_bac['75%']['charges']+1.5*IQR.charges], value = df['charges'].median(), inplace = True)



# Check whether values are correctly replaced or not?

# 14      39611.75770 index 14 has this value in df dataframe, this should become median value

print(df['charges'][14])

print(df_copy['charges'][14])

# Perfectly fine

# The question is whether median is a good choice for replacing the outlier values in this case?
# Let us plot the the distributions of charges of df(with outliers) and df_copy (without outliers)

fig, ax =plt.subplots(1,2)

sns.distplot(df['charges'],ax=ax[0])

ax[0].set_title("With Outlier"); 

sns.distplot(df_copy['charges'],ax=ax[1])

ax[1].set_title("Without Outlier");



# One can observe that right-tail in with outlier plot is now reduced in without outlier plot.
# Let us analyze categorical variables "smoker" and "sex" (as hue) with respect to "charges" as continuous variable

sns.catplot(x="smoker", y="charges", hue="sex", kind="box", data=df);

# One can observe that one belonging to smoker category is charged more than the non smoker irrespective of gender
# Let us analyze categorical variables "region" and "smoker" (as hue) with respect to "charges" as continuous variable

sns.catplot(x="region", y="charges", hue="smoker", kind="box", data=df);

# One can observe that southeast charges relatively higher than other regions to smokers
# Let us analyze categorical variables "region" and "sex" (as hue) with respect to charges as continuous variable

sns.catplot(x="region", y="charges", hue="sex", kind="box", data=df);

# One can observe that southeast has wider range of charges than other regions
# Let us analyze scatterplot between age and charges with smoker as a hue
sns.catplot(x="age", y="charges", hue="smoker", kind="swarm", data=df);

# One can see that over increasing age charges incurred are more and smokers are charged relatively higher 

# than non-smoker
# Let us analyze scatterplot between bmi and charges with smoker as a hue

sns.catplot(x="bmi", y="charges", hue="smoker", kind="swarm", data=df);

# One can see that smokers with increasing bmi tend to pay more charges than non-smokers..!
# Plotting swarmplot between charges and categorical variable smoker

sns.swarmplot(df['smoker'], df['charges']);
# Plotting swarmplot between charges and categorical variable sex

sns.swarmplot(df['sex'], df['charges']);
# Plotting swarmplot between charges and categorical variable region

sns.swarmplot(df['region'], df['charges']);



# region doesn't seem to play any major role in charges
# Let's plot barplot between charges and smoker

sns.barplot(df['smoker'],df['charges']);



# One can see that smoker's tend to pay higher charges than non-smokers
# Let's plot barplot between charges and sex

sns.barplot(df['sex'],df['charges']);



# One can see that males have wider range of charges than females
# Let's plot barplot between charges and region

sns.barplot(df['region'],df['charges']);



# Region doesn't seem to be a good choice to predict charges
sns.pairplot(data=df);

# One can oberve that age and charges seem to have large correlation than other numeric attributes 
# Pairplot with smoker as hue

sns.pairplot(data=df, hue='smoker');

# One can oberve that smokers with increasing bmi's tend to pay more charges than non-smokers
# Pairplot with sex as hue

sns.pairplot(data=df, hue='sex');

# One can oberve that smokers with increasing bmi's tend to pay more charges than non-smokers
# Pairplot with region as hue

sns.pairplot(data=df, hue='region');

# One can oberve that the distributions of age,bmi,children and charges columns over region as a hue do not 

# provide any new information since they are fairly similar plots
corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values);
# Another way toprint correlation matrix

corr.style.background_gradient(cmap='coolwarm').set_precision(2)



# One can see that correlation coefficient for charges<----->age and charges<----->bmi is positive and high

# showing strong correlation among these quantities. bmi<--->age are also correlated. 
# group1 : smoker

group1 = df[df['smoker']=='yes'].charges

# group2 : Non-smoker

group2 = df[df['smoker']=='no'].charges
t_statistic, p_value = ttest_ind(group1, group2)

print(t_statistic, p_value)



# if p_value < 0.05, we reject the null hypothesis else we fail to reject the null hypothesis
df['sex'].value_counts()
# group1_b:  bmi values for males only

group1_b = df[df['sex']=='male'].bmi



# group2_b:  bmi values for females only

group2_b = df[df['sex']=='female'].bmi



t_statistic, p_value = ttest_ind(group1_b, group2_b)

print(t_statistic, p_value)



# if p_value < 0.05, we reject the null hypothesis else we fail to reject the null hypothesis
# In order to find the proportions, let us first calculate number of males and females

n_males = df[df['sex']=='male'].sex.count() # 676

n_females = df[df['sex']=='female'].sex.count()  # 662



# Let us calculate number of male smokers

ms_all = df[df['sex']=='male'].smoker

male_smokers = ms_all[ms_all=='yes']

# so number of males who smoke are 

n_male_smokers = male_smokers.count()

print("Number of male smokers are :",n_male_smokers,"and total number of males are :",n_males)



# Let us calculate number of female smokers

fs_all = df[df['sex']=='female'].smoker

female_smokers = fs_all[fs_all=='yes']

# so number of females who smoke are 

n_female_smokers = female_smokers.count()

print("Number of female smokers are :",n_female_smokers,"and total number of females are :",n_females)



# Calculate the proportions of male and female smokers..

male_smoker_prop = n_male_smokers/n_males

female_smoker_prop = n_female_smokers/n_females



print("Proportion of males who smoke is: {:.2}".format(float(male_smoker_prop)))

print("Proportion of females who smoke is: {:.2}".format(float(female_smoker_prop)))



# One can see that proportions are not equal however are they significantly different, statistically



# Let us carry out the z-proportions test



'''We formulate our null and alternate hypothesis for this problem as follows:



H0: smoker proportions are equal in different genders



H1: smoker proportions are not equal in different genders



We assume the level of significance in this case as alpha = 0.05

'''





stat, pval = proportions_ztest([n_male_smokers, n_female_smokers] , [n_males, n_females])

print("The value of p is ",pval)
# Boxplots for group1: women with no children

# Boxplots for group2: women with 1 children

# Boxplots for group3: women with 2 children



grp1 = df[(df['sex']=='female') & (df['children']==0)]['bmi']

grp2 = df[(df['sex']=='female') & (df['children']==1)]['bmi']

grp3 = df[(df['sex']=='female') & (df['children']==2)]['bmi']



fig, ax =plt.subplots(3,1);

ax[0].set_title("Women with no children");

sns.boxplot(grp1,ax=ax[0]);

ax[1].set_title("Women with 1 children");

sns.boxplot(grp2,ax=ax[1]);

ax[2].set_title("Women with 2 children");

sns.boxplot(grp3,ax=ax[2]);



# From plots one can see the boxplots for females with no, one and two children are almost similar

# Let us check the same fact from the hypothesis testing..
f_stat, p_value_anova = stats.f_oneway(grp1,grp2,grp3)

print(p_value_anova)