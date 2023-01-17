import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder

import statsmodels.api         as     sm

from   statsmodels.formula.api import ols

from   statsmodels.stats.anova import anova_lm

import copy
#I've modified the name of the data columns and its position for clear reference,

#use the placement_data file with same data
%matplotlib inline
df = pd.read_csv('../input/placement-data/Placement_Data.csv')
df.head()
df.info()
#null value check.

df.isna().apply(pd.value_counts)
df.describe(include='all').T
df.describe().T
#Plots to see the distribution of the continuous features individually



plt.figure(figsize= (20,15))

plt.subplot(3,3,1)

plt.hist(df.X_P, color='crimson', edgecolor = 'black', alpha = 1)

plt.xlabel('Xth')



plt.subplot(3,3,2)

plt.hist(df.XII_P, color='darkgrey', edgecolor = 'black', alpha = 0.7)

plt.xlabel('XII')



plt.subplot(3,3,3)

plt.hist(df.UG_P, color='lime', edgecolor = 'black', alpha = 0.7)

plt.xlabel('UG')



plt.subplot(3,3,4)

plt.hist(df.PG_P, color='gold', edgecolor = 'black', alpha = 0.7)

plt.xlabel('PG ')



plt.subplot(3,3,5)

plt.hist(df.Etest_P, color='cornflowerblue', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Employability Test')





plt.subplot(3,3,6)

plt.hist(df.Salary, color='hotpink', edgecolor = 'black', alpha = 0.7)

plt.xlabel('Salary')

plt.show()

df.Salary.dropna(inplace=True)
# Measuring the skewness of required columns.

Skewness = pd.DataFrame({'Skewness' : [stats.skew(df.X_P),stats.skew(df.XII_P),stats.skew(df.UG_P),

                                      stats.skew(df.PG_P),stats.skew(df.Etest_P),stats.skew(df.Salary)]},

                        index=['X_P','XII_P','UG_P','PG_P','Etest_P','Salary'])  # Measure the skeweness of the required columns

Skewness
df['Salary'].plot(kind='density')

plt.vlines(df['Salary'].mean(),ymin=0,ymax=0.000007,color='red')

plt.vlines(df['Salary'].median(),ymin=0,ymax=0.000007,color='green')
df = pd.read_csv('../input/placement-data/Placement_Data.csv')
#boxplot



plt.figure(figsize= (15,15))

plt.subplot(3,1,1)

sns.boxplot(x= df.X_P)



plt.subplot(3,1,2)

sns.boxplot(x= df.XII_P,color='pink')



plt.subplot(3,1,3)

sns.boxplot(x= df.UG_P, color='magenta')



plt.figure(figsize= (20,15))

plt.subplot(3,1,1)

sns.boxplot(x= df.PG_P, color='darkturquoise')



plt.subplot(3,1,2)

sns.boxplot(x= df.Etest_P, color='mediumspringgreen')



plt.subplot(3,1,3)

sns.boxplot(x= df.Salary, color='lightblue')

plt.show()
#Students placed according to Gender

gender_placed_record = df.Status.groupby(df.Gender)

gender_placed_record.value_counts()
# countplot for the above observation.

sns.countplot(df.Gender, hue=df.Status,palette='winter');
#Students placed according to their department in UG

dept_status_record = df.Status.groupby([df.UG_Field])

dept_status_record.value_counts()
#countplot for above observation

sns.violinplot(x="UG_Field", y="Salary", data=df)

sns.stripplot(x="UG_Field", y="Salary", data=df,hue='Status')
#similarly we can also see the stats for PG specialization.

dept_status_pg_record = df.Status.groupby([df.PG_Specialization])

dept_status_pg_record.value_counts()
#countplot for above observation

sns.countplot(df.PG_Specialization, hue=df.Status);
#Label encoding the variables before doing a pairplot because pairplot ignores strings

df_encoded = copy.deepcopy(df)

df_encoded.loc[:,['Gender','X_Board','XII_Board','XII_Stream','UG_Field','PG_Specialization','Work_exp']] = df_encoded.loc[:,['Gender','X_Board','XII_Board','XII_Stream','UG_Field','PG_Specialization','Work_exp']].apply(LabelEncoder().fit_transform) 

plt.figure(figsize= (25,25))

sns.pairplot(df_encoded)  #pairplot

plt.show()
sns.pairplot(df, hue='Status')

#Using Pearson Correlation

df = pd.read_csv('../input/placement-data/Placement_Data.csv')

del df['Sl_No']

df.Salary.fillna(value=0,inplace=True)

numeric_data = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12,10))

cor = numeric_data.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
df.corr()


#Scatter plot to look for visual evidence of dependency between attributes smoker and charges accross different ages

plt.figure(figsize=(8,6))

sns.scatterplot(df.PG_P, df.Salary,hue=df.PG_Specialization,palette= ['red','green'],alpha=0.9)

plt.show()

sns.catplot(x="Status", y="X_P", data=df,kind="swarm")

sns.catplot(x="Status", y="XII_P", data=df,kind="swarm",hue='Gender')

sns.catplot(x="Status", y="UG_P", data=df,kind="swarm",hue='Gender')
df.Salary = df.Salary.fillna(0)
formula = 'Salary ~ C(Gender) + C(X_Board) + C(XII_Board) + C(XII_Stream) + C(UG_Field) + C(PG_Specialization) + C(Work_exp)'

model = ols(formula, data= df).fit()

aov_table = anova_lm(model, typ=1)



print(aov_table)
ata_crosstab = pd.crosstab(df['Gender'], df['Status'], margins = False) 

ata_crosstab
# Chi_square test to check if Placement status are different for different Genders

Ho = "Gender has no effect on Job status"   # Stating the Null Hypothesis

Ha = "Gender has no effect on Job status"   # Stating the Alternate Hypothesis



chi, p_value, dof, expected =  stats.chi2_contingency(ata_crosstab)



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')
ata_crosstab = pd.crosstab(df['PG_Specialization'], df['Status'], margins = False) 

ata_crosstab
# Chi_square test to check if Placement status are different for different Specialisation

Ho = "Specialization has no effect on Placement status"   # Stating the Null Hypothesis

Ha = "Specialization has an effect on Placement status"   # Stating the Alternate Hypothesis



chi, p_value, dof, expected =  stats.chi2_contingency(ata_crosstab)



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value.round(3)}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value.round(3)}) > 0.05')

# T-test to check dependency of percentage on placement

Ho = "PG_P of Placed and non-Placed are same"   # Stating the Null Hypothesis

Ha = "PG_P of Placed and non-Placed are not the same"   # Stating the Alternate Hypothesis



x = df.loc[df.Status == 'Placed', "PG_P"].values  

y = df.loc[df.Status == 'Not Placed', "PG_P"].values



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
# T-test to check dependency for

Ho = "MBA_P of Placed and non-Placed are same"   # Stating the Null Hypothesis

Ha = "MBA_P of Placed and non-Placed are not the same"   # Stating the Alternate Hypothesis



x = df.loc[(df.Status == 'Placed') & (df.PG_Specialization =='Mkt&Fin'), "PG_P"].values  

y = df.loc[(df.Status == 'Not Placed') & (df.PG_Specialization == 'Mkt&Fin'), "PG_P"].values 



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
# T-test to check dependency 

Ho = "MBA_P of Placed and non-Placed are same"   # Stating the Null Hypothesis

Ha = "MBA_P of Placed and non-Placed are not the same"   # Stating the Alternate Hypothesis



x = df.loc[(df.Status == 'Placed') & (df.PG_Specialization =='Mkt&HR'), "PG_P"].values 

y = df.loc[(df.Status == 'Not Placed') & (df.PG_Specialization == 'Mkt&HR'), "PG_P"].values 



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
# T-test to check dependency 

Ho = "Etest_P of Placed and non-Placed are same"   # Stating the Null Hypothesis

Ha = "Etest_P of Placed and non-Placed are not the same"   # Stating the Alternate Hypothesis



x = df.loc[df.Status == 'Placed', "Etest_P"].values  

y = df.loc[df.Status == 'Not Placed', "Etest_P"].values 



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
# T-test to check dependency 

Ho = "Etest_P of Placed and non-Placed are same"   # Stating the Null Hypothesis

Ha = "Etest_P of Placed and non-Placed are not the same"   # Stating the Alternate Hypothesis



x = df.loc[(df.Status == 'Placed') & (df.PG_Specialization =='Mkt&HR'), "Etest_P"].values  

y = df.loc[(df.Status == 'Not Placed') & (df.PG_Specialization == 'Mkt&HR'), "Etest_P"].values  



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
# T-test to check dependency 

Ho = "Etest_P of Placed and non-Placed are same"   # Stating the Null Hypothesis

Ha = "Etest_P of Placed and non-Placed are not the same"   # Stating the Alternate Hypothesis



x = df.loc[(df.Status == 'Placed') & (df.PG_Specialization =='Mkt&Fin'), "Etest_P"].values  

y = df.loc[(df.Status == 'Not Placed') & (df.PG_Specialization == 'Mkt&Fin'), "Etest_P"].values 



t, p_value  = stats.ttest_ind(x,y, axis = 0)  #Performing an Independent t-test



if p_value < 0.05:  # Setting our significance level at 5%

    print(f'{Ha} as the p_value ({p_value}) < 0.05')

else:

    print(f'{Ho} as the p_value ({p_value}) > 0.05')
series1 = df.Salary.fillna(value=0)

series2 = df.PG_P

series3 = df.Etest_P



def central_limit_theorem(data,n_samples = 500, sample_size = 100):

    """ Use this function to demonstrate Central Limit Theorem. 

        data = 1D array, or a pd.Series

        n_samples = number of samples to be created

        sample_size = size of the individual sample """

    %matplotlib inline

    import pandas as pd

    import numpy as np

    import matplotlib.pyplot as plt

    import seaborn as sns

    min_value = 0  # minimum index of the data

    max_value = data.count()  # maximum index of the data

    b = {}

    for i in range(n_samples):

        x = np.unique(np.random.randint(min_value, max_value, size = sample_size)) # set of random numbers with a specific size

        b[i] = data[x].mean()   # mean of each sample

    c = pd.DataFrame()

    c['sample'] = b.keys()  # sample number 

    c['Mean'] = b.values()  # mean of that particular sample

    plt.figure(figsize= (15,5))



    plt.subplot(1,2,2)

    sns.distplot(c.Mean)

    plt.title(f"Sampling Distribution. \n \u03bc = {round(c.Mean.mean(), 3)} & SE = {round(c.Mean.std(),3)}")

    plt.xlabel('data')

    plt.ylabel('freq')



    plt.subplot(1,2,1)

    sns.distplot(data)

    plt.title(f"Population Distribution. \n \u03bc = {round(data.mean(), 3)} & \u03C3 = {round(data.std(),3)}")

    plt.xlabel('data')

    plt.ylabel('freq')



    plt.show()
central_limit_theorem(series1,n_samples = 500, sample_size = 100)

central_limit_theorem(series2,n_samples = 500, sample_size = 100)

central_limit_theorem(series3,n_samples = 500, sample_size = 100)
