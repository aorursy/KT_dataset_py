import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stat

sns.set_color_codes()

%matplotlib inline



#setting up for customized printing

from IPython.display import Markdown, display

from IPython.display import HTML

def printmd(string, color=None):

    colorstr = "<span style='color:{}'>{}</span>".format(color, string)

    display(Markdown(colorstr))
data = pd.read_csv("../input/insurance.csv")

data.head()
print('The total number of rows :', data.shape[0])

print('The total number of columns :', data.shape[1])
data.info()
print(data.isna().sum())

print('===================')

print(data.isnull().sum())

print('===================')

printmd('**CONCLUSION**: As seen from the data above, we conclude there are **"NO Missing"** values in the data', color="blue")
data.describe().transpose()
f, axes = plt.subplots(1, 3, figsize=(20, 8))

bmi = sns.distplot(data['bmi'], color="red", ax=axes[0], kde=True, hist_kws={"edgecolor":"k"})

bmi.set_xlabel("BMI",fontsize=20)



age = sns.distplot(data['age'], color='green', ax = axes[1], kde=True, hist_kws={"edgecolor":"k"})

age.set_xlabel("Age",fontsize=20)



charges = sns.distplot(data['charges'], color='blue', ax = axes[2], kde=True, hist_kws={"edgecolor":"k"})

charges.set_xlabel("Charges",fontsize=20)

pd.DataFrame.from_dict(dict(

    {

        'bmi':data.bmi.skew(), 

        'age': data.age.skew(), 

        'charges': data.charges.skew()

    }), orient='index', columns=['Skewness'])
f, axes = plt.subplots(3, 1, figsize=(15, 15))

bmi = sns.boxplot(data['bmi'], color="olive", ax=axes[0])

bmi.set_xlabel("BMI",fontsize=20)



age = sns.boxplot(data['age'], color='lightgreen', ax=axes[1])

age.set_xlabel("Age",fontsize=20)



charges = sns.boxplot(data['charges'], color='teal', ax=axes[2])

charges.set_xlabel("Charges",fontsize=20)

f, axes = plt.subplots(2, 2, figsize=(20, 12))

sex = sns.countplot(data['sex'], color="red", ax=axes[0,0])

sex.set_xlabel("Sex",fontsize=20)



smoker = sns.countplot(data['smoker'], color='green', ax = axes[0,1])

smoker.set_xlabel("Smoker",fontsize=20)



region = sns.countplot(data['region'], color='blue', ax = axes[1,0])

region.set_xlabel("Region",fontsize=20)



children = sns.countplot(data['children'], color='teal', ax = axes[1,1])

children.set_xlabel("Children",fontsize=20)
f, axes = plt.subplots(3, 1, figsize=(12, 15))

sns.boxplot('region', 'charges', 'sex', data, ax=axes[0])



sns.boxplot('region', 'charges', 'smoker', data, ax = axes[1])



sns.boxplot('region', 'charges', 'children', data, ax = axes[2])

data_copy = data.copy()

data_copy.sex.value_counts()
data_copy.smoker.value_counts()
data_copy.region.value_counts()
# Replace categorical columns with numerical equivalents



data_copy['sex'] = data_copy['sex'].replace({'male': 1, 'female': 2})

data_copy['smoker'] = data_copy['smoker'].replace({'yes': 1, 'no': 0})

data_copy['region'] = data_copy['region'].replace({'southeast': 1, 'southwest': 2, 'northwest': 3, 'northeast': 4})



# Pair plot with all the columns

sns.pairplot(data_copy)
sns.pairplot(data, hue='sex')
sns.pairplot(data, hue='smoker')
sns.pairplot(data, hue='region')
data_copy.corr()
pd.DataFrame.from_dict(dict(

    {

        'charges_smokers':data[data.smoker == 'yes'].charges.skew(),    

        'charges_non-smokers': data[data.smoker == 'no'].charges.skew(),

    }), orient='index', columns=['Skewness'])
data.charges  = np.log1p(data.charges)
pd.DataFrame.from_dict(dict(

    {

        'charges_smokers':data[data.smoker == 'yes'].charges.skew(),    

        'charges_non-smokers': data[data.smoker == 'no'].charges.skew(),

    }), orient='index', columns=['Skewness'])
sns.boxplot('smoker', 'charges', data=data)
import scipy.stats as stats



#Split the charges column into two parts between smokers and non-smokers



X = np.array(data[data.smoker == 'yes'].charges) #Smokers

Y = np.array(data[data.smoker == 'no'].charges) #Non-Smokers



#executing the independent t-test to run tests on single variable

t_stat, p_value = stats.ttest_ind(X,Y)



# Setting our significance level at 5%

if p_value < 0.05:  

    printmd(f'As the p_value **({p_value}) < 0.05**, we reject the Null Hypothesis. Hence **charges of smokers differ significantly from non-smokers**', color='blue')

else:

    printmd(f'As the p_value **({p_value}) > 0.05**, we fail to reject Null Hypothesis. Hence **charges of smokers are same as charges of non-smokers**', color='blue')
sns.boxplot('sex', 'bmi', data=data)
#Split the bmi column into two parts between male and female



X = np.array(data[data.sex == 'male'].bmi) #Males

Y = np.array(data[data.sex == 'female'].bmi) #Females



#executing the independent t-test to run tests on single variable

t_stat, p_value = stats.ttest_ind(X,Y)



# Setting our significance level at 5%

if p_value < 0.05:  

    printmd(f'As the p_value **({p_value}) < 0.05**, we reject the Null Hypothesis. Hence **BMI of Males are significantly different from that of Females**', color='blue')

else:

    printmd(f'As the p_value **({p_value}) > 0.05**, we fail to reject Null Hypothesis. Hence **BMI of Males are similar to that of Females**', color='blue')
sns.countplot('smoker', hue='sex', data=data[data.smoker == 'yes'])
#Since Smokers and Sex are categorical columns, choosing Chi-Square test for testing



contigencytable = pd.crosstab(data['sex'],data['smoker'])

chi_sq_Stat, p_value, deg_freedom, exp_freq =  stats.chi2_contingency(contigencytable)



# Setting our significance level at 5%

if p_value < 0.05:  

    printmd(f'As the p_value **({p_value}) < 0.05**, we reject the Null Hypothesis. Hence **Proportion of smokers are significantly different in different Genders**', color='blue')

else:

    printmd(f'As the p_value **({p_value}) > 0.05**, we fail to reject Null Hypothesis. Hence **Proportion of smokers are similar in both the Genders**', color='blue')

sns.boxplot('children', 'bmi', data=data[(data.children>0) & (data.children<4)])
#get the female data 

female_data = data[data['sex'] == 'female']



#get the bmi samples based on number of children

no_children_bmi = female_data[female_data['children'] == 0].bmi

one_child_bmi = female_data[female_data['children'] == 1].bmi

two_children_bmi = female_data[female_data['children'] == 2].bmi



#Since there are multiple samples and we need to check the variances of multiple samples, choosing ANOVA testing for this



f_stat, p_value = stats.f_oneway(no_children_bmi,one_child_bmi ,two_children_bmi )





# Setting our significance level at 5%

if p_value < 0.05:  

    printmd(f'As the p_value **({p_value}) < 0.05**, we reject the Null Hypothesis. Hence **BMI is different across women with different number of children**', color='blue')

else:

    printmd(f'As the p_value **({p_value}) > 0.05**, we fail to reject Null Hypothesis. Hence **BMI is uniform across women with different number of children**', color='blue')
