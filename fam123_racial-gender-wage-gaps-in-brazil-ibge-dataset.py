#1

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns; sns.set(color_codes=True)

from operator import itemgetter

%matplotlib inline

%config InlineBackend.figure_formats=['svg']



import warnings

warnings.filterwarnings("ignore")



# Our 1st task will be to upload our IBGE dataset to Kaggle and then command it to open.

#Below is a pretty cool video by Kaggle on Youtube showing how to upload your dataset, 

#plus the innitial commands to start using your data. Hope it helps!



from IPython.display import HTML

HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/0jQwAp7po00" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')

     

     

#it worked!!! Dataset open...Let's test it with a file.head command

df = pd.read_csv("../input/data.csv")

df.head()

#gives you the command's default top 5 rows in your table
df.tail(10)

#gives you the bottom 10 rows of your table
df.shape



#df.shape: Gives you the #of rows and columns
df.dtypes



#types of characters and spacing allowed
df.info()



#with total entries, rows and columns
df.describe()



#basic stats information
df.columns



#introduce you to the columns contained in our table
for col in df:

    if col != 'seq' and col != 'id':

        print('')

        print(df[col].value_counts(dropna=False).nlargest(10))

        

df.describe()



#!= Not Equal To
df['regiao'].value_counts(dropna=False)



# Absence of NaN values

#For 'regiao' calculate how many rows there are for each of the 5 regions using value_counts()
df.regiao.value_counts().plot(kind = 'bar')
df.sort_values(['regiao'], ascending = True).head(10)





#Here we want to sort our table by 'regiao' in alphabetical ascending order, and to bring only the top 10 rows
df.columns
df.rename(columns={'cor/raca':'cor_raca'}, inplace=True)

df.columns



#noticed the change?
df.cor_raca.value_counts().plot.bar()



#all 5 categories plotted in a bar graph
x1 = list(df[df['cor_raca'] == 'Branca']['regiao'])

x2 = list(df[df['cor_raca'] == 'Parda']['regiao'])

x3 = list(df[df['cor_raca'] == 'Preta']['regiao'])

x4 = list(df[df['cor_raca'] == 'Indigena']['regiao'])

x5 = list(df[df['cor_raca'] == 'Amarela']['regiao'])



# Assign colors for each race and the names



colors = ['#E69F00', '#56B4E9', '#D55E00', '#009E73', '#F0E442']

names = ['Branca', 'Parda', 'Preta','Indigena', 'Amarela']

         

# Make the histogram using a list of lists

# Normalize the flights and assign colors and names

plt.hist([x1, x2, x3, x4, x5], bins = 5, normed=True,

         color = colors, label=names)



# Plot formatting

plt.legend()

df['estado_civil'] = df['estado_civil'].map({0:'solteiro',1:'casado'})
df.estado_civil.value_counts().plot.bar()
df['idade'].value_counts()

#For 'idade' calculate how many rows there are for all different ages using value_counts()
df['idade'].value_counts(dropna=False).nlargest(10)



#We can see no NaN have come up.
(df.idade.value_counts()).shape




plt.hist(df['idade'], color = 'green', edgecolor = 'black', bins = (41))
df['idade'].plot.kde()



#
df['sexo'].value_counts()
df = df.replace('gestante', 'mulher')
df['sexo'].value_counts(dropna=False)
df['salario'].value_counts(dropna=False).nlargest(10)



#nlargest for the program to bring the largest 10 values



#When we call a variable in Python, the program does not bring NaN values by default. For that reason 

#we need to use function dropna = True to force it to show such values.By

#using dropna=False, Python will bring the values not filled in the dataset.

#We can also use function fillna() to fill the the missing or NaN values in the pandas dataframe

#with a suitable data as decided by the user.
#Now, for some graph.



df['salario'].plot.hist(bins=100)

df['salario'].plot.hist(bins=500, xlim=(-2000, 100000))
df['salario'].value_counts(dropna=False).nlargest(7)

df['salario'].fillna(df['salario'].median(), inplace=True)
df['salario'].value_counts(dropna=False).nlargest(7)
df['salario'].plot.hist(bins=500, xlim=(-2000, 100000))
UPPER_BOUND_SALARY = 100000



mask_salary = (df['salario'] > 0) & (df['salario'] < UPPER_BOUND_SALARY)

filtered_df = df[mask_salary]

filtered_df['salario'].max()


df_white_male = filtered_df[filtered_df['cor_raca'] == 'Branca']

df_white_male[['salario']].plot.hist(bins = 20)

mean_white_male = df_white_male['salario'].mean()

std_white_male = df_white_male['salario'].std()



df_black_male = filtered_df[filtered_df['cor_raca'] == 'Preta']

df_black_male[['salario']].plot.hist(bins = 20)

mean_black_male = df_black_male['salario'].mean()

std_black_male = df_black_male['salario'].std()



df_brown_male = filtered_df[filtered_df['cor_raca'] == 'Parda']

df_brown_male[['salario']].plot.hist(bins = 20)

mean_brown_male = df_brown_male['salario'].mean()

std_brown_male = df_brown_male['salario'].std()



df_indigenous_male = filtered_df[filtered_df['cor_raca'] == 'Indigena']

df_indigenous_male[['salario']].plot.hist(bins = 20)

mean_indigenous_male = df_indigenous_male['salario'].mean()

std_indigenous_male = df_indigenous_male['salario'].std()



df_yellow_male = filtered_df[filtered_df['cor_raca'] == 'Indigena']

df_yellow_male[['salario']].plot.hist(bins = 20)

mean_yellow_male = df_yellow_male['salario'].mean()

std_yellow_male = df_yellow_male['salario'].std()



print('White male average salary:',mean_white_male)

print('Black male average salary:',mean_black_male)

print('White males earn more, on average, than black males')



#std_white_male
print('White male average salary:',mean_white_male)

print('Brown male average salary:',mean_brown_male)

print('White males earn more, on average, than brown males')

print('White male average salary:', mean_white_male)

print('Indigenous male average salary:', mean_indigenous_male)

print('White males earn more, on average, than indigenous males')

print('White male average salary:', mean_white_male)

print('Yellow male average salaries:',mean_yellow_male)

print('White males earn more, on average, than yellow males')

mask_1 = (df['sexo'] == 'mulher') & (df['salario']>=0) & (df['salario']<600000)

mask_2 = (df['sexo'] == 'homem') & (df['salario']>=0) & (df['salario']<600000)



#by limiting our returns to values >=0 & <=600k we are eliminating outliers that could disturb our analysis.
df['salario'][mask_1].describe()
df['salario'][mask_2].describe()

df['salario'][mask_1].plot.kde()



#In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function (PDF)

#of a random variable. This function uses Gaussian kernels and includes automatic bandwidth determination.
df['salario'][mask_2].plot.kde()
df_men = df[df['sexo'] == 'homem']

df_women = df[df['sexo'] == 'mulher']



#separating gender data into two distinct dataframes (men & women)
mean_salary_men= df_men['salario'].mean()

median_salary_men = df_men['salario'].median()

mode_salary_men= df_men['salario'].idxmax()



mean_salary_women = df_women['salario'].mean()

median_salary_women = df_women['salario'].median()

moda_salary_women = df_women['salario'].idxmax()



#Testing our hypothesis ==> men have higher salaries than women by ways of mean, median, and mode
mean_salary_men > mean_salary_women
median_salary_men > median_salary_women
mode_salary_men > moda_salary_women
print('Men mean salary =',"%.2f"% mean_salary_men,'Women mean salary =',"%.2f"% mean_salary_women)

print('Men median salary =',"%.2f"% median_salary_men,'Women median salary =',"%.2f"% median_salary_women)

print('Most frequent salary for men =',"%.2f"% mode_salary_men,'Most frequent salary for women =',"%.2f"% moda_salary_women)
df_men.dropna(inplace=True)

df_women.dropna(inplace=True)
from matplotlib import pyplot



bins = 100



ax = pyplot.hist(df_men['salario'], bins, alpha=0.5, label='Men Salary',color='blue')

ax = pyplot.hist(df_women['salario'], bins, alpha=0.5, label='Women Salary',color='red')



pyplot.legend(loc='upper right')

pyplot.show()