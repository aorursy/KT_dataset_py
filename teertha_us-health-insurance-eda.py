# Lodaing the necessary Libraries.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-whitegrid')

pd.set_option('display.max_columns', 500)

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/ushealthinsurancedataset/insurance.csv')

data.head()
print(f'Shape of the data: {data.shape}')

print(f'There are {data.shape[0]} rows in the data.')
data.info()
# What are the different datatypes present in the data?

data.dtypes.unique()
# Let's check out individual columns:

data.columns
# Transform the Index object to a series, and grouping by data types:

g = data.columns.to_series().groupby(data.dtypes).groups

g
# let's create a dictionary containing various datatypes (Integer, Float and object) and the columns that have this datatype:

dt = {k.name: v for k, v in g.items()}

# Display the columns by different datatypes:

attributes_by_datatype = pd.DataFrame(list(dt.values()), index = dt.keys(), columns = ['Attr 1', 'Attr 2', 'Attr 3'])

attributes_by_datatype
# Unique values for 'children':

sorted(data['children'].unique())
data.isnull().any()
data.describe().transpose()
# Let's construct a function that shows the summary and density distribution of a numerical attribute:

def summary(x):

    x_min = data[x].min()

    x_max = data[x].max()

    Q1 = data[x].quantile(0.25)

    Q2 = data[x].quantile(0.50)

    Q3 = data[x].quantile(0.75)

    print(f'5 Point Summary of {x.capitalize()} Attribute:\n'

          f'{x.capitalize()}(min) : {x_min}\n'

          f'Q1                    : {Q1}\n'

          f'Q2(Median)            : {Q2}\n'

          f'Q3                    : {Q3}\n'

          f'{x.capitalize()}(max) : {x_max}')



    fig = plt.figure(figsize=(16, 10))

    plt.subplots_adjust(hspace = 0.6)

    sns.set_palette('pastel')

    

    plt.subplot(221)

    ax1 = sns.distplot(data[x], color = 'r')

    plt.title(f'{x.capitalize()} Density Distribution')

    

    plt.subplot(222)

    ax2 = sns.violinplot(x = data[x], palette = 'Accent', split = True)

    plt.title(f'{x.capitalize()} Violinplot')

    

    plt.subplot(223)

    ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)

    plt.title(f'{x.capitalize()} Boxplot')

    

    plt.subplot(224)

    ax3 = sns.kdeplot(data[x], cumulative=True)

    plt.title(f'{x.capitalize()} Cumulative Density Distribution')

    

    plt.show()
# Let's take a closer look at the Boxplot, and calculate the measure of skewness and totalnumber of outlier values for various attributes through a function:



def box_plot(x = 'bmi'):

    def add_values(bp, ax):

        """ This actually adds the numbers to the various points of the boxplots"""

        for element in ['whiskers', 'medians', 'caps']:

            for line in bp[element]:

                # Get the position of the element. y is the label you want

                (x_l, y),(x_r, _) = line.get_xydata()

                # Make sure datapoints exist 

                # (I've been working with intervals, should not be problem for this case)

                if not np.isnan(y): 

                    x_line_center = x_l + (x_r - x_l)/2

                    y_line_center = y  # Since it's a line and it's horisontal

                    # overlay the value:  on the line, from center to right

                    ax.text(x_line_center, y_line_center, # Position

                            '%.2f' % y, # Value (3f = 3 decimal float)

                            verticalalignment='center', # Centered vertically with line 

                            fontsize=12, backgroundcolor="white")



    fig, axes = plt.subplots(1, figsize=(4, 8))



    red_diamond = dict(markerfacecolor='r', marker='D')



    bp_dict = data.boxplot(column = x, 

                             grid=True, 

                             figsize=(4, 8), 

                             ax=axes, 

                             vert = True, 

                             notch=False, 

                             widths = 0.7, 

                             showmeans = True, 

                             whis = 1.5,

                             flierprops = red_diamond,

                             boxprops= dict(linewidth=3.0, color='black'),

                             whiskerprops=dict(linewidth=3.0, color='black'),

                             return_type = 'dict')



    add_values(bp_dict, axes)



    plt.title(f'{x.capitalize()} Boxplot', fontsize=16)

    plt.ylabel(f'{x.capitalize()}', fontsize=14)

    plt.show()

    

    skew = data[x].skew()

    Q1 = data[x].quantile(0.25)

    Q3 = data[x].quantile(0.75)

    IQR = Q3 - Q1

    total_outlier_num = ((data[x] < (Q1 - 1.5 * IQR)) | (data[x] > (Q3 + 1.5 * IQR))).sum()

    print(f'Mean {x.capitalize()} = {data[x].mean()}')

    print(f'Median {x.capitalize()} = {data[x].median()}')

    print(f'Skewness of {x}: {skew}.')

    print(f'Total number of outliers in {x} distribution: {total_outlier_num}.')   
summary('age')
box_plot('age')
# How many of the insured have the age of 64?  

df = data[data['age'] == data['age'].max()]

print(df.head())

print()

print(f'Total number of insured people with the age of 64: {len(df)}.')
summary('bmi')
box_plot('bmi')
# Who is the insured with the highest BMI, and how does his charges compare to the rest?

data[data['bmi'] == data['bmi'].max()]
data['charges'].mean(), data['charges'].median()
summary('charges')
box_plot('charges')
# Who is paying the highest charges?

data[data['charges'] == data['charges'].max()]
# Who is the insured with the highest BMI, and how does his charges compare to the rest?

data[data['bmi'] == data['bmi'].max()]
data['charges'].mean(), data['charges'].median()
# Create a function that returns a Pie chart for categorical variable:

def pie_chart(x = 'smoker'):

    """

    Function creates a Pie chart for categorical variables.

    """

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))



    s = data.groupby(x).size()



    mydata_values = s.values.tolist()

    mydata_index = s.index.tolist()



    def func(pct, allvals):

        absolute = int(pct/100.*np.sum(allvals))

        return "{:.1f}%\n({:d})".format(pct, absolute)





    wedges, texts, autotexts = ax.pie(mydata_values, autopct=lambda pct: func(pct, mydata_values),

                                      textprops=dict(color="w"))



    ax.legend(wedges, mydata_index,

              title="Index",

              loc="center left",

              bbox_to_anchor=(1, 0, 0.5, 1))



    plt.setp(autotexts, size=12, weight="bold")



    ax.set_title(f'{x.capitalize()} Piechart')



    plt.show()
sns.countplot(x = 'sex', data = data)
pie_chart('sex')
sns.countplot(x = 'smoker', hue = 'sex', data = data)
pie_chart('smoker')
# Are average premium charges for smokers significantly higher than non-smokers?

data['charges'].groupby(data['smoker']).mean()
data.groupby(['smoker', 'sex']).agg('count')
# yes, average premium charges for smokers are indeed significantly higher than non-smokers.

sns.barplot(x = "smoker", y = "charges", data = data)
sns.catplot(x="smoker", y="charges", hue="sex",

            kind="violin", inner="quartiles", split=True,

            palette="pastel", data=data);
data.groupby(['smoker', 'sex']).agg('count')['age']
sns.countplot(x = 'region', data = data)
pie_chart('region')
sns.countplot(x = 'children', data = data)
pie_chart('children')
data.groupby(['children']).agg('count')['age']
for x in ['sex', 'children', 'smoker', 'region']:

    data[x] = data[x].astype('category')



data.dtypes 
type(data.dtypes)
# Next, we select all columns of the dataFrame with datatype = category:

cat_columns = data.select_dtypes(['category']).columns

cat_columns
# Finally, we transform the original columns by replacing the elements with their category codes:

data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

data.head()
# Now we can plot all columns of our dataset in a pairplot!

sns.pairplot(data, hue  = 'smoker')
data.plot(kind="scatter", x="age", y="charges", 

    s=data["smoker"]*25, label="smoker", figsize=(14,10),

    c='bmi', cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

plt.legend()
corr = data.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, cmap = 'summer_r')
smokers = data[data['smoker'] == 0]

nonsmokers = data[data['smoker'] == 1]

charge_smokers = smokers['charges']

charge_nonsmokers = nonsmokers['charges']



print(f'Number of smokers: {smokers.shape[0]}')

print(f'Variance in charges of smokers: {np.var(charge_smokers)}')

print(f'Number of non - smokers: {nonsmokers.shape[0]}')

print(f'Variance in charges of non - smokers: {np.var(charge_nonsmokers)}')
from scipy.stats import ttest_ind



t_statistic, p_value = ttest_ind(charge_smokers, charge_nonsmokers, equal_var=False)

print(f't_statistic: {t_statistic}\np_value: {p_value}')
print ("two-sample t-test p-value=", p_value)
p_value > 0.05
smokers = data[data['smoker'] == 0]

nonsmokers = data[data['smoker'] == 1]

charge_smokers = smokers['charges']

charge_nonsmokers = nonsmokers['charges']



print(f'Number of smokers: {smokers.shape[0]}')

print(f'Variance in charges of smokers: {np.var(charge_smokers)}')

print(f'Number of non - smokers: {nonsmokers.shape[0]}')

print(f'Variance in charges of non - smokers: {np.var(charge_nonsmokers)}')
#Visualizing the collected data:

g = sns.catplot(x="smoker", y="charges", hue="sex",

            kind="violin", inner="quartiles", split=True,

            palette="RdBu_r", data=data, legend_out = True);



xlabels = ['non-smoker', 'smoker']

g.set_xticklabels(xlabels)



new_title = 'Sex'

g._legend.set_title(new_title)

g._legend.set_bbox_to_anchor([1.1, 0.5])

# replace labels

new_labels = ['female', 'male']

for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
from scipy.stats import ttest_ind



t_statistic, p_value = ttest_ind(charge_smokers, charge_nonsmokers, equal_var=False)

print(f't_statistic: {t_statistic}\np_value: {p_value}')
print ("two-sample t-test p-value=", p_value)
p_value > 0.05
males = data[data['sex'] == 1]

females = data[data['sex'] == 0]

bmi_males = males['bmi']

bmi_females = females['bmi']



print(f'Number of males: {males.shape[0]}')

print(f'Variance in BMI of males: {np.var(bmi_males)}')

print(f'Number of females: {females.shape[0]}')

print(f'Variance in BMI of females: {np.var(bmi_females)}')
#Visualizing the collected data:

g = sns.catplot(x="sex", y="bmi", hue="smoker",

            kind="violin", inner="quartiles", split=True,

            palette="pastel", data=data, legend_out = True);



xlabels = ['female', 'male']

g.set_xticklabels(xlabels)



new_title = 'Smoker'

g._legend.set_title(new_title)

g._legend.set_bbox_to_anchor([1.1, 0.5])

# replace labels

new_labels = ['non-smoker', 'smoker']

for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
from scipy.stats import ttest_ind



t_statistic, p_value = ttest_ind(bmi_males, bmi_females, equal_var=False)

print(f't_statistic: {t_statistic}\np_value: {p_value}')
print ("two-sample t-test p-value=", p_value)
p_value > 0.05
n_females = data['sex'].value_counts()[0] # number of females in the data

n_males = data['sex'].value_counts()[1] # number of females in the data



female_smokers = data[data['sex'] == 0].smoker.value_counts()[1] # number of female smokers

male_smokers = data[data['sex'] == 1].smoker.value_counts()[1] # number of male smokers



print([female_smokers, male_smokers] , [n_females, n_males])

print(f' Proportion of smokers in females, males = {round(115/662,4)*100}%, {round(159/676,4)*100}% respectively.')
# Visualization of the collected data:

plt.figure(figsize=(6,5))

chart = sns.countplot(y = 'sex', hue = 'smoker', data = data)

chart.set_yticklabels(['female', 'male'])
# The proportions are different, but are they statistically significant?

from statsmodels.stats.proportion import proportions_ztest



stat, pval = proportions_ztest([female_smokers, male_smokers] , [n_females, n_males])

print(f'Statistic: {stat}\np_value: {pval}')
if pval < 0.05:

    print(f'With a p-value of {pval} the difference is significant. We reject the Null Hypothesis.')

else:

    print(f'With a p-value of {pval} the difference is not significant. We fail to reject the Null Hypothesis')
df = data[data['children'] <= 2]

female = df[df['sex'] == 0]

female.head()
# Visualizing the collected data:

fig = plt.figure(figsize=(12, 8))

box_plot = sns.boxplot(x = "children", y = "bmi", data = female, width = 0.5)



medians = female.groupby(['children'])['bmi'].median().round(2)

vertical_offset = female['bmi'].median() * 0.05 # offset from median for display



medians

for xtick in box_plot.get_xticks():

    box_plot.text(xtick, medians[xtick] + vertical_offset,medians[xtick], 

            horizontalalignment='center',color='w',weight='semibold')





plt.title('BMI by No. of children')

plt.show()
import statsmodels.api as sm

from statsmodels.formula.api import ols



mod = ols('bmi ~ children', data = female).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print(pairwise_tukeyhsd(female['bmi'], female['children']))