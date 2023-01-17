import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



students_df = pd.read_csv('../input/StudentsPerformance.csv')

students_df.head()
students_df.dtypes
categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

students_df[categorical_columns] = students_df[categorical_columns].astype('category')



new_column_names = [column + '(categorical)' for column in categorical_columns]

col_name_pairs = dict(zip(categorical_columns, new_column_names))



students_df.rename(columns=col_name_pairs, inplace=True)



students_df.dtypes
for column in students_df.columns:

    print((column + ':    {}\n').format(students_df[column].unique()))
students_df.isnull().sum()
#  draw's the y-axis values of bars of the bar_plot passed to it 

def draw_freq_on_bars(bar_plot, category_freqs):

    for barIndex, freq in enumerate(category_freqs):

        bar_plot.text(x=barIndex, y=freq - (0.16 * bar_plot.get_ylim()[1]), s=str(freq) + '\n(' + str(round((freq / category_freqs.sum()) * 100)) + '%)', color='white', horizontalAlignment='center', fontsize=15)

        

        

#  formats xtickLabels of the plot passed to it so that they don't overlap with each other when shown.

def format_xtickLabels(plot, df, x, show_count=False):

    xtickLabels = plot.get_xticklabels()

    if(show_count):

        for count, xtickLabel in zip(df[x].value_counts(), xtickLabels):

            xtickLabel.set_text(xtickLabel.get_text().replace(' ', '\n') + '\n(n=' + str(count) + ')')

    else:

        for xtickLabel in xtickLabels:

            xtickLabel.set_text(xtickLabel.get_text().replace(' ', '\n'))

    plot.set_xticklabels(xtickLabels)
#  changing color palette of Seaborn

sns.set(palette='tab10')
plt.figure()

category_freqs = students_df['gender(categorical)'].value_counts()

bar_plot = sns.barplot(x=category_freqs.index.get_values(), y=category_freqs)



bar_plot.set_ylabel('count (by gender)')



draw_freq_on_bars(bar_plot, category_freqs)
plt.figure(figsize=(7,4))

category_freqs = students_df['race/ethnicity(categorical)'].value_counts()

bar_plot = sns.barplot(x=category_freqs.index.get_values(), y=category_freqs)



bar_plot.set_ylabel('count (by race/ethnicity)')



draw_freq_on_bars(bar_plot, category_freqs)
plt.figure(figsize=(9,4))

category_freqs = students_df['parental level of education(categorical)'].value_counts()

bar_plot = sns.barplot(x=category_freqs.index.get_values(), y=category_freqs)



bar_plot.set_ylabel('count (by parental level of education)')



format_xtickLabels(bar_plot, students_df, 'parental level of education(categorical)')



draw_freq_on_bars(bar_plot, category_freqs)
plt.figure()

category_freqs = students_df['lunch(categorical)'].value_counts()

bar_plot = sns.barplot(x=category_freqs.index.get_values(), y=category_freqs)



bar_plot.set_ylabel('count (by lunch)')



draw_freq_on_bars(bar_plot, category_freqs)
plt.figure()

category_freqs = students_df['test preparation course(categorical)'].value_counts()

bar_plot = sns.barplot(x=category_freqs.index.get_values(), y=category_freqs)



bar_plot.set_ylabel('count (by test preparation course)')



draw_freq_on_bars(bar_plot, category_freqs)
sns.pairplot(students_df, diag_kws={'bins':20, 'ec':'white'})
#  draw's medians of scores(column y), for each group/category in column x of DataFrame df,

#  on the passed violin_plot

def draw_median_on_violinplot(violin_plot, df, x, y):

    medians = df.groupby([x])[y].median().values

    for violinIndex, median in enumerate(medians):

        violin_plot.text(violinIndex + 0.06, median - 2, str(median), color='black', fontsize='small')
#  this'll be handy in iterating over violin-plots of each scores column

score_columns = ['math score', 'reading score', 'writing score']
sns.pairplot(students_df, hue='gender(categorical)', diag_kind='kde')
plt.figure(figsize=(12, 4))

plt.subplots_adjust(wspace=1, bottom=0.2)



count_of_subplots = len(score_columns)

for i, column in enumerate(score_columns):

    

    plt.subplot(1, count_of_subplots, i + 1)

    

    violin_plot = sns.violinplot(data=students_df, x='gender(categorical)', y=column)



    format_xtickLabels(violin_plot, students_df, 'gender(categorical)', True)



    draw_median_on_violinplot(violin_plot=violin_plot, df=students_df, x='gender(categorical)', y=column)
sns.pairplot(students_df, hue='race/ethnicity(categorical)', diag_kind='kde')
plt.figure(figsize=(20, 5))

plt.subplots_adjust(wspace=0.3, bottom=0.2)



count_of_subplots = len(score_columns)

for i, column in enumerate(score_columns):

    

    plt.subplot(1, count_of_subplots, i + 1)

    

    violin_plot = sns.violinplot(data=students_df, x='race/ethnicity(categorical)', y=column)

    

    format_xtickLabels(violin_plot, students_df, 'race/ethnicity(categorical)', True)



    draw_median_on_violinplot(violin_plot=violin_plot, df=students_df, x='race/ethnicity(categorical)', y=column)
sns.pairplot(students_df, hue='parental level of education(categorical)', diag_kind='kde')
plt.figure(figsize=(25, 5))

plt.subplots_adjust(wspace=0.3, bottom=0.2)



count_of_subplots = len(score_columns)

for i, column in enumerate(score_columns):

    

    plt.subplot(1, count_of_subplots, i + 1)

    

    violin_plot = sns.violinplot(data=students_df, x='parental level of education(categorical)', y=column)

    

    format_xtickLabels(violin_plot, students_df, 'parental level of education(categorical)', True)



    draw_median_on_violinplot(violin_plot=violin_plot, df=students_df, x='parental level of education(categorical)', y=column)
sns.pairplot(students_df, hue='lunch(categorical)', diag_kind='kde')
plt.figure(figsize=(12, 5))

plt.subplots_adjust(wspace=1, bottom=0.2)



count_of_subplots = len(score_columns)

for i, column in enumerate(score_columns):

    

    plt.subplot(1, count_of_subplots, i + 1)

    

    violin_plot = sns.violinplot(data=students_df, x='lunch(categorical)', y=column)

    

    format_xtickLabels(violin_plot, students_df, 'lunch(categorical)', True)



    draw_median_on_violinplot(violin_plot=violin_plot, df=students_df, x='lunch(categorical)', y=column)
sns.pairplot(students_df, hue='test preparation course(categorical)', diag_kind='kde')
plt.figure(figsize=(12, 5))

plt.subplots_adjust(wspace=1, bottom=0.2)



count_of_subplots = len(score_columns)

for i, column in enumerate(score_columns):

    

    plt.subplot(1, count_of_subplots, i + 1)

    

    violin_plot = sns.violinplot(data=students_df, x='test preparation course(categorical)', y=column)

    

    format_xtickLabels(violin_plot, students_df, 'test preparation course(categorical)', True)



    draw_median_on_violinplot(violin_plot=violin_plot, df=students_df, x='test preparation course(categorical)', y=column)
#  format of comments below:    (<numeric label 1>, <category 1>), (<numeric label 2>, <category 2>), .... and so on.



#  (0, 'some high school'), (1, 'high school'), (2, 'some college'), (3, "associate's degree"),

#  (4, "bachelor's degree"), (5, "master's degree")

students_df['parental level of education(numeric)'] = students_df['parental level of education(categorical)']

students_df['parental level of education(numeric)'].cat.categories = [3, 4, 1, 5, 2, 0]

students_df['parental level of education(numeric)'] = students_df['parental level of education(numeric)'].astype('int')



#  (0, 'female'), (1, 'male')

students_df['gender(numeric)'] = students_df['gender(categorical)']

students_df['gender(numeric)'].cat.categories = [0, 1]

students_df['gender(numeric)'] = students_df['gender(numeric)'].astype('int')



#  (0, 'group A'), (1, 'group B'), (2, 'group C'), (3, 'group D'), (4, 'groud E')

students_df['race/ethnicity(numeric)'] = students_df['race/ethnicity(categorical)']

students_df['race/ethnicity(numeric)'].cat.categories = [0, 1, 2, 3, 4]

students_df['race/ethnicity(numeric)'] = students_df['race/ethnicity(numeric)'].astype('int')



#  (0, 'free/reduced'), (1, 'standard')

students_df['lunch(numeric)'] = students_df['lunch(categorical)']

students_df['lunch(numeric)'].cat.categories = [0, 1]

students_df['lunch(numeric)'] = students_df['lunch(numeric)'].astype('int')



#  (0, 'none'), (1, 'completed')

students_df['test preparation course(numeric)'] = students_df['test preparation course(categorical)']

students_df['test preparation course(numeric)'].cat.categories = [1, 0]

students_df['test preparation course(numeric)'] = students_df['test preparation course(numeric)'].astype('int')



students_df.head()
plt.figure()

correlation_coeffs = students_df.corr()

#  mask gets rid of the same coefficients repeated in the triangle above the diagonal

mask = np.tril(np.ones(correlation_coeffs.shape)).astype('bool')

mask = ~mask



sns.heatmap(correlation_coeffs, mask=mask, annot=True, vmin=-1, vmax=1, cmap='viridis', annot_kws={'size':9})