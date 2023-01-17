# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

% matplotlib inline 

# Necessary to display plots in notebook

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as stats # Stats API 

import matplotlib.pyplot as plt # Plotting API

import seaborn as sns 

sns.set(style='whitegrid', palette='colorblind')

from IPython.core.display import display, HTML # Allows to print HTML tables

import warnings # Necessary to ignore warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/abalone_original.csv")

df.head()
df.info()
df['rings'] = df['rings'].astype('int32')

float_cols = ['length', 'diameter', 'height', 'whole-weight', 'shucked-weight', 'viscera-weight', 'shell-weight']

for col in float_cols:

    df[col] = df[col].astype('float32')

df.info()
df.sex.describe()
df.sex.value_counts(normalize=True).sort_index()
df_sex_category = df.sex.value_counts(normalize=True).sort_index()

x = range(len(df_sex_category))

figure = plt.figure(figsize=(10, 6))

axes1 = figure.add_subplot(1, 2, 1)

axes1.bar(x, df_sex_category, color="steelblue",align="center")

axes1.set_xticks(x)

# Set x axis tick labels

axes1.set_xticklabels(df_sex_category.axes[0])

# Set x and y axis chart label

axes1.set_title("Sex Categories")

axes1.set_ylabel("Density")

axes1.xaxis.grid(False)

# Remove all of the axis tick marks

axes1.tick_params(bottom=False, top=False, left=False, right=False)

# Hide all of the spines

for spine in axes1.spines.values():

    spine.set_visible(False)

axes1.yaxis.grid(b=True, which="major");
df.length.describe()
def calculate_tukey_five(data):

    min, q1, q2, q3, max = np.concatenate(

        [[np.min(data)], stats.mstats.mquantiles(data, [0.25, 0.5, 0.75]), [np.max(data)]])

    data = {"Min": min, "Q1": q1, "Q2": q2, "Q3": q3, "Max": max}

    return data



def calculate_tukey_dispersion(five):

    data = {

        "Range": five["Max"] - five["Min"],

        "IQR": five["Q3"] - five["Q1"],

        "QCV": (five["Q3"] - five["Q1"]) / five["Q2"]

    }

    return data



def display_dict(m, precision=3):

    table = "<table>"

    for item in m.items():

        table += ("<tr><th>{0}</th><td>{1:." +

                  str(precision) + "f}</td></tr>").format(*item)

    table += "</table>"

    return display(HTML(table))



data = calculate_tukey_five(df.length)

data_dict = calculate_tukey_dispersion(data)



display_dict(data_dict)
def restyle_boxplot(patch):

    # change color and linewidth of the whiskers

    for whisker in patch['whiskers']:

        whisker.set(color='#000000', linewidth=1)



    # change color and linewidth of the caps

    for cap in patch['caps']:

        cap.set(color='#000000', linewidth=1)



    # change color and linewidth of the medians

    for median in patch['medians']:

        median.set(color='#000000', linewidth=2)



    # change the style of fliers and their fill

    for flier in patch['fliers']:

        flier.set(marker='o', color='#000000', alpha=0.2)



    for box in patch["boxes"]:

        box.set(facecolor='#FFFFFF', alpha=0.5)



def numeric_boxplot(numeric_df, label, title):

    figure = plt.figure(figsize=(20, 6))

    # Add Main Title

    figure.suptitle(title)

    # Left side: Boxplot 1

    axes1 = figure.add_subplot(1, 2, 1)

    patch = axes1.boxplot(numeric_df, labels=[label], vert=False, showfliers = True, patch_artist=True, zorder=1)

    restyle_boxplot(patch)

    axes1.set_title('Boxplot 1')

    # Right side: Boxplot 2

    axes2 = figure.add_subplot(1, 2, 2)

    patch = axes2.boxplot(numeric_df, labels=[label], vert=False, patch_artist=True, zorder=1)

    restyle_boxplot(patch)

    axes2.set_title('Boxplot 2')

    y = np.random.normal(1, 0.01, size=len(numeric_df))

    axes2.plot(numeric_df, y, 'o', color='steelblue', alpha=0.4, zorder=2)

    plt.show()

    plt.close()



numeric_boxplot(df.length, 'Length', 'Distribution of Length')
figure = plt.figure(figsize=(20, 6))



axes = figure.add_subplot(1, 2, 2)

axes.hist(df.length, density=True, alpha=0.75)

axes.set_title("Density Histogram of Length: default bins")

axes.set_ylabel("Density")

axes.set_xlabel("Length")

axes.xaxis.grid(False)

plt.show()

plt.close()
figure = plt.figure(figsize=(6, 6))

axes = figure.add_subplot(1, 1, 1)

stats.probplot(df.length, dist="norm", plot=axes)

axes.set(title="Q-Q Plot of Length")

axes.xaxis.grid(False)

plt.show()

plt.close()
def cdf_plot(numeric_df):

    figure = plt.figure(figsize=(20, 8))



    mn = np.min(numeric_df)

    mx = np.max(numeric_df)

    mean = np.mean(numeric_df)

    std = np.std(numeric_df)



    axes = figure.add_subplot(1, 2, 1)



    values, base = np.histogram(numeric_df, bins=11, density=True)

    cumulative = np.cumsum(values)

    axes.plot(base[:-1], cumulative, color="steelblue")

    axes.set_xlim((mn, mx))



    sampled_data = [mean + r * std for r in np.random.standard_normal(10000)]

    values2, base = np.histogram(sampled_data, bins=base, density=True)

    cumulative2 = np.cumsum(values2)

    axes.plot( base[:-1], cumulative2, color="firebrick")

    axes.set_xlim((np.min(df.length), np.max(df.length)))

    axes.set_xlabel( "Empirical v. Theoretical: Normal Distribution")

    axes.xaxis.grid(False)

    

    axes = figure.add_subplot(1, 2, 2)



    differences = cumulative2 - cumulative

    axes.plot(base[:-1], differences, color='firebrick')

    axes.set_xlim((mn, mx))

    axes.hlines(0, 0, 14000, linestyles="dotted")

    axes.set_xlabel( "Empirical v. Theoretical: Normal Distribution, Difference")

    axes.xaxis.grid(False)

    plt.show()

    plt.close()

    

cdf_plot(df.length)
df.height.describe()
df[df['height'] == 0]
df[df['height'] == 226]
numeric_boxplot(df.height, 'Height', 'Distribution of Height')
df[['whole-weight', 'shucked-weight', 'viscera-weight', 'shell-weight']].describe()
# I am trying out freeman diaconis rule to find optimal binwidth as it is less sensitive to outliers in data

# Reference: https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width

def freeman_diaconis(data):

    quartiles = stats.mstats.mquantiles(data, [0.25, 0.5, 0.75])

    iqr = quartiles[2] - quartiles[0]

    n = len(data)

    h = 2.0 * (iqr/n**(1.0/3.0))

    return int(h)



weights = ['whole-weight', 'shucked-weight', 'viscera-weight', 'shell-weight']

figure = plt.figure(figsize=(20, 10))

for i, k in enumerate(weights):

    axes = figure.add_subplot(2, 3, i + 1)

    subdata = df[k]

    binwidth = freeman_diaconis(subdata)

    bins = np.arange(min(subdata), max(subdata) + binwidth, binwidth)

    axes.hist(subdata, color="steelblue", bins=bins, density=True, alpha=0.75)

    axes.xaxis.grid(False)

    axes.set_title("Density of {}: adjusted bin size".format(k))

    if (i % 3 == 0):

        axes.set_ylabel("Density")

    axes.set_xlabel(k)

plt.tight_layout()
# Create weight-diff feature

df['weight-diff'] = df['whole-weight'] - (df['shell-weight'] + df['shucked-weight'] + df['viscera-weight'])

df['weight-diff'].describe()
print("Number of weigh-diff observations that are negative:", len(df[df['weight-diff'] < 0]))

df[df['weight-diff'] < 0].head()
df[df['weight-diff'] < 0]['weight-diff'].describe()
figure = plt.figure(figsize=(10, 6))



axes = figure.add_subplot(1, 1, 1)

axes.hist(df['weight-diff'], bins=30, density=True, alpha=0.75)

axes.set_ylabel("Density")

axes.set_xlabel("Weight Difference")

axes.set_title("Density Histogram of Weight Difference: adjusted bin size")

axes.xaxis.grid(False)

plt.show()
df.rings.describe()
bins = np.arange(min(df.rings), max(df.rings) + binwidth, binwidth)

figure = plt.figure(figsize=(10, 6))

axes = figure.add_subplot(1, 1, 1)

axes.hist(df.rings, bins=20, density=True, alpha=0.75)

axes.set_ylabel("Density")

axes.set_xlabel("Rings")

axes.set_title("Density Histogram of Rings")

axes.xaxis.grid(False)

plt.show()
# Taken from the seaborn example at:

# http://seaborn.pydata.org/examples/many_pairwise_correlations.html

corr = df.corr('spearman')

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(8, 6))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, vmax=1,square=True, 

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True)

plt.show()

plt.close()
df.plot.scatter("height", "whole-weight", figsize=(10, 6),

                title="Plot of Height vs. Whole Weight ", alpha=0.75)

plt.show()

plt.close()
def lowess_scatter(data, x, y, jitter=0.0, skip_lowess=False):

    import statsmodels.api as sm

    if skip_lowess:

        fit = np.polyfit(data[x], data[y], 1)

        line_x = np.linspace(data[x].min(), data[x].max(), 10)

        line = np.poly1d(fit)

        line_y = list(map(line, line_x))

    else:

        lowess = sm.nonparametric.lowess(data[y], data[x], frac=.3)

        line_x = list(zip(*lowess))[0]

        line_y = list(zip(*lowess))[1]



    figure = plt.figure(figsize=(10, 6))

    axes = figure.add_subplot(1, 1, 1)

    xs = data[x]

    if jitter > 0.0:

        xs = data[x] + stats.norm.rvs(0, 0.5, data[x].size)



    axes.scatter(xs, data[y], marker="o", color="steelblue", alpha=0.5)

    axes.plot(line_x, line_y, color="DarkRed")

    title = "Plot of {0} v. {1}".format(x, y)

    if not skip_lowess:

        title += " with LOESS"

    axes.set_title(title)

    axes.set_xlabel(x)

    axes.set_ylabel(y)

    axes.xaxis.grid(False)

    plt.show()

lowess_scatter(df, "length", "whole-weight")
lowess_scatter(df, "diameter", "whole-weight")
lowess_scatter(df, "whole-weight", "rings")
lowess_scatter(df, "length", "rings")
lowess_scatter(df, "diameter", "rings")
df.plot.scatter("height", "rings", figsize=(10, 6),

                title="Plot of Height vs. Rings", alpha=0.75)

axes = plt.gca()

axes.xaxis.grid(False)

plt.show()
no_outlier_df = df[(df['height'] > 0) & (df['height'] < 100)]

lowess_scatter(no_outlier_df, "height", "rings")
lowess_scatter(df, "shell-weight", "rings")
sns.pairplot(no_outlier_df, vars = ['rings', 'length', 'diameter', 'height', 'whole-weight', 'shell-weight'], hue='sex');