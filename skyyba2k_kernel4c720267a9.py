# Essential Data Analysis Ecosystem

import numpy as np

import pandas as pd

from pandas.api.types import CategoricalDtype



# Visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# Python Standard Libraries

import os  # For os file operations.

import re  # Used for data cleaning purposes.

import webbrowser  # Used to see sample reviews in glassdoor.com



# Ensures plots to be embedded inline.

%matplotlib inline



# Plot size frequently used.

two_in_row = (12, 4)  

# Style and Base color used for seaborn plots.

bcolor = sns.color_palette()[0]

sns.set(style='ticks', palette='pastel')



# Suppress warnings from final output.

import warnings

warnings.simplefilter("ignore")
dataset_path = '../input/'

df = pd.read_csv(os.path.join(dataset_path, 'employee_reviews.csv'), index_col=0)
print('Number of rows (reviews) and columns:', df.shape)

df_samples = df.sample(3)

df_samples
random_review = np.random.randint(0, df.shape[0]-1)

df.iloc[random_review] # A detailed look at a random review
df.info()
df.nunique()
links = df_samples['link']

print(links)

answer = input('Enter,  y  if you would like to open and see these sample reviews\` urls? ')

if answer.lower()=='y':

    [webbrowser.open(link) for link in links]
df.columns
df.columns = df.columns.str.replace('-', '_') 
companies_by_founded_date = ['microsoft', 'apple', 'amazon', 'netflix', 'google', 'facebook']

company_cat = CategoricalDtype(ordered=True, categories=companies_by_founded_date)

df['company'] = df['company'].astype(company_cat)

# TEST

df['company'].values
def plot_cat_counts(data=None, x=None):

    """Plot a categorical value with side by side horizantal bar and pie charts"""

    

    plt.figure(figsize=two_in_row)



    plt.subplot(1, 2, 1)

    sns.countplot(data=data, y=x, color=bcolor)

    plt.ylabel('')

    plt.xlabel('Review Counts')

    sns.despine() # remove the top and right borders





    plt.subplot(1, 2, 2)

    sorted_counts = data[x].value_counts()

    labels = sorted_counts.index



    plt.pie(sorted_counts, labels=None, 

            startangle=90, counterclock=False, wedgeprops = {'width' : 0.35})

    plt.axis('square')



    plt.legend(labels,

              title="Companies Proportions",

              loc="top left",

              bbox_to_anchor=(1, 0, .25, 1));
plot_cat_counts(df, 'company')
# Replace string "none" with NaN in entire dataset.



df = df.replace('none', np.nan)

df = df.replace('None', np.nan)

df = df.replace('None.', np.nan)
# Plot top 30 frequent locations

plt.figure(figsize=two_in_row)

(df['location'].value_counts().head(30) / len(df)).plot.bar();
btween_parentheses = r'\(([^)]+)\)'  # Regular expression to get a string between parentheses



def get_country(location):

    """Extracts and returns country name from location string.

    Returns NaN if 'none'."""

    

    if pd.isnull(location):

        return np.nan

    

    not_usa = re.findall(btween_parentheses, location)

    if not_usa:

        return not_usa[0]

    else:

        return 'USA'

    



def get_state(location):

    """Extracts and returns state name (if aby) from location string.

    Returns Nan if 'none or not applicable."""

    

    if pd.isnull(location):

        return np.nan

    

    not_usa = re.findall(btween_parentheses, location)

    if not_usa:

        if ',' in location:

            return location.split(',')[1].split()[0]

        else:

            return np.nan

    else:

        return location.strip()[-2:]



    

def get_city(location):

    """Extracts and returns city name from location string.

    Returns Nan if 'none'."""

    

    if pd.isnull(location):

        return np.nan

    

    not_usa = re.findall(btween_parentheses, location)

    if not_usa:

        if ',' in location:

            return location.split(',')[0]

        else:

            return location.split()[0]

    else:

        return location.split(',')[0]    
# Creating three new columns for location data

df['city'] = df['location'].apply(get_city)  # New column for the city.

df['state'] = df['location'].apply(get_state)  # New column for the State/Region.

df['country'] = df['location'].apply(get_country)  # New Column for the Country.



# Drop the untidy and no longer needed location column.

del df['location']
# TEST location columns

df[['city', 'state', 'country']].sample(5)
def plot_top_cats(col, top_percentage):

    """Plot members of a categorical variable that make up the top_percentage."""

    

    mask = df[col].value_counts(normalize=True).cumsum() < top_percentage

    top_items = mask[mask].index



    def group_top_itesm(x):

        if x in top_items:

            return x

        elif pd.isna(x):

            return np.nan

        else:

            return 'Other Countries'

    items = df[col].apply(group_top_itesm)



    plt.figure(figsize=two_in_row)

    sns.countplot(y=items, color=bcolor, order=items.value_counts().index)

    plt.ylabel(f'TOP {top_percentage*100}% in {col.upper()}')

    sns.despine()
plot_top_cats('country', 0.90)
plot_top_cats('country', 0.925)
countries_mask = (df.country == 'USA') | (df.country == 'UK') | (df.country == 'Ireland') | (df.country == 'Canada')

df = df[countries_mask]

del countries_mask

df.shape
df['dates'] = pd.to_datetime(df['dates'], errors='coerce')  # Type Casting to date

df.sort_values(by='dates', ascending=False, inplace=True)  # Sort reviews by date

df.rename(columns={'dates': 'date_posted'}, inplace=True)
yearly = df.groupby(df['date_posted'].dt.year).size()

positions = yearly.index

plt.bar(positions, yearly.values)

plt.title(f'{df.date_posted.min()} to {df.date_posted.max()}');
plt.figure(figsize=two_in_row)

(df['job_title'].value_counts().head(30) / len(df)).plot.bar();
def clean_text(col):

    """Cleaning text from formatings."""

    col = col.str.strip()

    col = col.str.replace("(<br/>)", "")

    col = col.str.replace('(<a).*(>).*(</a>)', '')

    col = col.str.replace('(&amp)', '')

    col = col.str.replace('(&gt)', '')

    col = col.str.replace('(&lt)', '')

    col = col.str.replace('(\xa0)', ' ')  

    return col



df['job_title'] = clean_text(df['job_title'])
df['current_emp'] = df['job_title'].apply(lambda x: True if x.split()[0] == 'Current' else False)

df['anonymous'] = df['job_title'].apply(lambda x: True if 'Anonymous' in str(x) else False)



df['job_title'] = df['job_title'].apply(lambda x: x.split('-')[1])

df['job_title'] = df['job_title'].apply(lambda x: np.nan if 'Anonymous' in str(x) else x)
# Test

plt.figure(figsize=two_in_row)

(df['job_title'].value_counts().head(30) / len(df)).plot.bar();

df[['job_title', 'current_emp', 'anonymous']].sample(5)
# Most popular Job Titles in entire dataset i.e. all companies combined.

df['job_title'].value_counts()[:20]
## Most popular group of employee wrote review in each company

df.groupby(['company', 'job_title']).size().sort_values(ascending=False)[:25]
ax = sns.countplot(df['current_emp'], hue=df['anonymous'], color=bcolor)

ax.set_xticklabels(['Past Employees', 'Current Employees'])

ax.set_xlabel('')



ax.legend(['Identified', 'Anonymous'], 

          title="Job-Title")



sns.despine();
# we leave summary (Review Headline) out of our analysis.

#text_cols = ['summary', 'pros', 'cons', 'advice_to_mgmt']

text_cols = [           'pros', 'cons', 'advice_to_mgmt']

for col in text_cols:

    df[col] = clean_text(df[col])



df[text_cols] = df[text_cols].replace('none', np.nan)

df[text_cols] = df[text_cols].replace('None', np.nan)

df['summary'][df['summary']=='.'] = np.nan # These are actually missing values
df.sample(5)[text_cols]
rating_cols = ['overall_ratings', 'work_balance_stars', 'culture_values_stars',

              'carrer_opportunities_stars', 'comp_benefit_stars', 'senior_mangemnet_stars']

df[rating_cols] = df[rating_cols].replace('none', np.nan)

df['overall_ratings'].nunique()
# Rating values to Numeric

for col in rating_cols:

    df[col] = pd.to_numeric(df[col], downcast='unsigned')

    

for col in rating_cols:

    if df[col].nunique() > 5:

        print(df[col].value_counts())
def five_ratings_only(col):

    for idx in col.value_counts().index:

        col[col==idx] = int(float(idx))

    return col



for col in rating_cols:

    df[col] = five_ratings_only(df[col])

    df[col].astype(np.unsignedinteger, errors='ignore')
fig, ax = plt.subplots(figsize=two_in_row)



fig.suptitle('Employees\' Overall Ratings\' Distributions', fontsize=14, fontweight='bold')



color = sns.color_palette()[1]



sns.countplot(data=df, x='overall_ratings', hue='current_emp', color=color)



ax.set(title='1 Star to 5 Stars')

ax.legend(['Past Employees', 'Current Employees'])

ax.set_axis_off()



locs = ax.get_xticks()

labels = ax.get_xlabel()



counts = list(df['overall_ratings'].value_counts(normalize=True).iloc[::-1])

for loc, lable, count in zip(locs, labels, counts):



    text = '{:0.0f}%'.format(100*count)

    ax.text(loc, 0, text, color='black', va='top', ha='center', fontsize=14)
fig, ax = plt.subplots(figsize=two_in_row)



fig.suptitle('Employees\' Overall Ratings\' Distributions', fontsize=14, fontweight='bold')



color = sns.color_palette()[3]



sns.countplot(data=df, x='overall_ratings', hue=None, color=color)



ax.set(title='1 Star to 5 Stars')

ax.set_axis_off()



locs = ax.get_xticks()

labels = ax.get_xlabel()



counts = list(df['overall_ratings'].value_counts(normalize=True).iloc[::-1])

for loc, lable, count in zip(locs, labels, counts):



    text = '{:0.0f}%'.format(100*count)

    ax.text(loc, 0, text, color='black', va='top', ha='center', fontsize=14)
plt.hist(df['helpful_count']);
def hist_magnifier(df, x, xlim1, xlim2, binsize):

    plt.hist(data=df, x=x, bins=np.arange(xlim1, xlim2+binsize, binsize))

    plt.xlim(xlim1, xlim2);
plt.figure(figsize=(18, 4))



plt.subplot(1, 3, 1)

hist_magnifier(df, df['helpful_count'], 0, 11, 1)



plt.subplot(1, 3, 2)

hist_magnifier(df, df['helpful_count'], 11, 100, 10)



plt.subplot(1, 3, 3)

hist_magnifier(df, df['helpful_count'], 100, df['helpful_count'].max()+100, 100)
cum_hist = df['helpful_count'].value_counts(normalize=True).cumsum()

cum_hist[cum_hist<0.95]
del df['link']
def plot_missings(df, figsize=(12, 4)):

    """Plot missing values bar visualization for each column of a DataFrame."""

    

    print(f'The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.')

    

    fig, ax = plt.subplots(figsize=figsize)

    sns.set(style='ticks', palette='pastel')

    color = sns.color_palette()[3]

    

    x = df.isna().sum().index.values

    y = df.isna().sum()

    sns.barplot(x, y, color=color, ax=ax)

    locs, labels = plt.xticks(rotation=90)

    for loc, label, missings, in zip(locs, labels, y):

        if not missings:

            ax.text(loc, 0, 'None', rotation=0, va='bottom', ha='center')

        else:

            ax.text(loc, missings, missings, rotation=0, va='bottom', ha='center')



    ax.set(title='Missing Value Counts in all Columns', xlabel='Columns', ylabel='Counts')

    sns.despine() # remove the top and right borders
plot_missings(df)
# Test; Checking the review with missing cons comment.

df[df['cons'].isna()]
# Test; Checking 4 reviews with missing date_posted.

df[df['date_posted'].isna()]
# Drop rows/reviews with no 'advice_to_managment`

df = df[df['advice_to_mgmt'].notna()]
df.sample()
df.describe().transpose()  # Numeric Columns
df.info(null_counts=False)
fig, axes = plt.subplots(nrows=1, ncols=6, sharey=True, figsize=(20, 5))

fig.suptitle('Star-Ratings\' Distributions', fontsize=22, fontweight='bold')

xticks=[1, 2, 3, 4, 5]

for ax, col in zip(axes, rating_cols):

    

    if col=='overall_ratings':

        color = sns.color_palette()[2]

    else:

        color = sns.color_palette()[1]

        

    ax = sns.countplot(ax=ax, data=df, x=df[col], color=color, order=xticks, hue=None)

    # plt.ylim(0, 12000)

    mean = '{:0.2f}'.format(df[col].mean())

    ax.set(title=ax.get_xlabel(), xlabel=mean, ylabel='')



    # TODO: Print percentage of each bar on each bar on it.
def ratings_trend(df=df, rating_cols=rating_cols):

    plt.figure(figsize=(7, 7))



    colors = ['grey', 'blue', 'green', 'red', 'brown']

    ypos = 4.25

    for col, color in zip(rating_cols[1:], colors):

        sns.pointplot(data=df, x='overall_ratings', y=col, color=color)

        plt.text(0.5, ypos, str(col), color=color)

        ypos += 0.15



    plt.ylim(1, 5)

    plt.grid()

    plt.xlabel('Overall Rating Stars')

    plt.ylabel('Star-Level average ratings of each Sub-Ratings vs. Overall Rating');
ratings_trend(df)
review_cols = ['pros', 'cons', 'advice_to_mgmt']

correction_dict = {r'-': '',

                   r'w/': 'with',

                   r' i ': ' I ',

                   r' & ': ' and '}



df[review_cols] = df[review_cols].replace(regex=correction_dict)
# Counting Words

df['wordcount'] = 0

for col in review_cols:

    df['wordcount'] += df[col].astype(str).apply(lambda text: len(text.split()))
df['wordcount'].describe()
df = df[df['wordcount'] >= 39]
def distplot_closelook(series, **kwarg):

    """"""

    

    fig, ax0 = plt.subplots(1, 1, figsize=(20, 2))

    sns.boxplot(series, color=bcolor)

    

    ax0.set_xlabel(f'All {len(series)} observations')

    

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 2))

    sns.boxplot(series, ax=ax0, **kwarg)

    ax0.set_xlim(0, np.percentile(series, 25))

    ax0.set_xlabel(f'Bottom (left) 25% Distribution')



    sns.boxplot(series, ax=ax1, **kwarg)

    ax1.set_xlim(np.percentile(series, 75), series.max())

    ax1.set_xlabel(f'Top (right) 25% Distribution')    
distplot_closelook(df['wordcount'])

df['wordcount'].describe()
# Assigns values outside 92.50% boundary to boundary value. 

# In other words capping word_count to a set ceiling value.

df['wordcount'] = df['wordcount'].clip(0, np.percentile(df['wordcount'], 90.0))
distplot_closelook(df['wordcount'])
plt.figure(figsize=(16, 4))

bin_size = 10

bins = np.arange(5, np.max(df['wordcount'])+bin_size, bin_size)

plt.hist(df.wordcount, bins)

plt.xticks(np.arange(0, 220+10, 10));
bins
# assign/map each review to the bin it belongs

bin_id = pd.cut(df['wordcount'], bins=bins, right=False, include_lowest=True)



# We linearly assign a score for each bin

bv = 1 / bins.shape[0]

f'Number of bins: {bins.shape[0]} - Each bins\' value: {bv} (evenly distributed over all bins)'
bins_table =  bin_id.value_counts().sort_index().to_frame().reset_index()

# calculate each bin wc_score increamentally from 0 to 1

bins_table['bin_score'] = (bins_table.index + 2) * bv  

bins_table
mapping_series = pd.Series(data=bins_table['bin_score'].values, index=bins_table['index'])  # make a series with bins' names (edges) as index and bin_score as value

mapping_series
# for each review, map the weight/score of the bin it beloges to

df['detail_factor'] = bin_id.map(mapping_series)
# Test

df[['wordcount', 'detail_factor']].sample(7)
max_days = 5 * 364.25 

df['review_days'] = pd.to_numeric((pd.datetime.today() - df['date_posted']).dt.days)

df['time_factor'] = df['review_days'].apply(lambda x: 1 - x/max_days if x < max_days else 0.0)
# Test

sns.lineplot(data=df, x=df['date_posted'], y=df['time_factor'])  # reviews time_factor values for the last 5 year
df['helpful_count'] = df['helpful_count'].clip(0, 100)

df['helful_factor'] = df['helpful_count'].apply(lambda x: 1 + x / 10)
plt.figure()



plt.subplot(2, 1, 1)

plt.hist(df['helful_factor'], bins=50)





plt.subplot(2, 1, 2)

plt.hist(df['helful_factor'], bins=100)

plt.xlim(1, 3);
# Test

df[['helpful_count', 'helful_factor']].sample(7)
df['stars_score'] = df['overall_ratings'].apply(lambda x: (int(x) - 1) / 4)
# Test

df[['stars_score','overall_ratings']].sample(7)
plt.figure(figsize=(18,6))



plt.subplot(1, 3, 1)

sns.boxplot(data=df, x=df['overall_ratings'], y=df['wordcount'], hue='country')



plt.subplot(1, 3, 2)

sns.boxplot(data=df, x=df['overall_ratings'], y=df['detail_factor'], hue='anonymous')



plt.subplot(1, 3, 3)

sns.boxplot(data=df, x=df.overall_ratings, y=df.culture_values_stars)
sns.boxplot(data=df, x=df.overall_ratings, y=df.helpful_count)
plt.figure(figsize=(18,6))

sns.barplot(data=df, x='company', y='detail_factor', hue='country');
df.groupby(['company', 'country']).count()
plt.figure(figsize=(18,6))

sns.barplot(data=df, x='company', y='overall_ratings', hue=df.country);
df.groupby('company').mean()[['detail_factor', 'overall_ratings']].sort_values('detail_factor', ascending=False)
plt.figure(figsize=(18,6))

sns.barplot(data=df, x=df.helpful_count, y=df.overall_ratings);

plt.xlim(-0.5, 10+0.5)
cum_hist = df['helpful_count'].value_counts(normalize=True).cumsum()

cum_hist[cum_hist < 0.975].index.max()
df.groupby('country').mean().sort_values(by='overall_ratings', ascending=False)
fig, axes = plt.subplots(nrows=1, ncols=6, sharey=True, figsize=(20, 5))

fig.suptitle('Ratings\' Distributions', fontsize=22, fontweight='bold')

xticks=[1, 2, 3, 4, 5]

for ax, col in zip(axes, rating_cols):

    

    if col=='overall_ratings' or col=='comp_benefit_stars':

        color = sns.color_palette()[0]

    elif col=='work_balance_stars' or col=='senior_mangemnet_stars':

        color = sns.color_palette()[1]

    else:

        color = sns.color_palette()[2]

        

    ax = sns.countplot(ax=ax, data=df, x=df[col], color=color, order=xticks, hue=None)

    mean = '{:0.2f}'.format(df[col].mean())

    std = '{:0.2f}'.format(df[col].std())

    ax.set(title=ax.get_xlabel(), xlabel=f'Mean:{mean}\nSD:{std}', ylabel='')
fig, axes = plt.subplots(nrows=1, ncols=6, sharey=True, figsize=(20, 5))

fig.suptitle('Ratings\' Distributions between Current and Past Employees', fontsize=22, fontweight='bold')

xticks=[1, 2, 3, 4, 5]

for ax, col in zip(axes, rating_cols):

    

    if col=='overall_ratings' or col=='comp_benefit_stars':

        color = sns.color_palette()[0]

    elif col=='work_balance_stars' or col=='senior_mangemnet_stars':

        color = sns.color_palette()[1]

    else:

        color = sns.color_palette()[2]

        

    ax = sns.countplot(ax=ax, data=df, x=df[col], color=color, order=xticks, hue='current_emp')

    mean = '{:0.2f}'.format(df[col].mean())

    std = '{:0.2f}'.format(df[col].std())

    ax.set(title=ax.get_xlabel(), xlabel=f'Mean:{mean}\nSD:{std}', ylabel='')
def ratings_trend(df=df, x='overall_ratings', rating_cols=rating_cols, hue=None):

    plt.figure(figsize=(7, 7))



    colors = ['grey', 'blue', 'green', 'red', 'brown']

    ypos = 4.25

    for col, color in zip(rating_cols[1:], colors):

        sns.pointplot(data=df, x=x, y=col, color=color, hue=hue)

        plt.text(0.5, ypos, str(col), color=color)

        ypos += 0.15



    plt.ylim(1, 5)

    plt.grid()

    plt.xlabel('Overall Rating Stars')

    plt.ylabel('Star-Level average ratings of each Sub-Ratings vs. Overall Rating');
ratings_trend()
ratings_trend(hue='anonymous')
ratings_trend(hue='current_emp')