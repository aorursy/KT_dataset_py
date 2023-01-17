import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='darkgrid')
# import warnings; warnings.filterwarnings('ignore')

# Auto reload any script without the need to restart jupyter notebook server
%reload_ext autoreload
%autoreload 2
%matplotlib inline

# Change the format of precisions of decimals
pd.set_option('display.float_format', lambda x: '{:f}'.format(x))

SEED = 42
PATH = '../input/'
color = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#feb308", "e78ac3"]

%matplotlib inline
%config InlineBackend.figure_format='retina'
# Change defualt size of matplotlibs plots
plt.rcParams['figure.figsize'] = (8.0, 5.0)

## Data Description
def describe(df):
    print('======================================')
    print('No. of Rows.:{0}\nNo. of Columns:{1}\n'.format(df.shape[0], df.shape[1]))
    print('======================================')
    data_type = DataFrame(df.dtypes, columns=['Data Type'])
    null_count =  DataFrame(df.isnull().sum(), columns=['Null Count'])
    not_null_count = DataFrame(df.notnull().sum(), columns=['Not Null Count'])
    unique_count = DataFrame(df.nunique(), columns=['Unique Count'])
    categorical = loans.describe(include='O').T
    count = loans.isnull().sum().sort_values(ascending=False)
    percent = loans.isnull().sum().sort_values(ascending=False)/loans.shape[0]*100
    missing = pd.DataFrame({'Missing Count':count , 'Missing count Percent':percent})#, columns=['Missing Count', 'Percent count'])
    joined = pd.concat([data_type, null_count, not_null_count, unique_count, categorical, missing], axis=1)
    display(joined)
    display(df.describe().T)
    return None

## Adding more time columns
def add_datepart(df, date_column):
    date_series = df[date_column]
    df[date_column] = pd.to_datetime(date_series, infer_datetime_format=True)
    for n in ('Year', 'Month', 'Week', 'Day', 'Weekday_Name', 'Dayofweek', 'Dayofyear'):
        df['Date'+'_'+n] = getattr(date_series.dt, n.lower())
        
## Proportion Plot
def proportion_plot(column, title='', figsize=(15, 5), top=20):
    value_count = (100*column.value_counts()/column.shape[0])[:top]
    plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")
    g = sns.barplot(x=value_count.index, y=value_count.values)
    plt.title(title, size=20)
    plt.xlabel(column.name, size=15); plt.ylabel('percentage', size=15)
    plt.xticks(size=12, rotation=90); plt.yticks(size=15)
loans = pd.read_csv(f'{PATH}kiva_loans.csv')
loans.sample(n=5).T
df_describe = describe(loans)
sns.set_style("darkgrid")
sns.kdeplot(loans.loan_amount[loans.loan_amount], color='r', shade=True);
sns.kdeplot(loans.funded_amount[loans.funded_amount], color='b', shade=True);
appr_equal_to_apply = 100*( (loans.funded_amount == loans.loan_amount).sum())/loans.shape[0]
appr_less_than_apply = 100*(loans.funded_amount < loans.loan_amount).sum()/loans.shape[0]
appr_more_than_apply = 100*(loans.funded_amount > loans.loan_amount).sum()/loans.shape[0]

temp = pd.Series([appr_equal_to_apply, appr_less_than_apply, appr_more_than_apply], 
                 index=['appr_equal_to_apply', 'appr_less_than_apply', 'appr_more_than_apply'])

sns.barplot(x=temp.values, 
            y=temp.index)

plt.title('', size=20)
plt.ylabel('percentage', size=15)
plt.xticks(size=15, rotation=-15)
plt.yticks(size=15);
for index, value in enumerate(temp.values):
    plt.text(0, index, np.round(value, 4), fontdict={'size':18})
loans[(loans.loan_amount - loans.funded_amount) < 0]
sns.set_style("darkgrid")

country = 'United States'

plt.figure(figsize=(16, 5))
plt.subplot(1,2, 1)
loans[(loans.funded_amount != 0) & (loans.country == country)]['loan_amount'].plot.hist(bins=20, 
                                                                                        title='USA\nApproved', 
                                                                                        color=color[5])
plt.subplot(1,2,2)
loans[(loans.funded_amount == 0) & (loans.country == country)]['loan_amount'].plot.hist(bins=20, 
                                                                                        title='USA\nNot approved', 
                                                                                        color=color[3]);
country = 'Kenya'

plt.figure(figsize=(16, 5))
plt.subplot(1,2, 1)
loans[(loans.funded_amount != 0) & (loans.country == country)]['loan_amount'].plot.hist(bins=20, 
                                                                                        title='Kenya\nApproved', 
                                                                                        color=color[5])
plt.subplot(1,2,2)
loans[(loans.funded_amount == 0) & (loans.country == country)]['loan_amount'].plot.hist(bins=20, 
                                                                                        title='Kenya\nNot approved', 
                                                                                        color=color[3]);
proportion_plot(loans.country, 
                title='Proportion of Applications by countries', 
                figsize=(15, 5), top=30)


proportion_plot(loans[loans.funded_amount != 0].country, 
                title='Proportion of loans APPROVED, by countries', 
                figsize=(15, 5), top=30)

proportion_plot(loans[loans.funded_amount == 0].country, 
                title='Proportion of loan NOT APPROVED, by countries', 
                figsize=(15, 5), top=30)
sns.set_style("darkgrid")
plt.figure(figsize=(18, 5))
lender_count =loans[loans.lender_count <= 50].lender_count.value_counts()

sns.barplot(x=lender_count.index, y=lender_count.values)
plt.xlabel('Lenders Count');
loans.groupby(['country']).agg({'lender_count':'mean'}).sort_values(by='lender_count',ascending=False).plot.bar(figsize=(16, 5));
proportion_plot(loans.term_in_months, top=40);
plt.figure(figsize=(16, 8))
sns.regplot(x="term_in_months", y="funded_amount", data=loans.sample(frac=.1, random_state=SEED), 
            color=color[0],scatter_kws={'alpha':0.3});
import plotly.plotly as py #For World Map
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
temp=loans['country'].value_counts().reset_index()
temp.columns = ['country', 'value']
data = dict(type = 'choropleth', 
           locations = temp['country'],
           locationmode = 'country names',
           z = temp['value'], 
           text = temp['country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Loan Application Count', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
# from wordcloud import WordCloud, STOPWORDS
# stopwords = set(STOPWORDS)

# wordcloud = WordCloud(
#                           background_color='white',
#                           stopwords=stopwords,
#                           max_words=200,
#                           max_font_size=40, 
#                           random_state=42
#                          ).generate(str(loans['tags']))
proportion_plot(loans.currency, title='Currency Proportion', top=40)
proportion_plot(loans.activity, title='Proportion of loan count by activity', figsize=(15, 5), top=40)
proportion_plot(loans.sector, figsize=(15, 5), title='Proportion of loan count by sector')
# sector_repayment = ['sector', 'repayment_interval']
# cm = sns.light_palette("red", as_cmap=True)
# pd.crosstab(loans[sector_repayment[0]], loans[sector_repayment[1]]).style.background_gradient(cmap = cm)
repayment = loans.repayment_interval.value_counts()
sns.barplot(x = repayment.values, y = repayment.index);
for i, v in enumerate(repayment.values):
    plt.text(0.8, i, v, color='k',fontsize=19)
plt.xlabel('Repayment Interval')
plt.ylabel('Frequency')
plt.title('Repayment Interval Proportion');
countries_funded_amount = loans.groupby('country').mean()['funded_amount'].sort_values(ascending = False)
print("Top Countries with funded_amount(Dollar value of loan funded on Kiva.org)(Mean values)\n",countries_funded_amount.head(10))
# a = loans.borrower_genders[15213]
loans['female_borrower_count']= loans.borrower_genders.apply(lambda x: str(x).split(', ').count('female'))
loans['male_borrower_count']= loans.borrower_genders.apply(lambda x: str(x).split(', ').count('male'))
borrowers_percent = dict()
borrowers_percent['female'] = 100*loans['female_borrower_count'].sum()/(loans['female_borrower_count'].sum() + loans['male_borrower_count'].sum())
borrowers_percent['male'] = 100*loans['male_borrower_count'].sum()/(loans['female_borrower_count'].sum() + loans['male_borrower_count'].sum())
print('Female borrowers percentage: {:.2f} %'.format(borrowers_percent['female']))
print('Male borrowers percentage: {:.2f} %'.format(borrowers_percent['male']))
gender_prop = pd.Series(borrowers_percent)
g = sns.barplot(y=gender_prop.index, x=gender_prop.values)
plt.title('Proportion of female and male borrowers', size=20)
plt.ylabel('percentage', size=15)
plt.xticks(size=15)
plt.yticks(size=15);
for gender, value in enumerate(gender_prop.values):
    plt.text(x=0, y=gender, s=(str(np.round(value, 2))+ "%"),fontdict={'size':20})