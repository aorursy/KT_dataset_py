# Import packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist, word_tokenize

# plt.style.use('seaborn')
# print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
# Load data
loans = pd.read_csv('../input/kiva_loans.csv', parse_dates = ['posted_time', 'disbursed_time', 'funded_time', 'date'])
themes_by_region = pd.read_csv('../input/loan_themes_by_region.csv')
region_loc = pd.read_csv('../input/kiva_mpi_region_locations.csv')
themes = pd.read_csv('../input/loan_theme_ids.csv')
loans.shape, themes_by_region.shape, region_loc.shape, themes.shape
# Sometimes .sample() is better than .head() to if I want to inspect random lines
loans.drop(['use'], axis=1).sample(5) # use column is too long
themes.sample(5)
loans = pd.merge(loans, themes.drop(['Partner ID'], axis=1), how='left', on='id')
loans.drop(['use'],axis=1).head()
plt.figure(figsize=(10,6))
sec = loans.groupby('sector').sum()['loan_amount'].reset_index()
sec = sec.sort_values(by='loan_amount', ascending = False)
sec['loan_amount'] = sec['loan_amount']/1000000
g = sns.barplot(x='sector', y='loan_amount', ci=None, palette = 'spring', data=sec)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.title('Total Loan Amount by Sector', fontsize=18)
plt.xlabel('Sector', fontsize=12)
plt.ylabel('Loan amount (M USD)', fontsize=12)
plt.xticks(fontsize=11)
plt.show()
loans['Loan Theme Type'] = loans['Loan Theme Type'].fillna('Unknown')
plt.figure(figsize=(12,6))
theme = loans.groupby('Loan Theme Type').sum()['loan_amount'].reset_index()
theme['loan_amount'] = theme['loan_amount']/1000000
theme = theme.sort_values(by='loan_amount', ascending = False)
g = sns.barplot(y='Loan Theme Type', x='loan_amount', ci=None, palette = 'spring', data=theme.head(20))
plt.title('Top 20 Loan Themes', fontsize=16)
plt.ylabel('Theme', fontsize=12)
plt.xlabel('Loan amount (USD M)', fontsize=12)
plt.yticks(fontsize=11)
plt.show()
plt.figure(figsize=(10,6))
top10c = loans.groupby('country').sum()['loan_amount'].reset_index()
top10c = top10c.sort_values(by='loan_amount', ascending = False)
top10c['loan_amount'] = top10c['loan_amount']/1000000
g = sns.barplot(x='country', y='loan_amount', ci=None, palette = "cool", data=top10c.head(10))
g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.title('Top 10 Countries in Loan Amount', fontsize=18)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Loan amount (USD M)', fontsize=12)
plt.xticks(fontsize=12)
plt.show()
lcst = loans.loc[:,['loan_amount', 'country','sector', 'posted_time']]
lcst.set_index('posted_time', inplace=True)

# Overall loan amount evolution
plt.figure(figsize=(12,6))
monthly = lcst.resample('M').sum()/1000000
month_label=[]
for dt in monthly.index:
    month_label.append(dt.strftime('%Y-%m'))
sns.barplot(x=month_label, y=monthly['loan_amount'], color='orchid', alpha=0.7)
plt.title('Loan Amount by Month', fontsize=18)
plt.xticks(rotation=90, fontsize=12)
plt.ylabel('Loan Amount (USD M)')
plt.show()
plt.figure(figsize=(12,6))

def plot_monthly_by_cty(cty_name):
    
    lctm_cty = lcst[lcst.country==cty_name].resample('M').sum()/1000000
    month_label=[]
    for dt in lctm_cty.index:
        month_label.append(dt.strftime('%Y-%m'))
    sns.barplot(x=month_label, y=lctm_cty['loan_amount'], color='Blue', alpha=0.7)
    plt.title('Loan Amount by Month - ' + cty_name, fontsize=18)
    plt.xticks(rotation=90, fontsize=12)
    plt.ylabel('Loan Amount (USD M)', fontsize=12)
    plt.show()

plot_monthly_by_cty('Kenya')
plt.figure(figsize=(12,6))

def plot_monthly_by_sector(sec_name):
    
    lsm_sec = lcst[lcst.sector==sec_name].resample('M').sum()/1000000
    month_label=[]
    for dt in lsm_sec.index:
        month_label.append(dt.strftime('%Y-%m'))
    sns.barplot(x=month_label, y=lsm_sec['loan_amount'], color='Magenta', alpha=0.7)
    plt.title('Loan Amount by Month - ' + sec_name, fontsize=18)
    plt.ylabel('Loan Amount (USD M)', fontsize=12)
    plt.xticks(rotation=90, fontsize=12)
    plt.show()

plot_monthly_by_sector('Agriculture')
by_month = pd.DataFrame()

for cty in loans.country.unique():
    lctm_cty = lcst[lcst.country==cty].resample('M').sum()/1000000
    lctm_cty.columns = [cty]
    by_month = pd.concat([by_month, lctm_cty],axis=1)

by_month = by_month.fillna(0)

top10list = top10c.head(10)['country'].tolist()

month_label=[]
for dt in by_month.index:
    month_label.append(dt.strftime('%Y-%m'))
by_month.loc[:,top10list].plot(kind='bar', stacked=True, x = np.array(month_label), figsize=(15,7), colormap = 'Set3')
plt.title('Loan Amount Evolution of Top 10 Countries', fontsize=18)
plt.ylabel('Loan Amount (USD M)', fontsize=12)
plt.xticks(fontsize=12)
plt.legend(bbox_to_anchor=(1.01,0.95), fontsize=10)

plt.show()
by_month_sec = pd.DataFrame()

for sector in loans.sector.unique():
    lsm_sec = lcst[lcst.sector==sector].resample('M').sum()/1000000
    lsm_sec.columns = [sector]
    by_month_sec = pd.concat([by_month_sec, lsm_sec],axis=1)

by_month_sec = by_month_sec.fillna(0)

month_label=[]
for dt in by_month_sec.index:
    month_label.append(dt.strftime('%Y-%m'))
by_month_sec.plot(kind='bar', stacked=True, x = np.array(month_label), figsize=(15,7))
plt.title('Loan Amount Evolution of All Sectors', fontsize=18)
plt.ylabel('Loan Amount (USD M)', fontsize=12)
plt.xticks(fontsize=12)
plt.legend(bbox_to_anchor=(1.01, 0.95), fontsize=10)
plt.show()
loans.borrower_genders.fillna('Unknown', inplace=True)
cv = CountVectorizer()
gender_count = cv.fit_transform(loans.borrower_genders)
df_gender = pd.DataFrame(gender_count.toarray())
df_gender.columns = ['borrower_' + str(i) for i in cv.vocabulary_.keys()]
df_gender['borrower_total'] = df_gender['borrower_female']+df_gender['borrower_male']
df_gender.describe()
time_gender = pd.concat([df_gender[['borrower_female','borrower_male']], loans['posted_time']], axis=1)
time_gender.set_index('posted_time', inplace=True)
gender_trend = time_gender.resample('M').sum()

month_label=[]
for dt in gender_trend.index:
    month_label.append(dt.strftime('%Y-%m'))
gender_trend.plot(kind='bar', stacked=True, x = np.array(month_label), figsize=(15,7))
plt.title('Loan Count Evolution by gender', fontsize=18)
plt.ylabel('Loan Count', fontsize=12)
plt.xticks(fontsize=12)
plt.legend(bbox_to_anchor=(1.01, 0.95), fontsize=10)
plt.show()
themes_by_region.head()
partner_info = themes_by_region.loc[:,['Partner ID','Field Partner Name']].drop_duplicates()

# Loan amount by partner and country
top10p = loans.groupby(['partner_id','country']).sum()['loan_amount'].reset_index()
top10p['loan_amount'] = top10p['loan_amount']/1000000
top10p = top10p.sort_values(by='loan_amount', ascending = False)
# Add names to partner statistics
top10p_name = pd.merge(top10p, partner_info, left_on='partner_id', right_on='Partner ID', how='left')
top10p_name['Partner_country'] = top10p_name['Field Partner Name'] + '@' + top10p_name['country']
top10p_name.head()
plt.figure(figsize=(12,6))
g = sns.barplot(x='loan_amount', y='Partner_country', ci=None, color = 'navy', data=top10p_name.head(10), alpha=0.9)
plt.title('Top 10 Country-Specific Partners', fontsize=16)
plt.ylabel('Partner_Country', fontsize=12)
plt.xlabel('Loan amount (USD M)', fontsize=12)
plt.show()
top10p2 = top10p_name.groupby(['partner_id','Field Partner Name']).sum()['loan_amount'].reset_index()
top10p2 = top10p2.sort_values(by='loan_amount', ascending = False).head(10)
plt.figure(figsize=(12,6))
g = sns.barplot(x='loan_amount', y='Field Partner Name', ci=None, color='mediumblue', data=top10p2, alpha=0.9)
plt.title('Top 10 Partners', fontsize=16)
plt.ylabel('Partner Name', fontsize=12)
plt.xlabel('Loan amount (USD M)', fontsize=12)
plt.show()
stopwords = set(STOPWORDS)

names = loans["use"][~pd.isnull(loans["use"])]
word_freq = FreqDist(w for w in word_tokenize(' '.join(names).lower()) if (w not in stopwords) & (w.isalpha()))

wordcloud = WordCloud(background_color = 'white', width=600, height=400, max_words=150).generate_from_frequencies(word_freq)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Use", fontsize=25)
plt.axis("off")
plt.show() 
names = loans["activity"][~pd.isnull(loans["activity"])]
word_freq = FreqDist(word_tokenize(' '.join(names).lower()))

stopwords = set(STOPWORDS).add('&')
wordcloud = WordCloud(background_color = 'white', width=600, height=400, max_words=75, stopwords = stopwords).generate_from_frequencies(word_freq)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Activity", fontsize=25)
plt.axis("off")
plt.show() 
asian_codes = ['PK','IN','PH','KH','VN','IQ','PS','MN','TJ','JO','TL','ID','LB','NP','TH','LA','MM','CN','AF','BT']
asian_sub = loans.loc[loans.country_code.isin(asian_codes)]
asian_sub.shape
cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(asian_sub.country, asian_sub.sector).style.background_gradient(cmap = cm)
t_temp = pd.crosstab(asian_sub.country, asian_sub.sector, values=asian_sub.loan_amount, aggfunc=sum, normalize='index').round(4)*100
t_temp.style.background_gradient(cmap = cm)
# Code to find out specific partner
partner_info[partner_info['Field Partner Name'].str.contains('Arariwa')]
mfi_bob = [119, 156, 161, 145, 106, 62, 77]
bob = loans.loc[loans.partner_id.isin(mfi_bob), ['partner_id','sector','loan_amount','posted_time']]
bob.set_index('posted_time', inplace=True)
bob.shape
plt.figure(figsize=(14,7))

def plot_monthly_mfi(partner_id):
    
    partner_name = partner_info.loc[partner_info['Partner ID']==partner_id, 'Field Partner Name'].iloc[0]
    mfi_month = bob[bob.partner_id==partner_id].resample('M').sum()
    month_label=[]
    for dt in mfi_month.index:
        month_label.append(dt.strftime('%Y-%m'))
    plt.plot(month_label, mfi_month.loan_amount, label=partner_name)
    plt.xticks(rotation=90)
    
for x in mfi_bob:
    plot_monthly_mfi(x)

plt.title('Loan Trends in Partners Mentioned by Bob', fontsize=16)
plt.ylabel('Loan Amount (USD)', fontsize=12)
plt.legend(bbox_to_anchor=(1.01, 0.95), fontsize=11)
plt.show()
x = pd.merge(bob, partner_info, left_on='partner_id', right_on='Partner ID', how='left')
t_temp = pd.crosstab(x['Field Partner Name'], x.sector, values=x.loan_amount, aggfunc=sum, normalize='index').round(4)*100
t_temp.style.background_gradient(cmap = cm)
loans['fully_funded'] = ~loans.funded_time.isnull()*1
loans.fully_funded.value_counts()
time_fund = loans.funded_time - loans.posted_time
loans['days_to_fund'] = time_fund.apply(lambda x: x.days)
loans['days_to_fund'].describe()
plt.figure(figsize=(16,7))
plt.subplot(121)
plt.hist(loans.days_to_fund[loans.days_to_fund>0], bins=30)
plt.title('Distribution of Days to Funded', fontsize=14)
plt.subplot(122)
plt.hist(loans.days_to_fund[(loans.days_to_fund>0) & (loans.days_to_fund<90)], bins=30)
plt.title('Distribution of Days to Funded (Less Than 90 Days)', fontsize=14)
plt.show()
g = sns.jointplot(loans.days_to_fund[loans.days_to_fund>0], np.log10(loans[loans.days_to_fund>0].loan_amount), size=8, s=6)
plt.xlabel('Days to Fund')
plt.ylabel('Log Loan Amount')
plt.show()
ff_c = loans.groupby('country').mean()['fully_funded'].reset_index()
count_c = loans['country'].value_counts().reset_index()
ff_c = pd.merge(ff_c, count_c, left_on='country', right_on='index').drop(['index'],axis=1)
ff_c.columns = ['country','fully_funded_ratio','loan_count']
ff_c2 = ff_c.loc[ff_c.loan_count>1000]

plt.figure(figsize=(10,14))
ff_c2 = ff_c2.sort_values('fully_funded_ratio')
g = sns.barplot(x='fully_funded_ratio', y='country', ci=None, color='tomato', data=ff_c2, alpha=0.6)
plt.title('Ratio of Fully Funded Loans (Countries >1000 Loans Only)', fontsize=14)
plt.xlabel('Ratio of Fully Funded Loans', fontsize=12)
plt.ylabel('')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.axvline(loans.fully_funded.mean(),color='olive')
plt.show()
ff_c2.head(4)
ff_s = loans.groupby('sector').mean()['fully_funded'].reset_index()
ff_s.sort_values(by='fully_funded')
ff_s.columns = ['sector', 'fully_funded_ratio']
ff_s = ff_s.sort_values('fully_funded_ratio')

plt.figure(figsize=(7,8))
g = sns.barplot(x='fully_funded_ratio', y='sector', ci=None, color='tomato', data=ff_s, alpha=0.6)
plt.title('Ratio of Fully Funded Loans', fontsize=14)
plt.xlabel('Ratio of Fully Funded Loans', fontsize=12)
plt.ylabel('')
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.axvline(loans.fully_funded.mean(),color='olive')
plt.show()