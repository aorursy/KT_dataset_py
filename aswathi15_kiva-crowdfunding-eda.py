# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import geopandas as gpd
import squarify
%matplotlib inline
sns.set(style="darkgrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import dataset
loans = pd.read_csv("../input/kiva_loans.csv")
mpi = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme = pd.read_csv("../input/loan_theme_ids.csv")
loan_region = pd.read_csv("../input/loan_themes_by_region.csv")
loans.head()
loans.info()
loans['Posted_Date'],loans["Posted_Time"] = zip(*loans["posted_time"].map(lambda x: x.split(' ')))

loans= loans.drop('posted_time',axis=1)
loans = loans[['id','funded_amount','loan_amount','activity','sector','use','country_code',
                    'country','region','currency','partner_id','Posted_Date','Posted_Time','disbursed_time',
                    'funded_time','term_in_months','lender_count','borrower_genders','repayment_interval']]
loans.describe()
#convert id and partner id to object type
loans['id'] = loans['id'].astype('object')
loans['partner_id'] = loans['partner_id'].astype('object')
loans.rename(index=str,columns = {'id' :'loan_id','borrower_genders':'genders'},inplace=True)
f, ax = plt.subplots(ncols = 1)
ax = sns.kdeplot(loans['loan_amount'], shade = 'True', color = "red")
ax = sns.kdeplot(loans['funded_amount'],shade = 'True', color = "blue")
ax.set_xlim(0,100000)
ax.set_title("Distribution of Loan Amount and Funded Amount",fontsize = 20)
f.set_size_inches(15,10)
loans['activity'].unique()
activity = pd.DataFrame(loans.groupby(['activity'])['loan_amount'].sum()).reset_index()
activity.sort_values(by = 'loan_amount',ascending = False,inplace = True)
top_10_activity = activity.head(10)
top_10_activity
bottom_10_activity = activity.tail(10)
# Two subplots
f, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
sns.barplot(y = 'activity',x = 'loan_amount', data = top_10_activity,ax=ax1)
ax1.set(xlabel = "Loan Amount in Crores")
ax1.set_title("Top 10 activities",fontsize = 20)
sns.barplot(y = 'activity',x = 'loan_amount', data = bottom_10_activity,ax=ax2)
ax2.set(xlabel = "Loan Amount in Thousands")
ax2.set_title("Bottom 10 activities",fontsize = 20)
f.set_size_inches(15, 11)
filter = loans[loans['activity'] == 'Farming']
farming = filter[['loan_amount','activity']]
farming.sort_values(by='loan_amount',ascending=False,inplace=True)
f,ax = plt.subplots(ncols=1)
ax = sns.kdeplot(farming['loan_amount'],shade=True)
ax.set_title ('Loan amount taken for Farming',fontsize = 20)
f.set_size_inches(15,10)
loans['sector'].unique()
sector = loans.groupby(['sector'])['loan_amount'].sum().reset_index()
sector.sort_values(by = 'loan_amount',ascending = False, inplace = True)
sector
f,ax = plt.subplots(ncols = 1)
sns.barplot(y = 'sector',x = 'loan_amount', data= sector)
ax.set_title ('Loan Amount taken sector-wise',fontsize = 20)
f.set_size_inches(15,10)
value = ['Agriculture','Food','Retail']
sector_1 = loans[loans['sector'].isin(value)]
sector_1.head()
f,ax = plt.subplots(ncols=1)
ax = sns.boxplot(x = 'sector', y = 'loan_amount',data=loans)
f.set_size_inches(30, 30)
# There is an outlier, who has taken a loan of 100000, I am ignoring that outlier and reducing the limit
ax.set_ylim(0,50000)
ax.set_title('Sector-wise loan amount distribution',fontsize = 20)
f,ax = plt.subplots(ncols=1)
ax = plt.scatter(x='loan_amount',y='term_in_months',data=loans,marker = 'o',alpha = 0.1,color = 'red')
f.set_size_inches(15, 10)

f,ax = plt.subplots(ncols = 1)
ax = sns.countplot(x ='repayment_interval',data = loans)
ax.set_title('Count of Repayment Interval',fontsize = 20)
f.set_size_inches(15,10)
bullet = loans[loans['repayment_interval'] == 'bullet']
irregular = loans[loans['repayment_interval'] == 'irregular']
irregular_country = pd.DataFrame(irregular['country'].unique())
irregular_country.columns = ["Country"]
irregular_country.head(20)
country_loan = loans.groupby(['country'])['loan_amount'].sum().reset_index()
country_loan.sort_values(by = 'loan_amount',ascending = False, inplace = True)
country_loan.head(10)
f,ax = plt.subplots(ncols = 1)
sns.barplot(x = 'loan_amount',y = 'country', data = country_loan.head(10))
f.set_size_inches(15,10)
ax.set_title('Loan amount given Country-Wise',fontsize = 20)
loan_theme.head()
loan_type = loan_theme.groupby(['Loan Theme Type'])['id'].count().reset_index()
loan_type.sort_values(by = 'id',ascending = False,inplace = True)
loan_type.rename(index = str, columns = {'id' : 'Count'},inplace = True)
loan_type_top10 = loan_type.head(10)
loan_type_bottom10 = loan_type.tail(10)
f, (ax1, ax2) = plt.subplots(ncols=2, sharey=False)
ax1 = sns.barplot(y = 'Loan Theme Type', x = 'Count', data = loan_type_top10,ax = ax1)
ax1.set_title ("Top 10 Loan Theme Type",fontsize=20)
ax2 = sns.barplot(y = 'Loan Theme Type', x = 'Count', data = loan_type_bottom10,ax = ax2)
ax2 .set_title ("Bottom 10 Loan Theme Type",fontsize=20)
f.set_size_inches(20,15)
loans.head(2)
loans_use = loans['use'].astype(str)
type(loans_use)
loans_use.dropna(axis = 0, how ='any')
from wordcloud import WordCloud,STOPWORDS
import nltk
words = []
for i in range(0,len(loans_use)):
    words.append(nltk.word_tokenize(loans_use[i]))

words = [i for i in words for i in i]
words
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('buy')
stopwords.append('purchase')
stopwords.append('sell')
wordcloud = WordCloud(max_font_size = 30,width = 600, height = 300,stopwords = stopwords).generate(" ".join(words))
plt.figure(figsize =(15,8))
plt.imshow(wordcloud)
plt.title('Wordcloud for loan uses',fontsize = 20)
plt.axis('off')
plt.show()
mpi.head()
mpi.drop(['LocationName','geo'],axis = 1, inplace = True)
mpi['world_region'].unique()
mpi.info()
count_loan_region = pd.DataFrame(mpi['world_region'].value_counts()).reset_index()
count_loan_region.columns = ['world_region','count_of_loans']
count_loan_region
f,ax = plt.subplots(ncols = 1)
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')
ax = plt.pie(count_loan_region['count_of_loans'], labels = count_loan_region['world_region'],wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })
p=plt.gcf()
p.gca().add_artist(my_circle)
f.set_size_inches(10,10)
plt.show()

f,ax = plt.subplots(ncols = 1)
sns.kdeplot(mpi['MPI'], kernel ='gau',shade = True,bw ='scott',color ='green')
f.set_size_inches(15,10)
ax.set_title('Multidimensional Poverty Index Distribution',fontsize = 20)
g = sns.FacetGrid(data = mpi, col = 'world_region',hue = 'world_region',dropna = True)
g.map(sns.kdeplot, "MPI",shade = True)
loan_region.head()
loan_region['Field Partner Name'].unique()
loan_field_partner = loan_region.groupby(['Field Partner Name'])['amount'].sum().reset_index().sort_values(by = 'amount',ascending = False)
theme_type = loan_region.groupby(['Loan Theme Type'])['amount'].sum().reset_index().sort_values(by = 'amount',ascending = False)
loan_field_partner.head()
theme_type.head()
loan_field_partner_top20 = loan_field_partner.head(20)
theme_type_top20 = theme_type.head(20)
#create a color palette matching the values 
cmap = matplotlib.cm.viridis
mini=min(loan_field_partner_top20['amount'])
maxi=max(loan_field_partner_top20['amount'])
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in loan_field_partner_top20['amount']]


f,ax = plt.subplots(ncols=1)
ax = squarify.plot(sizes = loan_field_partner_top20['amount'],value = loan_field_partner_top20['amount'], norm_x = 100, norm_y = 100, label = loan_field_partner_top20['Field Partner Name'],color = colors )
f.set_size_inches(20,10)
ax.set_title('Top 20 Loan Field Partners providing loans',fontsize = 20)
#create a color palette matching the values 
cmap = matplotlib.cm.viridis
mini=min(theme_type_top20['amount'])
maxi=max(theme_type_top20['amount'])
norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in theme_type_top20['amount']]


f,ax = plt.subplots(ncols=1)
ax = squarify.plot(sizes = theme_type_top20['amount'],value = theme_type_top20['amount'], norm_x = 100, norm_y = 100, label = theme_type_top20['Loan Theme Type'],color = colors )
f.set_size_inches(20,10)
ax.set_title ('Top 20 Loan theme types',fontsize=20)


