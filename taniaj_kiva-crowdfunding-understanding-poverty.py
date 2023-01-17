import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from scipy.stats.mstats import gmean

%matplotlib inline

sns.set(rc={"figure.figsize": (16,8), "axes.titlesize" : 18, "axes.labelsize" : 12, 
            "xtick.labelsize" : 14, "ytick.labelsize" : 14 }, 
        palette=sns.color_palette("OrRd_d", 20))

import warnings
warnings.filterwarnings('ignore')

!cp ../input/images/regional-intensity-of-deprivation.png .
# Original Kiva datasets
kiva_loans_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
kiva_mpi_locations_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")
loan_theme_ids_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
loan_themes_by_region_df = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")

# Additional MPI datasets
mpi_national_df = pd.read_csv("../input/mpi/MPI_national.csv")
# The subnational MPI data has been enhanced with lat/long data
mpi_subnational_df = pd.read_csv("../input/kiva-mpi-subnational-with-coordinates/mpi_subnational_coords.csv")

# World Bank population data
world_pop_df = pd.read_csv("../input/world-population/WorldPopulation.csv")
# Plot loans per country
sns.countplot(y="country", data=kiva_loans_df, 
              order=kiva_loans_df.country.value_counts().iloc[:20].index).set_title("Distribution of Kiva Loans by Country")
plt.ylabel('')
# Plot loans per region
sns.countplot(y="region", data=kiva_loans_df, 
              order=kiva_loans_df.region.value_counts().iloc[:20].index).set_title("Distribution of Kiva Loans by Region")
plt.ylabel('')
#kiva_loans_df.loc[kiva_loans_df['region'] == 'Kaduna'].head()
countries_number_loans = kiva_loans_df.groupby('country').count()['loan_amount'].sort_values(ascending = False)
data = [dict(
        type='choropleth',
        locations= countries_number_loans.index,
        locationmode='country names',
        z=countries_number_loans.values,
        text=countries_number_loans.index,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='# Loans'),
)]
layout = dict(title = 'Number of Loans Requested by Country', 
        geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=50, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='loans-total-map')
countries_loan_amount = kiva_loans_df.groupby('country').sum()['loan_amount'].sort_values(ascending = False)
data = [dict(
        type='choropleth',
        locations= countries_loan_amount.index,
        locationmode='country names',
        z=countries_loan_amount.values,
        text=countries_loan_amount.index,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Loan Amount'),
)]
layout = dict(title = 'Total Loan Amount Requested by Country', 
        geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=50, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='loans-total-map')
# Get loan_per_country data
kiva_loan_country_df = kiva_loans_df[['id', 'country']].groupby(['country'])['id'].agg({'loan_amount': ['sum','count']}).reset_index()
kiva_loan_country_df.columns = kiva_loan_country_df.columns.droplevel()
kiva_loan_country_df.columns = ['country', 'loan_amount', 'loan_count']

# Join world population data to kiva loan_per_country data
kiva_loan_country_df = kiva_loan_country_df.merge(world_pop_df[['Country', '2016']], left_on=['country'], right_on=['Country'])
kiva_loan_country_df.drop('Country', axis=1, inplace=True)

# Calculate values per million population
kiva_loan_country_df['loans_per_mil'] = kiva_loan_country_df['loan_count'] / (kiva_loan_country_df['2016'] / 1000000)
kiva_loan_country_df['loan_amount_per_mil'] = kiva_loan_country_df['loan_amount'] / (kiva_loan_country_df['2016'] / 1000000)
# Plot loans per million per country
with sns.color_palette("OrRd_d", 10), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,4))
    kiva_loan_country_df.sort_values('loans_per_mil', ascending=False, inplace=True)
    sns.barplot(kiva_loan_country_df.head(10).loans_per_mil, kiva_loan_country_df.head(10).country).set_title("Number of Loans (population adjusted) per Country")
    plt.ylabel('')
kiva_loan_country_df.head(10)
# Plot loan amount per million per country
with sns.color_palette("OrRd_d", 10), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,4))
    kiva_loan_country_df.sort_values('loan_amount_per_mil', ascending=False, inplace=True)
    sns.barplot(kiva_loan_country_df.head(10).loan_amount_per_mil, kiva_loan_country_df.head(10).country).set_title("Loan Amount (population adjusted) per Country")
    plt.ylabel('')
data = [dict(
        type='choropleth',
        locations= kiva_loan_country_df.country,
        locationmode='country names',
        z=kiva_loan_country_df.loan_amount_per_mil,
        text=kiva_loan_country_df.index,
        colorscale = [[0,'rgb(128, 0, 0)'],[0.95,'rgb(180, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Loan Amount per Mil.'),
)]
layout = dict(title = 'Total Loan Amount (adjusted) Requested by Country', 
        geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=50, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='loans-total-map')
# Plot loans per sector
with sns.color_palette("OrRd_d", 15), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,5))
    sns.countplot(y="sector", data=kiva_loans_df, 
              order=kiva_loans_df.sector.value_counts().iloc[:20].index).set_title("Distribution of Loans by Sector")
    plt.ylabel('')
# Truncate outliers
percentile_99 = np.percentile(kiva_loans_df.loan_amount.values, 99)
kiva_loans_df['loan_amount_trunc'] = kiva_loans_df['loan_amount'].copy()
kiva_loans_df.loc[kiva_loans_df['loan_amount_trunc'] > percentile_99, 'loan_amount_trunc'] = percentile_99
# Plot loan amount histogram
sns.distplot(kiva_loans_df.loan_amount_trunc.values, kde=False)
plt.title("Loan Amount Distribution")
plt.xlabel('Loan Amount')
plt.ylabel('Number of Loans')
# Plot repayent term histogram
sns.distplot(kiva_loans_df.term_in_months.values, kde=False)
plt.title("Loan Term Distribution")
plt.xlabel('Loan Term')
plt.ylabel('Number of Loans')
# Plot repayment interval of loans
with sns.color_palette("YlOrBr_d", 4):
    plt.figure(figsize=(6,6))
    plt.title("Repayment Interval")
    kiva_loans_df.repayment_interval.value_counts().T.plot.pie(labeldistance=1.1)
    plt.ylabel('')
def parse_genders(borrower_genders):
    gender_list = borrower_genders.split(",")
    gender_list = list(set(gender_list))
    gender_list = [borrower_genders.strip() for borrower_genders in gender_list]
    if len(gender_list)==2:
        if 'female' in gender_list and 'male' in gender_list:
            return "both"
        elif 'female' in gender_list:
            return "multiple female"
        elif 'male' in gender_list:
            return "multiple male"
    elif gender_list[0]=="female":
        return "single female"
    elif gender_list[0]=="male":
        return "single male"
    else:
        return "unknown"
    
# Plot loans by borrower gender
with sns.color_palette("YlOrBr_d", 8):
    plt.figure(figsize=(6,6))
    plt.title("Borrower Gender")
    kiva_loans_df.borrower_genders[kiva_loans_df.borrower_genders.isnull()]= 'unknown'
    kiva_loans_df['gender'] = kiva_loans_df.borrower_genders.apply(parse_genders)
    kiva_loans_df.gender.value_counts().plot.pie(labeldistance=1.1, explode = (0, 0.025, 0.05, 0.1, 0.3, 0.7))
    plt.ylabel('')
# Plot Kiva MPI Locations
data = [ dict(
        type = 'scattergeo',
        lon = kiva_mpi_locations_df['lon'],
        lat = kiva_mpi_locations_df['lat'],
        text = kiva_mpi_locations_df['LocationName'],
        drawmapboundary = dict(fill_color = '#A6CAE0', linewidth = 0.1),
        mode = 'markers',
        marker = dict(
            size = 6,
            opacity = 0.9,
            symbol = 'circle',
            line = dict(width = 1, color = 'rgba(80, 80, 80)'),
            colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
            reversescale=True,
            cmin = 0,
            color = kiva_mpi_locations_df['MPI'],
            cmax = kiva_mpi_locations_df['MPI'].max(),
            colorbar=dict(title="MPI")
        ))]
layout = dict(
            title = 'Kiva MPI Locations',
            geo = dict(
            showframe = False, 
            showcoastlines = True,
            showcountries=True,
            showland = True,
            landcolor = 'rgb(245, 241, 213)',
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
print(kiva_mpi_locations_df.shape)
kiva_mpi_locations_df.sample(5)
print("Original MPI dataset: ", kiva_mpi_locations_df.shape)
region_mpi_df = kiva_mpi_locations_df[['world_region', 'LocationName', 'country','region', 'MPI', 'lat', 'lon']]
region_mpi_df = region_mpi_df.dropna()
print("Cleaned MPI dataset: ", region_mpi_df.shape)
# Plot MPI by World Region
with sns.color_palette("OrRd", 6), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,6))
    plt.subplot(211).set_title("MPI count by World Region")
    world_region_mpi_count_df = region_mpi_df.groupby(['world_region'])['MPI'].count().reset_index(name='count_mpi')
    sns.barplot(world_region_mpi_count_df.count_mpi, world_region_mpi_count_df.world_region)
    plt.ylabel('')

    plt.subplot(212).set_title("MPI average by World Region")
    world_region_mpi_mean_df = region_mpi_df.groupby(['world_region'])['MPI'].mean().reset_index(name='mean_mpi')
    sns.barplot(world_region_mpi_mean_df.mean_mpi, world_region_mpi_mean_df.world_region)
    plt.ylabel('')

    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
print("Original Kiva Loans dataset: ", kiva_loans_df.shape)

# Merging Kiva loans to MPI using loan_themes
kiva_loans_mpi_df = pd.merge(kiva_loans_df, loan_theme_ids_df, how='left', on='id')
kiva_loans_mpi_df = kiva_loans_mpi_df.merge(loan_themes_by_region_df, how='left', on=['Partner ID', 'Loan Theme ID', 'country', 'region'])
kiva_loans_mpi_df = kiva_loans_mpi_df.merge(kiva_mpi_locations_df, how='left', left_on=['country', 'mpi_region'], right_on=['country', 'LocationName'])

# Drop entries with null MPI
kiva_loans_mpi_df = kiva_loans_mpi_df.dropna(subset=['MPI'])

# Remove some information that is no longer needed
kiva_loans_mpi_df.drop('mpi_region', axis=1, inplace=True)
kiva_loans_mpi_df.drop('LocationName_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('sector_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('Loan Theme Type_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('ISO_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('region_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('geo_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('lat_y', axis=1, inplace=True)
kiva_loans_mpi_df.drop('lon_y', axis=1, inplace=True)

# Rename some columns
kiva_loans_mpi_df = kiva_loans_mpi_df.rename(index=str, columns={'region_x': 'region', 'sector_x' : 'sector', 'Loan Theme Type_x':'loan_theme_type',
                                                     'ISO_x':'ISO', 'LocationName_x':'location_name', 'geo_x':'geo', 'lat_x':'lat', 'lon_x':'lon'
                                      })

print("Merged Loans MPI dataset: ", kiva_loans_mpi_df.shape)
# Scatter plot of number of loans per MPI
total_loans_mpi_df = kiva_loans_mpi_df.groupby(['country','region','MPI'])['loan_amount'].count().reset_index(name='total_number_loans')
total_loans_mpi_df.sample(5)
sns.regplot(x = total_loans_mpi_df.MPI, y = total_loans_mpi_df.total_number_loans, fit_reg=False)
plt.title("Total number of Loan Requests vs. Regional MPI")
plt.show()
# Examine outliers
percentile_95_df = total_loans_mpi_df[total_loans_mpi_df.total_number_loans > total_loans_mpi_df.total_number_loans.quantile(.95)]
percentile_95_df.sort_values('total_number_loans', ascending=False).head(10)
# Scatter plot of total loan amount per MPI
total_loan_amt_df = kiva_loans_mpi_df.groupby(['country','region','MPI'])['loan_amount'].sum().reset_index(name='total_loan_amt')
sns.regplot(x = total_loan_amt_df.MPI, y = total_loan_amt_df['total_loan_amt'], fit_reg = False)
plt.title("Total of Loan Amount Requested vs. Regional MPI")
plt.show()
# Examine outliers
percentile_95_df = total_loan_amt_df[total_loan_amt_df.total_loan_amt > total_loan_amt_df.total_loan_amt.quantile(.95)]
percentile_95_df.sort_values('total_loan_amt', ascending=False).head(10)
# Scatter plot of total loan amount per MPI
total_funded_amt_df = kiva_loans_mpi_df.groupby(['country','region','MPI'])['funded_amount'].sum().reset_index(name='total_funded_amt')
total_loan_amt_df= pd.merge(total_loan_amt_df, total_funded_amt_df, how='left')
sns.regplot(x = total_loan_amt_df.MPI, y = total_loan_amt_df['total_funded_amt']/total_loan_amt_df['total_loan_amt'], fit_reg = False)
plt.title("Percentage funded vs. Regional MPI")
plt.show()
# Plot MPI per sector
kiva_sector_mpi = kiva_loans_mpi_df.groupby(['sector'])['MPI'].mean().reset_index(name='mean_mpi')
kiva_sector_mpi.sort_values(['mean_mpi'], ascending=False, inplace=True)

with sns.color_palette("OrRd_d", 20), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,5))
    sns.barplot(x='mean_mpi', y='sector', data=kiva_sector_mpi)
    plt.ylabel('')
    plt.title("Average MPI per Sector")
def color_func(word, font_size, position, orientation,random_state=None, **kwargs):
    return("hsl(0,100%%, %d%%)" % np.random.randint(5,55))

# Plot word clouds
plt.subplot(221).set_title("Sector: Health")
wc = WordCloud(background_color='white', stopwords=STOPWORDS,max_words=20).generate(" ".join(kiva_loans_df.loc[kiva_loans_df['sector'] == 'Health'].use.astype(str)))
plt.imshow(wc.recolor(color_func=color_func))
plt.axis('off')

plt.subplot(222).set_title("Sector: Food")
wc = WordCloud(background_color='white', stopwords=STOPWORDS,max_words=20).generate(" ".join(kiva_loans_df.loc[kiva_loans_df['sector'] == 'Food'].use.astype(str)))
plt.imshow(wc.recolor(color_func=color_func))
plt.axis('off')

plt.subplot(223).set_title("Sector: Agriculture")
wc = WordCloud(background_color='white', stopwords=STOPWORDS,max_words=20).generate(" ".join(kiva_loans_df.loc[kiva_loans_df['sector'] == 'Agriculture'].use.astype(str)))
plt.imshow(wc.recolor(color_func=color_func))
plt.axis('off')

plt.subplot(224).set_title("Sector: Personal Use")
wc = WordCloud(background_color='white', stopwords=STOPWORDS,max_words=20).generate(" ".join(kiva_loans_df.loc[kiva_loans_df['sector'] == 'Personal Use'].use.astype(str)))
plt.imshow(wc.recolor(color_func=color_func))
plt.axis('off')

plt.suptitle("Loan Use")
plt.tight_layout(pad=0.4, w_pad=0.0, h_pad=2.0, rect=[0, 0, 0.95, 0.95])
plt.figure(figsize=(16,20))
# Plot MPI per Activity
kiva_activity_mpi = kiva_loans_mpi_df.groupby(['activity'])['MPI'].mean().reset_index(name='mean_mpi')
kiva_activity_mpi = kiva_activity_mpi.sort_values(by=['mean_mpi'], ascending=False).head(30)

with sns.color_palette("OrRd_d", 30), sns.plotting_context("notebook", font_scale=1.2):
    sns.barplot(x='mean_mpi', y='activity', data=kiva_activity_mpi)
    plt.ylabel('')
    plt.title("Average MPI per Activity")
# Plot MPI per gender
kiva_gender_mpi = kiva_loans_mpi_df.groupby(['gender'])['MPI'].mean().reset_index(name='mean_mpi')
kiva_gender_mpi.sort_values(['mean_mpi'], ascending=False, inplace=True)

with sns.color_palette("OrRd_d", 5), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,2))
    sns.barplot(x='mean_mpi', y='gender', data=kiva_gender_mpi)
    plt.ylabel('')
    plt.title("Borrower Gender vs. average MPI")
# Encode some categorical features
category_mapping = {'world_region':{'Sub-Saharan Africa':1, 'South Asia':2, 'East Asia and the Pacific':3, 
                                      'Arab States':4,'Latin America and Caribbean':5,'Europe and Central Asia':6},
                    'repayment_interval':{'irregular':1, 'bullet':2, 'monthly':3, 'weekly':4, },
                    'sector':{'Personal Use':1, 'Agriculture':2, 'Housing':3, 'Education':4, 'Retail':5, 
                                'Clothing':6, 'Food':7, 'Wholesale':8, 'Services':9, 'Health':10, 
                                'Construction':11, 'Manufacturing':12, 'Transportation':13, 'Arts':14, 'Entertainment':15}}
kiva_loans_corr_df = kiva_loans_mpi_df.replace(category_mapping)

# Get dummies for gender
gender_encoded_df = pd.get_dummies(kiva_loans_corr_df['gender'], prefix='gender')
kiva_loans_corr_df = pd.concat([kiva_loans_corr_df, gender_encoded_df], axis=1, join_axes=[kiva_loans_corr_df.index])


# Plot correlation between MPI and loan factors
kiva_loans_corr = kiva_loans_corr_df[['loan_amount', 'term_in_months', 'repayment_interval', 'world_region',
                                      'sector', 'lat', 'lon', 'rural_pct', 'MPI',
                                     'gender_both', 'gender_multiple female', 'gender_multiple male', 
                                      'gender_single female', 'gender_single male']].corr()

cmap =  sns.diverging_palette(220, 20, sep=20, as_cmap=True)
sns.heatmap(kiva_loans_corr, 
            xticklabels=kiva_loans_corr.columns.values,
            yticklabels=kiva_loans_corr.columns.values, 
            cmap=cmap, vmin=-0.7, vmax=0.7, annot=True, square=True)
plt.title('Kiva Loan Feature Correlation')
kiva_loans_corr_df.sample()
kiva_loans_mpi_df['use_filtered'] = kiva_loans_mpi_df.use.fillna('')
kiva_loans_mpi_df['use_filtered'] = kiva_loans_mpi_df['use_filtered'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in (STOPWORDS)]))
kiva_loans_mpi_df['use_filtered'].sample(5)
sanitation_words = ['water', 'filter', 'drinking', 'latrine', 'waste', 'wastes', 'toilet', 'toilets']
food_words = ['farm', 'food', 'corn', 'maize', 'rice', 'bread', 'oil', 'grain', 'meat', 'yield', 'harvest', 
              'potatoes', 'cooking', 'milk', 'sugar', 'beans', 'fruit', 'fruits' 'vegetables', 'fertilizer', 
              'seed', 'grow', 'growing', 'cultivation', 'crops', 'plant']
shelter_words = ['house', 'home', 'household', 'roof', 'repair', 'maintenance', 'broken', 'yard', 'bathroom', 'fix']
clothing_words = ['clothing', 'shoes', 'sewing', 'skirts', 'blouses']
education_words = ['university', 'tuition', 'education', 'study', 'studies', 'teach', 'teaching', 'course', 'degree']
family_words = ['family', 'child', 'children', 'daughter', 'son', 'father', 'mother',
                'provide', 'eliminating', 'pressure', 'medical']
building_words = ['supplies', 'materials', 'build', 'solar', 'cement']
improvement_words = ['buy', 'purchase', 'invest', 'improved', 'sell', 'business', 'fees', 'income', 'pay', 'store', 'yields', 'stock', 
                     'products', 'prices', 'increase', 'inputs', 'shop', 'hire', 'snacks', 'restock', 'trade']

def assign_keyword(words):
    result = assign_word_from_list(sanitation_words, "sanitation", words)
    if result != "other":
        return result
    result = assign_word_from_list(food_words, "food", words)
    if result != "other":
        return result
    result = assign_word_from_list(shelter_words, "shelter", words)
    if result != "other":
        return result
    result = assign_word_from_list(clothing_words, "clothing", words)
    if result != "other":
        return result
    result = assign_word_from_list(education_words, "education", words)
    if result != "other":
        return result
    result = assign_word_from_list(family_words, "family", words)
    if result != "other":
        return result
    result = assign_word_from_list(building_words, "building", words)
    if result != "other":
        return result
    result = assign_word_from_list(improvement_words, "improvement", words)

    return result
                 
def assign_word_from_list(category_words, keyword, words):
    result = "other"
    word_list = words.lower().split(" ")
#    print("words: ", word_list)
    for category_word in category_words:
        for word in word_list:
            if category_word == word:
                result = keyword
#                print("keyword: ", word)
                return result
    return result
            
kiva_loans_mpi_df['keyword'] =  kiva_loans_mpi_df.use_filtered.apply(assign_keyword)
kiva_loans_mpi_df.loc[kiva_loans_mpi_df['keyword'] == 'other'].sample(3)
# Plot keyword counts
loan_keyword_mpi = kiva_loans_mpi_df.groupby(['keyword']).keyword.count().reset_index(name='keyword_count')
loan_keyword_mpi.sort_values(['keyword_count'], ascending=False, inplace=True)

with sns.color_palette("OrRd_d", 10), sns.plotting_context("notebook", font_scale=1.2):
    plt.figure(figsize=(16,4))
    sns.barplot(x='keyword_count', y='keyword', data=loan_keyword_mpi)
    plt.ylabel('')
    plt.title("Keyword counts")
# Encode the keyword feature and check the correlation to MPI
keyword_mapping = {'keyword' : {'sanitation':1, 'food':2, 'shelter':3, 'clothing':4, 'education':5, 
                'family':6, 'building':7, 'improvement':8, 'other':9}}
keyword_mpi_corr_df = kiva_loans_mpi_df.replace(keyword_mapping)

keyword_mpi_corr_df['keyword'].corr(keyword_mpi_corr_df['MPI'])
# correlation using dummy encoding
encoded_feature = pd.get_dummies(kiva_loans_mpi_df.sector)
corr = encoded_feature.corrwith(kiva_loans_mpi_df.MPI)
corr.sort_values(ascending=False)
data = [dict(
        type='choropleth',
        locations= mpi_national_df.Country,
        locationmode='country names',
        z=mpi_national_df['MPI Urban'],
        text=mpi_national_df.Country,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
)]
layout = dict(
            title = 'Urban MPI, Country Level',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
data = [dict(
        type='choropleth',
        locations= mpi_national_df.Country,
        locationmode='country names',
        z=mpi_national_df['MPI Rural'],
        text=mpi_national_df.Country,
        colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
        reversescale=True,
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
)]
layout = dict(
            title = 'Rural MPI, Country Level',
            geo = dict(
            showframe = False, 
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
# Sort
mpi_national_20_df = mpi_national_df.sort_values(by=['MPI Rural'], ascending=False).head(20)

# Transform the dataframe
mpi_national_urban = mpi_national_20_df[['Country', 'MPI Urban']]
mpi_national_urban.rename(columns={'MPI Urban':'value'}, inplace=True)
mpi_national_urban['indicator'] = 'MPI Urban'

mpi_national_rural = mpi_national_20_df[['Country', 'MPI Rural']]
mpi_national_rural.rename(columns={'MPI Rural':'value'}, inplace=True)
mpi_national_rural['indicator'] = 'MPI Rural'

mpi_urban_rural = mpi_national_urban.append(mpi_national_rural)

# Plot the urban and rural MPI per country (top 20)
with sns.color_palette("OrRd_d", 4), sns.plotting_context("notebook", font_scale=2):
    sns.factorplot(x='Country', y='value', hue='indicator', data=mpi_urban_rural, 
                   kind='bar', legend_out=False,  size=12, aspect=2)
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('')
    plt.title("Urban and Rural MPI per country (top 20)")
    #plt.savefig('urban_rural_mpi.png');
# Sort
mpi_national_20_df = mpi_national_df.sort_values(by=['Headcount Ratio Rural'], ascending=False).head(20)

# Transform the dataframe
mpi_national_hr_urban = mpi_national_20_df[['Country', 'Headcount Ratio Urban']]
mpi_national_hr_urban.rename(columns={'Headcount Ratio Urban':'value'}, inplace=True)
mpi_national_hr_urban['indicator'] = 'Headcount Ratio Urban'

mpi_national_hr_rural = mpi_national_20_df[['Country', 'Headcount Ratio Rural']]
mpi_national_hr_rural.rename(columns={'Headcount Ratio Rural':'value'}, inplace=True)
mpi_national_hr_rural['indicator'] = 'Headcount Ratio Rural'

mpi_hr_urban_rural  = mpi_national_hr_urban.append(mpi_national_hr_rural)

# Plot the urban and rural Headcount Ratio per country (top 20)
with sns.color_palette("OrRd_d", 4), sns.plotting_context("notebook", font_scale=2):
    sns.factorplot(x='Country', y='value', hue='indicator', data=mpi_hr_urban_rural, 
                   kind='bar', legend_out=False, size=12, aspect=2)
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('')
    plt.title("Urban and Rural Headcount Ratio per country (top 20)")
# Sort
mpi_national_20_df = mpi_national_df.sort_values(by=['Intensity of Deprivation Rural'], ascending=False).head(20)

# Transform the dataframe
mpi_national_id_urban = mpi_national_20_df[['Country', 'Intensity of Deprivation Urban']]
mpi_national_id_urban.rename(columns={'Intensity of Deprivation Urban':'value'}, inplace=True)
mpi_national_id_urban['indicator'] = 'Intensity of Deprivation Urban'

mpi_national_id_rural = mpi_national_20_df[['Country', 'Intensity of Deprivation Rural']]
mpi_national_id_rural.rename(columns={'Intensity of Deprivation Rural':'value'}, inplace=True)
mpi_national_id_rural['indicator'] = 'Intensity of Deprivation Rural'

mpi_id_urban_rural  = mpi_national_id_urban.append(mpi_national_id_rural)

# Plot the urban and rural Intensity of Deprivation per country (top 20)
with sns.color_palette("OrRd_d", 4), sns.plotting_context("notebook", font_scale=2):
    sns.factorplot(x='Country', y='value', hue='indicator', data=mpi_id_urban_rural, 
                   kind='bar', legend_out=False, size=12, aspect=2)
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('')
    plt.title("Urban and Rural Intensity of Deprivation per country (top 20)")
data = [ dict(
        type = 'scattergeo',
        lon = mpi_subnational_df['lng'],
        lat = mpi_subnational_df['lat'],
        text = mpi_subnational_df['Sub-national region'],
        #drawmapboundary = dict(fill_color = '#A6CAE0', linewidth = 0.1),
        mode = 'markers',
        marker = dict(
            symbol = 'circle',
            sizemode = 'diameter',
            opacity = 0.7,
            line = dict(width=0),
            sizeref = 5,
            size= mpi_subnational_df['Headcount Ratio Regional'],
            color = mpi_subnational_df['Intensity of deprivation Regional'],
            colorscale = [[0,'rgb(128, 0, 0)'],[1,'rgb(217, 179, 140)']],
            reversescale=True,
            cmin = 0,
            cmax = mpi_subnational_df['Intensity of deprivation Regional'].max(),
            colorbar=dict(title="Intensity")
        ))]
layout = dict(
            title = 'Regional Headcount Ratio and Intensity of deprivation',
            geo = dict(
            showframe = False, 
            showwater=True, 
            showcountries=True,
            showland = True,
            landcolor = 'rgb(245, 241, 213)',
            showcoastlines = True, 
            projection = dict(type = 'Mercator')),
            margin = dict(t=30, b=30, l=10, r=10))
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)