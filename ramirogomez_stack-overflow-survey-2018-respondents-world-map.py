import pandas as pd

df_public = pd.read_csv('../input/survey_results_public.csv', low_memory=False)
df_countries = pd.DataFrame(df_public.Country.value_counts())
df_countries.head(5)
from iso3166 import countries

country_index = {name: countries.get(name).alpha3 for name in df_countries.index if name in countries}
set(df_countries.index) - set(country_index)
from collections import Counter

country_index.update({
    'Bolivia': 'BOL',
    'Cape Verde': 'CPV',
    'Congo, Republic of the...': 'COG',
    'Czech Republic': 'CZE',
    "Democratic People's Republic of Korea": 'PRK',
    'Democratic Republic of the Congo': 'COD',
    'Hong Kong (S.A.R.)': 'HKG',
    'Iran, Islamic Republic of...': 'IRN',
    'Libyan Arab Jamahiriya': 'LBY',
    'Micronesia, Federated States of...': 'FSM',
    'North Korea': 'PRK',
    'Republic of Korea': 'KOR',
    'Republic of Moldova': 'MDA',
    'South Korea': 'KOR',
    'The former Yugoslav Republic of Macedonia': 'MKD',
    'United Kingdom': 'GBR',
    'United Republic of Tanzania': 'TZA',
    'Venezuela, Bolivarian Republic of...': 'VEN'
})

pd.Series(country_index).value_counts().head()
df_countries['iso'] = df_countries.index.map(lambda x: country_index.get(x))
iso_index = df_countries.groupby('iso').sum()
iso_index.sort_values('Country', ascending=False).head()
import geopandas as gpd

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).to_crs('+proj=robin')
world = world[world.name != 'Antarctica']

world['respondents'] = world['iso_a3'].apply(lambda x: int(iso_index.loc[x]) if x in iso_index.index else None)
world['respondent_ratio'] = world['respondents'] / world['pop_est'] * 1_000_000
world.sort_values('respondent_ratio', ascending=False).head(10)
known = world.dropna(subset=['respondent_ratio'])
unknown = world[world['respondent_ratio'].isna()]

ax = known.plot(column='respondent_ratio', cmap='viridis_r', figsize=(20, 12), scheme='fisher_jenks', k=7, legend=True, edgecolor='#aaaaaa')
unknown.plot(ax=ax, color='#ffffff', hatch='//', edgecolor='#aaaaaa')

ax.set_title('Stack Overflow Developer Survey 2018 Respondents per 1 Million People', fontdict={'fontsize': 20}, loc='left')
descripton = '''
Survey data: kaggle.com/stackoverflow/stack-overflow-2018-developer-survey • Population estimates: naturalearthdata.com • 
Source code: kaggle.com/ramirogomez/stack-overflow-survey-2018-respondents-world-map • Author: Ramiro Gómez - ramiro.org'''.strip()
ax.annotate(descripton, xy=(0.065, 0.12), size=12, xycoords='figure fraction')

ax.set_axis_off()
legend = ax.get_legend()
legend.set_bbox_to_anchor((.11, .4))
legend.prop.set_size(12)