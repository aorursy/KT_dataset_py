!pip install negspacy
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz

import os
import numpy as np
import pandas as pd
import json
import spacy
import scispacy
from negspacy.negation import Negex
from spacy import displacy
import networkx as nx
from scispacy.umls_linking import UmlsEntityLinker
from itertools import chain
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
# Data libraries
import pandas as pd
import re
import pycountry

# Visualisation libraries
import plotly.express as px
import plotly.graph_objects as go

%matplotlib inline

pd.set_option('display.max_columns', 500)

# Load data
metadata_file = '../input/CORD-19-research-challenge/metadata.csv'
df = pd.read_csv(metadata_file,
                 dtype={'Microsoft Academic Paper ID': str,
                        'pubmed_id': str})

def doi_url(d):
    if d.startswith('http'):
        return d
    elif d.startswith('doi.org'):
        return f'http://{d}'
    else:
        return f'http://doi.org/{d}'
    
df.doi = df.doi.fillna('').apply(doi_url)

print(f'loaded DataFrame with {len(df)} records')
# Helper function for filtering df on abstract + title substring
def abstract_title_filter(search_string):
    return (df.abstract.str.lower().str.replace('-', ' ').str.contains(search_string, na=False) |
            df.title.str.lower().str.replace('-', ' ').str.contains(search_string, na=False))
# Helper function for Cleveland dot plot visualisation of count data
def dotplot(input_series, title, x_label='Count', y_label='Regex'):
    subtitle = '<br><i>Hover over dots for exact values</i>'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=input_series.sort_values(),
    y=input_series.sort_values().index.values,
    marker=dict(color="crimson", size=12),
    mode="markers",
    name="Count",
    ))
    fig.update_layout(title=f'{title}{subtitle}',
                  xaxis_title=x_label,
                  yaxis_title=y_label)
    fig.show()
# Helper function which counts synonyms and adds tag column to DF
def count_and_tag(df: pd.DataFrame,
                  synonym_list: list,
                  tag_suffix: str) -> (pd.DataFrame, pd.Series):
    counts = {}
    df[f'tag_{tag_suffix}'] = False
    for s in synonym_list:
        synonym_filter = abstract_title_filter(s)
        counts[s] = sum(synonym_filter)
        df.loc[synonym_filter, f'tag_{tag_suffix}'] = True
    return df, pd.Series(counts)
# Function for printing out key passage of abstract based on key terms
def print_key_phrases(df, key_terms, n=5, chars=300):
    for ind, item in enumerate(df[:n].itertuples()):
        print(f'{ind+1} of {len(df)}')
        print(item.title)
        print('[ ' + item.doi + ' ]')
        try:
            i = len(item.abstract)
            for kt in key_terms:
                kt = kt.replace(r'\b', '')
                term_loc = item.abstract.lower().find(kt)
                if term_loc != -1:
                    i = min(i, term_loc)
            if i < len(item.abstract):
                print('    "' + item.abstract[i-30:i+chars-30] + '"')
            else:
                print('    "' + item.abstract[:chars] + '"')
        except:
            print('NO ABSTRACT')
        print('---')
covid19_synonyms = ['covid',
                    'coronavirus disease 19',
                    'sars cov 2', # Note that search function replaces '-' with ' '
                    '2019 ncov',
                    '2019ncov',
                    r'2019 n cov\b',
                    r'2019n cov\b',
                    'ncov 2019',
                    r'\bn cov 2019',
                    'coronavirus 2019',
                    'wuhan pneumonia',
                    'wuhan virus',
                    'wuhan coronavirus',
                    r'coronavirus 2\b'
                   'Covid-19',
'Covid19',
'Covid-2019',
'Covid2019',
'Coronavirus',
'Sars-cov-2',
'Sars-cov',
'Mars-cov',
'Hcov',
'Alpha coronavirus',
'Beta coronavirus',
'229e alpha coronavirus',
                   ' Nl63 alpha coronavirus',
'Oc43 beta coronavirus',
'Hku1 beta coronavirus'

]
df, covid19_counts = count_and_tag(df, covid19_synonyms, 'disease_covid19')
covid19_counts.sort_values(ascending=False)
dotplot(covid19_counts, 'Covid-19 synonyms in title / abstract metadata')
novel_corona_filter = (abstract_title_filter('novel corona') &
                       df.publish_time.str.startswith('2020', na=False))
print(f'novel corona (published 2020): {sum(novel_corona_filter)}')
df.loc[novel_corona_filter, 'tag_disease_covid19'] = True
df.tag_disease_covid19.value_counts()
# SENSE CHECK: Confirm these all published 2020 (or missing date)
df[df.tag_disease_covid19].publish_time.str.slice(0, 4).value_counts(dropna=False)
# Fix the earlier papers that are about something else
df.loc[df.tag_disease_covid19 & ~df.publish_time.str.startswith('2020', na=True),
       'tag_disease_covid19'] = False
sars_synonyms = [r'\bsars\b',
                 'severe acute respiratory syndrome']
df, sars_counts = count_and_tag(df, sars_synonyms, 'disease_sars')
sars_counts
df.tag_disease_sars.value_counts()
df.groupby('tag_disease_covid19').tag_disease_sars.value_counts()
mers_synonyms = [r'\bmers\b',
                 'middle east respiratory syndrome']
df, mers_counts = count_and_tag(df, mers_synonyms, 'disease_mers')
mers_counts
df.tag_disease_mers.value_counts()
df.groupby('tag_disease_covid19').tag_disease_mers.value_counts()
corona_synonyms = ['corona', r'\bcov\b']
df, corona_counts = count_and_tag(df, corona_synonyms, 'disease_corona')
corona_counts
df.tag_disease_corona.value_counts()
df.groupby('tag_disease_covid19').tag_disease_corona.value_counts()
ards_synonyms = ['acute respiratory distress syndrome',
                 r'\bards\b']
df, ards_counts = count_and_tag(df, ards_synonyms, 'disease_ards')
ards_counts
df.tag_disease_ards.value_counts()
n = (df.tag_disease_covid19 & df.tag_disease_ards).sum()
print(f'There are {n} papers on Covid-19 and ARDS.')
riskfac_synonyms = [
    'risk factor analysis',
    'cross sectional case control',
    'prospective case control',
    'matched case control',
    'medical records review',
    'seroprevalence survey',
    'syndromic surveillance'
]
df, riskfac_counts = count_and_tag(df, riskfac_synonyms, 'design_riskfac')
dotplot(riskfac_counts, 'Risk factor analysis synonyms in title / abstract metadata')
riskfac_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_design_riskfac).sum()
print(f'There are {n} papers on Covid-19 with a Risk Factor Analysis research design.')
risk_factor_synonyms = ['risk factor',
                        'risk model',
                        'risk by',
                        'comorbidity',
                        'comorbidities',
                        'coexisting condition',
                        'co existing condition',
                        'clinical characteristics',
                        'clinical features',
                        'demographic characteristics',
                        'demographic features',
                        'behavioural characteristics',
                        'behavioural features',
                        'behavioral characteristics',
                        'behavioral features',
                        'predictive model',
                        'prediction model',
                        'univariate', # implies analysis of risk factors
                        'multivariate', # implies analysis of risk factors
                        'multivariable',
                        'univariable',
                        'odds ratio', # typically mentioned in model report
                        'confidence interval', # typically mentioned in model report
                        'logistic regression',
                        'regression model',
                        'factors predict',
                        'factors which predict',
                        'factors that predict',
                        'factors associated with',
                        'underlying disease',
                        'underlying condition']
df, risk_generic_counts = count_and_tag(df, risk_factor_synonyms, 'risk_generic')
dotplot(risk_generic_counts,
        'Count of generic risk factor indicated in title / abstract')
risk_generic_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_generic).sum()
print(f'There are {n} papers on Covid-19 and generic risk factors.')
print_key_phrases(df[df.tag_disease_covid19 & df.tag_risk_generic],
                  risk_factor_synonyms)
age_synonyms = ['median age',
                'mean age',
                'average age',
                'elderly',
                r'\baged\b',
                r'\bold',
                'young',
                'teenager',
                'adult',
                'child'
               ]
df, age_counts = count_and_tag(df, age_synonyms, 'risk_age')
dotplot(age_counts, 'Age synonyms in title / abstract metadata')
age_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_age).sum()
print(f'There are {n} papers on Covid-19 and age.')
sex_synonyms = ['sex',
                'gender',
                r'\bmale\b',
                r'\bfemale\b',
                r'\bmales\b',
                r'\bfemales\b',
                r'\bmen\b',
                r'\bwomen\b'
               ]
df, sex_counts = count_and_tag(df, sex_synonyms, 'risk_sex')
dotplot(sex_counts, 'Sex / gender synonyms in title / abstract metadata')
sex_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_sex).sum()
print(f'There are {n} papers on Covid-19 and sex / gender.')
bodyweight_synonyms = [
    'overweight',
    'over weight',
    'obese',
    'obesity',
    'bodyweight',
    'body weight',
    r'\bbmi\b',
    'body mass',
    'body fat',
    'bodyfat',
    'kilograms',
    r'\bkg\b', # e.g. 70 kg
    r'\dkg\b'  # e.g. 70kg
]
df, bodyweight_counts = count_and_tag(df, bodyweight_synonyms, 'risk_bodyweight')
dotplot(bodyweight_counts, 'Bodyweight synonyms in title / abstract data')
bodyweight_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_bodyweight).sum()
print(f'There are {n} papers on Covid-19 and bodyweight')
print_key_phrases(df[df.tag_disease_covid19 & df.tag_risk_bodyweight],
                  bodyweight_synonyms)
smoking_synonyms = ['smoking',
                    'smoke',
                    'cigar', # this picks up cigar, cigarette, e-cigarette, etc.
                    'nicotine',
                    'cannabis',
                    'marijuana']
df, smoking_counts = count_and_tag(df, smoking_synonyms, 'risk_smoking')
dotplot(smoking_counts, 'Smoking synonym counts in title / abstract metadata')
smoking_counts.sort_values(ascending=False)
df.groupby('tag_disease_covid19').tag_risk_smoking.value_counts()
n = (df.tag_disease_covid19 & df.tag_risk_smoking).sum()
print(f'tag_disease_covid19 x tag_risk_smoking currently returns {n} papers')
print_key_phrases(df[df.tag_disease_covid19 & df.tag_risk_smoking],
                  smoking_synonyms, n=12)
diabetes_synonyms = [
    'diabet', # picks up diabetes, diabetic, etc.
    'insulin', # any paper mentioning insulin likely to be relevant
    'blood sugar',
    'blood glucose',
    'ketoacidosis',
    'hyperglycemi', # picks up hyperglycemia and hyperglycemic
]
df, diabetes_counts = count_and_tag(df, diabetes_synonyms, 'risk_diabetes')
dotplot(diabetes_counts, 'Diabetes synonym counts in title / abstract metadata')
diabetes_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_diabetes).sum()
print(f'There are {n} papers on Covid-19 and diabetes')
print_key_phrases(df[df.tag_disease_covid19 & df.tag_risk_diabetes],
                  diabetes_synonyms, n=49)
hypertension_synonyms = [
    'hypertension',
    'blood pressure',
    r'\bhbp\b', # HBP = high blood pressure
    r'\bhtn\b' # HTN = hypertension
]
df, hypertension_counts = count_and_tag(df, hypertension_synonyms, 'risk_hypertension')
dotplot(hypertension_counts, 'Hypertension synonyms in title / abstract metadata')
hypertension_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_hypertension).sum()
print(f'There are {n} papers on Covid-19 and hypertension')
immunodeficiency_synonyms = [
    'immune deficiency',
    'immunodeficiency',
    r'\bhiv\b',
    r'\baids\b'
    'granulocyte deficiency',
    'hypogammaglobulinemia',
    'asplenia',
    'dysfunction of the spleen',
    'spleen dysfunction',
    'complement deficiency',
    'neutropenia',
    'neutropaenia', # alternate spelling
    'cell deficiency' # e.g. T cell deficiency, B cell deficiency
]
df, immunodeficiency_counts = count_and_tag(df,
                                            immunodeficiency_synonyms,
                                            'risk_immunodeficiency')
dotplot(immunodeficiency_counts, 'Immunodeficiency synonyms in title / abstract metadata')
immunodeficiency_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_immunodeficiency).sum()
print(f'tag_disease_covid19 x tag_risk_immunodeficiency currently returns {n} papers')
df[df.tag_disease_covid19 & df.tag_risk_immunodeficiency].head()
cancer_synonyms = [
    'cancer',
    'malignant tumour',
    'malignant tumor',
    'melanoma',
    'leukemia',
    'leukaemia',
    'chemotherapy',
    'radiotherapy',
    'radiation therapy',
    'lymphoma',
    'sarcoma',
    'carcinoma',
    'blastoma',
    'oncolog'
]
df, cancer_counts = count_and_tag(df, cancer_synonyms, 'risk_cancer')
dotplot(cancer_counts, 'Cancer synonyms in title / abstract metadata')
cancer_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_cancer).sum()
print(f'There are {n} papers on Covid-19 and cancer')
chronicresp_synonyms = [
    'chronic respiratory disease',
    'asthma',
    'chronic obstructive pulmonary disease',
    r'\bcopd',
    'chronic bronchitis',
    'emphysema'
]
df, chronicresp_counts = count_and_tag(df, chronicresp_synonyms, 'risk_chronicresp')
dotplot(chronicresp_counts, 'Chronic respiratory disease terms in title / abstract metadata')
chronicresp_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_risk_chronicresp).sum()
print(f'There are {n} papers on Covid-19 and chronic respiratory disease')
print_key_phrases(df[df.tag_disease_covid19 & df.tag_risk_chronicresp],
                  chronicresp_synonyms, n=15)
# Only really one term for asthma
df, asthma_counts = count_and_tag(df, ['asthma'], 'risk_asthma')
asthma_counts
n = (df.tag_disease_covid19 & df.tag_risk_asthma).sum()
print(f'There are {n} papers on Covid-19 and asthma')
print_key_phrases(df[df.tag_disease_covid19 & df.tag_risk_asthma],
                  ['asthma'])
immunity_synonyms = [
    'immunity',
    r'\bvaccin',
    'innoculat'
]
df, immunity_counts = count_and_tag(df, immunity_synonyms, 'immunity_generic')
immunity_counts
n = (df.tag_disease_covid19 & df.tag_immunity_generic).sum()
print(f'There are {n} papers on Covid-19 and immunity / vaccines')
print('Intersection of tag_disease_covid19, tag_risk_generic & tag_immunity_generic')
print('=' * 76)
print_key_phrases(df[df.tag_disease_covid19 &
                     df.tag_risk_generic &
                     df.tag_immunity_generic],
                  risk_factor_synonyms + immunity_synonyms)
tag_columns = df.columns[df.columns.str.startswith('tag_')].tolist()
# Note that this section needs more work - have been focusing on later sections
continental_regions = {
    'asia': 'asia|china|korea|japan|hubei|wuhan|malaysia|singapore|hong kong',
    'east_asia': 'east asia|china|korea|japan|hubei|wuhan|hong kong',
    'south_asia': 'south asia|india|pakistan|bangladesh|sri lanka',
    'se_asia': r'south east asia|\bse asia|malaysia|thailand|indonesia|vietnam|cambodia|viet nam',
    'europe': 'europe|italy|france|spain|germany|austria|switzerland|united kingdom|ireland',
    'africa': 'africa|kenya',
    'middle_east': 'middle east|gulf states|saudi arabia|\buae\b|iran|persian',
    'south_america': 'south america|latin america|brazil|argentina',
    'north_america': 'north america|usa|united states|canada|caribbean',
    'australasia': 'australia|new zealand|oceania|australasia|south pacific'
}

counts = {}
for cr, s in continental_regions.items():
    con_filter = abstract_title_filter(s)
    counts[cr] = sum(con_filter)
    df.loc[con_filter, f'tag_continent_{cr}'] = True
    df[f'tag_continent_{cr}'].fillna(False, inplace=True)
counts = pd.Series(counts)
dotplot(counts, 'Continent counts in title / abstract metadata')
### THIS SECTION TAKES A LONG TIME TO RUN SO COMMENTED OUT WHILE DEVELOPING
# MIN_PAPERS_PER_COUNTRY = 50
# counts = {}

# for i, country in enumerate(pycountry.countries):
#     if i % 20 == 0:
#         print(f'Checking country {i} ({country.name})')
#     country_filter = abstract_title_filter(r'\b' + re.escape(country.name.lower()) + r'\b')
#     n = sum(country_filter)
#     if n >= MIN_PAPERS_PER_COUNTRY:
#         counts[country.name] = n
#         df.loc[country_filter, f'tag_country_{country.alpha_3.lower()}'] = True
#         df[f'tag_country_{country.alpha_3.lower()}'].fillna(False, inplace=True)
# counts = pd.Series(counts)
# plt.figure(figsize=(5,7))
# dotplot(counts, 'Country counts in title / abstract metadata')
# df.groupby('tag_disease_covid19').tag_country_chn.value_counts()
climate_synonyms = [
    'climate',
    'weather',
    'humid',
    'sunlight',
    'air temperature',
    'meteorolog', # picks up meteorology, meteorological, meteorologist
    'climatolog', # as above
    'dry environment',
    'damp environment',
    'moist environment',
    'wet environment',
    'hot environment',
    'cold environment',
    'cool environment'
]
df, climate_counts = count_and_tag(df, climate_synonyms, 'climate_generic')
dotplot(climate_counts, 'Climate synonyms by title / abstract metadata')
climate_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_climate_generic).sum()
print(f'There are {n} papers on Covid-19 and climate:\n')
print_key_phrases(df[df.tag_disease_covid19 & df.tag_climate_generic],
                  climate_synonyms, n=n)

transmission_synonyms = [
    'transmiss', # Picks up 'transmission' and 'transmissibility'
    'transmitted',
    'incubation',
    'environmental stability',
    'airborne',
    'via contact',
    'human to human',
    'through droplets',
    'through secretions',
    r'\broute',
    'exportation'
]
df, transmission_counts = count_and_tag(df, transmission_synonyms, 'transmission_generic')
dotplot(transmission_counts, 'Transmission / incubation synonyms by title / abstract metadata')
transmission_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_transmission_generic).sum()
print(f'There are {n} papers on Covid-19 and transmission / incubation / environmental stability')
print('\nThis entire dataset is exported to thematic_tagging_output_transmission.csv')
repr_synonyms = [
    r'reproduction \(r\)',
    'reproduction rate',
    'reproductive rate',
    '{r}_0',
    r'\br0\b',
    r'\br_0',
    '{r_0}',
    r'\b{r}',
    r'\br naught',
    r'\br zero'
]
df, repr_counts = count_and_tag(df,repr_synonyms, 'transmission_repr')
dotplot(repr_counts, 'R<sub>0</sub> synonyms by title / abstract metadata')
repr_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_transmission_repr).sum()
print(f'There are {n} papers on Covid-19 and R or R_0')
print('=')
print_key_phrases(df[df.tag_disease_covid19 & df.tag_transmission_repr], 
                  repr_synonyms, n=52, chars=500)
# DATA_FOLDER = '../input/CORD-19-research-challenge'

# import json
# import os

# json_list = []

# for row in df[df.tag_disease_covid19 &
#               df.tag_transmission_repr & 
#               df.has_full_text].itertuples():
#     filename = f'{row.sha}.json'
#     sources = ['biorxiv_medrxiv', 'comm_use_subset',
#                'custom_license', 'noncomm_use_subset']
#     for source in sources:
#         if filename in os.listdir(os.path.join(DATA_FOLDER, source, source)):
#             with open(os.path.join(DATA_FOLDER, source, source, filename), 'rb') as f:
#                 json_list.append(json.load(f))
# candidate_sections = [
#     'results',
#     'conclusion',
#     'conclusions',
#     'reproduction',
#     'r_0',
#     'r0',
#     'reproductive'
# ]
# for i, item in enumerate(json_list):
#     print(i)
#     body_text = item['body_text']
#     for sub_item in body_text:
#         found = False
#         for cs in candidate_sections:
#             if cs in sub_item['section'].lower():
#                 found = True
#         if found:
#             print(sub_item['section'])
#             print(sub_item['text'])
#             print()
#     print()
# for i, item in enumerate(json_list):
#     print(i)
#     body_text = item['body_text']
#     for sub_item in body_text:
#         if sub_item['section'] in ['Methods and Results', 'Results', 'Conclusions']:
#             print(sub_item['text'])
#     print()
filename = 'thematic_tagging_output_covid19_only.csv'
print(f'Outputting {df.tag_disease_covid19.sum()} records to {filename}')
df[df.tag_disease_covid19].to_csv(filename, index=False)
file_filter = df.tag_disease_covid19 & df.tag_risk_generic
filename = 'thematic_tagging_output_risk_factors.csv'
print(f'Outputting {file_filter.sum()} records to {filename}')
df[file_filter].to_csv(filename, index=False)
file_filter = df.tag_disease_covid19 & df.tag_risk_diabetes
filename = 'thematic_tagging_output_risk_diabetes.csv'
print(f'Outputting {file_filter.sum()} records to {filename}')
df[file_filter].to_csv(filename, index=False)
file_filter = df.tag_disease_covid19 & df.tag_risk_smoking
filename = 'thematic_tagging_output_risk_smoking.csv'
print(f'Outputting {file_filter.sum()} records to {filename}')
df[file_filter].to_csv(filename, index=False)
file_filter = df.tag_disease_covid19 & df.tag_climate_generic
filename = 'thematic_tagging_output_climate.csv'
print(f'Outputting {file_filter.sum()} records to {filename}')
df[file_filter].to_csv(filename, index=False)
file_filter = df.tag_disease_covid19 & df.tag_transmission_generic
filename = 'thematic_tagging_output_transmission.csv'
print(f'Outputting {file_filter.sum()} records to {filename}')
df[file_filter].to_csv(filename, index=False)
file_filter = df.tag_disease_covid19 & df.tag_transmission_repr
filename = 'thematic_tagging_output_repr.csv'
print(f'Outputting {file_filter.sum()} records to {filename}')
df[file_filter].to_csv(filename, index=False)
filename = 'thematic_tagging_output_full.csv'
print(f'Outputting {len(df)} records to {filename}')
df.to_csv(filename, index=False)
df
df.columns
df[['tag_disease_covid19',
       'tag_disease_sars', 'tag_disease_mers', 'tag_disease_corona',
       'tag_disease_ards', 'tag_design_riskfac', 'tag_risk_generic',
       'tag_risk_age', 'tag_risk_sex', 'tag_risk_bodyweight',
       'tag_risk_smoking', 'tag_risk_diabetes', 'tag_risk_hypertension',
       'tag_risk_immunodeficiency', 'tag_risk_cancer', 'tag_risk_chronicresp',
       'tag_risk_asthma', 'tag_immunity_generic', 'tag_continent_asia',
       'tag_continent_east_asia', 'tag_continent_south_asia',
       'tag_continent_se_asia', 'tag_continent_europe', 'tag_continent_africa',
       'tag_continent_middle_east', 'tag_continent_south_america',
       'tag_continent_north_america', 'tag_continent_australasia',
       'tag_climate_generic', 'tag_transmission_generic',
       'tag_transmission_repr']] *= 1
dfbol=df[['tag_disease_covid19',
       'tag_disease_sars', 'tag_disease_mers', 'tag_disease_corona',
       'tag_disease_ards', 'tag_design_riskfac', 'tag_risk_generic',
       'tag_risk_age', 'tag_risk_sex', 'tag_risk_bodyweight',
       'tag_risk_smoking', 'tag_risk_diabetes', 'tag_risk_hypertension',
       'tag_risk_immunodeficiency', 'tag_risk_cancer', 'tag_risk_chronicresp',
       'tag_risk_asthma', 'tag_immunity_generic', 'tag_continent_asia',
       'tag_continent_east_asia', 'tag_continent_south_asia',
       'tag_continent_se_asia', 'tag_continent_europe', 'tag_continent_africa',
       'tag_continent_middle_east', 'tag_continent_south_america',
       'tag_continent_north_america', 'tag_continent_australasia',
       'tag_climate_generic', 'tag_transmission_generic',
       'tag_transmission_repr']]
met = pd.read_csv(metadata_file, sep = ",")
met["publish_time"] = pd.to_datetime(met["publish_time"])

## Keep only records with a PDF hash

has_sha = met[met["sha"].notnull()]
df2 = df[df["sha"].isin(has_sha["sha"])]
root_dir = "/kaggle/input/CORD-19-research-challenge" #***
sub_dirs = [root_dir + d for d in ["/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json",
                                   "/comm_use_subset/comm_use_subset/pdf_json",
                                   "/custom_license/custom_license/pdf_json",
                                   "/noncomm_use_subset/noncomm_use_subset/pdf_json"]]
# corpus = []

# ind = 0
# for sd in sub_dirs:
#         for dn, _, all_nms in os.walk(sd):
#             for fl in all_nms:
#                 ind += 1
#                 this_json = json.loads(open(os.path.join(dn, fl)).read())

#                 para_ind = 0
#                 for para in this_json["body_text"]:
#                     para_ind += 1
#                     corpus.append([this_json["paper_id"], para_ind, para["section"], para["text"]])

#                 if ind % 500 == 0:
#                     print(ind)
            
# corpus = pd.DataFrame(corpus, columns = ["FileID", "Paragraph", "Section", "Text"])
# corpus["Section"] = corpus["Section"].str.lower()
corpus = []

papers_filtered = df2[df2["tag_disease_corona"] == 1]["sha"].drop_duplicates()
corpus_docs = (papers_filtered + ".json").tolist()

ind = 0
for sd in sub_dirs:
    for fl in corpus_docs:
        if (ind % 500 == 0) & (ind != 0):
            print(ind)
        try:
            ind += 1
            this_json = json.loads(open(os.path.join(sd, fl)).read())

            para_ind = 0
            for para in this_json["body_text"]:
                para_ind += 1
                corpus.append([this_json["paper_id"], para_ind, para["section"], para["text"]])
        except FileNotFoundError:
            pass

corpus = pd.DataFrame(corpus, columns = ["FileID", "Paragraph", "Section", "Text"])
corpus["Section"] = corpus["Section"].str.lower()
nlp_proc = spacy.load("en_core_sci_sm")
linky = UmlsEntityLinker(resolve_abbreviations=True)
nlp_proc.add_pipe(linky)
negx = Negex(nlp_proc, language = "en_clinical")
nlp_proc.add_pipe(negx, last=True)
corpus_use = corpus.groupby("FileID").apply(lambda gr: " ".join(gr["Text"])).reset_index()
corpus_use["Section"] = "Full Document"
corpus_use = corpus_use.sample(100) # for example purposes: remove or enlargen for a full analysis
corpus_use = corpus_use[["FileID", "Section", 0]].values
cui_table = True
umls_one_verb = False

doc_cui_track = []
trk = []

for d in range(corpus_use.shape[0]):
    td = corpus_use[d,2]
    p = nlp_proc(td)
    if cui_table:
        for e in p.ents:
            for u in e._.umls_ents:
                doc_cui_track.append((corpus_use[d,0], corpus_use[d,1], u[0], u[1], e._.negex))
    if umls_one_verb:
        pass
    if (d % 50) == 0:
        print(d)

if cui_table:
    cui_data = pd.DataFrame(doc_cui_track, columns = ["FileID", "Section", "CUI", "Confidence", "NegatedInd"])

from IPython.core.display import HTML

cui_data["Combo"] = pd.Series(np.where(cui_data["NegatedInd"], "N_", "Y_")) + cui_data["CUI"]
as_list = cui_data[cui_data["Confidence"] > 0.9999].groupby("FileID").apply(lambda gr: list(gr["Combo"]))
mlb = MultiLabelBinarizer()
for_km = pd.DataFrame(mlb.fit_transform(as_list), columns = mlb.classes_)
dense_cols = for_km.mean(); dense_cols = dense_cols[dense_cols > 0.5].index.tolist()
for_km = for_km[dense_cols]
for_km["FileID"] = as_list.index
cols_tda = dense_cols


mpr = km.KeplerMapper()
proj = mpr.fit_transform(for_km[cols_tda], projection = TruncatedSVD(n_components=2))
cov = km.Cover(n_cubes = 5, perc_overlap = 0.50)
simp_comp = mpr.map(proj, X = for_km[cols_tda], cover = cov,
                    clusterer = KMeans(n_clusters = 3))
                    #clusterer = DBSCAN(metric = "hamming", eps = 0.001))

tda_html = mpr.visualize(simp_comp, save_file=False)

# Would need to export to HTML and open in separate page
display(HTML(tda_html))
nodes = pd.DataFrame([(ky,v) for ky, l in simp_comp["nodes"].items() for v in l], columns = ["Node", "ind"])
nd_fl = for_km.iloc[nodes["ind"]]["FileID"]
nd_fl.index = nodes.index
nodes["FileID"] = nd_fl
nodes = nodes.drop("ind", axis=1)

combo = pd.merge(nodes, for_km, on = "FileID")
combo = pd.merge(combo,
                    df2[[c for c in df2.columns if c[:3] == "tag"] + ["sha"]].rename(columns={"sha":"FileID"}),
                    on = "FileID")
G=nx.Graph()

G.add_edges_from([(ky,v) for ky, l in simp_comp["links"].items() for v in l])
color = combo.groupby("Node")["tag_risk_generic"].mean()
color_sub = color[color > color.quantile(0.2)].index.tolist()

cc = [c for c in nx.connected_components(G.subgraph(color_sub))]
out_grp = set().union(*cc)
out_grp = set(nodes["Node"]).difference(out_grp)
lcc = len(cc)
cc = [(i+1, n) for i in range(lcc) for n in cc[i]]
cc += [("OutGroup", v) for v in out_grp]
cc = pd.DataFrame(cc, columns = ["Hotspot", "Node"])
hot_fl = pd.merge(cc, nodes, on = "Node")[["Hotspot", "FileID"]]
ignore = ["FileID", "Node"]
covidout=pd.merge(hot_fl, combo, on = "FileID").drop(ignore, axis=1).groupby("Hotspot").mean().T
covidout.to_csv('covidout.csv')