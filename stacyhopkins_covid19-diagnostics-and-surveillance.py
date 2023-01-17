#import libraries and load data



# Data libraries

import pandas as pd





pd.set_option('display.max_columns', 500)



# Load data

metadata_file = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

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


# Helper functions



# Helper function for filtering df on abstract + title substring

def abstract_title_filter(search_string):

    return (df.abstract.str.lower().str.replace('-', ' ').str.contains(search_string, na=False) |

            df.title.str.lower().str.replace('-', ' ').str.contains(search_string, na=False))

  



# Helper function which counts synonyms (keywords) and adds tag column to DF

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
# initial covid-19 filters



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

                    r'coronavirus 2\b']



df, covid19_counts = count_and_tag(df, covid19_synonyms, 'disease_covid19')



covid19_counts.sort_values(ascending=False)



novel_corona_filter = (abstract_title_filter('novel corona') &

                       df.publish_time.str.startswith('2020', na=False))



# novel corona records published in 2020

print(f'novel corona (published 2020): {sum(novel_corona_filter)}')



# covid-19 tagged records published in 2020

df.loc[novel_corona_filter, 'tag_disease_covid19'] = True

df.tag_disease_covid19.value_counts()



# Confirm all covid-19 records published in 2020 (or missing date)

df[df.tag_disease_covid19].publish_time.str.slice(0, 4).value_counts(dropna=False)



# Mark out earlier papers that are about something else

df.loc[df.tag_disease_covid19 & ~df.publish_time.str.startswith('2020', na=True),

       'tag_disease_covid19'] = False

# Surveillance filter



# Covid-19 relationship with surveillance



surv_synonyms = ['surveillance',

                         'syndromic surveillance',

                         'assessment',

                         'testing',

                         'diagnostics',

                         'screening'] 



df, surv_counts = count_and_tag(df, surv_synonyms, 'surv')



surv_counts

    

df.tag_surv.value_counts()  



##crosstab between covid19 and surveillance

df.groupby('tag_disease_covid19').tag_surv.value_counts()



# 1222 papers on Covid-19 and surveillance

n = (df.tag_disease_covid19 & df.tag_surv).sum()

print(f'There are {n} papers on Covid-19 and surveillance.')

# Abstract excerpt example



# Printing out default 5 examples and key text from  Abstract.



print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv],

                  surv_synonyms)





# Optional: adjust Abstract quantity and key text quantity (n=3, chars=100)

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv], surv_synonyms, n=3, chars=100)



bullet1_synonyms =['policy recommendations',

                         'sampling methods',

                         'early detection',]

                         

                         

df, bullet1_counts = count_and_tag(df, bullet1_synonyms, 'bullet1')

bullet1_counts

    

df.tag_bullet1.value_counts()  



# Papers on Covid-19, surveillance, and bullet1

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet1).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet1:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet1],

                  bullet1_synonyms)
bullet2_synonyms =['existing surveillance',

                   'existing diagnostic']



                         

df, bullet2_counts = count_and_tag(df, bullet2_synonyms, 'bullet2')



bullet2_counts

    

df.tag_bullet2.value_counts()  





# Papers on Covid-19, surveillance, and bullet2

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet2).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet2:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet2],

                  bullet2_synonyms)
bullet3_synonyms =['expertise',

                   'capacity']



                         

df, bullet3_counts = count_and_tag(df, bullet3_synonyms, 'bullet3')



bullet3_counts

    

df.tag_bullet3.value_counts()  





# Papers on Covid-19, surveillance, and bullet3

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet3).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet3:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet3],

                  bullet3_synonyms)

bullet4_synonyms =['guidance',

                   'guidelines',

                   'best practices']



                         

df, bullet4_counts = count_and_tag(df, bullet4_synonyms, 'bullet4')



bullet4_counts

    

df.tag_bullet4.value_counts()  





# Papers on Covid-19, surveillance, and bullet4

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet4).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet4:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet4],

                  bullet4_synonyms)

bullet5_synonyms =['rapid test',

                   'rapid tests',

                   'rapid testing']

                   

                         

df, bullet5_counts = count_and_tag(df, bullet5_synonyms, 'bullet5')



bullet5_counts

    

df.tag_bullet5.value_counts()  





# Papers on Covid-19, surveillance, and bullet5

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet5).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet5:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet5],

                  bullet5_synonyms)



bullet6_synonyms =['pcr']

                 

                         

df, bullet6_counts = count_and_tag(df, bullet6_synonyms, 'bullet6')



bullet6_counts

    

df.tag_bullet6.value_counts()  



# Papers on Covid-19, surveillance, and bullet6

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet6).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet6:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet6],

                  bullet6_synonyms)


bullet7_synonyms =['assay', 'assays']

                  

  

                         

df, bullet7_counts = count_and_tag(df, bullet7_synonyms, 'bullet7')



bullet7_counts

    

df.tag_bullet7.value_counts()  





# Papers on Covid-19, surveillance, and bullet7

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet7).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet7:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet7],

                  bullet7_synonyms)
bullet8_synonyms =['genetic drift', 'mutations']

                  

                          

df, bullet8_counts = count_and_tag(df, bullet8_synonyms, 'bullet8')



bullet8_counts

    

df.tag_bullet8.value_counts()  





# Papers on Covid-19, surveillance, and bullet8

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet8).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet8:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet8],

                  bullet8_synonyms)





bullet9_synonyms =['latency']

                  

                       

df, bullet9_counts = count_and_tag(df, bullet9_synonyms, 'bullet9')



bullet9_counts

    

df.tag_bullet9.value_counts()  





# Papers on Covid-19, surveillance, and bullet9

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet9).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet9:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet9],

                  bullet9_synonyms)
bullet10_synonyms =['markers']

                  

                         

df, bullet10_counts = count_and_tag(df, bullet10_synonyms, 'bullet10')



bullet10_counts

    

df.tag_bullet10.value_counts()  





# Papers on Covid-19, surveillance, and bullet10

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet10).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet10:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet10],

                  bullet10_synonyms)
bullet11_synonyms =['policies','protocols']

                  

                         

df, bullet11_counts = count_and_tag(df, bullet11_synonyms, 'bullet11')



bullet11_counts

    

df.tag_bullet11.value_counts()  





# Papers on Covid-19, surveillance, and bullet11

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet11).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet11:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet11],

                  bullet11_synonyms)



bullet12_synonyms =['supplies',]

                  

                          

df, bullet12_counts = count_and_tag(df, bullet12_synonyms, 'bullet12')



bullet12_counts

    

df.tag_bullet12.value_counts()  





# Papers on Covid-19, surveillance, and bullet12

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet12).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet12:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet12],

                  bullet12_synonyms)
bullet13_synonyms =['technology']

                  

  

                         

df, bullet13_counts = count_and_tag(df, bullet13_synonyms, 'bullet13')



bullet13_counts

    

df.tag_bullet13.value_counts()  





# Papers on Covid-19, surveillance, and bullet13

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet13).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet13:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet13],

                  bullet13_synonyms)


bullet14_synonyms =['genomics','genes']

                  

                           

df, bullet14_counts = count_and_tag(df, bullet14_synonyms, 'bullet14')



bullet14_counts

    

df.tag_bullet14.value_counts()  





# Papers on Covid-19, surveillance, and bullet14

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet14).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet14:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet14],

                  bullet14_synonyms)

bullet15_synonyms =['sequencing','bioinformatics']

                  

                      

df, bullet15_counts = count_and_tag(df, bullet15_synonyms, 'bullet15')



bullet15_counts

    

df.tag_bullet15.value_counts()  





# Papers on Covid-19, surveillance, and bullet15

n = (df.tag_disease_covid19 & df.tag_surv & df.tag_bullet15).sum()

print(f'There are {n} papers on Covid-19, surveillance, and bullet15:')

print_key_phrases(df[df.tag_disease_covid19 & df.tag_surv & df.tag_bullet15],

                  bullet15_synonyms)