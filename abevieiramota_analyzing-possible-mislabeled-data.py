# see: https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/81331#488716

mislabeleds = """development-36	mislabeled as A

development-41	mislabeled as B, should be N

development-95	mislabeled as A

development-171	mislabeled as A

development-219	mislabeled as A

development-260	mislabeled as A

development-291	mislabeled as B, should be A

development-302	mislabeled as A

development-325	mislabeled as A

development-328	mislabeled as B, should be N

development-337	mislabeled as A

development-346	mislabeled as B, should be N

development-401	mislabeled as B, should be A

development-444	mislabeled as A, it's actually N (refers to Theresa the granddaughter, not Theresa the grandmother)

development-528	mislabeled as A

development-541	mislabeled as N, should be B

development-566	mislabeled as B, should be A

development-578	mislabeled as B, should be A

development-697	mislabeled as A, it's actually N

development-779	mislabeled as A

development-822	mislabeled as A

development-837	mislabeled as B, should be N (character, not actor)

development-857	mislabeled as B, should be N (official, not Reagan)

development-871	mislabeled as B, should be N (heroine, not Mouna Ragam)

development-921	mislabeled as B, should be A

development-955	mislabeled as B, should be N

development-967	mislabeled as A

development-1086	mislabeled as A, it's actually N (Amanda Plummer)

development-1089	mislabeled as N, it's both A and B

development-1030	mislabeled as B, should be N

development-1051	mislabeled as B, should be N

development-1161	mislabeled as N, should be B

development-1192	mislabeled as B, should be A

development-1204	mislabeled as A

development-1216	mislabeled as B, should be A

development-1292	mislabeled as A, it's actually N (Rainn Wilson)

development-1369	mislabeled as B, should be A

development-1422	mislabeled as B, should be A

development-1565	mislabeled as A

development-1569	mislabeled as A, it's actually N (Budweiser Wickersham)

development-1680	mislabeled as A

development-1710	mislabeled as A

development-1722	mislabeled as A, it's actually N (Kushi)

development-1745	mislabeled as N, should be B

development-1874	mislabeled as B, should be A

development-1908	mislabeled as A

development-1934	mislabeled as B, should be A

development-1994	mislabeled as A"""
mislabeled_ids_message = [line.split('\t') for line in mislabeleds.split('\n')]
import pandas as pd



from spacy import displacy



def display_entry(entry):



    data = entry.to_dict()

    

    colors = {

        'Pronoun': '#aa9cfc',

        'A': '#fc9ce7' if not 'A-coref' in data or not data['A-coref'] else '#FFE14D',

        'B': '#fc9ce7' if not 'B-coref' in data or not data['B-coref'] else '#FFE14D'

    }



    options = {

        'colors': colors

    }

    

    render_data = {

        'text': data['Text'],

        'ents': sorted([

            {

                'start': data['Pronoun-offset'],

                'end': data['Pronoun-offset-end'],

                'label': 'Pronoun'

            },

            {

                'start': data['A-offset'],

                'end': data['A-offset-end'],

                'label': 'A'

            },

            {

                'start': data['B-offset'],

                'end': data['B-offset-end'],

                'label': 'B'

            }

        ], key=lambda x: x['start'])

    }

    

    displacy.render(render_data, style='ent', manual=True, jupyter=True, options=options)



    

def read_df(path):

    

    df = pd.read_csv(path, index_col='ID', sep='\t', encoding='utf-8')



    # add some columns



    # the ending offset of the pronoun and the candidates referring entities

    df['Pronoun-offset-end'] = df['Pronoun-offset'] + df['Pronoun'].str.len()

    df['A-offset-end'] = df['A-offset'] + df['A'].str.len()

    df['B-offset-end'] = df['B-offset'] + df['B'].str.len()



    # text length



    df['Text-length'] = df['Text'].str.len()

    

    return df



df = read_df('../input/test_stage_1.tsv')
for idx, message in mislabeled_ids_message:

    

    entry = df.loc[idx]

    

    print(message)

    display_entry(entry)

    print()