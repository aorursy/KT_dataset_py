import json
from pathlib import Path
import spacy
import pandas as pd
import csv
from IPython.display import display
!pip install scispacy --quiet
en_core_sci_lg = '../input/languagemodel/en_core_sci_lg-0.2.4/en_core_sci_lg/en_core_sci_lg-0.2.4'
nlp = spacy.load(en_core_sci_lg)
def make_df(file):
    """Read text fields from JSON file into spacy docs; create dataframe with
    entities, title (so we can later filter by topic) and source file (so we
    can locate the source data)."""
    p = Path(file)
    with p.open() as f:
        data = json.load(f)
        texts = []
        try:
            abstract = data['abstract'][0]['text']
            texts.append(abstract)
        except (IndexError, KeyError):
            print('No abstract.')
        for i in range(len(data['body_text'])):
            text = data['body_text'][i]['text']
            texts.append(text)
        # for efficiency, use nlp.pipe on small chunks of texts and NER only
        for doc in nlp.pipe(texts, disable=["tagger", "parser"]):
            try:
                ents = [[ent.text, p.stem] for ent in doc.ents]
                return pd.DataFrame(ents, columns=['Entity', 'Source'])
            except AttributeError:
                print('No entities found.')
p = Path('/kaggle/input/CORD-19-research-challenge')
df = pd.concat(make_df(file) for file in p.glob('**/*.json'))

entities = df['Entity'].value_counts().to_dict()
sorted_entities = sorted(entities.items())

with open('entities.csv', 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerows(sorted_entities)
df.describe()
print(sorted_entities)
import pandas as pd
metadata = '/kaggle/input/CORD-19-research-challenge/metadata.csv'
sources = pd.read_csv(metadata)
sources.head()
def get_context(file, keyword):
    """Read text fields from JSON file into spacy docs; return list of
    sentences that include the target word."""
    Path('/kaggle/input/CORD-19-research-challenge')
    q = sorted(Path().rglob(file))
    for item in q:
        with item.open() as f:
            data = json.load(f)
            texts = []
            try:
                abstract = data['abstract'][0]['text']
                texts.append(abstract)
            except (IndexError, KeyError):
                print(file)
            for i in range(len(data['body_text'])):
                text = data['body_text'][i]['text']
                texts.append(text)
            sents = []
            for doc in nlp.pipe(texts):
                for sent in doc.sents:
                    for token in sent:
                        if token.text == keyword:
                            if sent not in sents:
                                sents.append(sent)
        return sents

    
def make_final_table(entity, keyword):
    """Filter the original dataframe of named entities to find the source files.
    Get the original sentences where they appear. From the dataframe of source metadata,
    get the date, title, and url of the paper. Return a list."""
    entity_filter = df[df['Entity'].str.contains(entity)]  # entities
    entity_source = entity_filter['Source'].drop_duplicates().values.tolist()
    out = []
    for sha in entity_source:
        file = f'{sha}.json'
        snippets = str(get_context(file, keyword))  # sentence can contain a different keyword
        info = sources[sources['sha'].str.contains(sha, na=False)]
        if not info['publish_time'].empty:
            date = info['publish_time'].item()
        if not info['title'].empty:
            title = info['title'].item()
        if not info['url'].empty:
            url = info['url'].item()
            out.append([date, title, url, snippets])
    return out
sample = '/kaggle/input/risk-factors/risk_factors_output.csv'
risk_factors = pd.read_csv(sample)
display(risk_factors)
entities = ["risk factor", "risk factor analysis"]

# For each named entity, get the relevant background data. 
# Note that the keyword (the word to be identified in the full text) needs to be one word.
content = [make_final_table(entity, keyword='risk') for entity in entities]


def make_results_df(data):
    """Create the dataframe for each piece of relevant content, listing the
    date, title, and URL of the study, with list of matching sentences."""
    return pd.DataFrame(data, columns=['Date', 'Study', 'URL', 'Snippet'])

# Concatenate dataframe with all results
results = pd.concat(make_results_df(data) for data in content)

# Sort with newest first
date_sort = results.sort_values(by='Date', ascending=False)

# Write csv
results.to_csv('risk_factors.csv', index=False)
date_sort.head(25)
entities = ["incubation"]
content = [make_final_table(entity, keyword='incubation') for entity in entities]
results = pd.concat(make_results_df(data) for data in content)
date_sort = results.sort_values(by='Date', ascending=False)
results.to_csv('incubation.csv', index=False)
date_sort.head(25)
entities = ["asymptomatic"]
content = [make_final_table(entity, keyword='asymptomatic') for entity in entities]
results = pd.concat(make_results_df(data) for data in content)
date_sort = results.sort_values(by='Date', ascending=False)
results.to_csv('asymptomatic.csv', index=False)
date_sort.head(25)
entities = ["randomized-controlled trial", "randomized controlled trial", "randomised controlled trial"]
content = [make_final_table(entity, keyword='trial') for entity in entities]
results = pd.concat(make_results_df(data) for data in content)
date_sort = results.sort_values(by='Date', ascending=False)
results.to_csv('randomized.csv', index=False)
date_sort.head(25)
entities = ["systematic review"]
content = [make_final_table(entity, keyword='review') for entity in entities]
results = pd.concat(make_results_df(data) for data in content)
date_sort = results.sort_values(by='Date', ascending=False)
results.to_csv('systematic_review.csv', index=False)
date_sort.head(25)
entities = ["cohort study", "retrospective cohort study", "retrospective analysis"]
content = [make_final_table(entity, keyword='study') for entity in entities]
results = pd.concat(make_results_df(data) for data in content)
date_sort = results.sort_values(by='Date', ascending=False)
results.to_csv('cohort_study.csv', index=False)
date_sort.head(25)
entities = ["retrospective study", "retrospective analysis"]
content = [make_final_table(entity, keyword='retrospective') for entity in entities]
results = pd.concat(make_results_df(data) for data in content)
date_sort = results.sort_values(by='Date', ascending=False)
results.to_csv('retrospective_study.csv', index=False)
date_sort.head(25)