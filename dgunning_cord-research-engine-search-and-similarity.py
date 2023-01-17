!pip install -U git+https://github.com/dgunning/cord19.git
from cord import ResearchPapers



research_papers = ResearchPapers.load()
research_papers.search('antiviral treatment')
research_papers.searchbar('antiviral treatment')
since_covid = research_papers.since_sarscov2()

since_covid
research_papers.match('H[0-9]N[0-9]')
research_papers.contains("Fauci", column='authors').since_sarscov2()
paper = research_papers[197]

paper
paper = research_papers['asf5c7xu'] # or research_papers['5c31897d01f3edc7f58a0f03fceec2373fcfdc3d']

paper
research_papers.query("sha=='5c31897d01f3edc7f58a0f03fceec2373fcfdc3d'")
from ipywidgets import interact

from IPython.display import display



paper = research_papers['asf5c7xu']



def view_paper(ViewPaperAs):

    if ViewPaperAs == 'Overview':

        display(paper)

    elif ViewPaperAs == 'Abstract':

        display(paper.abstract)

    elif ViewPaperAs == 'Summary of Abstract':

        display(paper.summary)

    elif ViewPaperAs == 'HTML':

        display(paper.html)

    elif ViewPaperAs == 'Text':

        display(paper.text)

    elif ViewPaperAs == 'Summary of Text':

        display(paper.text_summary)

    

interact(view_paper,

         ViewPaperAs=['Overview', # Show an overview of the paper's important fields and statistics

                      'Abstract', # Show the paper's abstract

                      'Summary of Abstract', # Show a summary of the paper's abstract

                      'HTML', # Show the paper's contents as (slightly) formatted HTML

                      'Text', # Show the paper's contents

                      'Summary of Text' # Show a summary of the paper's content

                     ]

        );
research_papers.similar_to('asf5c7xu')
import pandas as pd

import numpy as np

import ipywidgets as widgets

from cord.core import cord_support_dir



# Load the document vectors

vectors = pd.read_parquet(cord_support_dir()/ 'DocumentVectors.pq').reset_index()

display(widgets.HTML('<h4>Document Vectors</h4>'))

display(vectors.head())



# Use the metadata of the research_papers

metadata = research_papers.metadata.copy()



vector_metadata_merge = vectors.merge(metadata, on='cord_uid', how='right')

vectors['covid_related'] = vector_metadata_merge.covid_related

vectors['published'] = vector_metadata_merge.published

display(widgets.HTML('<h4>Document Vectors with covid_related and published</h4>'))

display(vectors.head())
import altair as alt





vector_df = pd.DataFrame({'x':vectors.x,

                          'y': vectors.y,

                          '1d': vectors['1d'],

                          'cluster': vectors.cluster,

                          'covid_related': vectors.covid_related,

                          'published': vectors.published})  # Ensure 5000 limit



alt.Chart(vector_df.sample(5000)).mark_point().encode(

       x=alt.X('x', axis=None),

       y=alt.Y('y', axis=None),

       color= 'cluster:N'

    ).properties(

        title='CORD Research Papers in 2D space',

        width=600,

        height=400

    ).configure_axis(

        grid=False

    ).configure_view(

        strokeWidth=0

    )
import matplotlib.style as style

style.use('fivethirtyeight')

covid_count = vector_df[['cluster', 'covid_related']].groupby(['cluster']).sum() 

cluster_count = vector_df[['cluster', 'covid_related']].groupby(['cluster']).count() 

covid_cluster_stats = (covid_count / cluster_count) * 100

covid_cluster_stats = covid_cluster_stats.sort_values(['covid_related'])

fig = covid_cluster_stats.plot.barh(grid=False, figsize=(6, 3), legend=False, title='% Covid Related');
vector_since = vector_df.query('published > "2015-01-01" & (cluster==2 | cluster==6)').copy()

vector_since.loc[vector_since.published> '2020-06-30', 'published'] = pd.to_datetime('2020-03-30')

if len(vector_since) > 5000:

    vector_since = vector_since.sample(5000)

alt.Chart(vector_since).mark_point().encode(

       x=alt.X('published:T'),

       y=alt.Y('1d'),

       color= 'cluster:N'

    ).properties(

        title='CORD Research Papers since 2015',

        width=600,

        height=400

    ).configure_axis(

        grid=False

    ).configure_view(

        strokeWidth=0

    )
query ="""

Efforts to identify the underlying drivers of fear, anxiety and stigma that

fuel misinformation and rumor, particularly through social media.

"""

research_papers.search_2d(query)
import pandas as pd

import numpy as np

from rank_bm25 import BM25Okapi

pd.options.display.max_colwidth=160



meta_df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv') # Or pd.read_csv()

meta_df = meta_df[['title', 'abstract', 'publish_time']].head(1000)

meta_df
from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import preprocess_documents, preprocess_string



meta_df_tokens = meta_df.abstract.fillna('').apply(preprocess_string)
from rank_bm25 import BM25Okapi

import numpy as np



bm25_index = BM25Okapi(meta_df_tokens.tolist())



def search(search_string, num_results=10):

    search_tokens = preprocess_string(search_string)

    scores = bm25_index.get_scores(search_tokens)

    top_indexes = np.argsort(scores)[::-1][:num_results]

    return top_indexes



indexes = search('novel coronavirus treatment')

indexes
meta_df.loc[indexes, ['abstract', 'publish_time']]
meta_df.loc[search('novel coronavirus treatment')]
import multiprocessing

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Collection, Any



def is_notebook():

    try:

        from IPython import get_ipython

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"

    except (NameError, ImportError):

        return False



if is_notebook():

    from tqdm.notebook import tqdm

else:

    from tqdm import tqdm

    

def ifnone(a: Any, b: Any) -> Any:

    return b if a is None else a



def parallel(func, arr: Collection, max_workers: int = None):

    "Call `func` on every element of `arr` in parallel using `max_workers`."

    max_workers = ifnone(max_workers, multiprocessing.cpu_count())

    progress_bar = tqdm(arr)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:

        futures_to_index = {ex.submit(func, o): i for i, o in enumerate(arr)}

        results = []

        for f in as_completed(futures_to_index):

            results.append((futures_to_index[f], f.result()))

            progress_bar.update()

        for n in range(progress_bar.n, progress_bar.total):

            time.sleep(0.1)

            progress_bar.update()

        results.sort(key=lambda x: x[0])

    return [result for i, result in results]
from functools import partial



def get_text(paper_json, text_key) -> str:

    """

    :param paper_json: The json

    :param text_key: the text_key - "body_text" or "abstract"

    :return: a text string with the sections

    """

    body_dict = collections.defaultdict(list)

    for rec in paper_json[text_key]:

        body_dict[rec['section']].append(rec['text'])



    body = ''

    for section, text_sections in body_dict.items():

        body += section + '\n\n'

        for text in text_sections:

            body += text + '\n\n'

    return body





get_body = partial(get_text, text_key='body_text')

get_abstract = partial(get_text, text_key='abstract')



def author_name(author_json):

    first = author_json.get('first')

    middle = "".join(author_json.get('middle'))

    last = author_json.get('last')

    if middle:

        return ' '.join([first, middle, last])

    return ' '.join([first, last])





def get_affiliation(author_json):

    affiliation = author_json['affiliation']

    institution = affiliation.get('institution', '')

    location = affiliation.get('location')

    if location:

        location = ' '.join(location.values())

    return f'{institution}, {location}'





def get_authors(paper_json, include_affiliation=False):

    if include_affiliation:

        return [f'{author_name(a)}, {get_affiliation(a)}'

                for a in paper_json['metadata']['authors']]

    else:

        return [author_name(a) for a in paper_json['metadata']['authors']]
research_papers.display('dao10kx9', 'rjc3b4br',  'r0lduvs1', '7i422cht', 'pa9h6d0a', 'dbzrd23n', '5gbkrs73', '94tdt2rv', 

                        'xsgxd5sy', 'jf36as70', 'uz91cd6h')
from typing import Dict, List

import requests



SPECTER_URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"

MAX_BATCH_SIZE = 16



def chunks(lst, chunk_size=MAX_BATCH_SIZE):

    """Splits a longer list to respect batch size"""

    for i in range(0, len(lst), chunk_size):

        yield lst[i: i + chunk_size]





def get_embeddings_for_papers(papers: List[Dict[str, str]]):

    embeddings_by_paper_id: Dict[str, List[float]] = {}

    for chunk in chunks(papers):

        # Allow Python requests to convert the data above to JSON

        response = requests.post(SPECTER_URL, json=chunk)



        if response.status_code != 200:

            print("Something went wrong on the spector API side .. try again")

            return None



        for paper in response.json()["preds"]:

            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]



    return embeddings_by_paper_id



def get_embeddings(title: str, abstract: str = None):

    abstract = abstract or title

    paper = {"paper_id": "paper", "title": title, "abstract": abstract}

    embeddings = get_embeddings_for_papers([paper])

    return embeddings['paper'] if embeddings else None



def plot_embeddings(vector):

    df = pd.DataFrame(vector)

    ax = df.plot.bar(figsize=(10,1))

    ax.get_legend().remove()

    ax.axes.set_xticklabels([])

    ax.axes.set_yticklabels([])
embeddings = get_embeddings('Animal to human transmission')



embeddings[:10]
plot_embeddings(embeddings)
plot_embeddings(get_embeddings('Animal to human viral transmission'))

plot_embeddings(get_embeddings('Bat to human viral transmission'))