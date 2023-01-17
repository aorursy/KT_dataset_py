%reload_ext autoreload

%autoreload 2

%matplotlib inline
!pip install git+https://github.com/amrmkayid/recovid.git

# !pip uninstall recovid -y
#@title Libraries & Dependencies



from recovid import *

import recovid.data as data

import recovid.process as process

import recovid.search as search

import recovid.utils as utils
#@title Configuration



plt.style.use("ggplot")

plt.rcParams["figure.figsize"] = (25, 10)

warnings.filterwarnings("ignore")



nltk.download("punkt")

nltk.download("stopwords")

nltk.download('wordnet')



ROOT_PATH = Path("/kaggle/input/CORD-19-research-challenge/")

METADATA_PATH = ROOT_PATH / "metadata.csv"

ROOT_PATH, METADATA_PATH
metadata = pd.read_csv(

    METADATA_PATH,

    dtype={

        "doi": str,

        "title": str,

        "pubmed_id": str,

        "Microsoft Academic Paper ID": str,

    },

)

metadata.head(3)
metadata.info()
# Visualizing null values in each column

metadata.isna().sum().plot(kind="bar", stacked=True)
metadata.isna().sum()
# Distribution of title length



sns.distplot(metadata["title"].str.len())

plt.title("Distribution of title length")

plt.show()
#@title Visualizing Most Common Words from Title



utils.most_common_words_from_title(metadata)
#@title Visualizing Most Common Journals



utils.most_common_journals(metadata)
# Set the abstract to the paper title if it is null



metadata.abstract = metadata.abstract.fillna(metadata.title)

print("Number of articles before removing duplicates: %s " % len(metadata))



duplicate_paper = ~(metadata.title.isnull() | metadata.abstract.isnull() | metadata.publish_time.isnull()) & (metadata.duplicated(subset=['title', 'abstract']))

metadata.dropna(subset=['publish_time', 'journal'])

metadata = metadata[~duplicate_paper].reset_index(drop=True)

print("Number of articles AFTER removing duplicates: %s " % len(metadata))
papers = data.ResearchPapers(metadata)
paper = papers[0]

print(f'Example paper \n\nTitle: {paper.title()} \n\nAuthors: {paper.authors(split=True)} \n\nAbstract: {paper.abstract()} \n\n')
# Summary for a single paper

paper
paper.html()
display(HTML(paper.text()))
search_engine = search.SearchEngine(metadata)

search_engine
keywords = 'virus pandemic' #@param {type:"string"}

results = search_engine.search(keywords, 50)

results.results.sort_values(by=['publish_time'], ascending=False).head(5)
search_terms = 'virus pandemic' #@param {type:"string"}

searchbar = widgets.interactive(lambda search_terms: search.search_papers(search_engine, search_terms), search_terms=search_terms)

searchbar
tasks = [

    ('What is known about transmission, incubation, and environmental stability?',

     'transmission incubation environment coronavirus'),

    ('What do we know about COVID-19 risk factors?', 'risk factors'),

    ('What do we know about virus genetics, origin, and evolution?',

     'genetics origin evolution'),

    ('What has been published about ethical and social science considerations',

     'ethics ethical social'),

    ('What do we know about diagnostics and surveillance?',

     'diagnose diagnostic surveillance'),

    ('What has been published about medical care?', 'medical care'),

    ('What do we know about vaccines and therapeutics?',

     'vaccines vaccine vaccinate therapeutic therapeutics')

]

tasks = pd.DataFrame(tasks, columns=['Task', 'Keywords'])
results = interact(lambda task: search.show_task(search_engine, tasks, task), task=tasks.Task.tolist());