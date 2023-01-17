import numpy as np

import pandas as pd

import json

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

import plotly.express as px
def get_metadata():

    with open('../input/arxiv/arxiv-metadata-oai-snapshot.json', 'r') as f:

        for line in f:

            yield line
metadata = get_metadata()
category_dict = {'astro-ph': 'Astrophysics',

'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',

'astro-ph.EP': 'Earth and Planetary Astrophysics',

'astro-ph.GA': 'Astrophysics of Galaxies',

'astro-ph.HE': 'High Energy Astrophysical Phenomena',

'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',

'astro-ph.SR': 'Solar and Stellar Astrophysics',

'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',

'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',

'cond-mat.mtrl-sci': 'Materials Science',

'cond-mat.other': 'Other Condensed Matter',

'cond-mat.quant-gas': 'Quantum Gases',

'cond-mat.soft': 'Soft Condensed Matter',

'cond-mat.stat-mech': 'Statistical Mechanics',

'cond-mat.str-el': 'Strongly Correlated Electrons',

'cond-mat.supr-con': 'Superconductivity',

'cs.AI': 'Artificial Intelligence',

'cs.AR': 'Hardware Architecture',

'cs.CC': 'Computational Complexity',

'cs.CE': 'Computational Engineering, Finance, and Science',

'cs.CG': 'Computational Geometry',

'cs.CL': 'Computation and Language',

'cs.CR': 'Cryptography and Security',

'cs.CV': 'Computer Vision and Pattern Recognition',

'cs.CY': 'Computers and Society',

'cs.DB': 'Databases',

'cs.DC': 'Distributed, Parallel, and Cluster Computing',

'cs.DL': 'Digital Libraries',

'cs.DM': 'Discrete Mathematics',

'cs.DS': 'Data Structures and Algorithms',

'cs.ET': 'Emerging Technologies',

'cs.FL': 'Formal Languages and Automata Theory',

'cs.GL': 'General Literature',

'cs.GR': 'Graphics',

'cs.GT': 'Computer Science and Game Theory',

'cs.HC': 'Human-Computer Interaction',

'cs.IR': 'Information Retrieval',

'cs.IT': 'Information Theory',

'cs.LG': 'Machine Learning',

'cs.LO': 'Logic in Computer Science',

'cs.MA': 'Multiagent Systems',

'cs.MM': 'Multimedia',

'cs.MS': 'Mathematical Software',

'cs.NA': 'Numerical Analysis',

'cs.NE': 'Neural and Evolutionary Computing',

'cs.NI': 'Networking and Internet Architecture',

'cs.OH': 'Other Computer Science',

'cs.OS': 'Operating Systems',

'cs.PF': 'Performance',

'cs.PL': 'Programming Languages',

'cs.RO': 'Robotics',

'cs.SC': 'Symbolic Computation',

'cs.SD': 'Sound',

'cs.SE': 'Software Engineering',

'cs.SI': 'Social and Information Networks',

'cs.SY': 'Systems and Control',

'econ.EM': 'Econometrics',

'eess.AS': 'Audio and Speech Processing',

'eess.IV': 'Image and Video Processing',

'eess.SP': 'Signal Processing',

'gr-qc': 'General Relativity and Quantum Cosmology',

'hep-ex': 'High Energy Physics - Experiment',

'hep-lat': 'High Energy Physics - Lattice',

'hep-ph': 'High Energy Physics - Phenomenology',

'hep-th': 'High Energy Physics - Theory',

'math.AC': 'Commutative Algebra',

'math.AG': 'Algebraic Geometry',

'math.AP': 'Analysis of PDEs',

'math.AT': 'Algebraic Topology',

'math.CA': 'Classical Analysis and ODEs',

'math.CO': 'Combinatorics',

'math.CT': 'Category Theory',

'math.CV': 'Complex Variables',

'math.DG': 'Differential Geometry',

'math.DS': 'Dynamical Systems',

'math.FA': 'Functional Analysis',

'math.GM': 'General Mathematics',

'math.GN': 'General Topology',

'math.GR': 'Group Theory',

'math.GT': 'Geometric Topology',

'math.HO': 'History and Overview',

'math.IT': 'Information Theory',

'math.KT': 'K-Theory and Homology',

'math.LO': 'Logic',

'math.MG': 'Metric Geometry',

'math.MP': 'Mathematical Physics',

'math.NA': 'Numerical Analysis',

'math.NT': 'Number Theory',

'math.OA': 'Operator Algebras',

'math.OC': 'Optimization and Control',

'math.PR': 'Probability',

'math.QA': 'Quantum Algebra',

'math.RA': 'Rings and Algebras',

'math.RT': 'Representation Theory',

'math.SG': 'Symplectic Geometry',

'math.SP': 'Spectral Theory',

'math.ST': 'Statistics Theory',

'math-ph': 'Mathematical Physics',

'nlin.AO': 'Adaptation and Self-Organizing Systems',

'nlin.CD': 'Chaotic Dynamics',

'nlin.CG': 'Cellular Automata and Lattice Gases',

'nlin.PS': 'Pattern Formation and Solitons',

'nlin.SI': 'Exactly Solvable and Integrable Systems',

'nucl-ex': 'Nuclear Experiment',

'nucl-th': 'Nuclear Theory',

'physics.acc-ph': 'Accelerator Physics',

'physics.ao-ph': 'Atmospheric and Oceanic Physics',

'physics.app-ph': 'Applied Physics',

'physics.atm-clus': 'Atomic and Molecular Clusters',

'physics.atom-ph': 'Atomic Physics',

'physics.bio-ph': 'Biological Physics',

'physics.chem-ph': 'Chemical Physics',

'physics.class-ph': 'Classical Physics',

'physics.comp-ph': 'Computational Physics',

'physics.data-an': 'Data Analysis, Statistics and Probability',

'physics.ed-ph': 'Physics Education',

'physics.flu-dyn': 'Fluid Dynamics',

'physics.gen-ph': 'General Physics',

'physics.geo-ph': 'Geophysics',

'physics.hist-ph': 'History and Philosophy of Physics',

'physics.ins-det': 'Instrumentation and Detectors',

'physics.med-ph': 'Medical Physics',

'physics.optics': 'Optics',

'physics.plasm-ph': 'Plasma Physics',

'physics.pop-ph': 'Popular Physics',

'physics.soc-ph': 'Physics and Society',

'physics.space-ph': 'Space Physics',

'q-bio.BM': 'Biomolecules',

'q-bio.CB': 'Cell Behavior',

'q-bio.GN': 'Genomics',

'q-bio.MN': 'Molecular Networks',

'q-bio.NC': 'Neurons and Cognition',

'q-bio.OT': 'Other Quantitative Biology',

'q-bio.PE': 'Populations and Evolution',

'q-bio.QM': 'Quantitative Methods',

'q-bio.SC': 'Subcellular Processes',

'q-bio.TO': 'Tissues and Organs',

'q-fin.CP': 'Computational Finance',

'q-fin.EC': 'Economics',

'q-fin.GN': 'General Finance',

'q-fin.MF': 'Mathematical Finance',

'q-fin.PM': 'Portfolio Management',

'q-fin.PR': 'Pricing of Securities',

'q-fin.RM': 'Risk Management',

'q-fin.ST': 'Statistical Finance',

'q-fin.TR': 'Trading and Market Microstructure',

'quant-ph': 'Quantum Physics',

'stat.AP': 'Applications',

'stat.CO': 'Computation',

'stat.ME': 'Methodology',

'stat.ML': 'Machine Learning',

'stat.OT': 'Other Statistics',

'stat.TH': 'Statistics Theory'}
titles_tags_dict = {"title":[], "tags":[]}

for paper in metadata:

    parsed = json.loads(paper)

    titles_tags_dict["title"].append(parsed['title'])

    titles_tags_dict["tags"].append(parsed['categories'])
titles_tags_df = pd.DataFrame.from_dict(titles_tags_dict)
subset_categories = {'astro-ph.EP': 'Earth and Planetary Astrophysics',

'cond-mat.mtrl-sci': 'Materials Science',

'cs.AI': 'Artificial Intelligence',

'cs.DB': 'Databases'}
astro_ph_df = titles_tags_df[titles_tags_df["tags"] == "astro-ph.EP"]

mtrl_sci_df = titles_tags_df[titles_tags_df["tags"] == "cond-mat.mtrl-sci"]

ai_df = titles_tags_df[titles_tags_df["tags"] == "cs.AI"]

db_df = titles_tags_df[titles_tags_df["tags"] == "cs.DB"]
stop = stopwords.words('english')

def get_wc_df(df):

    data = df.copy(deep=True)

    data['title'] = data['title'].apply(lambda x: x.lower())

    data['title'] = data["title"].apply(lambda x: x.split(' '))

    data["title"] = data["title"].apply(lambda x: [item for item in x if item not in stop])

    data['title'] = data['title'].apply(lambda x: ' '.join(x))

    vectorizer = CountVectorizer()

    wc_mat = vectorizer.fit_transform(list(data['title']))

    return pd.DataFrame({"word":vectorizer.get_feature_names(), "count":list(np.sum(wc_mat.toarray(), axis=0))}).sort_values(by="count", ascending=False).reset_index(drop=True)
top_astro = get_wc_df(astro_ph_df)[:10]

top_mtrl_sci = get_wc_df(mtrl_sci_df)[:10]

top_ai = get_wc_df(ai_df)[:10]

top_db = get_wc_df(db_df)[:10]
px.bar(data_frame=top_astro, x = "word", y = "count", title="Most Common Title Words: Earth and Planetary Astrophysics", color='count')
px.bar(data_frame=top_mtrl_sci, x = "word", y = "count", title="Most Common Title Words: Material Science", color='count')
px.bar(data_frame=top_ai, x = "word", y = "count", title="Most Common Title Words: AI", color='count')
px.bar(data_frame=top_db, x = "word", y = "count", title = "Most Common Title Words: Databases", color='count')