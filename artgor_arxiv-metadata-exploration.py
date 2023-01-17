# import libraries



import numpy as np

import pandas as pd

import gc

import os

import json

from collections import Counter, defaultdict

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import re

year_pattern = r'([1-2][0-9]{3})'
with open('/kaggle/input/arxiv/authors-parsed.json', 'r') as f:

    authors = json.load(f)

# with open('/kaggle/input/arxiv/internal-citations.json', 'r') as f:

#     citations = json.load(f)
"""

Json files in the dataset are huge. Sadly, python has certain memory problems when loading huge json files.

As a result, I read this file using `yield` and get necessary information from in in the loop.

"""



def get_metadata():

    with open('/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json', 'r') as f:

        for line in f:

            yield line
metadata = get_metadata()

for paper in metadata:

    for k, v in json.loads(paper).items():

        print(f'{k}: {v}')

    break
# https://arxiv.org/help/api/user-manual

category_map = {'astro-ph': 'Astrophysics',

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
year_categories = {}

year_abstract_words = {}

year_authors = {}

metadata = get_metadata()

for ind, paper in tqdm(enumerate(metadata)):

    paper = json.loads(paper)

    

    # try to extract year

    if paper['journal-ref']:

        year = re.match(year_pattern, paper['journal-ref']).groups() if re.match(year_pattern, paper['journal-ref']) else None

        if year:

            year = [int(i) for i in year if int(i) < 2020 and int(i) >= 1991]

            if year == []:

                year = None

            else:

                year = min(year)

    else:

        year = None

                    

    if year:   

        if year not in year_categories.keys():

            year_categories[year] = defaultdict(int)

            year_abstract_words[year] = defaultdict(int)

            year_authors[year] = defaultdict(int)

    # collect counts of various things over years

    for cat in paper['categories']:

        for c in cat.split():

            if year:

                year_categories[year][c] += 1

    for word in paper['abstract'].replace('\n', ' ').split():

        if year:

            year_abstract_words[year][word] += 1

    paper_authors = authors.get(paper['id'])

    if paper_authors:

        if year:

            for author in paper_authors:

                year_authors[year][' '.join(author)] += 1
df = pd.DataFrame(year_categories)

cats = []

for col in df.columns:

    top_cats = [i for i in df[col].fillna(0).sort_values().index][-10:]

    cats.extend(top_cats)

cats = list(set(cats))



df1 = df.T[cats]

df1 = df1.sort_index()

df2 = df1.reset_index().melt(id_vars=['index'])

df2.columns = ['year', 'category', 'count']

fig = px.line(df2, x="year", y="count", color='category')

fig.show()
for c in sorted(cats):

    if c in category_map:

        print(f"{c}: {category_map[c]}")
df = pd.DataFrame(year_authors)

authors = []

for col in df.columns:

    top_authors = [i for i in df[col].fillna(0).sort_values().index][-10:]

    authors.extend(top_authors)

authors = list(set(authors))



df1 = df.T[authors]

df1 = df1.sort_index()

df2 = df1.reset_index().melt(id_vars=['index'])

df2.columns = ['year', 'author', 'count']

fig = px.line(df2, x="year", y="count", color='author', width=1600, height=600)

fig.show()