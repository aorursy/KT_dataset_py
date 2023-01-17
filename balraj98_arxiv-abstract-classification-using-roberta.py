import numpy as np
import pandas as pd
import os, json, gc, re, random
from tqdm.notebook import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
%%time

!pip uninstall -q torch -y > /dev/null
!pip install -q torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html > /dev/null
!pip install -q -U transformers > /dev/null
!pip install -q -U simpletransformers > /dev/null
import torch, transformers, tokenizers
torch.__version__, transformers.__version__, tokenizers.__version__
data_file = '../input/arxiv/arxiv-metadata-oai-snapshot-2020-08-14.json'

""" Using `yield` to load the JSON file in a loop to prevent Python memory issues if JSON is loaded directly"""

def get_metadata():
    with open(data_file, 'r') as f:
        for line in f:
            yield line
metadata = get_metadata()
for paper in metadata:
    for k, v in json.loads(paper).items():
        print(f'{k}: {v} \n')
    break
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
%%time

titles = []
abstracts = []
categories = []

# Consider all categories in the `category_map` to be used during training and prediction
paper_categories = np.array(list(category_map.keys())).flatten()

# Uncomment & edit below line to use specific paper categories to be used during training and prediction
# paper_categories = ["astro-ph", # Astrophysics
#                     "cs.CV", # Computer Vision and Pattern Recognition
#                     'q-fin.EC'] # Economics

metadata = get_metadata()
for paper in tqdm(metadata):
    paper_dict = json.loads(paper)
    category = paper_dict.get('categories')
    try:
        try:
            year = int(paper_dict.get('journal-ref')[-4:])    ### Example Format: "Phys.Rev.D76:013009,2007"
        except:
            year = int(paper_dict.get('journal-ref')[-5:-1])    ### Example Format: "Phys.Rev.D76:013009,(2007)"

        if category in paper_categories and 2010<year<2021:
            titles.append(paper_dict.get('title'))
            abstracts.append(paper_dict.get('abstract'))
            categories.append(paper_dict.get('categories'))
    except:
        pass 

len(titles), len(abstracts), len(categories)
%%time

papers = pd.DataFrame({
    'title': titles,
    'abstract': abstracts,
    'categories': categories
})

papers['abstract'] = papers['abstract'].apply(lambda x: x.replace("\n",""))
papers['abstract'] = papers['abstract'].apply(lambda x: x.strip())
papers['text'] = papers['title'] + '. ' + papers['abstract']

papers['categories'] = papers['categories'].apply(lambda x: tuple(x.split()))

# Choosing paper categories based on their frequency & eliminating categories with very few papers
shortlisted_categories = papers['categories'].value_counts().reset_index(name="count").query("count > 250")["index"].tolist()
papers = papers[papers["categories"].isin(shortlisted_categories)].reset_index(drop=True)

# Shuffle DataFrame
papers = papers.sample(frac=1).reset_index(drop=True)

# Sample roughtly equal number of texts from different paper categories (to reduce class imbalance issues)
papers = papers.groupby('categories').head(250).reset_index(drop=True)

multi_label_encoder = MultiLabelBinarizer()
multi_label_encoder.fit(papers['categories'])
papers['categories_encoded'] = papers['categories'].apply(lambda x: multi_label_encoder.transform([x])[0])

papers = papers[["text", "categories", "categories_encoded"]]
del titles, abstracts, categories
papers
%%time

from simpletransformers.classification import MultiLabelClassificationModel

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "save_model_every_epoch": False, # Model occupies 1.4GB size per epoch (Total Disk Space Available: 4.9GB)
    "save_eval_checkpoints": False,
    "max_seq_length": 512,
    "train_batch_size": 16,
    "num_train_epochs": 4,
}

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel('roberta', 
                                      'roberta-base', 
                                      num_labels=len(shortlisted_categories), 
                                      args=model_args)
%%time

train_df, eval_df = train_test_split(papers, test_size=0.1, stratify=papers["categories"], random_state=42)

# Train the model
model.train_model(train_df[["text", "categories_encoded"]])

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df[["text", "categories_encoded"]])
print(result)
predicted_categories_encoded = list(map(lambda x: np.argmax(x), model_outputs))
eval_gt_labels = eval_df["categories_encoded"].apply(lambda x: np.argmax(x)).tolist()

plt.figure(figsize=(22,18))
cf_matrix = confusion_matrix(predicted_categories_encoded, eval_gt_labels)
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix))
ax.set_xlabel('Predicted labels', fontsize=16)
ax.set_ylabel('True labels', fontsize=16)
ax.set_title('Confusion Matrix', fontsize=20)

plt.show()
predicted_categories_argmax = list(map(lambda x: np.argmax(x), model_outputs))
# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
predicted_categories_encoded = np.eye(len(shortlisted_categories))[predicted_categories_argmax]
predicted_categories = multi_label_encoder.inverse_transform(predicted_categories_encoded)

eval_gt_labels = eval_df["categories"].tolist()

shortlisted_categories_formatted = list(map(lambda x: list(x)[0], shortlisted_categories))

plt.figure(figsize=(22,18))
cf_matrix = confusion_matrix(predicted_categories, eval_gt_labels, shortlisted_categories_formatted)
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix))
ax.set_xlabel('Predicted labels', fontsize=18)
ax.set_ylabel('True labels', fontsize=18)
ax.set_title('Confusion Matrix', fontsize=20)
ax.set_xticklabels(shortlisted_categories_formatted, rotation=90, fontsize=16)
ax.set_yticklabels(shortlisted_categories_formatted, rotation=0, fontsize=16)

plt.show()
for _ in range(50):

    random_idx = random.randint(0, len(eval_df)-1)
    text = eval_df.iloc[random_idx]['text']
    true_categories = eval_df.iloc[random_idx]['categories']

    # Predict with trained multilabel classification model
    predicted_categories_encoded, raw_outputs = model.predict([text])
    predicted_categories_encoded = np.array(predicted_categories_encoded)
    predicted_categories_encoded[0][np.argmax(raw_outputs[0])] = 1
    predicted_categories = multi_label_encoder.inverse_transform(predicted_categories_encoded)[0]

    print(f'True Categories:'.ljust(21,' '), f'{true_categories} - {category_map[true_categories[0]]}\n')
    print(f'Predicted Categories: {predicted_categories} - {category_map[predicted_categories[0]]}\n')
    print(f'Abstract: {text}\n\n')