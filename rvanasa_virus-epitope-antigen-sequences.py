import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dfa = pd.read_csv('/kaggle/input/epitopes-antigens/iedb_antigens.csv')
dfa
dfe = pd.read_csv('/kaggle/input/epitopes-antigens/iedb_epitopes.csv')

# Show columns
print(list(dfe))

aid_prefix = 'http://www.ncbi.nlm.nih.gov/protein/'

# Extract AID (Antigen ID) from database URL
dfe['AID'] = dfe['Antigen IRI'].map(lambda u: u[len(aid_prefix):] if str(u).startswith(aid_prefix) else None)

# Drop epitopes with unresolved antigens
dfe_na = dfe
dfe = dfe_na[['Object Type', 'Description', 'AID', 'Antigen Name']].dropna()

print(len(dfe_na), '/', len(dfe))

dfe
dfv = pd.read_csv('/kaggle/input/epitopes-antigens/ncbi_virus_protein_seqs.csv')
dfv
# Associate epitope and antigen sequences
df = dfe.merge(dfv, on='AID')
df
amino_acids = 'ARNDCEQGHILKMFPSTWYVX'

def normalize_seq(s):
    s = s.upper()
    if any(c not in amino_acids for c in s):
        return
    return s

# Focus on linear peptides
df = df[df['Object Type'] == 'Linear peptide']
df = df.copy()

df['Epitope Sequence'] = df['Description'].map(normalize_seq)
df['Antigen Sequence'] = df['Sequence'].map(normalize_seq)
df = df.dropna()

df[['Epitope Sequence', 'Antigen Name', 'Antigen Sequence']]
# Access pre-normalized data
dfn = pd.read_csv('/kaggle/input/epitopes-antigens/virus_normalized_seqs.csv')
dfn