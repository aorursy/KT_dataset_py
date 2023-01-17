# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install biopython
!pip install squiggle
import numpy as np
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
from Bio import SeqIO
for sequence in SeqIO.parse('../input/coronavirus-genome-sequence/MN908947.fna', "fasta"):
    print(sequence.seq)

print(len(sequence),'nucliotides')
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
DNAsequence = SeqIO.read('../input/coronavirus-genome-sequence/MN908947.fna', "fasta")
print(DNAsequence)
DNAfile = open("../input/coronavirus-genome-sequence/MN908947.txt" , 'r')
file_contents = DNAfile.read()
print(file_contents)
#reading the pdffile
!pip install PyPDF2
import PyPDF2
pdffile = open('../input/coronavirus-genome-sequence/A_new_coronavirus_associated_with_human_respirator.pdf' , 'rb')
pdfreader = PyPDF2.PdfFileReader(pdffile)
print(pdfreader.numPages)
pageobj = pdfreader.getPage(0)
print(pageobj.extractText())
pdffile.close()
DNA = DNAsequence.seq
#Convert DNA into mRNA Sequence
mRNA = DNA.transcribe() #Transcribe a DNA sequence into RNA.
print(mRNA)
print(len(mRNA) , 'nucleotides')
Amino_Acid = mRNA.translate(table=1, cds=False)
print('Amino Acid', Amino_Acid)

print("Length of Protein:",len(Amino_Acid))
print("Length of Original mRNA:",len(mRNA))
import re
delimeter = '*'
splited_amino_acid = Amino_Acid.split()
print(splited_amino_acid)
from Bio.Data import CodonTable
print(CodonTable.unambiguous_rna_by_name['Standard'])
#Identify all the Proteins (chains of amino acids)
Proteins = Amino_Acid.split('*') # * is translated stop codon
df = pd.DataFrame(Proteins)
df.describe()
print('Total proteins:', len(df))

print(df)
def conv(item):
    return len(item)
def to_str(item):
    return str(item)
df['sequence_str'] = df[0].apply(to_str)
df['length'] = df[0].apply(conv)
df.rename(columns={0: "sequence"}, inplace=True)
df.head()
# Take only longer than 20
functional_proteins = df.loc[df['length'] >= 20]
print('Total functional proteins:', len(functional_proteins))
functional_proteins.describe()

df
print(df['length'].max())
print(df['length'].min())
print(df['sequence_str'])
from __future__ import division
poi_list = []
MW_list = []
from Bio.SeqUtils import ProtParam
for record in Proteins[:]:
    print("\n")
    X = ProtParam.ProteinAnalysis(str(record))
    POI = X.count_amino_acids()
    poi_list.append(POI)
    MW = X.molecular_weight()
    MW_list.append(MW)
    print("Protein of Interest = ", POI)
    print("Amino acids percent =    ",str(X.get_amino_acids_percent()))
    print("Molecular weight = ", MW_list)
    print("Aromaticity = ", X.aromaticity())
    print("Flexibility = ", X.flexibility())
    print("Isoelectric point = ", X.isoelectric_point())
    print("Secondary structure fraction = ",   X.secondary_structure_fraction())
MoW = pd.DataFrame(data = MW_list,columns = ["Molecular Weights"] )
#plot POI
poi_list = poi_list[10]
                    
plt.figure(figsize=(10,6));
plt.bar(poi_list.keys(), list(poi_list.values()), align='center')
