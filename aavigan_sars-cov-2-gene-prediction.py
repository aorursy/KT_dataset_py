import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from Bio import SeqIO

from Bio.Alphabet import IUPAC



#read in SARS-COV-1 DNA sequence

DNAseq = SeqIO.read('../input/coronavirus-genome-sequence/MN908947.fna', "fasta", alphabet = IUPAC.unambiguous_dna)

DNA = DNAseq.seq

#transcribe sequence to rna

mRNA = DNA.transcribe()



#list to store open reading frames

Frames = []



#loop through all three possible reading frames

for frame in range(3):

    #ensures reading frame is multiple of 3

    length = 3 * ((len(mRNA)-frame) // 3)

    #translate reading frame into amino acid sequence and split by stop codons to find all possible open reading frames

    ORFs = mRNA[frame: frame+length].translate().split('*')

    #append the set of open reading frames for each reading frame to list Frames

    Frames.append(ORFs)    



#list to store dfs for concatenation

COV19_dfs = []



#loop through three possible reading frames

for frame in range(3):

    #create df having for storing amino acid sequences for each ORF

    df = pd.DataFrame(Frames[frame], columns= ['Amino Acid Sequence'])

    #create sequence length column

    df['length'] = df['Amino Acid Sequence'].apply(len)

    #create columns for ORF start and stop positions 

    df['reading_frame_start_pos'] = frame + 1

    df['reading_frame_end_pos'] = 0

    #reading frame column

    df['frame'] = frame

    #calculate nucleotide start position for each ORF (not zero indexed)

    for i in range(1,len(df)+1):

        df['reading_frame_start_pos'].at[i] = df.iloc[i-1]['reading_frame_start_pos'] + 3*(df.iloc[i-1]['length']+1)

    #calculate nucleotide end position for each ORF

    for i in range(0,len(df)):

        df['reading_frame_end_pos'].at[i]=  df.loc[i]['reading_frame_start_pos'] + 3*(df.loc[i]['length']+1)-1

    #append df to list

    COV19_dfs.append(df)

#concatenate dfs in list to single df

COV19_df = pd.concat(COV19_dfs, ignore_index = True)

#filter to drop amino acid sequences less than 39 which is the length of the shortest SARS-2 protein

COV19_df = COV19_df.loc[COV19_df.length >= 39]

#reset index

COV19_df.reset_index(drop =True, inplace =True)



COV19_df.head()
#read in SARS-1 proteome

SARS_Proteome = SeqIO.parse('../input/human-sars-coronavirus-sarscov-proteome/uniprot-proteome_UP000000354.fasta', "fasta")

SARS_Proteome = [p for p in SARS_Proteome]

# list of protein sequences

sequence = [p.seq for p in SARS_Proteome]

#list of protein ids

identity = [p.id for p in SARS_Proteome]

#instantiate dataframe indexed by protein for storing protein amino acid sequences

SARS_df = pd.DataFrame({'Amino Acid Sequence':sequence}, index = identity)

#create columns for later use

SARS_df['best_match'] = 0

SARS_df['match_score'] = 0.0

SARS_df['offset'] = 0

#calcualte protein amino acid sequence lengths

SARS_df['length'] = SARS_df['Amino Acid Sequence'].apply(len)





from Bio import pairwise2



#iterate over SARS-1 proteins

for i, r in SARS_df.iterrows():

    #dictionaries for storing match scores and alignment offsets

    scores = {}

    offsets = {}

    #for each SARS-1 protein iterate over all SARS-2 open reading frames

    for index, row in COV19_df.iterrows():

        #calculate local pairwise alignment between protein and ORF amino acid sequence

        alignment = pairwise2.align.localxx(r['Amino Acid Sequence'], row['Amino Acid Sequence'], one_alignment_only = True)

        #calculate match score and store in dictionary

        scores[index]= alignment[0][2]/(alignment[0][4]-alignment[0][3])

        #store offset in dictionary

        offsets[index] = alignment[0][3]

    #retrieve index of best match score and store value in best_match column

    SARS_df.at[i, 'best_match'] = max(scores, key = scores.get) 

    #store best match score in match_score column

    SARS_df.at[i,'match_score'] = max([val for k, val in scores.items()])

    #store offset in offset column

    SARS_df.at[i, 'offset'] = offsets[max(scores, key = scores.get)]

#sort dataframe by match_score in descending order

SARS_df.sort_values(by = 'match_score',ascending = False)

alignment = pairwise2.align.localxx(SARS_df.loc['sp|P59637|VEMP_CVHSA', 'Amino Acid Sequence'], COV19_df.loc[8]['Amino Acid Sequence'],  penalize_extend_when_opening= True)

print(pairwise2.format_alignment(*alignment[0]))

from Bio.SeqRecord import SeqRecord

from Bio.SeqFeature import SeqFeature, FeatureLocation



#list for storing sequence feature

features = []

#iterate over SARS_df by index and row

for i, r in SARS_df.iterrows():

    #calculate the nucleotide start position for the region of best match in the SARS-2 genome

    start = (COV19_df.loc[r['best_match']]['reading_frame_start_pos']-1) + r['offset']*3

    #calcualte the nucleotide end position for the region of best match in teh SARS-2 genome

    end = COV19_df.loc[r['best_match']]['reading_frame_end_pos']

    #create feature

    feature = SeqFeature(FeatureLocation(int(start), int(end)), type="CDS", id = 'homolog-'+i, qualifiers = {'match_score': r['match_score'], 'homolog':i} )

    #append to feature list

    features.append(feature)

#create sequenc record with features

record = SeqRecord(DNA, id = 'MN908947', name = 'SARS-CoV2', description = 'potential homologous genes to SARS-CoV1', features = features)

#write to genbank file for later use

SeqIO.write(record, "cov2_homologs.gb", "genbank")