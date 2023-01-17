import gensim
import copy
import os
import sys
import glob

!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

FILE=open('../input/drugdata/fda.csv','r')
exist={}
for line in FILE:
    line=line.strip()
    line=line.lower()
    exist[line]=1
FILE.close()

all_drugs=exist.keys()

all_files=glob.glob(f'{root_path}/comm_use_subset/**/*.json', recursive=True)
print(len(all_files))

NEW=open(('result.comm'),'w')
i=0
for the_drug in all_drugs:
    os.system("find /kaggle/input/CORD-19-research-challenge/comm_use_subset/*  -type f | xargs grep -i '"+the_drug+"' >aaa.txt")
#    print("find ../../data/comm_use_subset/*  -type f | xargs grep -i '"+the_drug+"' >aaa.txt")
    FILE=open('aaa.txt','r')
    for line in FILE:
        NEW.write(the_drug)
        NEW.write('\t')
        table=line.split('json')
        NEW.write(table[0])
        NEW.write('json')
        NEW.write('\t')
        NEW.write(line)
    FILE.close()
    i=i+1
    if (i%10==0):
        print ('finished ', i)
        break
#    print ('finished '+the_drug)


all_files=glob.glob(f'{root_path}/noncomm_use_subset/**/*.json', recursive=True)
print(len(all_files))
NEW=open(('result.noncomm'),'w')
for the_drug in all_drugs:
    os.system("find /kaggle/input/CORD-19-research-challenge/noncomm_use_subset/*  -type f | xargs grep -i '"+the_drug+"' >aaa.txt")
#    print("find ../../data/comm_use_subset/*  -type f | xargs grep -i '"+the_drug+"' >aaa.txt")
    FILE=open('aaa.txt','r')
    for line in FILE:
        NEW.write(the_drug)
        NEW.write('\t')
        table=line.split('json')
        NEW.write(table[0])
        NEW.write('json')
        NEW.write('\t')
        NEW.write(line)
    FILE.close()
    i=i+1
    if (i%10==0):
        print ('finished ', i)
        break
#    print ('finished '+the_drug)



all_files=glob.glob(f'{root_path}/custom_license/**/*.json', recursive=True)
print(len(all_files))
NEW=open(('result.pmc'),'w')
for the_drug in all_drugs:
    os.system("find /kaggle/input/CORD-19-research-challenge/custom_license/*  -type f | xargs grep -i '"+the_drug+"' >aaa.txt")
#    print("find ../../data/comm_use_subset/*  -type f | xargs grep -i '"+the_drug+"' >aaa.txt")
    FILE=open('aaa.txt','r')
    for line in FILE:
        NEW.write(the_drug)
        NEW.write('\t')
        table=line.split('json')
        NEW.write(table[0])
        NEW.write('json')
        NEW.write('\t')
        NEW.write(line)
    FILE.close()
    i=i+1
    if (i%10==0):
        print ('finished ', i)
        break
#    print ('finished '+the_drug)



all_files=glob.glob(f'{root_path}/biorxiv_medrxiv/**/*.json', recursive=True)
print(len(all_files))
NEW=open(('result'),'w')
for the_drug in all_drugs:
    os.system("find /kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/*  -type f | xargs grep -i '"+the_drug+"' >aaa.txt")
#    print("find ../../data/comm_use_subset/*  -type f | xargs grep -i '"+the_drug+"' >aaa.txt")
    FILE=open('aaa.txt','r')
    for line in FILE:
        NEW.write(the_drug)
        NEW.write('\t')
        table=line.split('json')
        NEW.write(table[0])
        NEW.write('json')
        NEW.write('\t')
        NEW.write(line)
    FILE.close()
    i=i+1
    if (i%10==0):
        print ('finished ', i)
        break
#    print ('finished '+the_drug)



!cat ../input/drugdata/result.* |cut -f 1-2|sort|uniq|cut -f 1|sort|uniq -c|sort -g >sorted_alresult
!cat ../input/drugdata/result.* |grep -i -E -- 'COVID|coronavirus|SARS'|cut -f 1-2|sort|uniq|cut -f 1|sort|uniq -c|sort -g >sorted_alresult.coronavirus
!cat ../input/drugdata/result.* |grep -i -E -- 'COVID-19|COVID19|sars-cov-2'|cut -f 1-2|sort|uniq|cut -f 1|sort|uniq -c|sort -g >sorted_alresult.covid19
!tail -30 sorted_alresult
!tail -30 sorted_alresult.coronavirus
!tail -30 sorted_alresult.covid19
!find ../input/CORD-19-research-challenge/*  -type f | xargs grep -i antibody |grep -E -- 'COVID-19|COVID19|sars-cov-2' >antibody_paragraphs.txt

import numpy as np

exist={}
LIST=open('../input/drugdata/sorted_alresult.coronavirus','r')
for line in LIST:
    line=line.replace('\s\s+','\t')
    line=line.strip()
    table=line.split(' ')
    
    if (float(table[0])>4):
        exist[table[1]]=0
    
    
REF=open('../input/drugdata/combine_drug_name_id.csv','r')
DATA=open('../input/drugdata/combined_fp2_data.csv','r')

drug=[]
all_drug={}
for ref in REF:
    #if ('pos' in ref):
        ref=ref.strip()
        rrr=ref.split(',')
        data=DATA.readline()
        if (rrr[1].lower() in exist):
            drug.append(rrr[1])
    
            
            data=data.strip()
            data=data.split(',')
            kkk=0
            for i in data:
                data[kkk]=float(i)
                kkk+1
            all_drug[rrr[1]]=np.asarray(data).astype(np.float)
    
REF.close()
DATA.close()

connections1=[]
connections2=[]
for drug1 in drug:
    for drug2 in drug:
        if (drug1<drug2):
            cor=np.corrcoef(all_drug[drug1],all_drug[drug2])
            if (cor[0,1]>0.40):
                connections1.append(drug1)
                connections2.append(drug2)
                
import sys
import plotly.graph_objects as go
import networkx as nx

node_list=list(all_drug.keys())
G = nx.Graph()
for i in node_list:
    G.add_node(i)

i=0
for drug1 in connections1:
    drug2=connections2[i]
    G.add_edges_from([(drug1,drug2)])
    i=i+1



pos = nx.spring_layout(G, k=0.5, iterations=50)
for n, p in pos.items():
    G.nodes[n]['pos'] = p
    
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=1,color='#888'),
    hoverinfo='none',
    mode='lines')


for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='RdBu',
        reversescale=True,
        color=[],
        size=15,
        colorbar=dict(
            thickness=5,
       #     title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=0)))

for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
   # node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
    node_info = adjacencies[0]
    node_trace['text']+=tuple([node_info])

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Similarity of chemical structures among the drugs that are related to coronavirus in literature',
                titlefont=dict(size=12),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=50,l=100,r=100,t=50),
                annotations=[ dict(
                   # text="No. of connections",
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
fig.show()


import numpy as np

exist={}
LIST=open('../input/drugdata/sorted_alresult.covid19','r')
for line in LIST:
    line=line.replace('\s\s+','\t')
    line=line.strip()
    table=line.split(' ')
    if (float(table[0])>0):
        exist[table[1]]=0
    
    
REF=open('../input/drugdata/combine_drug_name_id.csv','r')
DATA=open('../input/drugdata/combined_fp2_data.csv','r')

drug=[]
all_drug={}
for ref in REF:
    #if ('pos' in ref):
        ref=ref.strip()
        rrr=ref.split(',')
        data=DATA.readline()
        if (rrr[1].lower() in exist):
            drug.append(rrr[1])
    
            
            data=data.strip()
            data=data.split(',')
            kkk=0
            for i in data:
                data[kkk]=float(i)
                kkk+1
            all_drug[rrr[1]]=np.asarray(data).astype(np.float)
    
REF.close()
DATA.close()

connections1=[]
connections2=[]
for drug1 in drug:
    for drug2 in drug:
        if (drug1<drug2):
            cor=np.corrcoef(all_drug[drug1],all_drug[drug2])
            if (cor[0,1]>0.35):
                connections1.append(drug1)
                connections2.append(drug2)
                
import sys
import plotly.graph_objects as go
import networkx as nx

node_list=list(all_drug.keys())
G = nx.Graph()
for i in node_list:
    G.add_node(i)

i=0
for drug1 in connections1:
    drug2=connections2[i]
    G.add_edges_from([(drug1,drug2)])
    i=i+1



pos = nx.spring_layout(G, k=0.5, iterations=50)
for n, p in pos.items():
    G.nodes[n]['pos'] = p
    
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=1,color='#888'),
    hoverinfo='none',
    mode='lines')


for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='RdBu',
        reversescale=True,
        color=[],
        size=15,
        colorbar=dict(
            thickness=5,
       #     title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=0)))

for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
   # node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
    node_info = adjacencies[0]
    node_trace['text']+=tuple([node_info])

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Similarity of chemical structures among the drugs that are related to COVID-19 in literature',
                titlefont=dict(size=12),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=50,l=100,r=100,t=50),
                annotations=[ dict(
                   # text="No. of connections",
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
fig.show()

import sys
import plotly.graph_objects as go
import networkx as nx

node_list=list(['Chloroquine phosphate','Spike (S) antibody','IL-6 antibody','Remdesivir','Favipiravir','Fluorouracil','Ribavirin','Acyclovir','Ritonavir','Lopinavir','Kaletra','Darunavir','Arbidol','Hydroxychloroquine','Oseltamivir'])
G = nx.Graph()
for i in node_list:
    G.add_node(i)

G.add_edges_from([('Spike (S) antibody','IL-6 antibody')])
G.add_edges_from([('Remdesivir','Favipiravir')])
G.add_edges_from([('Remdesivir','Fluorouracil')])
G.add_edges_from([('Remdesivir','Ribavirin')])
G.add_edges_from([('Remdesivir','Acyclovir')])
G.add_edges_from([('Fluorouracil','Favipiravir')])
G.add_edges_from([('Ribavirin','Favipiravir')])
G.add_edges_from([('Acyclovir','Favipiravir')])
G.add_edges_from([('Fluorouracil','Ribavirin')])
G.add_edges_from([('Fluorouracil','Acyclovir')])
G.add_edges_from([('Ribavirin','Acyclovir')])

G.add_edges_from([('Ritonavir','Lopinavir')])
G.add_edges_from([('Ritonavir','Kaletra')])
G.add_edges_from([('Ritonavir','Darunavir')])
G.add_edges_from([('Lopinavir','Kaletra')])
G.add_edges_from([('Lopinavir','Darunavir')])
G.add_edges_from([('Kaletra','Darunavir')])

G.add_edges_from([('Arbidol','Hydroxychloroquine')])
G.add_edges_from([('Chloroquine phosphate','Hydroxychloroquine')])
G.add_edges_from([('Chloroquine phosphate','Arbidol')])

pos = nx.spring_layout(G, k=0.5, iterations=50)
for n, p in pos.items():
    G.nodes[n]['pos'] = p
    
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=1,color='#888'),
    hoverinfo='none',
    mode='lines')


for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='RdBu',
        reversescale=True,
        color=[],
        size=15,
        colorbar=dict(
            thickness=5,
       #     title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=0)))

for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
   # node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
    node_info = adjacencies[0]
    node_trace['text']+=tuple([node_info])

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Groups of drugs in clinical trials by working mechanisms',
                titlefont=dict(size=12),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=100,l=100,r=100,t=100),
                annotations=[ dict(
                   # text="No. of connections",
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
fig.show()
#!/usr/bin/env python
# coding: utf-8
import sys
!conda install --yes --prefix {sys.prefix} -c rdkit rdkit
!pip install pubchempy
!conda install --yes --prefix {sys.prefix} -c openbabel openbabel
import numpy as np
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from pubchempy import *
from rdkit.Chem import MACCSkeys, AllChem
import csv
import openbabel
import pybel
import random
import time

### map fingerprinting to features; the appended label is later removed in the training code

def fp_to_feature_v1(fp, data_set, label):
    tmp = fp.ToBitString()
    the_feature = list(map(int, tmp))
    if label == 1:
        the_feature.append(1)
    else:
        the_feature.append(0)
    data_set.append(the_feature)
    return data_set

def fp_to_feature_v2(fp, data_set, label):
    the_feature = [0 for i in range(1024)]
    for n in fp.bits:
        the_feature[n] = 1
    if label == 1:
        the_feature.append(1)
    else:
        the_feature.append(0)
    data_set.append(the_feature)
    return data_set

def fp_prepare(fp, max_bit, fps):
    if len(fp.bits) > 0 and max(fp.bits) > max_bit:
        max_bit = max(fp.bits)
    fps.append(fp.bits)
    return max_bit, fps

def fp_to_feature_v3(max_bit, fps, num_valid_pos, data_set):
    m = 1
    for bits in fps:
        the_feature = [0 for i in range(max_bit + 1)]
        for n in bits:
            the_feature[n] = 1
        if m <= num_valid_pos:
            the_feature.append(1)
        else:
            the_feature.append(0)
        data_set.append(the_feature)
        m += 1
    return data_set

def get_all_data(smile, maccs_data, morgan_data, fp2_data, fp3_max_bit, fp3s, fp4_max_bit, fp4s, label, num):
    ms = Chem.MolFromSmiles(smile)
    mol = pybel.readstring("smi", smile)
    if ms and mol:
        fp = MACCSkeys.GenMACCSKeys(ms)
        maccs_data = fp_to_feature_v1(fp, maccs_data, label)
        fp = AllChem.GetMorganFingerprintAsBitVect(ms,2,nBits=1024)
        morgan_data = fp_to_feature_v1(fp, morgan_data, label)
        fp2 = mol.calcfp('FP2')
        fp2_data = fp_to_feature_v2(fp2, fp2_data, label)
        fp3 = mol.calcfp('FP3')
        fp3_max_bit, fp3s = fp_prepare(fp3, fp3_max_bit, fp3s)
        fp4 = mol.calcfp('FP4')
        fp4_max_bit, fp4s = fp_prepare(fp4, fp4_max_bit, fp4s)
        num += 1
    return maccs_data, morgan_data, fp2_data, fp3_max_bit, fp3s, fp4_max_bit, fp4s, num


maccs_data = []
morgan_data = []
fp2_data = []
fp3_data = []
fp3_max_bit = 0
fp3s = []
fp4_data = []
fp4_max_bit = 0
fp4s = []

## the random time sleep is inserted because continuous search from pubchem will result in blocking of our IP 
## I was told by pubchem that there are better methods to download, but we never had a chance to explore
num_pos = 0
num_valid_pos = 0
with open("../input/drugdata/bioarchive.list.csv") as f:
    a = 0
    for line in f:
        line=line.strip()
        if a % 100 == 0:
            print(a // 100)
        time.sleep(random.randint(0, 5))
        m = get_compounds(line, 'name')
        if len(m) > 0:
            smile = m[0].isomeric_smiles
            maccs_data, morgan_data, fp2_data, fp3_max_bit, fp3s, fp4_max_bit, fp4s, num_valid_pos = get_all_data(smile, maccs_data, morgan_data, fp2_data, fp3_max_bit, fp3s, fp4_max_bit, fp4s, 1, num_valid_pos)
            #num_valid_pos += 1
        num_pos += 1
        a += 1

fp3_data = fp_to_feature_v3(fp3_max_bit, fp3s, num_valid_pos, fp3_data)
fp4_data = fp_to_feature_v3(fp4_max_bit, fp4s, num_valid_pos, fp4_data)

### generate individual feature files in the format of features, followed by the last element to be labels
with open("maccs_data.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(maccs_data)

with open("morgan_data.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(morgan_data)

with open("fp2_data.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(fp2_data)

with open("fp3_data.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(fp3_data)

with open("fp4_data.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(fp4_data)
#!/usr/bin/env python
# coding: utf-8
import numpy as np
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from pubchempy import *
from rdkit.Chem import rdmolops
import pybel
import csv
import time
import random

data = []
drug_name_id = []

num_pos = 0
num_valid_pos = 0
with open("../input/drugdata/bioarchive.list.csv") as f:
    a = 0
    for line in f:
        line=line.strip()
        if a % 100 == 0:
            print(a // 100)
        m = get_compounds(line, 'name')
        if len(m) > 0:
            smile = m[0].isomeric_smiles
            #labels.append(1)
            ms = Chem.MolFromSmiles(smile)
            mol = pybel.readstring("smi", smile)
            if ms and mol:
                fp = rdmolops.RDKFingerprint(ms)
                tmp = fp.ToBitString()
                the_feature = list(map(int, tmp))
                the_feature.append(1)
                data.append(the_feature)
                drug_name_id.append(['pos_name', line[30:-1]])
                num_valid_pos += 1
        num_pos += 1
        a += 1



csvfile.close()
with open("top_data.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

with open("drug_name_id.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(drug_name_id)

import numpy as np

REF=open('../input/drugdata/combine_drug_name_id.csv','r')
DATA=open('../input/drugdata/combined_fp2_data.csv','r')

drug=[]
all_drug={}
for ref in REF:
    data=DATA.readline()
    if ('pos' in ref):
        ref=ref.strip()
        rrr=ref.split(',')
        drug.append(rrr[1])

        
        data=data.strip()
        data=data.split(',')
        kkk=0
        for i in data:
            data[kkk]=float(i)
            kkk+1
        all_drug[rrr[1]]=np.asarray(data).astype(np.float)

REF.close()
DATA.close()

connections1=[]
connections2=[]
for drug1 in drug:
    for drug2 in drug:
        if (drug1<drug2):
            cor=np.corrcoef(all_drug[drug1],all_drug[drug2])
            if (cor[0,1]>0.35):
                connections1.append(drug1)
                connections2.append(drug2)
                
import sys
import plotly.graph_objects as go
import networkx as nx

node_list=list(all_drug.keys())
G = nx.Graph()
for i in node_list:
    G.add_node(i)

i=0
for drug1 in connections1:
    drug2=connections2[i]
    G.add_edges_from([(drug1,drug2)])
    i=i+1



pos = nx.spring_layout(G, k=0.5, iterations=50)
for n, p in pos.items():
    G.nodes[n]['pos'] = p
    
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=1,color='#888'),
    hoverinfo='none',
    mode='lines')


for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])
    
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='RdBu',
        reversescale=True,
        color=[],
        size=15,
        colorbar=dict(
            thickness=5,
       #     title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=0)))

for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
   # node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1]))
    node_info = adjacencies[0]
    node_trace['text']+=tuple([node_info])

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='Similarity of chemical structures among the drugs that were proposed by in vitro virus protein binding',
                titlefont=dict(size=12),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=50,l=100,r=100,t=50),
                annotations=[ dict(
                   # text="No. of connections",
                    text="",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
fig.show()

#!/usr/bin/env python
# coding: utf-8


import numpy as np
import csv
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
from scipy import interp
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle

import os
os.system('rm -rf pred')
os.system('mkdir pred')



def readcsv(filename):
    csvfile = open(filename, encoding='utf-8')
    reader = csv.reader(csvfile)
    X = []
    Y = []
    num_all = 0
    num_positive = 0
    for item in reader:
        X.append(list(map(int, item[:-1])))
        Y.append(int(item[-1]))
        if Y[-1] == 1:
            num_positive += 1
        num_all += 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, num_all, num_positive

def train_cv(train, test, X, Y):
    X_train, X_test = X[train], X[test]
    Y_train, Y_test = Y[train], Y[test]
    train_data = lgb.Dataset(X_train, label=Y_train)
    bst = lgb.train(param, train_data, num_round)
    prediction = bst.predict(X_test)
    return prediction, Y_test

X1, Y1, num_all, num_positive = readcsv("../input/drugdata/combined_top_data.csv")
X2, Y2, num_all, num_positive = readcsv("../input/drugdata/combined_maccs_data.csv")
X3, Y3, num_all, num_positive = readcsv("../input/drugdata/combined_morgan_data.csv")
X4, Y4, num_all, num_positive = readcsv("../input/drugdata/combined_fp2_data.csv")
X5, Y5, num_all, num_positive = readcsv("../input/drugdata/combined_fp3_data.csv")
X6, Y6, num_all, num_positive = readcsv("../input/drugdata/combined_fp4_data.csv")
X = np.hstack((X1, X2, X3, X4, X5, X6))
Y = Y1

print(len(X))
print(len(X[0]))

the_round=0
while (the_round<20):
    num_fold = 5
    kf = KFold(n_splits=num_fold, random_state=None, shuffle=True)
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 10

    AUC = []
    AUPRC = []
    AUC_score = []
    predictions = []
    test_sets = []
    num = 0
    for train, test in kf.split(X):
        #prediction1, Y_test1 = train_cv(train, test, X1, Y1)
        #prediction2, Y_test2 = train_cv(train, test, X2, Y2)
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]
        train_data = lgb.Dataset(X_train, label=Y_train)
        bst = lgb.train(param, train_data, num_round)
        num += 1
        pickle.dump(bst, open('pred/all_lightgbm_model-'+str(num)+'.'+str(the_round)+'.sav', 'wb'))
        with open("pred/all_lightgbm_Xtest-"+str(num)+'.'+str(the_round)+".csv","w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(X_test)
        with open("pred/all_lightgbm_testid-"+str(num)+'.'+str(the_round)+".csv","w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows([test])
        prediction = bst.predict(X_test)
        np.savetxt(('pred/predictions.txt.'+str(num)+'.'+str(the_round)),prediction)
        predictions.append(prediction)
        test_sets.append(Y_test)
    the_round=the_round+1
import glob

all_dir=glob.glob("/kaggle/working/pred/all_lightgbm_testid-*")
ref={}
count={}
for the_dir in all_dir:
    FILE=open(the_dir,'r')
    line=FILE.readline()
    line=line.strip()

    lll=line.split(',')
    FILE.close()
    pred=the_dir
    pred=pred.replace('all_lightgbm_testid-','predictions.txt.')
    pred=pred.replace('.csv','')
    PRED=open(pred,'r')
    i=0

    for line in PRED:
        line=line.strip()
        if int(lll[i]) in ref:
            ref[int(lll[i])]=ref[int(lll[i])]+float(line)
            count[int(lll[i])]=count[int(lll[i])]+1
        else:
            ref[int(lll[i])]=float(line)
            count[int(lll[i])]=1
        i=i+1



REF=open("../input/drugdata/combine_drug_name_id.csv",'r')
NEW=open('assembled_prediction.dat','w')
i=0;
for line in REF:
    line=line.strip()
    t=line.split(',')
    t[1]=t[1].replace(' ','_')
    NEW.write(t[0])
    NEW.write('\t')
    NEW.write(t[1])
    try:
        val=ref[int(i)]/count[int(i)];
        NEW.write('\t')
        NEW.write(str(val))
        NEW.write('\t')
        NEW.write(count[i])
        
    except:
        pass
    NEW.write('\n')
    i=i+1
REF.close()
NEW.close()


!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

!find ../input/CORD-19-research-challenge/*  -type f | xargs grep -i vaccine |grep -E -- 'COVID-19|COVID19|sars-cov-2' >vaccine_paragraphs

import numpy as np


all_epi={}
FILE=open('../input/drugdata/all_epitope','r')
for line in FILE:
    line=line.strip()
    all_epi[line]=1

the_map={}
for epi1 in all_epi.keys():
    for epi2 in all_epi.keys():
## now we calculate the maximal similarity score:
        if (len(epi1)<len(epi2)):
            if (epi1 in epi2):
                print (epi1,' is a subsequence' ,epi2)
                if (epi1 in the_map):
                    the_map[epi1]=the_map[epi1]+'\t'+epi2
                else:
                    the_map[epi1]=epi2
                if (epi2 in the_map):
                    the_map[epi2]=the_map[epi2]+'\t'+epi1
                else:
                    the_map[epi2]=epi1
            cut=1
            while (cut<5):
                epi1_cut=epi1[cut:]
                if (len(epi1_cut)>5):
                    if (epi1_cut==epi2[0:len(epi1_cut)]):
                        print (epi1,' is almost subsequence by ' ,cut,' to ',epi2)
                        if (epi1 in the_map):
                            the_map[epi1]=the_map[epi1]+'\t'+epi2
                        else:
                            the_map[epi1]=epi2
                        if (epi2 in the_map):
                            the_map[epi2]=the_map[epi2]+'\t'+epi1
                        else:
                            the_map[epi2]=epi1

                epi1_cut=epi1[:-cut]
                if (len(epi1_cut)>5):
                    if (epi1_cut == epi2[(len(epi2)-len(epi1_cut)):]):
                        print (epi1,' is almost subsequence by ' ,cut,' to ',epi2)
                        if (epi1 in the_map):
                            the_map[epi1]=the_map[epi1]+'\t'+epi2
                        else:
                            the_map[epi1]=epi2
                        if (epi2 in the_map):
                            the_map[epi2]=the_map[epi2]+'\t'+epi1
                        else:
                            the_map[epi2]=epi1

                cut=cut+1

group={}
for epi in all_epi.keys():
    if (epi in the_map):
        the_group=the_map[epi].split('\t')
        uniq_group={}
        for ggg in the_group:
            uniq_group[ggg]=1
        uniq_group[epi]=1
        the_member=[]
        for k in sorted(uniq_group, key=len, reverse=False):
            the_member.append(k)
        string=the_member.pop(0)
        k=1
        for mmm in the_member:
            string=string+'|'
            string=string+mmm
            k=k+1
        while(k<9):
            string=string+'|'
            k=k+1

        group[string]=1
    else:

        string=epi
        k=1
        while(k<9):
            string=string+'|'
            k=k+1
        group[string]=1

g_i=1
for uniq_group in sorted(group.keys()):
    string='|'+str(g_i)+'|'+uniq_group+'|'
    print(string)
    g_i=g_i+1

import numpy as np


all_epi={}
FILE=open('../input/drugdata/all_epitope','r')
for line in FILE:
    line=line.strip()
    all_epi[line]=1

the_connection={}
for epi1 in all_epi.keys():
    for epi2 in all_epi.keys():
## now we calculate the maximal similarity score:
        if (len(epi1)<len(epi2)):
            if (epi1 in epi2):
                string=epi1+'_'+epi2
                the_connection[string]=1
                string=epi2+'_'+epi1
                the_connection[string]=1
            cut=1
            while (cut<5):
                epi1_cut=epi1[cut:]
                if (len(epi1_cut)>5):
                    if (epi1_cut==epi2[0:len(epi1_cut)]):
                        string=epi1+'_'+epi2
                        the_connection[string]=1
                        string=epi2+'_'+epi1
                        the_connection[string]=1

                epi1_cut=epi1[:-cut]
                if (len(epi1_cut)>5):
                    if (epi1_cut == epi2[(len(epi2)-len(epi1_cut)):]):
                        string=epi1+'_'+epi2
                        the_connection[string]=1
                        string=epi2+'_'+epi1
                        the_connection[string]=1

                cut=cut+1

visited={}
group=1
for epi in all_epi.keys():
    if (epi in visited):
        pass
    else:
        ch=0
        for epi_ref in visited.keys():
            string=epi_ref+'_'+epi
            if (string in the_connection):
                value=visited[epi_ref]
                ch=1
        if (ch==1):
            visited[epi]=value
        if (ch==0):
            visited[epi]=group
            group=group+1


ggg=1
while (ggg<group):
    member=[]
    for epi in all_epi.keys():
        if (visited[epi]==ggg):
            member.append(epi)
    print('Group '+str(ggg),member)
    ggg=ggg+1