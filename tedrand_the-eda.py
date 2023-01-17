import pandas as pd
from IPython.display import HTML

df = pd.read_csv('../input/meta-simple-repeat.csv')

bases = {'A': {'n':'Adenine','c':'blue'}, 
         'G': {'n':'Guanine','c':'green'}, 
         'C': {'n':'Cytosine','c':'red'},
         'T': {'n':'Thymine','c':'yellow'}}

h = ''.join(["<span style='color:" + bases[x]['c'] + ";'>" + x + "</span>" for x in df['sequence'][0]])
h = "<div style='background-color:black;width:500px; word-break: break-all; word-wrap: break-word;'>" + h + "</div>"
print(df['chrom'][0])
HTML(h)
