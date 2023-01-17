import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
syd=pd.read_csv('../input/diffsydiw.csv').dropna()
dia=pd.read_csv('../input/dia_t.csv').dropna()
sym=pd.read_csv('../input/sym_t.csv').dropna()

print(sym.head())
print(dia.head())

# All diseases with 'Lower abdominal pain' symptom sorted by weight
print(syd[syd['syd']==2].sort_values('wei',ascending=False).head())

# Main abdominal pain disease
print(dia[dia['did']==56])
# All appendicitis symptoms
print(syd[syd['did']==56])
print(sym[sym['syd']==128])
print(sym[sym['syd']==264])
# If you're a baby and you don't talk
print(sym[sym['symptom'].str.contains("baby")]) 
# All baby related diseases sorted by weight
print(syd[syd['syd']==128].sort_values('wei',ascending=False).head())
print(dia[dia['did']==54])
print(dia[dia['did']==159])
print(dia[dia['did']==177])
print(dia[dia['did']==192])
