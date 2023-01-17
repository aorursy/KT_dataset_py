import numpy as np 

import pandas as pd 
!conda install -y -c bioconda forgi

!conda install -y -c bioconda viennarna
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
train[['id','sequence','structure','predicted_loop_type']].head(3)
!mkdir ../fx_files

for i in range(5): 

    fx_test=">{0}\n{1}\n{2}\n".format(train.loc[i, "id"],train.loc[i, "sequence"],train.loc[i, "structure"]) 

    textfile = open('../fx_files/'+train.loc[i, "id"]+'.fx', 'w')

    textfile.write(fx_test)

    textfile.close()
import matplotlib.pyplot as plt

import forgi.visual.mplotlib as fvm

import forgi
for i in range(3):

    ##Print Structure

    plt.figure(figsize=(20,20))

    cg = forgi.load_rna('../fx_files/'+train.loc[i, "id"]+'.fx', allow_many=False)

    fvm.plot_rna(cg, text_kwargs={"fontweight":"black"}, lighten=0.7,backbone_kwargs={"linewidth":3})

    plt.show()

    

    #Print 

    print('Second Structure info')

    print (cg.to_bg_string())

    

    #Get Pairs

    print('Secuence Pairs:')

    print(cg.to_pair_table())