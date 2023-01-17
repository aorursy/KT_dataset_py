import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
pd_candidates = pd.read_csv("../input/candidatos2018_merged.csv",encoding='latin_1')
pd_candidates = pd_candidates.drop_duplicates()
pd_cand_female1=pd_candidates[((pd_candidates['DS_GENERO']=='FEMININO') )]

tot=pd_cand_female1.shape[0]
print('Total de candidatas:',tot)
pd_cand_female1 = pd_cand_female1.groupby(['DS_GENERO','SG_PARTIDO']).size()
pd_cand_female1 = pd_cand_female1.nlargest(50).to_frame()

pd_cand_female1.head(50)