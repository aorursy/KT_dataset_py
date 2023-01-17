import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
pd_candidates = pd.read_csv("../input/candidatos2018_merged.csv",encoding='latin1')
pd_candidates = pd_candidates.drop_duplicates()
pd_cand_senador_sp=pd_candidates[((pd_candidates['DS_CARGO']=='SENADOR') & (pd_candidates['SG_UE']=='SP') & (pd_candidates['DS_SITUACAO_CANDIDATURA']=='APTO')  )]
pd_cand_senador_sp2 =pd_cand_senador_sp[['NM_CANDIDATO','NR_CANDIDATO','SG_PARTIDO','DS_OCUPACAO','DS_GRAU_INSTRUCAO','NM_MUNICIPIO_NASCIMENTO']]
pd_cand_senador_sp2 = pd_cand_senador_sp2.sort_values(by=['NR_CANDIDATO'])
pd_cand_senador_sp2
