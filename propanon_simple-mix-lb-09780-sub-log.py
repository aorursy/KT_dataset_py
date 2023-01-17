# I'm just taking these three results and naively mixing them together through different kinds of means
import pandas as pd
import numpy as np
print("Reading the data...\n")
df1 = pd.read_csv('../input/r-lgbm-single-model-40m-rows-lb-0-9736/lgb_Usrnewness.csv')
df2 = pd.read_csv('../input/simple-linear-stacking-with-ranks-lb-0-9760/sub_stacked.csv')
df3 = pd.read_csv('../input/talkingdata-wordbatch-fm-ftrl-lb-0-9752/wordbatch_fm_ftrl.csv')
models = { 'df1' : {
                    'name':'r-lgbm',
                    'score':97.36,
                    'df':df1 },
          'df2' : {
                    'name':'linstack',
                    'score':97.60,
                    'df':df2 },
           'df3' : {
                    'name':'wordbatch',
                    'score':97.52,
                    'df':df3 }
         }
df1.head()
# Making simple blendings of the models

isa_lg = 0
isa_hm = 0
print("Blending...\n")
for df in models.keys() : 
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
isa_lg = np.exp(isa_lg/3)
isa_hm = 1/isa_hm

print("Isa log\n")
print(isa_lg[:5])
print()
print("Isa harmo\n")
print(isa_hm[:5])
sub_log = pd.DataFrame()
sub_log['click_id'] = df1['click_id']
sub_log['is_attributed'] = isa_lg
sub_log.head()
sub_hm = pd.DataFrame()
sub_hm['click_id'] = df1['click_id']
sub_hm['is_attributed'] = isa_hm
sub_hm.head()
print("Writing...")
sub_log.to_csv('sub_log.csv', index=False, float_format='%.9f')
sub_hm.to_csv('sub_hm.csv', index=False, float_format='%.9f')