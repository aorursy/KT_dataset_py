# ## check gpu use - check with this kernel! 

# import torch

# torch.cuda.device(0)
from fastai.tabular import * 



## use fastAI's methods, notably default settings for embedding categoircals and missing value imputations
drop_cols = ['icu_id','hospital_id'] # and patient id?  
train = pd.read_csv("../input/widsdatathon2020/training_v2.csv").drop(drop_cols,axis=1)

test = pd.read_csv("../input/widsdatathon2020/unlabeled.csv").drop(drop_cols,axis=1)
procs = [FillMissing, Categorify, Normalize]

valid_idx = range(int(len(train)*0.9), len(train))
dep_var = 'hospital_death'
def to_cat(c): 

    train[c] = train[c].apply(str)

    test[c]  = test[c].apply(str)

    

[to_cat(c) for c in ['apache_3j_diagnosis', 'apache_2_diagnosis']]
cat_names = ['ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source',  

             'icu_stay_type', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem', 

             'elective_surgery', 'apache_post_operative', 'arf_apache',

#              'gcs_eyes_apache','gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache',

             'intubated_apache', 

             'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 

             'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis', 

              'apache_3j_diagnosis' , 'apache_2_diagnosis']  # 'icu_id',
cont_names = list(set(train)-set(cat_names)-

                  {dep_var,  'patient_id', 'encounter_id'}) # 'hospital_id',

cont_names
train["apache_4a_hospital_death_prob"]=train["apache_4a_hospital_death_prob"].replace({-1:np.nan})

test["apache_4a_hospital_death_prob"]=test["apache_4a_hospital_death_prob"].replace({-1:np.nan})



train["apache_4a_icu_death_prob"]=train["apache_4a_icu_death_prob"].replace({-1:np.nan})

test["apache_4a_icu_death_prob"]=test["apache_4a_icu_death_prob"].replace({-1:np.nan})
data = TabularDataBunch.from_df('.', train, dep_var, 

                                valid_idx=valid_idx, procs=procs, 

                                cat_names=cat_names, cont_names=cont_names)
## try classic pyramid shape of network based on input size

train.shape
## orig : learn = tabular_learner(data, layers=[100,100], ps=0.5, emb_drop=0.5, metrics=[accuracy, AUROC()])

learn = tabular_learner(data, layers=[400,200,128], ps=0.45, emb_drop=0.45, metrics=[accuracy, AUROC()])
# # learn.fit_one_cycle(3, 1e-2) ## orig

learn.fit_one_cycle(6, 1e-2)
learn2 = tabular_learner(data, layers=[256,128], ps=0.45, emb_drop=0.4, metrics=[accuracy, AUROC()])

learn2.fit_one_cycle(6, 1e-2)
data.add_test(TabularList.from_df(test,path='.' ,cat_names=cat_names, cont_names=cont_names))
probs = learn.get_preds(DatasetType.Test)[0][:,1]

probs
sub = pd.read_csv('../input/widsdatathon2020/solution_template.csv')

sub['hospital_death'] = probs

sub.head()
sub.to_csv("sub.csv", header=True, index=False)
! head sub.csv