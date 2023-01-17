%config Completer.use_jedi = False
%load_ext autoreload

%autoreload 2
import os

import platform



if platform.system() == 'Windows':

    os.environ['comspec'] = 'powershell'

    print(os.getenv('comspec'))
# ! pip install kaggle --user

# ! kaggle competitions download -c lish-moa -p ./

# ! 7z x lish-moa.zip -olish-moa -y

# ! unzip -o lish-moa.zip -d lish-moa 
base_path = '../input/lish-moa/'
# ! ls 'lish-moa/'

! ls '../input/lish-moa/'
import numpy as np



import pandas as pd



import seaborn as sns



import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV



import torch

from torch.utils.data import TensorDataset, DataLoader



import catalyst.dl

from catalyst.runners import SupervisedRunner
torch.backends.cudnn.benchmark = True

torch.manual_seed(123)

torch.cuda.random.manual_seed(123)



# Set proper device for computations

dtype, device, cuda_device_id = torch.float32, None, 0

os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(str(cuda_device_id) if cuda_device_id is not None else '')

if cuda_device_id is not None and torch.cuda.is_available():

    device = 'cuda:{0:d}'.format(0)

else:

    device = torch.device('cpu')



print(dtype, device)
sample_submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
train_features = pd.read_csv(os.path.join(base_path, 'train_features.csv'))
train_targets_nonscored = pd.read_csv(os.path.join(base_path, 'train_targets_nonscored.csv'))
train_targets_scored = pd.read_csv(os.path.join(base_path, 'train_targets_scored.csv'))
test_features = pd.read_csv(os.path.join(base_path, 'test_features.csv'))
sample_submission.sample(10)
train_targets_scored.sample(10)
train_targets_nonscored.sample(10)
print(np.mean(train_features['sig_id'] == train_targets_scored['sig_id']) == 1.0)

print(np.mean(train_targets_scored['sig_id'] == train_targets_nonscored['sig_id']) == 1.0)
train_targets = pd.concat([train_targets_scored, train_targets_nonscored], axis=1)
fig, axes = plt.subplots(1, 3, figsize=(21, 7))



sns.distplot(np.sum(train_targets, axis=1), ax=axes[0], kde=False)

sns.distplot(np.sum(train_targets_scored, axis=1), ax=axes[1], kde=False)

sns.distplot(np.sum(train_targets_nonscored, axis=1), ax=axes[2], kde=False)



plt.show()
has_one_label = (np.sum(train_targets, axis=1) <= 1)
X = pd.get_dummies(train_features.drop(['sig_id'], axis=1), columns=['cp_type', 'cp_time', 'cp_dose'])

y = np.argmax(np.concatenate([

    train_targets.drop(['sig_id'], axis=1).to_numpy(),

    (train_targets.drop(['sig_id'], axis=1).to_numpy().sum(axis=1) == 0).astype(np.int)[:, None]

], axis=1), axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



subX = pd.get_dummies(test_features.drop(['sig_id'], axis=1), columns=['cp_type', 'cp_time', 'cp_dose'])
sub_ds = TensorDataset(torch.tensor(subX.to_numpy(), dtype=torch.float32))

test_ds = TensorDataset(torch.tensor(X_test.to_numpy(), dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_ds = TensorDataset(torch.tensor(X_train.to_numpy(), dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))



sub_dl = DataLoader(sub_ds, batch_size=1024, shuffle=False, num_workers=3)

test_dl = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=3)

train_dl = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=3)
class SimpleModel(torch.nn.Module):

    def __init__(self, n_classes=609):

        super().__init__()

        

        self.n_classes = n_classes

        self.layers = torch.nn.Sequential(

            torch.nn.Linear(879, 100),

            torch.nn.ReLU(),

            torch.nn.Linear(100, n_classes),

        )

        

    def forward(self, x):

        return self.layers(x)
model = SimpleModel(n_classes=609).to(device=device, dtype=dtype)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=10.0)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
runner = SupervisedRunner()



runner.train(

    model=model,

    criterion=loss_fn,

    optimizer=optimizer,

    callbacks=[

        catalyst.dl.AccuracyCallback(),

    ],

    logdir='./logs/',

    loaders={

        'valid': test_dl,

        'train': train_dl 

    },

    num_epochs=100,

    scheduler=None,

    verbose=False,

    minimize_metric=True,

    fp16={"opt_level": "O3"},

)

model.eval()



sub_result = np.empty([sample_submission.shape[0], sample_submission.shape[1] - 1])

with torch.no_grad():

    idx = 0

    for (x, ) in sub_dl:

        preds = model(x.to(device=device, dtype=dtype)).cpu()

        sub_result[idx: idx + x.shape[0]] = torch.softmax(preds[:, :sample_submission.shape[1] - 1], axis=1).numpy()

        

        idx += x.shape[0]
with open('submission.csv', 'w') as file:

    file.write(

        'sig_id,5-alpha_reductase_inhibitor,11-beta-hsd1_inhibitor,acat_inhibitor,acetylcholine_receptor_agonist,acetylcholine_receptor_antagonist,acetylcholinesterase_inhibitor,adenosine_receptor_agonist,adenosine_receptor_antagonist,adenylyl_cyclase_activator,adrenergic_receptor_agonist,adrenergic_receptor_antagonist,akt_inhibitor,aldehyde_dehydrogenase_inhibitor,alk_inhibitor,ampk_activator,analgesic,androgen_receptor_agonist,androgen_receptor_antagonist,anesthetic_-_local,angiogenesis_inhibitor,angiotensin_receptor_antagonist,anti-inflammatory,antiarrhythmic,antibiotic,anticonvulsant,antifungal,antihistamine,antimalarial,antioxidant,antiprotozoal,antiviral,apoptosis_stimulant,aromatase_inhibitor,atm_kinase_inhibitor,atp-sensitive_potassium_channel_antagonist,atp_synthase_inhibitor,atpase_inhibitor,atr_kinase_inhibitor,aurora_kinase_inhibitor,autotaxin_inhibitor,bacterial_30s_ribosomal_subunit_inhibitor,bacterial_50s_ribosomal_subunit_inhibitor,bacterial_antifolate,bacterial_cell_wall_synthesis_inhibitor,bacterial_dna_gyrase_inhibitor,bacterial_dna_inhibitor,bacterial_membrane_integrity_inhibitor,bcl_inhibitor,bcr-abl_inhibitor,benzodiazepine_receptor_agonist,beta_amyloid_inhibitor,bromodomain_inhibitor,btk_inhibitor,calcineurin_inhibitor,calcium_channel_blocker,cannabinoid_receptor_agonist,cannabinoid_receptor_antagonist,carbonic_anhydrase_inhibitor,casein_kinase_inhibitor,caspase_activator,catechol_o_methyltransferase_inhibitor,cc_chemokine_receptor_antagonist,cck_receptor_antagonist,cdk_inhibitor,chelating_agent,chk_inhibitor,chloride_channel_blocker,cholesterol_inhibitor,cholinergic_receptor_antagonist,coagulation_factor_inhibitor,corticosteroid_agonist,cyclooxygenase_inhibitor,cytochrome_p450_inhibitor,dihydrofolate_reductase_inhibitor,dipeptidyl_peptidase_inhibitor,diuretic,dna_alkylating_agent,dna_inhibitor,dopamine_receptor_agonist,dopamine_receptor_antagonist,egfr_inhibitor,elastase_inhibitor,erbb2_inhibitor,estrogen_receptor_agonist,estrogen_receptor_antagonist,faah_inhibitor,farnesyltransferase_inhibitor,fatty_acid_receptor_agonist,fgfr_inhibitor,flt3_inhibitor,focal_adhesion_kinase_inhibitor,free_radical_scavenger,fungal_squalene_epoxidase_inhibitor,gaba_receptor_agonist,gaba_receptor_antagonist,gamma_secretase_inhibitor,glucocorticoid_receptor_agonist,glutamate_inhibitor,glutamate_receptor_agonist,glutamate_receptor_antagonist,gonadotropin_receptor_agonist,gsk_inhibitor,hcv_inhibitor,hdac_inhibitor,histamine_receptor_agonist,histamine_receptor_antagonist,histone_lysine_demethylase_inhibitor,histone_lysine_methyltransferase_inhibitor,hiv_inhibitor,hmgcr_inhibitor,hsp_inhibitor,igf-1_inhibitor,ikk_inhibitor,imidazoline_receptor_agonist,immunosuppressant,insulin_secretagogue,insulin_sensitizer,integrin_inhibitor,jak_inhibitor,kit_inhibitor,laxative,leukotriene_inhibitor,leukotriene_receptor_antagonist,lipase_inhibitor,lipoxygenase_inhibitor,lxr_agonist,mdm_inhibitor,mek_inhibitor,membrane_integrity_inhibitor,mineralocorticoid_receptor_antagonist,monoacylglycerol_lipase_inhibitor,monoamine_oxidase_inhibitor,monopolar_spindle_1_kinase_inhibitor,mtor_inhibitor,mucolytic_agent,neuropeptide_receptor_antagonist,nfkb_inhibitor,nicotinic_receptor_agonist,nitric_oxide_donor,nitric_oxide_production_inhibitor,nitric_oxide_synthase_inhibitor,norepinephrine_reuptake_inhibitor,nrf2_activator,opioid_receptor_agonist,opioid_receptor_antagonist,orexin_receptor_antagonist,p38_mapk_inhibitor,p-glycoprotein_inhibitor,parp_inhibitor,pdgfr_inhibitor,pdk_inhibitor,phosphodiesterase_inhibitor,phospholipase_inhibitor,pi3k_inhibitor,pkc_inhibitor,potassium_channel_activator,potassium_channel_antagonist,ppar_receptor_agonist,ppar_receptor_antagonist,progesterone_receptor_agonist,progesterone_receptor_antagonist,prostaglandin_inhibitor,prostanoid_receptor_antagonist,proteasome_inhibitor,protein_kinase_inhibitor,protein_phosphatase_inhibitor,protein_synthesis_inhibitor,protein_tyrosine_kinase_inhibitor,radiopaque_medium,raf_inhibitor,ras_gtpase_inhibitor,retinoid_receptor_agonist,retinoid_receptor_antagonist,rho_associated_kinase_inhibitor,ribonucleoside_reductase_inhibitor,rna_polymerase_inhibitor,serotonin_receptor_agonist,serotonin_receptor_antagonist,serotonin_reuptake_inhibitor,sigma_receptor_agonist,sigma_receptor_antagonist,smoothened_receptor_antagonist,sodium_channel_inhibitor,sphingosine_receptor_agonist,src_inhibitor,steroid,syk_inhibitor,tachykinin_antagonist,tgf-beta_receptor_inhibitor,thrombin_inhibitor,thymidylate_synthase_inhibitor,tlr_agonist,tlr_antagonist,tnf_inhibitor,topoisomerase_inhibitor,transient_receptor_potential_channel_antagonist,tropomyosin_receptor_kinase_inhibitor,trpv_agonist,trpv_antagonist,tubulin_inhibitor,tyrosine_kinase_inhibitor,ubiquitin_specific_protease_inhibitor,vegfr_inhibitor,vitamin_b,vitamin_d_receptor_agonist,wnt_inhibitor\n'

    )

    for idx in range(test_features.shape[0]):

        file.write(

            test_features.iloc[idx]['sig_id'] + ',' + ','.join(['{0:.4f}'.format(_) for _ in sub_result[idx]]) + '\n'

        )
pd.read_csv('submission.csv')