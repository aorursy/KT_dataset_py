import numpy as np

import pandas as pd

import os

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

from sklearn.linear_model import Ridge

from tqdm.notebook import tqdm
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')



ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
train_targets = train_targets[train_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

train_features = train_features[train_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(train_features)

test = preprocess(test_features)



del train_targets['sig_id']
# min max scaling of feature space

def scale(df, test_df):

    df_concat = pd.concat([df, test_df])

    

    for col in df_concat.columns:

        df_concat[col] = ( df_concat[col] - df_concat[col].min() ) / ( df_concat[col].max() - df_concat[col].min() )

    

    test_df = df_concat[len(df):]

    df = df_concat[:len(df)]

    

    return df, test_df



train, test = scale(train, test)
cat_features = [col for col in train.columns if col.startswith('cp')]

g_features = [col for col in train.columns if col.startswith('g')]

c_features = [col for col in train.columns if col.startswith('c')]



features = cat_features + g_features + c_features
top_feats = [  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  12,  14,  15,

        16,  18,  19,  20,  21,  23,  24,  25,  27,  28,  29,  30,  31,

        32,  33,  34,  35,  36,  37,  39,  40,  41,  42,  44,  45,  46,

        48,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,

        63,  64,  65,  66,  68,  69,  70,  71,  72,  73,  74,  75,  76,

        78,  79,  80,  81,  82,  83,  84,  86,  87,  88,  89,  90,  92,

        93,  94,  95,  96,  97,  99, 100, 101, 103, 104, 105, 106, 107,

       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,

       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134,

       135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,

       149, 150, 151, 152, 153, 154, 155, 157, 159, 160, 161, 163, 164,

       165, 166, 167, 168, 169, 170, 172, 173, 175, 176, 177, 178, 180,

       181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 195,

       197, 198, 199, 202, 203, 205, 206, 208, 209, 210, 211, 212, 213,

       214, 215, 218, 219, 220, 221, 222, 224, 225, 227, 228, 229, 230,

       231, 232, 233, 234, 236, 238, 239, 240, 241, 242, 243, 244, 245,

       246, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260,

       261, 263, 265, 266, 268, 270, 271, 272, 273, 275, 276, 277, 279,

       282, 283, 286, 287, 288, 289, 290, 294, 295, 296, 297, 299, 300,

       301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 315,

       316, 317, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331,

       332, 333, 334, 335, 338, 339, 340, 341, 343, 344, 345, 346, 347,

       349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377,

       378, 379, 380, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,

       392, 393, 394, 395, 397, 398, 399, 400, 401, 403, 405, 406, 407,

       408, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422,

       423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,

       436, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,

       452, 453, 454, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,

       466, 468, 469, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482,

       483, 485, 486, 487, 488, 489, 491, 492, 494, 495, 496, 500, 501,

       502, 503, 505, 506, 507, 509, 510, 511, 512, 513, 514, 516, 517,

       518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 531, 532, 533,

       534, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,

       549, 550, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,

       564, 565, 566, 567, 569, 570, 571, 572, 573, 574, 575, 577, 580,

       581, 582, 583, 586, 587, 590, 591, 592, 593, 595, 596, 597, 598,

       599, 600, 601, 602, 603, 605, 607, 608, 609, 611, 612, 613, 614,

       615, 616, 617, 619, 622, 623, 625, 627, 630, 631, 632, 633, 634,

       635, 637, 638, 639, 642, 643, 644, 645, 646, 647, 649, 650, 651,

       652, 654, 655, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668,

       669, 670, 672, 674, 675, 676, 677, 678, 680, 681, 682, 684, 685,

       686, 687, 688, 689, 691, 692, 694, 695, 696, 697, 699, 700, 701,

       702, 703, 704, 705, 707, 708, 709, 711, 712, 713, 714, 715, 716,

       717, 723, 725, 727, 728, 729, 730, 731, 732, 734, 736, 737, 738,

       739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,

       752, 753, 754, 755, 756, 758, 759, 760, 761, 762, 763, 764, 765,

       766, 767, 769, 770, 771, 772, 774, 775, 780, 781, 782, 783, 784,

       785, 787, 788, 790, 793, 795, 797, 799, 800, 801, 805, 808, 809,

       811, 812, 813, 816, 819, 820, 821, 822, 823, 825, 826, 827, 829,

       831, 832, 833, 834, 835, 837, 838, 839, 840, 841, 842, 844, 845,

       846, 847, 848, 850, 851, 852, 854, 855, 856, 858, 860, 861, 862,

       864, 867, 868, 870, 871, 873, 874]

print(len(top_feats))
features = [feat for feat in train.columns]

features = [features[i] for i in top_feats]
len(features)
seed = 0

target_cols = [t for t in train_targets.columns]

num_targets = len(target_cols)

nfolds = 7

overal_score = 0.0

kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
alphas = {'5-alpha_reductase_inhibitor': 1000, '11-beta-hsd1_inhibitor': 1000, 'acat_inhibitor': 10000, 'acetylcholine_receptor_agonist': 1000, 'acetylcholine_receptor_antagonist': 1000, 'acetylcholinesterase_inhibitor': 1000, 'adenosine_receptor_agonist': 1000, 'adenosine_receptor_antagonist': 1000, 'adenylyl_cyclase_activator': 100, 'adrenergic_receptor_agonist': 1000, 'adrenergic_receptor_antagonist': 1000, 'akt_inhibitor': 1000, 'aldehyde_dehydrogenase_inhibitor': 100, 'alk_inhibitor': 100, 'ampk_activator': 10000, 'analgesic': 10000, 'androgen_receptor_agonist': 1000, 'androgen_receptor_antagonist': 1000, 'anesthetic_-_local': 1000, 'angiogenesis_inhibitor': 1000, 'angiotensin_receptor_antagonist': 1000, 'anti-inflammatory': 1000, 'antiarrhythmic': 10000, 'antibiotic': 10000, 'anticonvulsant': 1000, 'antifungal': 100, 'antihistamine': 1000, 'antimalarial': 10000, 'antioxidant': 1000, 'antiprotozoal': 1000, 'antiviral': 1000, 'apoptosis_stimulant': 1000, 'aromatase_inhibitor': 1000, 'atm_kinase_inhibitor': 100, 'atp-sensitive_potassium_channel_antagonist': 10000, 'atp_synthase_inhibitor': 100, 'atpase_inhibitor': 1000, 'atr_kinase_inhibitor': 100, 'aurora_kinase_inhibitor': 100, 'autotaxin_inhibitor': 1000, 'bacterial_30s_ribosomal_subunit_inhibitor': 1000, 'bacterial_50s_ribosomal_subunit_inhibitor': 1000, 'bacterial_antifolate': 1000, 'bacterial_cell_wall_synthesis_inhibitor': 1000, 'bacterial_dna_gyrase_inhibitor': 1000, 'bacterial_dna_inhibitor': 1000, 'bacterial_membrane_integrity_inhibitor': 1000, 'bcl_inhibitor': 1000, 'bcr-abl_inhibitor': 10000, 'benzodiazepine_receptor_agonist': 1000, 'beta_amyloid_inhibitor': 10000, 'bromodomain_inhibitor': 1000, 'btk_inhibitor': 1000, 'calcineurin_inhibitor': 100, 'calcium_channel_blocker': 1000, 'cannabinoid_receptor_agonist': 1000, 'cannabinoid_receptor_antagonist': 1000, 'carbonic_anhydrase_inhibitor': 1000, 'casein_kinase_inhibitor': 1000, 'caspase_activator': 1000, 'catechol_o_methyltransferase_inhibitor': 10000, 'cc_chemokine_receptor_antagonist': 1000, 'cck_receptor_antagonist': 1000, 'cdk_inhibitor': 10, 'chelating_agent': 1000, 'chk_inhibitor': 100, 'chloride_channel_blocker': 1000, 'cholesterol_inhibitor': 10000, 'cholinergic_receptor_antagonist': 1000, 'coagulation_factor_inhibitor': 10000, 'corticosteroid_agonist': 100, 'cyclooxygenase_inhibitor': 100, 'cytochrome_p450_inhibitor': 1000, 'dihydrofolate_reductase_inhibitor': 10000, 'dipeptidyl_peptidase_inhibitor': 1000, 'diuretic': 1000, 'dna_alkylating_agent': 1000, 'dna_inhibitor': 1000, 'dopamine_receptor_agonist': 1000, 'dopamine_receptor_antagonist': 1000, 'egfr_inhibitor': 100, 'elastase_inhibitor': 10000, 'erbb2_inhibitor': 10000, 'estrogen_receptor_agonist': 100, 'estrogen_receptor_antagonist': 1000, 'faah_inhibitor': 1000, 'farnesyltransferase_inhibitor': 100, 'fatty_acid_receptor_agonist': 1000, 'fgfr_inhibitor': 1000, 'flt3_inhibitor': 1000, 'focal_adhesion_kinase_inhibitor': 100, 'free_radical_scavenger': 1000, 'fungal_squalene_epoxidase_inhibitor': 100, 'gaba_receptor_agonist': 1000, 'gaba_receptor_antagonist': 1000, 'gamma_secretase_inhibitor': 100, 'glucocorticoid_receptor_agonist': 100, 'glutamate_inhibitor': 1000, 'glutamate_receptor_agonist': 1000, 'glutamate_receptor_antagonist': 1000, 'gonadotropin_receptor_agonist': 1000, 'gsk_inhibitor': 100, 'hcv_inhibitor': 1000, 'hdac_inhibitor': 10, 'histamine_receptor_agonist': 1000, 'histamine_receptor_antagonist': 1000, 'histone_lysine_demethylase_inhibitor': 100, 'histone_lysine_methyltransferase_inhibitor': 1000, 'hiv_inhibitor': 1000, 'hmgcr_inhibitor': 1, 'hsp_inhibitor': 10000, 'igf-1_inhibitor': 1000, 'ikk_inhibitor': 10000, 'imidazoline_receptor_agonist': 1000, 'immunosuppressant': 1000, 'insulin_secretagogue': 1000, 'insulin_sensitizer': 1000, 'integrin_inhibitor': 1000, 'jak_inhibitor': 100, 'kit_inhibitor': 100, 'laxative': 10000, 'leukotriene_inhibitor': 1000, 'leukotriene_receptor_antagonist': 1000, 'lipase_inhibitor': 10000, 'lipoxygenase_inhibitor': 1000, 'lxr_agonist': 100, 'mdm_inhibitor': 100, 'mek_inhibitor': 10000, 'membrane_integrity_inhibitor': 1000, 'mineralocorticoid_receptor_antagonist': 1000, 'monoacylglycerol_lipase_inhibitor': 10000, 'monoamine_oxidase_inhibitor': 1000, 'monopolar_spindle_1_kinase_inhibitor': 100, 'mtor_inhibitor': 1000, 'mucolytic_agent': 1000, 'neuropeptide_receptor_antagonist': 1000, 'nfkb_inhibitor': 100, 'nicotinic_receptor_agonist': 10000, 'nitric_oxide_donor': 1000, 'nitric_oxide_production_inhibitor': 10000, 'nitric_oxide_synthase_inhibitor': 1000, 'norepinephrine_reuptake_inhibitor': 1000, 'nrf2_activator': 10000, 'opioid_receptor_agonist': 1000, 'opioid_receptor_antagonist': 1000, 'orexin_receptor_antagonist': 1000, 'p38_mapk_inhibitor': 1000, 'p-glycoprotein_inhibitor': 1000, 'parp_inhibitor': 100, 'pdgfr_inhibitor': 100, 'pdk_inhibitor': 1000, 'phosphodiesterase_inhibitor': 100, 'phospholipase_inhibitor': 1000, 'pi3k_inhibitor': 1000, 'pkc_inhibitor': 10000, 'potassium_channel_activator': 10000, 'potassium_channel_antagonist': 10000, 'ppar_receptor_agonist': 1000, 'ppar_receptor_antagonist': 1000, 'progesterone_receptor_agonist': 1000, 'progesterone_receptor_antagonist': 1000, 'prostaglandin_inhibitor': 1000, 'prostanoid_receptor_antagonist': 1000, 'proteasome_inhibitor': 10, 'protein_kinase_inhibitor': 1000, 'protein_phosphatase_inhibitor': 100, 'protein_synthesis_inhibitor': 10000, 'protein_tyrosine_kinase_inhibitor': 10000, 'radiopaque_medium': 1000, 'raf_inhibitor': 1, 'ras_gtpase_inhibitor': 10000, 'retinoid_receptor_agonist': 100, 'retinoid_receptor_antagonist': 100, 'rho_associated_kinase_inhibitor': 1000, 'ribonucleoside_reductase_inhibitor': 10000, 'rna_polymerase_inhibitor': 1000, 'serotonin_receptor_agonist': 1000, 'serotonin_receptor_antagonist': 1000, 'serotonin_reuptake_inhibitor': 1000, 'sigma_receptor_agonist': 1000, 'sigma_receptor_antagonist': 1000, 'smoothened_receptor_antagonist': 10000, 'sodium_channel_inhibitor': 1000, 'sphingosine_receptor_agonist': 1000, 'src_inhibitor': 1000, 'steroid': 1000, 'syk_inhibitor': 1000, 'tachykinin_antagonist': 1000, 'tgf-beta_receptor_inhibitor': 100, 'thrombin_inhibitor': 1000, 'thymidylate_synthase_inhibitor': 1000, 'tlr_agonist': 10000, 'tlr_antagonist': 10000, 'tnf_inhibitor': 1000, 'topoisomerase_inhibitor': 10000, 'transient_receptor_potential_channel_antagonist': 1000, 'tropomyosin_receptor_kinase_inhibitor': 10000, 'trpv_agonist': 10000, 'trpv_antagonist': 1000, 'tubulin_inhibitor': 100, 'tyrosine_kinase_inhibitor': 10000, 'ubiquitin_specific_protease_inhibitor': 100, 'vegfr_inhibitor': 100, 'vitamin_b': 10000, 'vitamin_d_receptor_agonist': 100, 'wnt_inhibitor': 1000}
for target in tqdm(target_cols):

    y_oof = np.zeros(train.shape[0])

    y_test = np.zeros((test.shape[0], nfolds))

    alpha = alphas[target]

    

    for fold, (train_idx, val_idx) in enumerate(kf.split(train, train)):

        xtrain, xval = train.iloc[train_idx], train.iloc[val_idx]

        ytrain, yval = train_targets.iloc[train_idx], train_targets.iloc[val_idx]

        model = Ridge(alpha=alpha, normalize=False)

        model.fit(xtrain[features], ytrain[target])

        

        y_oof[val_idx] = model.predict(xval[features])

        y_test[:, fold] = model.predict(test[features])

    

    target_score = log_loss(train_targets[target], y_oof)

    print("Score for target {} is {}".format(target, np.round(target_score, 4)))

    overal_score += (1/num_targets) * target_score

    

    ss[target] = y_test.mean(axis=1)
print("Overal score: ", np.round(overal_score, 4))
ss.loc[test['cp_type']==1, target_cols] = 0
ss.to_csv('submission.csv', index=False)