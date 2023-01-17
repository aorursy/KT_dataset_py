import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer

from xgboost import XGBRegressor

pd.set_option('display.max_columns', 100)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines = True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines = True)
sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
stop_codon = ["UAA", "UAG", "UGA"]
amino_acid = ["Ala", "Arg", "Asn", "Asp", "Cys", "Glu", "Gln", "Gly", "His", "Ile", "Leu", "Lys",
              "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]
codon_U = ["UUU", "UUC", "UUA", "UUG",
           "UCU", "UCC", "UCA", "UCG",
           "UAU", "UAC", "UAA", "UAG",
           "UGU", "UGC", "UGA", "UGG"]
codon_C = ["CUU", "CUC", "CUA", "CUG",
           "CCU", "CCC", "CCA", "CCG",
           "CAU", "CAC", "CAA", "CAG",
           "CGU", "CGC", "CGA", "CGG"]
codon_A = ["AUU", "AUC", "AUA", "AUG",
           "ACU", "ACC", "ACA", "ACG",
           "AAU", "AAC", "AAA", "AAG",
           "AGU", "AGC", "AGA", "AGG"]
codon_G = ["GUU", "GUC", "GUA", "GUG",
           "GCU", "GCC", "GCA", "GCG",
           "GAU", "GAC", "GAA", "GAG",
           "GGU", "GGC", "GGA", "GGG"]
#####Get the codons######
def codon(codon_in):
    
    translated_amino_acid = []

    #codon_in = train['sequence'].values[0]
    codon_txt = [(codon_in[i:i + 3]) for i in range(0, len(codon_in), 3)]
    for r in range(0, len(codon_txt)):
        if codon_txt[r] == "AUG":
            translated_amino_acid.append(amino_acid[12])
            r += 1
        elif codon_txt[r] == ("UUU" or "UUC"):
            translated_amino_acid.append(amino_acid[13])
            r += 1
        elif codon_txt[r] == ((("UUA" or "UUG") or ("CUU" or "CUA")) or ("CUG" or "CUC")):
            translated_amino_acid.append(amino_acid[10])
            r += 1
        elif codon_txt[r] == ((("UCU" or "UCA") or ("UCC" or "UCG")) or ("AGU" or "AGC")):
            translated_amino_acid.append(amino_acid[15])
            r += 1
        elif codon_txt[r] == ("UAU" or "UAC"):
            translated_amino_acid.append(amino_acid[18])
            r += 1
        elif codon_txt[r] == ("UGU" or "UGC"):
            translated_amino_acid.append(amino_acid[4])
            r += 1
        elif codon_txt[r] == "UGG":
            translated_amino_acid.append(amino_acid[17])
            r += 1
        elif codon_txt[r] == (("CCU" or "CCC") or ("CCA" or "CCG")):
            translated_amino_acid.append(amino_acid[14])
            r += 1
        elif codon_txt[r] == ("CAU" or "CAC"):
            translated_amino_acid.append(amino_acid[8])
            r += 1
        elif codon_txt[r] == ("CAA" or "CUG"):
            translated_amino_acid.append(amino_acid[6])
            r += 1
        elif codon_txt[r] == ((("CGU" or "CGC") or ("CGA" or "CGG")) or ("AGA" or "AGG")):
            translated_amino_acid.append(amino_acid[1])
            r += 1
        elif codon_txt[r] == (("AUU" or "AUC") or "AUA"):
            translated_amino_acid.append(amino_acid[9])
            r += 1
        elif codon_txt[r] == (("ACU" or "ACC") or ("ACA" or "ACG")):
            translated_amino_acid.append(amino_acid[16])
            r += 1
        elif codon_txt[r] == ("AAU" or "AAC"):
            translated_amino_acid.append(amino_acid[2])
            r += 1
        elif codon_txt[r] == ("AAA" or "AAG"):
            translated_amino_acid.append(amino_acid[11])
            r += 1
        elif codon_txt[r] == (("GUU" or "GUC") or ("GUA" or "GUG")):
            translated_amino_acid.append(amino_acid[19])
            r += 1
        elif codon_txt[r] == (("GCU" or "GCC") or ("GCA" or "GCG")):
            translated_amino_acid.append(amino_acid[0])
            r += 1
        elif codon_txt[r] == ("GAU" or "GAC"):
            translated_amino_acid.append(amino_acid[3])
            r += 1
        elif codon_txt[r] == ("GAA" or "GAG"):
            translated_amino_acid.append(amino_acid[5])
            r += 1
        elif codon_txt[r] == (("GGU" or "GGC") or ("GGA" or "GGG")):
            translated_amino_acid.append(amino_acid[7])
            r += 1
        elif codon_txt[r] == (("UAG" or "UAA") or "UGA"):
            translated_amino_acid.append("Stop")
            r += 1
    return translated_amino_acid
train['AminoAcid'] = ""
for i in range(len(train)):
    train['AminoAcid'][i] = codon(train['sequence'].values[i])
train_data = []
for mol_id in train['id'].unique():
    sample_data = train.loc[train['id']==mol_id]
    
    for i in range(68):
        sample_tuple = (sample_data['id'].values[0], sample_data['AminoAcid'].values[0],i,sample_data['sequence'].values[0][i],
                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i],
                        sample_data['reactivity'].values[0][i], sample_data['reactivity_error'].values[0][i],
                        sample_data['deg_Mg_pH10'].values[0][i], sample_data['deg_error_Mg_pH10'].values[0][i],
                        sample_data['deg_pH10'].values[0][i], sample_data['deg_error_pH10'].values[0][i],
                        sample_data['deg_Mg_50C'].values[0][i], sample_data['deg_error_Mg_50C'].values[0][i],
                        sample_data['deg_50C'].values[0][i], sample_data['deg_error_50C'].values[0][i])
        train_data.append(sample_tuple)
train_data = pd.DataFrame(train_data, columns=['id', 'AminoAcid','seqno','sequence', 'structure', 'predicted_loop_type', 'reactivity', 'reactivity_error', 'deg_Mg_pH10', 'deg_error_Mg_pH10',
                                  'deg_pH10', 'deg_error_pH10', 'deg_Mg_50C', 'deg_error_Mg_50C', 'deg_50C', 'deg_error_50C'])
test['AminoAcid'] = ""
for i in range(len(test)):
    test['AminoAcid'][i] = codon(test['sequence'].values[i])

test_data = []
for mol_id in test['id'].unique():
    sample_data = test.loc[test['id'] == mol_id]
    for i in range(sample_data['seq_scored'].values[0]):
        sample_tuple = (sample_data['id'].values[0] + f'_{i}', sample_data['AminoAcid'].values[0],i,sample_data['sequence'].values[0][i],
                        sample_data['structure'].values[0][i], sample_data['predicted_loop_type'].values[0][i])
        test_data.append(sample_tuple)
test_data = pd.DataFrame(test_data, columns=['id','AminoAcid', 'seqno','sequence', 'structure', 'predicted_loop_type'])
test_data.head(1)
train_data1 = pd.concat([train_data.drop('AminoAcid', 1), 
                         pd.get_dummies(train_data['AminoAcid'].apply(pd.Series).stack()).sum(level=0)], 1)

test_data1 = pd.concat([test_data.drop('AminoAcid', 1), 
                         pd.get_dummies(test_data['AminoAcid'].apply(pd.Series).stack()).sum(level=0)], 1)
map_sequence ={'G':'1','A':'2','C':'3','U':'4'}
map_structure = {'.':'1','(':'2',')':'3'}
map_loop = {'E':'1', 'S':'2', 'H':'3', 'B':'4', 'X':'5', 'I':'6', 'M':'7'}

train_data1['sequence'] = train_data1['sequence'].map(map_sequence)
train_data1['structure'] = train_data1['structure'].map(map_structure)
train_data1['predicted_loop_type'] = train_data1['predicted_loop_type'].map(map_loop)

test_data1['sequence'] = test_data1['sequence'].map(map_sequence)
test_data1['structure'] = test_data1['structure'].map(map_structure)
test_data1['predicted_loop_type'] = test_data1['predicted_loop_type'].map(map_loop)
# Split data in features and labels
X_train = train_data1.drop(['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C','reactivity_error','deg_error_Mg_pH10','deg_pH10',
                            'deg_error_pH10','deg_error_Mg_50C','deg_50C','deg_error_50C'], axis=1)
Y_train = train_data1[['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']]
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
def mcrmse_loss(y_true, y_pred, N=3):
    """
    Calculates competition eval metric
    """
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    return np.sum(np.sqrt(np.sum((y_true - y_pred)**2, axis=0)/n)) / N

custom_scorer = make_scorer(mcrmse_loss, greater_is_better=False)
# Basic XGB without hyperparameter tuning
xgb = XGBRegressor(
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=1,
    random_state=42,
    n_estimators = 1000,
    learning_rate = 0.1,
    max_depth = 5
)

reg = MultiOutputRegressor(xgb)

reg.fit(X_train.drop('id',1), Y_train)
# Train score
mcrmse_loss(reg.predict(X_train.drop('id',1)), np.array(Y_train))
# Predict
preds = pd.DataFrame(reg.predict(test_data1.drop('id',1)))
preds = preds.rename(columns={0: 'reactivity', 1: 'deg_Mg_pH10', 2: 'deg_Mg_50C'})
preds['id'] = test_data1['id']
sub = pd.merge(sub[['id_seqpos']], preds, left_on='id_seqpos', right_on='id', how='left').drop(['id'],axis=1)
sub = sub.fillna(0)
sub['deg_pH10'] = 0
sub['deg_50C'] = 0
sub = sub[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
sub.head()