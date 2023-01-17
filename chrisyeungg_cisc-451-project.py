import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Load data
data_dir = '/kaggle/input/stanford-covid-vaccine/'
train = pd.read_json(data_dir + 'train.json', lines=True)
test = pd.read_json(data_dir + 'test.json', lines=True)
sample_sub = pd.read_csv(data_dir + 'sample_submission.csv')
print(train.shape)
if ~train.isnull().values.any():
    print("No missing values.")
train.head()
print(test.shape)
if ~test.isnull().values.any():
    print("No missing values.")
test.head()
train_data = []
for mol_id in train['id'].unique():
    sample_data = train.loc[train['id'] == mol_id]
    sample_seq_length = sample_data.seq_length.values[0]
    
    for i in range(68):
        sample_dict = {'id' : sample_data['id'].values[0],
                       'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),
                       'sequence' : sample_data['sequence'].values[0][i],
                       'structure' : sample_data['structure'].values[0][i],
                       'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],
                       'reactivity' : sample_data['reactivity'].values[0][i],
                       'reactivity_error' : sample_data['reactivity_error'].values[0][i],
                       'deg_Mg_pH10' : sample_data['deg_Mg_pH10'].values[0][i],
                       'deg_error_Mg_pH10' : sample_data['deg_error_Mg_pH10'].values[0][i],
                       'deg_pH10' : sample_data['deg_pH10'].values[0][i],
                       'deg_error_pH10' : sample_data['deg_error_pH10'].values[0][i],
                       'deg_Mg_50C' : sample_data['deg_Mg_50C'].values[0][i],
                       'deg_error_Mg_50C' : sample_data['deg_error_Mg_50C'].values[0][i],
                       'deg_50C' : sample_data['deg_50C'].values[0][i],
                       'deg_error_50C' : sample_data['deg_error_50C'].values[0][i]}
        
        
#         shifts = [1,2,3,4,5]
#         shift_cols = ['sequence', 'structure', 'predicted_loop_type']
#         for shift,col in itertools.product(shifts, shift_cols):
#             if i - shift >= 0:
#                 sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]
#             else:
#                 sample_dict['b'+str(shift)+'_'+col] = -1
            
#             if i + shift <= sample_seq_length - 1:
#                 sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]
#             else:
#                 sample_dict['a'+str(shift)+'_'+col] = -1
        
        
        train_data.append(sample_dict)
train_data = pd.DataFrame(train_data)
train_data.head()
# Signal-to-noise and SN filter columns useful for incorporating the quality of the samples
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.kdeplot(train['signal_to_noise'], shade=True, ax=ax[0])
sns.countplot(train['SN_filter'], ax=ax[1])

ax[0].set_title('Signal/Noise Distribution')
ax[1].set_title('Signal/Noise Filter Distribution');
print(train.head())
train, test = train_test_split(train, test_size=0.2, random_state=2020)

train = train.query("signal_to_noise >= 1")
print(train.head())
#Value count for visualization
train_data['sequence'].value_counts().plot(title = 'Sequence Distribution', kind = 'barh')
train_data['predicted_loop_type'].value_counts().plot(kind = 'barh', title = 'Loop Type Distribution')
#Encode variables

# sequence_encmap = {'A': 0, 'G' : 1, 'C' : 2, 'U' : 3}
# structure_encmap = {'.' : 0, '(' : 1, ')' : 2}
# looptype_encmap = {'S':0, 'E':1, 'H':2, 'I':3, 'X':4, 'M':5, 'B':6}

# enc_targets = ['sequence', 'structure', 'predicted_loop_type']
# enc_maps = [sequence_encmap, structure_encmap, looptype_encmap]

# for t,m in zip(enc_targets, enc_maps):
#     for c in [c for c in train_data.columns if t in c]:
#         train_data[c] = train_data[c].replace(m)
#         test_data[c] = test_data[c].replace(m)

target_enc = {'sequence':{'A': 0, 'G' : 1, 'C' : 2, 'U' : 3},
                'structure': {'.' : 0, '(' : 1, ')' : 2}, 
                'looptype': {'S':0, 'E':1, 'H':2, 'I':3, 'X':4, 'M':5, 'B':6}}

train_data['sequence']=train_data.sequence.astype(str)
train_data['structure']=train_data.sequence.astype(str)
train_data['predicted_loop_type']=train_data.sequence.astype(str)

train_data.replace(target_enc, inplace=True)
train_data.head()
#Model training


#Linear Regression
X = train[['sequence', 'structure', 'predicted_loop_type']]
y = train[['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]

lin_reg = LinearRegression().fit(X, y)

#SVM
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X, y)
print(lin_reg.score(X, y))
#print(regr.score(X, y))