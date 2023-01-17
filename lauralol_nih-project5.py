!conda install -y -c rdkit rdkit
path ='../input/mlpredictdrugclass/'
from IPython.core.display import Image
Image(filename=path+'Img/SMILES-Figures.png')
import os, warnings
import numpy as np
import pandas as pd

from IPython.core.display import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor, PandasTools
from rdkit.Chem.Draw import IPythonConsole 
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions 
from concurrent import futures

warnings.filterwarnings('ignore')
IPythonConsole.molSize = (450,200)
imatinib = 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5'
imatinib_m = Chem.MolFromSmiles(imatinib)  #rdkit library
# generate 2D coordinates
rdDepictor.SetPreferCoordGen(True)
rdDepictor.Compute2DCoords(imatinib_m)
imatinib_m
! pip install py3Dmol
import py3Dmol

# The crystal structure of COVID-19 main protease in complex with an inhibitor N3
# The main protease (enzyme that catalyses/cuts proteins into smaller fragments) of coronavirus makes most of these cuts. The one shown here 
# (PDB entry 6lu7) is from the SARS-CoV-2 (or 2019-nCoV) coronavirus that is currently posing dangers in Wuhan

view = py3Dmol.view(query='pdb:6lu7')
view.setStyle({'cartoon':{'color':'spectrum'}})
Image(filename=path+'Img/FPComp.PNG',width = 300, height = 300 )
# (Following figure is based on an an online presentation)
IPythonConsole.molSize = (450,200)

# fever reducer
paracetamol = 'CC(=O)NC1=CC=C(O)C=C1'
paracetamol_m = Chem.MolFromSmiles(paracetamol)
rdDepictor.Compute2DCoords(paracetamol_m)

# withdrawn fever reducer
phenacetin = 'CCOC1=CC=C(NC(C)=O)C=C1'
phenacetin_m = Chem.MolFromSmiles(phenacetin)
rdDepictor.Compute2DCoords(phenacetin_m)

# save the molecules as a list
mols = [paracetamol_m, phenacetin_m]
Draw.MolsToGridImage(mols, subImgSize=(400, 300), molsPerRow = 2, legends = ['Paracetamol','Phenacetin'])
# instantiate a dictionary 
bi1 = {}

fp1 = AllChem.GetMorganFingerprintAsBitVect(paracetamol_m, radius=2, bitInfo=bi1)
bits1 = fp1.ToBitString()
print(len(bits1))
bits1
print(len(list(fp1.GetOnBits())))
print(list(fp1.GetOnBits()) )
# In its simplest form, the new code lets you display the atomic environment that sets a particular bit. Here we will look at bit 191:
Draw.DrawMorganBit(paracetamol_m,191,bi1)
bi2 = {}
fp2 = AllChem.GetMorganFingerprintAsBitVect(phenacetin_m, radius=2, bitInfo=bi2)
bits2 = fp2.ToBitString()

# In its simplest form, the new code lets you display the atomic environment that sets a particular bit. Here we will look at bit 191:
Draw.DrawMorganBit(phenacetin_m,191,bi2)
# Let us find common bits based on Dr. Jan Jensen's tutorial
# you use set operation by saving the result not as a list 

common_bits = set( fp1.GetOnBits()) & set(fp2.GetOnBits())

combined_bits = set( fp1.GetOnBits()) | set(fp2.GetOnBits())
print('Common_bits between Paracetamol and Phenacetin: ', common_bits,'\n')
print('Combined_bits between Paracetamol and Phenacetin: ', combined_bits)
# this will give the common bits and the proportion will tell us the similarity
print('Raw Calculation :', len(common_bits)/len(combined_bits),'\n')
# import the library
from rdkit import DataStructs

# Tanimoto Similarity
print('Tanimoto Similarity: ', DataStructs.TanimotoSimilarity(fp1, fp2))
!pip install mordred
# code credit from mordred manual
from mordred import Calculator, descriptors
n_all = len(Calculator(descriptors, ignore_3D=False).descriptors)
n_2D = len(Calculator(descriptors, ignore_3D=True).descriptors)

print("2D:    {:5}\n3D:    {:5}\n------------\ntotal: {:5}".format(n_2D, n_all - n_2D, n_all))
from rdkit import Chem
from mordred import Calculator, descriptors

# create descriptor calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=True)

IPythonConsole.molSize = (800,800)
dasatinib = 'CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=CC(=NC(=N3)C)N4CCN(CC4)CCO'
dasatinib_m = Chem.MolFromSmiles(dasatinib) 
rdDepictor.Compute2DCoords(dasatinib_m)

gnf5 = 'C1=CC(=CC(=C1)C(=O)NCCO)C2=CC(=NC=N2)NC3=CC=C(C=C3)OC(F)(F)F'
gnf5_m = Chem.MolFromSmiles(gnf5) 
rdDepictor.Compute2DCoords(gnf5_m)

dph = 'C1=CC=C(C=C1)N2C=C(C(=N2)C3=CC=C(C=C3)F)C4C(=O)NC(=O)N4'
dph_m = Chem.MolFromSmiles(dph)
rdDepictor.Compute2DCoords(dph_m)

molecules = [ imatinib_m, dasatinib_m, gnf5_m, dph_m ]
Draw.MolsToGridImage(molecules, molsPerRow = 2, subImgSize=(400, 250), legends = ['Imatinib','Dasatinib', 'GNF', 'DPH'])
# calculate multiple molecule
mols = [Chem.MolFromSmiles(smi) for smi in [imatinib, dasatinib, gnf5, dph]]

# as pandas
df = calc.pandas(mols)
df
from IPython.display import Image
Image(path+'Img/DrugFunctionModeling-banner.png', width=900, height=900)
## Preliminary library setup
import os, random, time, numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
df3  = pd.read_csv(path+'Data/3cls_rmsaltol.csv')

# five class dataset
df5  = pd.read_csv(path+'Data/5cls_rmsaltol.csv')

print("Here are few first/last 5 lines of the df3 data")
df3.iloc[0:6, [1,2]]
# All the data
print('Dimension of 3-class dataset', df3.shape)
print('Dimension of 5-class dataset', df5.shape)
# print('Dimension of 12-class dataset', df12.shape, '\n')
## Assign a dataset for analysis
df = df3
# convert the dataframe to numpy ndarray
x = df['smiles'].values
mols1 = [Chem.MolFromSmiles(smi) for smi in x]
outcome = df['class'].values

le = preprocessing.LabelEncoder()
le.fit(outcome);

print('What labels are available in classes?:', list(le.classes_))
ys_fit = le.transform(outcome)

print('transformed outcome:  ', ys_fit)
bin_count = np.bincount(ys_fit)
n_classes = len(bin_count)
print('How many classes? ',n_classes)
print('How many samples? ', len(ys_fit) )

print('How many from each class (raw numbers)? ', bin_count )
print('How many from each class (proportions)?: ', bin_count/(sum(bin_count)))
# Time to generate the Fingerprints: 8.323498249053955 seconds on core i7 laptop

time_start = time.time()

from rdkit.Chem import AllChem
fp1 = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols1]

# convert RDKit explicit vectors into NUMPY array
np_fps = np.asarray(fp1)

time_elapsed = time.time()-time_start
txt = 'Time to generate the Fingerprints: {} seconds '
print(txt.format(time_elapsed))
np_fps
print(np_fps[0:10,0:20])
from sklearn.model_selection import train_test_split
seed = 123

train_X, test_X, train_y, test_y = train_test_split(np_fps, ys_fit, 
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=seed,
                                                    stratify = ys_fit)
train_y = list(train_y)
test_y = list(test_y)
# Even outcome for this class
np.bincount(ys_fit)/len(ys_fit)
Image(path+'Img/PaperSummary1.png')
# get a random forest classifiert with 100 trees
seed = 1123
rf = RandomForestClassifier(n_estimators=50, random_state=seed)
from pprint import pprint
# View the parameters of the random forest
print('Parameters will be used for this model:\n')
pprint(rf.get_params())
# train the random forest
rf.fit(train_X, train_y);
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score

pred_y = rf.predict(test_X)
acc = metrics.accuracy_score(test_y, pred_y)
print("Test set accuracy: {:.2f}".format(acc))

balanced_acc_score = balanced_accuracy_score(test_y, pred_y)
print("Balanced set Accuracy Score: {:.2f}".format(balanced_acc_score))

# Plot non-normalized confusion matrix
# get a random forest classifiert with 100 trees
np.set_printoptions(precision=3)
from sklearn.metrics import plot_confusion_matrix

titles_options = [("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(rf, test_X, test_y,
                                 display_labels=le.classes_,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    plt.xticks(rotation=45)
    
    print(title)
    print(disp.confusion_matrix)

plt.show()
print(rf.predict(test_X[10:13]))
print(test_y[10:13])
# pred_y = rf_best_grid.predict(test_X)
# misclassified as a respiratory system drug 
cid_121878 = 'CC(=O)N(C1C(C(OC2=C1C=C(C=C2)C#N)(C)C)O)O'
cid_121878_m = Chem.MolFromSmiles(cid_121878)

rdDepictor.SetPreferCoordGen(True)
rdDepictor.Compute2DCoords(cid_121878_m)
# similarity with bronchodilator molecule 
cid_93504 = 'CC1(C(C(C2=C(O1)C=CC(=C2)C#N)N3CCCC3=O)O)C'
cid_93504_m = Chem.MolFromSmiles(cid_93504)
rdDepictor.Compute2DCoords(cid_93504_m)


molecules = [ cid_121878_m, cid_93504_m ]
Draw.MolsToGridImage(molecules, molsPerRow = 2, 
                     subImgSize=(450, 450), 
                     legends = ['PubChem ID 121878', 'Levcromakalim (bronchodilator)'])
