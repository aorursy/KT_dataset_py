import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
from colorama import Fore, Style
from tqdm.notebook import tqdm

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go

from IPython import display

sys.path.append('../input/iterative-stratification/iterative-stratification-master')
warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
plt.style.use("classic")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = "cuda" if torch.cuda.is_available() else "cpu"
def cout(string: str, color: str) -> str:
    """
    Prints a string in the required color
    """
    print(color+string+Style.RESET_ALL)
    
def statistics(dataframe, column):
    cout(f"The Average value in {column} is: {dataframe[column].mean():.2f}", Fore.RED)
    cout(f"The Maximum value in {column} is: {dataframe[column].max()}", Fore.BLUE)
    cout(f"The Minimum value in {column} is: {dataframe[column].min()}", Fore.YELLOW)
    cout(f"The 25th Quantile of {column} is: {dataframe[column].quantile(0.25)}", Fore.GREEN)
    cout(f"The 50th Quantile of {column} is: {dataframe[column].quantile(0.50)}", Fore.CYAN)
    cout(f"The 75th Quantile of {column} is: {dataframe[column].quantile(0.75)}", Fore.MAGENTA)
train_feats = pd.read_csv("../input/lish-moa/train_features.csv")
train_targets = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
test_feats = pd.read_csv("../input/lish-moa/test_features.csv")
sub = pd.read_csv("../input/lish-moa/sample_submission.csv")

data = pd.concat([train_feats, test_feats])
cout("Combined Train and Test Features", Fore.BLUE)
data.head()
cout("Train Targets", Fore.RED)
train_targets.head()
data[['cp_type']].describe()
df = data.groupby(['cp_type'])['sig_id'].count().reset_index()
df.columns = ['type', 'count']

fig = px.bar(
    df,
    x='type',
    y='count',
    color = 'type',
    width=650,
    height=500,
    title="Type of Treatment",
    labels={'type':"Type", 'count':"Count"},
)

fig.show()
fig = px.pie(
    df,
    names='type',
    values='count',
    hole=0.3,
    color_discrete_sequence=px.colors.sequential.Cividis,
    title="Type of Treatment - Pie Chart",
)

fig.show()
statistics(data, "cp_time")
df = data.groupby(['cp_time'])['sig_id'].count().reset_index()
df.columns = ['time', 'count']

fig = px.bar(
    df,
    x='time',
    y='count',
    color = 'count',
    width=650,
    height=500,
    title="Time of Treatment",
    labels={'time':"Time", 'count':"Count"},
)

fig.show()
fig = px.pie(
    df,
    names='time',
    values='count',
    hole=0.3,
    color_discrete_sequence=px.colors.sequential.Mint_r,
    title="Time of Treatment - Pie Chart",
)

fig.show()
data[['cp_dose']].describe()
df = data.groupby(['cp_dose'])['sig_id'].count().reset_index()
df.columns = ['dose', 'count']

fig = px.pie(
    df,
    names='dose',
    values='count',
    hole=0.3,
    color_discrete_sequence=px.colors.sequential.PuRd_r,
    title="Type of Dose Administered",
)

fig.show()
df = data.groupby(['cp_type', 'cp_time', 'cp_dose'])['sig_id'].count().reset_index()
df.columns = ['Type', 'Time', 'Dose', 'Count']

fig = px.sunburst(
    df, 
    path=[
        'Type',
        'Time',
        'Dose' 
    ], 
    values='Count', 
    title='Sunburst chart for Type, Time and Dose',
    width=600,
    height=600,
    color_discrete_sequence=px.colors.sequential.Sunset_r
)
fig.show()
train_targets.describe()
vals = train_targets.sum()[1:].sort_values().tolist()[:10]
names = list(dict(train_targets.sum()[1:].sort_values()).keys())[:10]
# Plot the sparsity of target matrix
plt.figure(figsize=(16, 9))
sns.barplot(names, vals)
plt.title("Top-10 Features with Most Sparse Target Matrix")
plt.xlabel("Feature Name")
plt.ylabel("Count of '1' in the feature")
plt.xticks(rotation=90)
plt.show()
correlation_matrix = pd.DataFrame()

columns = [i for i in train_feats.columns if i.startswith('g-')] + [i for i in train_feats.columns if i.startswith('c-')]

for t_col in tqdm(train_targets.columns):
    corr_list = list()
    if t_col == 'sig_id':
        continue
    for col in columns:
        res = train_feats[col].corr(train_targets[t_col])
        corr_list.append(res)
    correlation_matrix[t_col] = corr_list
correlation_matrix['train_features'] = columns
correlation_matrix = correlation_matrix.set_index('train_features')
correlation_matrix
# Encode categorical features in both training and testing sets

# DOSE
train_feats['cp_dose'] = train_feats['cp_dose'].map({'D1':0, 'D2':1})
test_feats['cp_dose'] = test_feats['cp_dose'].map({'D1':0, 'D2':1})

# CP_TYPE
train_feats['cp_type'] = train_feats['cp_type'].map({'trt_cp':0, 'ctl_vehicle':1})
test_feats['cp_type'] = test_feats['cp_type'].map({'trt_cp':0, 'ctl_vehicle':1})
# Remove column sig_id
train_feats = train_feats.drop(['sig_id'], axis=1)
test_feats = test_feats.drop(['sig_id'], axis=1)
train_targets = train_targets.drop(['sig_id'], axis=1)
important_feats = [  0,   1,   2,   3,   5,   6,   8,   9,  10,  11,  12,  14,  15,
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
class MOAData(Dataset):
    def __init__(self, feature_dataframe, target_dataframe=None, is_pred=False, important_features=important_feats):
        self.fd = feature_dataframe
        self.td = target_dataframe
        self.is_pred = is_pred
        self.important_features = important_feats
    
    def __getitem__(self, idx):
        item = self.fd.values[:, self.important_features][idx]
        
        if self.is_pred:
            return item, None
        
        else:
            target = self.td.astype(float).values[idx]
            return (item, target)
        
    def __len__(self):
        return len(self.fd)
class Network(nn.Module):
    def __init__(self, nb_feats):
        super(Network, self).__init__()
        
        self.nb_feats = nb_feats
        
        self.bn1 = nn.BatchNorm1d(num_features=self.nb_feats)
        self.fc1 = nn.utils.weight_norm(nn.Linear(in_features=nb_feats, out_features=512))
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.drp1 = nn.Dropout(0.2)
        self.fc2 = nn.utils.weight_norm(nn.Linear(in_features=512, out_features=256))
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.drp2 = nn.Dropout(0.3)
        self.out = nn.utils.weight_norm(nn.Linear(in_features=256, out_features=206))
        
    def forward(self, x):
        x = self.bn1(x)
        
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.drp1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.drp1(x)
        
        output = F.sigmoid(self.out(x))
        
        return output
model = Network(nb_feats=len(important_feats))
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()
# Training loop
nb_epochs = 5
save_path = "model_sav.pth"

result = train_targets.copy()

sub.loc[:, train_targets.columns] = 0
result.loc[:, train_targets.columns] = 0

for n, (train_split, val_split) in enumerate(MultilabelStratifiedKFold(n_splits=10, shuffle=True).split(train_targets, train_targets)):
#     cout(f"{'='*20} Fold: {n} {'='*20}", Fore.YELLOW)
    
    # Init a dataset object for every fold
    train_set = MOAData(
        feature_dataframe=train_feats.iloc[train_split].reset_index(drop=True),
        target_dataframe=train_targets.iloc[train_split].reset_index(drop=True),
    )
    
    valid_set = MOAData(
        feature_dataframe=train_feats.iloc[val_split].reset_index(drop=True),
        target_dataframe=train_targets.iloc[val_split].reset_index(drop=True),
    )
    
    # Connect them to dataloaders
    train = DataLoader(train_set, batch_size=32, num_workers=0)
    # valid = DataLoader(valid_set, batch_size=, shuffle=True, num_workers=2)
    
    # Run training epochs
    for epoch in range(nb_epochs):
        epoch_loss = 0
        model.train()
        
        for i, (x, y) in enumerate(train):
            x = torch.tensor(x, dtype=torch.float32).to(device)
            y = torch.tensor(y, dtype=torch.float32).to(device)
            
            # Train steps
            optim.zero_grad()
            z = model(x)
            loss = loss_fn(z, y)
            loss.backward()
            optim.step()
            
            # Calculate the loss
            # epoch_loss += loss.item()
            print(f"Fold: {n} | Epochs: {epoch}/{nb_epochs} | Batch: {i} | Loss: {loss.item():.4f}")
            display.clear_output(wait=True)
