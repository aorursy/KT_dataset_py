import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

from tqdm.notebook import tqdm



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



import warnings

warnings.filterwarnings('ignore')
train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')



ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
def preprocess(df):

    df = df.copy()

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(train_features)

test = preprocess(test_features)



del train_targets['sig_id']



train_targets = train_targets.loc[train['cp_type']==0].reset_index(drop=True)

train = train.loc[train['cp_type']==0].reset_index(drop=True)
def set_seed(seed):

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
nfolds = 7

nstarts = 7

nepochs = 50

batch_size = 128

val_batch_size = batch_size * 4

ntargets = train_targets.shape[1]

targets = [col for col in train_targets.columns]

criterion = nn.BCELoss()

kfold = MultilabelStratifiedKFold(n_splits=7, random_state=42, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MoaModel(nn.Module):

    def __init__(self, num_columns):

        super(MoaModel, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(num_columns)

        self.dropout1 = nn.Dropout(0.2)

        self.dense1 = nn.utils.weight_norm(nn.Linear(num_columns, 2048))

        

        self.batch_norm2 = nn.BatchNorm1d(2048)

        self.dropout2 = nn.Dropout(0.5)

        self.dense2 = nn.utils.weight_norm(nn.Linear(2048, 1048))

        

        self.batch_norm3 = nn.BatchNorm1d(1048)

        self.dropout3 = nn.Dropout(0.5)

        self.dense3 = nn.utils.weight_norm(nn.Linear(1048, 206))

    

    def forward(self, x):

        x = self.batch_norm1(x)

        x = self.dropout1(x)

        x = F.relu(self.dense1(x))

        

        x = self.batch_norm2(x)

        x = self.dropout2(x)

        x = F.relu(self.dense2(x))

        

        x = self.batch_norm3(x)

        x = self.dropout3(x)

        x = F.sigmoid(self.dense3(x))

        

        return x
top_feats = [  1,   2,   3,   4,   5,   6,   7,   9,  11,  14,  15,  16,  17,

        18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  29,  30,  31,

        32,  33,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  46,

        47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  58,  59,  60,

        61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,

        74,  75,  76,  78,  79,  80,  81,  82,  83,  84,  86,  87,  88,

        89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,

       102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,

       115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127, 128,

       129, 130, 131, 132, 133, 136, 137, 138, 139, 140, 141, 142, 143,

       144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157,

       158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,

       171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,

       184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 197,

       198, 199, 200, 202, 203, 204, 205, 206, 208, 209, 210, 211, 212,

       213, 214, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226,

       227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,

       240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,

       254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,

       267, 268, 269, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,

       281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294,

       295, 296, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,

       310, 311, 312, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323,

       324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336,

       337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,

       350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362,

       363, 364, 365, 366, 367, 368, 369, 370, 371, 374, 375, 376, 377,

       378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 391,

       392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,

       405, 406, 407, 408, 409, 411, 412, 413, 414, 415, 416, 417, 418,

       419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,

       432, 434, 435, 436, 437, 438, 439, 440, 442, 443, 444, 445, 446,

       447, 448, 449, 450, 453, 454, 456, 457, 458, 459, 460, 461, 462,

       463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475,

       476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487, 488, 489,

       490, 491, 492, 493, 494, 495, 496, 498, 500, 501, 502, 503, 505,

       506, 507, 509, 510, 511, 512, 513, 514, 515, 518, 519, 520, 521,

       522, 523, 524, 525, 526, 527, 528, 530, 531, 532, 534, 535, 536,

       538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 549, 550, 551,

       552, 554, 557, 559, 560, 561, 562, 565, 566, 567, 568, 569, 570,

       571, 572, 573, 574, 575, 577, 578, 580, 581, 582, 583, 584, 585,

       586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 599,

       600, 601, 602, 606, 607, 608, 609, 611, 612, 613, 615, 616, 617,

       618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630,

       631, 632, 633, 634, 635, 636, 637, 638, 639, 641, 642, 643, 644,

       645, 646, 647, 648, 649, 650, 651, 652, 654, 655, 656, 658, 659,

       660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,

       673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,

       686, 687, 688, 689, 691, 692, 693, 694, 695, 696, 697, 699, 700,

       701, 702, 704, 705, 707, 708, 709, 710, 711, 713, 714, 716, 717,

       718, 720, 721, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732,

       733, 734, 735, 737, 738, 739, 740, 742, 743, 744, 745, 746, 747,

       748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 759, 760, 761,

       762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774,

       775, 776, 777, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788,

       789, 790, 792, 793, 794, 795, 796, 797, 798, 800, 801, 802, 803,

       804, 805, 806, 808, 809, 811, 813, 814, 815, 816, 817, 818, 819,

       821, 822, 823, 825, 826, 827, 828, 829, 830, 831, 832, 834, 835,

       837, 838, 839, 840, 841, 842, 845, 846, 847, 848, 850, 851, 852,

       854, 855, 856, 858, 859, 860, 861, 862, 864, 866, 867, 868, 869,

       870, 871, 872, 873, 874]

print(len(top_feats))
# dataset class

class MoaDataset(Dataset):

    def __init__(self, df, targets, feats_idx, mode='train'):

        self.mode = mode

        self.feats = feats_idx

        self.data = df[:, feats_idx]

        if mode=='train':

            self.targets = targets

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        if self.mode == 'train':

            return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.targets[idx])

        elif self.mode == 'test':

            return torch.FloatTensor(self.data[idx]), 0
train = train.values

test = test.values

train_targets = train_targets.values
for seed in range(nstarts):

    print(f'Train seed {seed}')

    set_seed(seed)

    

    for n, (tr, te) in enumerate(kfold.split(train_targets, train_targets)):

        print(f'Train fold {n+1}')

        xtrain, xval = train[tr], train[te]

        ytrain, yval = train_targets[tr], train_targets[te]

        

        train_set = MoaDataset(xtrain, ytrain, top_feats)

        val_set = MoaDataset(xval, yval, top_feats)

        

        dataloaders = {

            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),

            'val': DataLoader(val_set, batch_size=val_batch_size, shuffle=False)

        }

        

        model = MoaModel(len(top_feats)).to(device)

        checkpoint_path = f'repeat:{seed}_Fold:{n+1}.pt'

        optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, eps=1e-4, verbose=True)

        best_loss = {'train': np.inf, 'val': np.inf}

        

        for epoch in range(nepochs):

            epoch_loss = {'train': 0.0, 'val': 0.0}

            

            for phase in ['train', 'val']:

                if phase == 'train':

                    model.train()

                else:

                    model.eval()

                

                running_loss = 0.0

                

                for i, (x, y) in enumerate(dataloaders[phase]):

                    x, y = x.to(device), y.to(device)

                    

                    optimizer.zero_grad()

                    

                    with torch.set_grad_enabled(phase=='train'):

                        preds = model(x)

                        loss = criterion(preds, y)

                        

                        if phase=='train':

                            loss.backward()

                            optimizer.step()

                        

                    running_loss += loss.item() / len(dataloaders[phase])

                

                epoch_loss[phase] = running_loss

            

            print("Epoch {}/{}   -   loss: {:5.5f}   -   val_loss: {:5.5f}".format(epoch+1, nepochs, epoch_loss['train'], epoch_loss['val']))

            

            scheduler.step(epoch_loss['val'])

            

            if epoch_loss['val'] < best_loss['val']:

                best_loss = epoch_loss

                torch.save(model.state_dict(), checkpoint_path)
oof = np.zeros((len(train), nstarts, ntargets))

oof_targets = np.zeros((len(train), ntargets))

preds = np.zeros((len(test), ntargets))
def mean_log_loss(y_true, y_pred):

    metrics = []

    for i, target in enumerate(targets):

        metrics.append(log_loss(y_true[:, i], y_pred[:, i].astype(float), labels=[0,1]))

    return np.mean(metrics)
for seed in range(nstarts):

    print(f"Inference for seed {seed}")

    seed_targets = []

    seed_oof = []

    seed_preds = np.zeros((len(test), ntargets, nfolds))

    

    for n, (tr, te) in enumerate(kfold.split(train_targets, train_targets)):

        xval, yval = train[te], train_targets[te]

        fold_preds = []

        

        val_set = MoaDataset(xval, yval, top_feats)

        test_set = MoaDataset(test, None, top_feats, mode='test')

        

        dataloaders = {

            'val': DataLoader(val_set, batch_size=val_batch_size, shuffle=False),

            'test': DataLoader(test_set, batch_size=val_batch_size, shuffle=False)

        }

        

        checkpoint_path = f'repeat:{seed}_Fold:{n+1}.pt'

        model = MoaModel(len(top_feats)).to(device)

        model.load_state_dict(torch.load(checkpoint_path))

        model.eval()

        

        for phase in ['val', 'test']:

            for i, (x, y) in enumerate(dataloaders[phase]):

                if phase == 'val':

                    x, y = x.to(device), y.to(device)

                elif phase == 'test':

                    x = x.to(device)

                

                with torch.no_grad():

                    batch_preds = model(x)

                    

                    if phase == 'val':

                        seed_targets.append(y)

                        seed_oof.append(batch_preds)

                    elif phase == 'test':

                        fold_preds.append(batch_preds)

                    

        fold_preds = torch.cat(fold_preds, dim=0).cpu().numpy()

        seed_preds[:, :, n] = fold_preds

        

    seed_targets = torch.cat(seed_targets, dim=0).cpu().numpy()

    seed_oof = torch.cat(seed_oof, dim=0).cpu().numpy()

    seed_preds = np.mean(seed_preds, axis=2)

    

    print("Score for this seed {:5.5f}".format(mean_log_loss(seed_targets, seed_oof)))

    oof_targets = seed_targets

    oof[:, seed, :] = seed_oof

    preds += seed_preds / nstarts



oof = np.mean(oof, axis=1)

print("Overall score is {:5.5f}".format(mean_log_loss(oof_targets, oof)))
ss[targets] = preds

ss.loc[test_features['cp_type']=='ctl_vehicle', targets] = 0

ss.to_csv('submission.csv', index=False)