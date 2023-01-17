import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from glob import glob

import os

import gc

import seaborn as sns

from tqdm import tqdm_notebook

import rsna_hemorrhage_detetction_preprocessing as rsna

# Sklearn stuff

from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score

from sklearn.externals.joblib import Parallel, delayed

from scipy.stats import ttest_ind, f_oneway





# Metric - see https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/110461#latest-636870

from sklearn.metrics import log_loss

sample_weights = [2/7, 1/7, 1/7, 1/7, 1/7, 1/7]



threshold = 5

threshold2 = 0.001

data_samples = 2500



sns.set_context('notebook')
PATH = '../input/rsna-intracranial-hemorrhage-detection/' # Set up the path.

# Load the stage 1 file

train_csv = pd.read_csv(f'{PATH}stage_1_train.csv')

# Create a path to the train image location:

image_path = os.path.join(PATH, 'stage_1_train_images') + os.sep 

print(image_path)
# Check out this kernel: https://www.kaggle.com/currypurin/simple-eda 

# This is a really nice preprocessing of ID and labels :)

train_csv['Image ID'] = train_csv['ID'].apply(lambda x: x.split('_')[1]) 

train_csv['Sub-type'] = train_csv['ID'].apply(lambda x: x.split('_')[2]) 

train_csv = pd.pivot_table(train_csv, index='Image ID', columns='Sub-type')
print(train_csv.sum(0))

print(f'Total number:{train_csv.shape[0]}')
fix, axes = plt.subplots(1,2, figsize=(12.5,5))

plot_data = train_csv.sum(0).reset_index()

a = sns.barplot(data=plot_data, x='Sub-type', y=0, palette='viridis', ax=axes[0])

a.tick_params(axis='x', rotation=45)

a.set_ylabel('Counts')

plot_data.loc[:, 0] = plot_data.loc[:, 0].values / train_csv.shape[0]

a = sns.barplot(data=plot_data, x='Sub-type', y=0, palette='viridis', ax=axes[1])

a.tick_params(axis='x', rotation=45)

a.set_ylabel('Percentage');
# Percentage of multiple labels: 

train_csv_dropped = train_csv.copy()

train_csv_dropped.columns = train_csv_dropped.columns.droplevel(0)

no_labels = train_csv_dropped.query('any == 1').shape[0]

l_names = list(train_csv_dropped.columns[1:])

collect_data = []



for nn, lesion in enumerate(l_names) :

    select_vec = np.ones(len(l_names) + 1).astype(bool)

    select_vec[nn + 1 ] = 0

    select_vec[0] = 0



    plot_data = train_csv_dropped.query(f'{lesion}==1').sum(0).reset_index()

    collect_data.append(plot_data[0].values)

    

# co-occurence matrix

collect_data = np.array(collect_data)

fig, ax = plt.subplots(1,3, figsize=(15,15))

sns.heatmap(collect_data[:, 1:], annot=True, fmt='', xticklabels=l_names, yticklabels=l_names, ax=ax[0], square=True, cbar=False, cmap='viridis')

ax[0].set_title("Count")



sns.heatmap(collect_data[:, 1:]/np.diag(collect_data[:,1:]).reshape(-1, 1), annot=True, fmt='.2', xticklabels=l_names, ax=ax[1], square=True, cbar=False, cmap='viridis')

ax[1].set_title("Co-occurences per label")



sns.heatmap(collect_data[:, 1:]/no_labels, annot=True, fmt='.2', xticklabels=l_names, ax=ax[2], square=True, cbar=False, cmap='viridis')

ax[2].set_title('Percentage of all positive labels')



plt.tight_layout()
fig, ax = plt.subplots()

temp_hist = plt.hist(train_csv_dropped.query("any==1")[l_names].sum(1), bins=(np.arange(10))/2 + 1)

[print(f' Overlapping labels: {i + 1}, count: {ii:4.2f}\n') for i, ii in enumerate(temp_hist[0][::2])];
plt.figure(figsize=(15,5))

img, dicom = rsna.image_to_hu(image_path, train_csv.index[5])

plt.subplot(221)

plt.title('Hounsfield Transformed Image')

plt.imshow(img, cmap='bone')

plt.subplot(223)

plt.hist(img.ravel());

plt.subplot(222)

plt.title('Windowed Image')

img = rsna.image_windowed(img, 50, 130, False)

plt.imshow(img, cmap='bone')

plt.subplot(224)

plt.hist(img.ravel());
min_img = np.min(img)

max_img = np.max(img)

no_bins = (max_img - min_img) / 5

# Create evenly spaced bins across histograms:

histogram_bins = np.linspace(min_img, max_img, np.int(no_bins + 1))

sino_bins = np.linspace(-4000, 12000, 25)

def get_histogram(image_path, img_id, train_csv, bin_spacing, sino_bins):

    from skimage.transform import radon



    img, dicom = rsna.image_to_hu(image_path, img_id)

    img = rsna.image_resample(img, dicom)

    img = rsna.image_windowed(img, 50, 130, False)

    # Discard values outside of the window

    histogram_vals = np.histogram(img.ravel(), histogram_bins)[0][1:-1]

    

    # Next to simple histograms, we are also applying the radon transform - i.e. creating the sinogram. 

    theta = np.linspace(0., 180., max(img.shape), endpoint=False)

    sinogram = radon(img, theta=theta, circle=True)



    sinogram_hist = np.histogram(sinogram.ravel(), bins=sino_bins)[0]

    

    if train_csv is not None:

        labels = train_csv.loc[img_id].values

    else:

        labels = np.zeros(6)



    return histogram_vals, sinogram_hist, labels
# Get IDs based on their labels:

imageID_dict = {}

for les in l_names:

    imageID_dict[les] = train_csv_dropped.query(f'{les}==1').index.values



imageID_dict['nolabel'] = train_csv_dropped.query(f'any==0').index.values
# 2000 samples of each hemorrhage + 10000 empty values

# Set seed

from sklearn.externals.joblib import Parallel, delayed

np.random.seed(2019)

# If you run this, get a coffee or something ;) 

noSamples = np.array([data_samples] * len(l_names) + [data_samples * 5])



# Create empty X, and y arrays for training:

data = []



c = 0

for noS, dict_key in zip(noSamples, imageID_dict.keys()):

    

    if noS < len(imageID_dict[dict_key]):

        stepIDs = np.random.choice(imageID_dict[dict_key], noS, replace=False)

    else:

        stepIDs = np.random.choice(imageID_dict[dict_key], noS, replace=True)

    

    out = Parallel(n_jobs=-1, prefer="threads")(delayed(get_histogram)(image_path, tmpid,

                                               train_csv_dropped, histogram_bins, 

                                               sino_bins) for tmpid in tqdm_notebook(stepIDs))

    data.extend(out)

    gc.collect



S = np.hstack([n + np.zeros(k) for n, k in enumerate(noSamples)])
X = np.zeros((len(data), data[0][0].shape[0] + data[0][1].shape[0]))

Y = np.zeros((len(data), data[0][2].shape[0]))



for n, (x1, x2, y1) in tqdm_notebook(enumerate(data)):

    x1 = (x1 - np.mean(x1)) / np.std(x1)

    x2 = (x2 - np.mean(x2)) / np.std(x2)

    X[n, : x1.shape[0]] = x1

    X[n, x1.shape[0] : ] = x2

    Y[n, :] = y1



X = np.nan_to_num(X, 0)
HU_hist = np.histogram(np.arange(100), bins=histogram_bins)[1][1:-2]

SIN_hist = np.histogram(np.arange(100), bins=sino_bins)[1][:-1]
# Check if there's a difference between no lesion and lesions

ts = []

for ii in range(X.shape[1]):

    ts.append(ttest_ind(X[S != 5, ii], X[S == 5, ii])[0])



ts = np.array(ts)



huTs = ts[: HU_hist.shape[0]]

plot_dim = np.ceil(np.sqrt(huTs.shape))



plt.figure(figsize=(15,15))

for nn, hidx in enumerate(np.arange(huTs.shape[0])[np.abs(huTs) > threshold]):

    plt.subplot(plot_dim, plot_dim, nn + 1)

    plt.hist(X[S!=5, hidx], alpha=0.5)

    plt.hist(X[S==5, hidx], alpha=0.5)

    plt.title(f'HU = {HU_hist[hidx]} \n t={ts[hidx]:4.2f}')



plt.tight_layout()
offset = HU_hist.shape[0]

sinTs = ts[HU_hist.shape[0] :]



plot_dim = np.ceil(np.sqrt(sinTs.shape))



plt.figure(figsize=(15,15))

for nn, hidx in enumerate(np.arange(sinTs.shape[0])[np.abs(sinTs) > threshold]):

    plt.subplot(plot_dim, plot_dim, nn + 1)

    plt.hist(X[S!=5, hidx  + offset ], alpha=0.5)

    plt.hist(X[S==5, hidx  + offset], alpha=0.5)

    plt.title(f'SIN-Value = {np.round(SIN_hist[hidx])} \n t={ts[hidx + offset]:4.2f}')



plt.tight_layout()

binsOfInterest = np.where(np.abs(ts) > threshold)[0]

labels = np.array(['HU_' + str(i) for i in HU_hist] + ['SIN_' + str(np.round(i)) for i in SIN_hist])

labels = labels[binsOfInterest]
# First do an ANOVA to figure out which bins have significant differences between regions:

p_aov = np.zeros(binsOfInterest.shape)

for n, bOi in enumerate(binsOfInterest):

    aov = f_oneway(X[S==0, bOi], X[S==1, bOi], X[S==2, bOi], X[S==3, bOi], X[S==4, bOi])

    p_aov[n] = aov[1]
print(binsOfInterest[p_aov < threshold2])
selected_binsOfInterest = binsOfInterest[p_aov < threshold2]

selLabels = labels[p_aov < threshold2]

subplot_dims = np.ceil(np.sqrt(selected_binsOfInterest.shape[0]))



diffArrays = []



plt.figure(figsize=(15,15))

for n, bOi in enumerate(selected_binsOfInterest):

    plt.subplot(subplot_dims, subplot_dims, n + 1)

    diffMap = np.zeros((5,5))

    for rr in range(5):

        for cc in range(5):

            diffMap[rr, cc] = ttest_ind(X[S==rr, bOi], X[S==cc, bOi])[0]

    

    if n in [0, 4, 8, 12]:

        yl = l_names

    else:

        yl = []

    

    if n >= 12:

        xl = l_names

    else:

        xl = []

        

    sns.heatmap(np.abs(diffMap), annot=True, fmt='.2', square=True, cbar=False, xticklabels=xl, yticklabels=yl)

    plt.title(selLabels[n])

    diffArrays.append(diffMap)



plt.suptitle("Absolut size of difference\n abs(t)")
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

Xtrain, Xval, Ytrain, Yval, Strain, Sval = train_test_split(X, Y, S, stratify=S)
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

SKF = StratifiedKFold(5, random_state=24)

CLF = make_pipeline(StandardScaler(), LogisticRegressionCV(cv=3))
# Create different models:

CLF_dict = {}



c = 0



for trIdx, teIdx in SKF.split(Xtrain, Strain):

    CLF_dict[c] = {}

    

    xTr = Xtrain[trIdx]; yTr = Ytrain[trIdx]

    xTe = Xtrain[teIdx]; yTe = Ytrain[teIdx]

    yPred = np.zeros(yTe.shape)

    

    CLF.fit(xTr[:, binsOfInterest], yTr[:, 0])

    CLF_dict[c]['any'] = CLF

    CLF_dict[c]['teIdx'] = teIdx

    

    yPred[:, 0] = CLF.predict_proba(xTe[:, binsOfInterest])[:, 1]

    

    for n, les in enumerate(l_names):

        CLF.fit(xTr[:, selected_binsOfInterest], yTr[:, n+1])

        CLF_dict[c][les] = CLF

        yPred[:, n+1] = CLF.predict_proba(xTe[:, selected_binsOfInterest])[:, 1]

        

    CLF_dict[c]['Prediction'] = yPred

    CLF_dict[c]['True'] = yTe

    c += 1
# Classification accuracies:

someNorm = 0

baac = { i : [] for i in range(6)}

LL = []



for n, c in enumerate(CLF_dict.keys()):

    tempPred = CLF_dict[c]['Prediction'].copy()

    

    if someNorm: 

        tempPred[tempPred[:, 0] < 0.5, 1:] = 0



    for ii in range(6):

            baac[ii].append(roc_auc_score(CLF_dict[c]['True'][:, ii], tempPred[:, ii]))

    

    ll= log_loss(CLF_dict[c]["True"].ravel(), tempPred.ravel(), sample_weight=sample_weights * CLF_dict[c]["Prediction"].shape[0])

    LL.append(ll)



for n, les in enumerate(['any'] + l_names):

    print(f'{les} - AUC: {np.mean(baac[n])} +/- {np.std(baac[n])}')

print(f'LL: {np.mean(LL)} +/- {np.std(LL)}')
baac = []



yPred = np.zeros(Yval.shape)



CLF.fit(Xtrain[:, binsOfInterest], Ytrain[:, 0])



yPred[:, 0] = CLF.predict_proba(Xval[:, binsOfInterest])[:, 1]



for n, les in enumerate(l_names):

    CLF.fit(Xtrain[:, selected_binsOfInterest], Ytrain[:, n+1])

    yPred[:, n+1] = CLF.predict_proba(Xval[:, selected_binsOfInterest])[:, 1]





tempPred = yPred.copy()





if someNorm: 

    tempPred[tempPred[:, 0] < 0.5, 1:] = 0



for ii in range(6):

        baac.append(roc_auc_score(Yval[:, ii], tempPred[:, ii]))

LL= log_loss(Yval.ravel(), tempPred.ravel(), sample_weight=sample_weights * Yval.shape[0])





for n, les in enumerate(['any'] + l_names):

    print(f'{les} - AUC: {np.mean(baac[n])} +/- {np.std(baac[n])}')

print(f'LL: {np.mean(LL)} +/- {np.std(LL)}')
sub_csv = pd.read_csv(f'{PATH}stage_1_sample_submission.csv')



sub_csv['Image ID'] = sub_csv['ID'].apply(lambda x: x.split('_')[1]) 

sub_csv['Sub-type'] = sub_csv['ID'].apply(lambda x: x.split('_')[2]) 

sub_csv = pd.pivot_table(sub_csv, index='Image ID', columns='Sub-type')
sub_csv_dropped = sub_csv.copy()

sub_csv_dropped.columns = sub_csv_dropped.columns.droplevel(0)
test_images = sub_csv_dropped.index.values

test_image_path =  os.path.join(PATH, 'stage_1_test_images') + os.sep 
CLF_dictTest = {}



CLF = RandomForestClassifier(n_estimators=10)

CLF.fit(X[:, binsOfInterest], Y[:,0])

CLF_dictTest['any'] = CLF

for n, l in enumerate(l_names):

    CLF = RandomForestClassifier(n_estimators=10)

    CLF.fit(X[:, selected_binsOfInterest], Y[:, n + 1])

    CLF_dictTest[l] = CLF
import gc

del X

del Y

del Xtrain

del Xval

gc.collect()
def calculate_test_sample(timg, binsOfInterest, histogram_bins, sino_bins, selected_binsOfInterest, CLF, test_image_path):

    

    pred = []

    

    tmpX = get_histogram(test_image_path, timg, None, histogram_bins, sino_bins)

    tmpX = np.hstack([(tmpX[0] - np.mean(tmpX[0])) / np.std(tmpX[0]), 

                      (tmpX[1] - np.mean(tmpX[1])) / np.std(tmpX[1])])

    tmpX = np.nan_to_num(tmpX, 0)

    pred.append(CLF_dictTest['any'].predict_proba(tmpX[np.newaxis, binsOfInterest])[0, 1])

    for les in  l_names:

        pred.append(CLF_dictTest[les].predict_proba(tmpX[np.newaxis, selected_binsOfInterest])[0, 1])



    return timg, np.hstack(pred)
test_out = Parallel(n_jobs=-1, prefer="threads")(delayed(calculate_test_sample)(timg, binsOfInterest, 

                                                                                histogram_bins, sino_bins, 

                                                                                selected_binsOfInterest, 

                                                                                CLF, test_image_path) 

                                                 for timg in tqdm_notebook(test_images))
for csv_id, preds in test_out:

    sub_csv_dropped.loc[csv_id, :] = preds
# Inspired by https://www.kaggle.com/erikgaasedelen/pandas-tricks-with-averaged-baseline

# Just far more hacky

out = sub_csv_dropped.copy()

out = out.reset_index()



value_vars = out.columns[1:]

out= pd.melt(out, id_vars=['Image ID'], value_vars=value_vars, value_name='Label')

out['ID'] = out['Image ID'].str.cat(out['Sub-type'], sep='_')

out.drop(columns=['Image ID', 'Sub-type'], inplace=True)

out['ID'] = 'ID_' + out['ID']

out = out[['ID', 'Label']] 
out.to_csv('submission_hist_sino.csv', index=False)
# Quick Test if the csv contains all the information.

sub_csv = pd.read_csv(f'{PATH}stage_1_sample_submission.csv')

np.sum(np.sort(sub_csv['ID']) == np.sort(out['ID'])) == out.shape[0]