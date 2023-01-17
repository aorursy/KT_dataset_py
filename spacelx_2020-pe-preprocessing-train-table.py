import numpy as np

import pandas as pd



# path management

from pathlib import Path



# progress bars

from tqdm import tqdm
comp_data_path = Path('../input/rsna-str-pulmonary-embolism-detection')

prep_data_path = Path('../input/2020pe-preprocessed-train-data')
# set sizing

NSCANS = 20

NPX = 128
# load train data table

train = pd.read_csv(comp_data_path / 'train.csv')

# put data file names into dataframes

train['dcmpath'] = train.StudyInstanceUID + '_' + train.SeriesInstanceUID
# modify train table to make it fit to our model (combine images to make a set of 20 for each exam)

allsamples = np.unique(train.dcmpath.values)

train_new = pd.DataFrame()

for sss in tqdm(allsamples):

    selec = train[train['dcmpath'] == sss]

    thisdata = selec.iloc[0].copy()



    # get order of files in exam

    thisfilelist = np.load(str(prep_data_path / f'proc_{NSCANS}_{NPX}_train' / (thisdata['dcmpath'] + '_list.npy')), allow_pickle=True)

    thisfilelist = [str(f).split('/')[-1].split('.')[0] for f in thisfilelist]

    # get corresponding order of PE observation true/false

    ordered_obs = np.array([selec[selec['SOPInstanceUID'] == f]['pe_present_on_image'].values for f in thisfilelist]).flatten()

    # split in 20 equal sections as done for the images

    split = np.linspace(0, len(ordered_obs), num=NSCANS+1).astype(int)

    pe_obs_binned = np.zeros((NSCANS))

    for sss in range(NSCANS):

        pe_obs_binned[sss] = int(np.mean(ordered_obs[split[sss]:split[sss+1]]) > 0.3)



    # add binned PE observations to dataframe

    for iii in range(NSCANS):

        thisdata[f'pe_in_image_bin_{iii}'] = pe_obs_binned[iii]

    # add acute PE label

    thisdata['acute_pe'] = ((thisdata['negative_exam_for_pe'] == 0) &

                            (thisdata['indeterminate'] == 0) &

                            (thisdata['chronic_pe'] == 0) &

                            (thisdata['acute_and_chronic_pe'] == 0)

                           ).astype(int)

    train_new = train_new.append(thisdata, ignore_index=True)



# drop unneeded labels

drop_labels = ['qa_motion', 'qa_contrast', 'flow_artifact', 'pe_present_on_image', 'true_filling_defect_not_pe', 'SOPInstanceUID', 'SeriesInstanceUID', 'StudyInstanceUID']

train_new.drop(labels=drop_labels, axis=1, inplace=True)

train_new.to_csv('train_proc.csv')