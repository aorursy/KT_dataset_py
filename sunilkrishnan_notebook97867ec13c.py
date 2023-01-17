import os

import numpy as np

import pandas as pd

from tqdm import tqdm

def verify_clean_raw(dft,dfl):

    ctrl_type = 0

    trtcp_no_label=0

    ctl_no_label = 0

    sigids_with_no_label = []

    label_distr = {str(x):0 for x in range(10)}



    for i in tqdm(range(len(dft))):

        s=dfl.loc[dfl['sig_id']==dft.loc[i,'sig_id']].sum(axis=1)

    

        #print(dft.loc[i,'sig_id'],dft.loc[i,'cp_type'],s.values[0])



        if  s.values[0] == 0:

            trtcp_no_label += 1

            sigids_with_no_label.append(dft.loc[i,'sig_id'])







        try:

                newcnt =  label_distr[str(s.values[0])] + 1

        except:

                newcnt = s.values[0]



        upd = {str(s.values[0]):newcnt}



        label_distr.update(upd)



    trt = dft[dft['cp_type']=='trt_cp']

    ctl = dft[dft['cp_type']=='ctl_vehicle']

    print('Total:{},trt_cp:{},ctl_vehicle:{},Samples with NO LABEL(incl of ctl_vehicle):{}'.format(len(dft),len(trt),len(ctl),trtcp_no_label))

    #print (label_distr)

    #print(sigids_with_no_label[0:5])

    dft = dft[~dft['sig_id'].isin(sigids_with_no_label)]

    #print(dft.shape)

    return dft

DATA_BASE_DIR = "/kaggle/input/lish-moa/"

dft= pd.read_csv(os.path.join(DATA_BASE_DIR,'train_features.csv'))

dfl= pd.read_csv(os.path.join(DATA_BASE_DIR,'train_targets_scored.csv'))

dftest = pd.read_csv(os.path.join(DATA_BASE_DIR,'test_features.csv'))

#some stats, and filter Training samples

dft= verify_clean_raw(dft,dfl)







all_tcols = list(dft)

g = ['g-'+str(i) for i in range(772)]

c = ['c-'+str(i) for i in range(100)]

all_gc =  g + c



#Labels/Targets

all_labels = list(dfl)

all_labels.remove('sig_id')



#label wise DS

lblds=[]

for lbl in all_labels:

     lblds.append(dfl.loc[dfl[lbl]==1])



label_wise_selected_features = []

new_train =[]

for i in tqdm(range(len(lblds))):

        temp_df=dft.loc[ (dft['sig_id'].isin(lblds[i]['sig_id']))]

        corr_matrix = temp_df.corr().abs()

        # Create a True/False mask and apply it

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        tri_df = corr_matrix.mask(mask)

        # List column names of highly correlated features (r > 0.8)

        to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.8) and c != 'cp_time']

        # Drop the features in the to_drop list

        reduced_df = temp_df.drop(to_drop, axis=1)

        label_wise_selected_features.append(list(reduced_df))

        new_train.append(reduced_df)



lbl_train = 0

train_counts =[]

stat_records =[]

for i in range(len(label_wise_selected_features)):



    lbl_train += len(lblds[i])

    train_counts.append(len(lblds[i]))

    stat_records.append([all_labels[i],len(lblds[i]),len(label_wise_selected_features[i])])

    print('{})Label:{}  Train:{}  features:{}'.format(i,all_labels[i],len(lblds[i]),len(label_wise_selected_features[i])))



stat_df = pd.DataFrame(stat_records,columns=['Label','TrainCount','Independent'])



stat_df