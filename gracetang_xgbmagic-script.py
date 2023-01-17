import scipy.io as sio

import os

import numpy as np

import pandas as pd

import xgbmagic

import matplotlib.pyplot as plt



WDIR = '/Path/to/working/directory/'



class EEG():

    def __init__(self, patient_number='3', subsample=False):

        self.subsample = subsample

        self.patient_number = patient_number

        self.train_folder = 'train_' + patient_number

        self.test_folder = 'test_' + patient_number

        print('patient_num', patient_number)

        self.raw_data = []

 

    def get_train_data(self):

        print('preparing training data')

        dir_folders =(os.listdir(WDIR))

        count = 0

        for folder in dir_folders:

            if folder.startswith('train_' + self.patient_number):

                count += 1

                if count == 1:

                    first = True

                else:

                    first = False

                train_data = []

                mat_files = os.listdir(WDIR + folder)

                file_count = 0

                if self.subsample:

                    permissible = [self.patient_number+'_'+str(x) for x in range(0,200)]

                for mat_file in mat_files: # debug

                    if self.subsample:

                        if mat_file[:-6] not in permissible:

                            continue

                    if file_count % 10 == 0:

                        print('PROGRESS:', file_count,'/',len(mat_files))

                    file_count += 1

                    hit = mat_file[-5]

                    row = self.extract(WDIR + folder, mat_file, target = hit, first=first)

                    if row:

                        train_data.append(row)

        self.train_data = train_data



    def get_test_data(self):

        print('preparing test data')

        dir_folders =(os.listdir(WDIR))

        count = 0

        for folder in dir_folders:

            if folder.startswith('test_' + self.patient_number):

                test_data = []

                mat_files = os.listdir(WDIR + folder)

                file_count = 0

                for mat_file in mat_files: # debug

                    if file_count % 10 == 0:

                        print('PROGRESS:', file_count,'/',len(mat_files))

                    file_count += 1

                    row = self.extract(WDIR + folder, mat_file, target = None, first=False, test=True)

                    if row:

                        test_data.append(row)

                    else:

                        test_data.append({'id': mat_file})

        self.test_data = test_data





    def extract(self, folder, filename, target=None, first=False, test=False):

        print('reading', filename)

        try:

            data = sio.loadmat(folder + '/' + filename)

            eeg = data['dataStruct'][0][0][0]

            self.raw_data.append(eeg)

            all_channel_data = self.preprocess(eeg, test=test)

            if all_channel_data:

                row = self.generate_features(all_channel_data, target, filename)

                return row

        except Exception as e:

            print(e)



    def preprocess(self, data, test=False):

        # if all zeros

        if np.max(np.abs(data)) == 0:

            return

        # if substantial dropout

        keep_rows = [x for x in range(data.shape[0]) if sum(data[x,:])!=0]

        data = data[keep_rows,:]

        if data.shape[0] < 0.8*240000 and not test:

            print('dropping cos more than 20% dropout')

            return

        #preproc

        all_channel_data = [data[:,x] for x in range(data.shape[1])]

        return all_channel_data





    def generate_features(self, data, target, filename):

        """

        format row

        """

        res_dict = {}

        if target != None:

            res_dict['target']  = int(target)

        res_dict['id']  = filename

        # TODO: derive more features from channel data

        res_dict['max_amplitude'], res_dict['mean_amplitude']  = _abs_max_mean(data)

        if res_dict['max_amplitude'] == 0:

            # if all data is zero, drop

            print('all values zero - drop run ', filename)

            return

        for idx, channel_dat in enumerate(data):

            channel = str(idx+1)

            res_dict['max_amplitude_'+str(channel)], res_dict['mean_amplitude_'+str(channel)]  = _abs_max_mean(channel_dat)

        return res_dict





    def train(self):

        print('training')

        df = pd.DataFrame(self.train_data)

        print(df['target'])

        xgb = xgbmagic.Xgb(df, target_column='target', id_column='id')

        xgb.train()

        test_df = pd.DataFrame(self.test_data)

        output = xgb.predict(test_df)

        xgb.write_csv('output-epilepsy_'+self.patient_number+'.csv')

        print(xgb.feature_importance())



def _abs_max_mean(dat):

    vals = np.abs(dat)

    return np.max(vals), np.mean(vals)

import numpy as np

import matplotlib.pyplot as plt



import pandas as pd
for num in range(1,2):

    patient = str(num + 1)

    print(patient)

    eeg = EEG(patient_number=patient, subsample=True)

    eeg.get_train_data()

    eeg.get_test_data()

    eeg.train()