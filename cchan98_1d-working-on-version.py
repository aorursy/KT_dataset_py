import glob
import random
import os
import numpy as np
import pandas as pd
import sklearn.preprocessing
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.python.keras import backend as k
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout, SeparableConv1D, LSTM, Bidirectional, GRU, Activation, BatchNormalization, Input, GlobalAveragePooling1D

from keras.layers.merge import add
from keras.utils.np_utils import to_categorical

!pip install git+https://github.com/raghakot/keras-vis.git -U
from vis.visualization import visualize_saliency
from vis.utils import utils
class DataSeries:
    def __init__(self, labelfunction, path, caselabels, classes, skiprows=1, include_only=None, regression=False):
        self.path = path
        self.caselabelspath = caselabels
        
        self.include_only = include_only
        self.skiprows = skiprows
        self.classes = classes
        self.n_classes = len(classes)
        self.regression = regression
        
        self.cases = self.get_cases()
        self.caselabels = labelfunction(self)

    def get_cases(self):
        cases = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and self.is_case_valid(d)]
        cases = sorted(cases)
        return cases

    def split_cases(self, ratio=0.75, seed=86, n_folds=4, fold_num=0):
        cases = np.copy(self.cases)
        if seed:
            random.seed(seed)
        random.shuffle(cases)
        training_sample_length = len(cases) // n_folds
        test = cases[training_sample_length * fold_num:training_sample_length * (fold_num + 1)] #0:1, 1:2, 2:3, 3:4
        train = np.hstack((cases[:training_sample_length * fold_num], cases[training_sample_length * (fold_num + 1):])) #:0 & 1: , :1 & 2:
        return train, test
    
    def get_labels_by_case_ap_left_right(self):
        labels = {}
        excelfile = pd.read_excel(self.caselabelspath, skiprows=self.skiprows)
        for case in self.cases:
            try:
                labels[case] = excelfile.loc[excelfile['Case ID'] == case, 'LeftVsRight'].values[0]
            except IndexError:
                print(f"Unable to find entry for {case}")
        return labels

    def get_labels_by_case_ap_lrs_regression(self):
        labels = {}
        excelfile = pd.read_excel(self.caselabelspath, skiprows=self.skiprows)
        for case in self.cases:
            try:
                label_lrs = excelfile.loc[excelfile['Case ID'] == case, 'LRS'].values[0]
                label_ap = excelfile.loc[excelfile['Case ID'] == case, 'AP'].values[0]
                if not np.isnan(label_lrs) and not np.isnan(label_ap):
                    labels[case] = np.array([label_lrs, label_ap])
            except IndexError:
                print(f"Unable to find entry for {case}")
        return labels

    def get_labels_by_case_ap_left_right_septal_other(self):
        labels = {}
        excelfile = pd.read_excel(self.caselabelspath, skiprows=self.skiprows)
        for case in self.cases:
            try:
                labels[case] = excelfile.loc[excelfile['Case ID'] == case, 'SeptalLateral'].values[0]
                if labels[case] == 'L':
                    labels[case] = excelfile.loc[excelfile['Case ID'] == case, 'LeftVsRight'].values[0]
                elif str(labels[case]) == 'nan':
                    labels[case] = 'O'
            except IndexError:
                print(f"Unable to find entry for {case} as {self.include_only} != 1")
        return labels

    def get_labels_by_case_ap_left_right_septal(self):
        labels = {}
        excelfile = pd.read_excel(self.caselabelspath, skiprows=self.skiprows)
        for case in self.cases:
            try:
                labels[case] = excelfile.loc[excelfile['Case ID'] == case, 'SeptalLateral'].values[0]
                if labels[case] == 'L':
                    labels[case] = excelfile.loc[excelfile['Case ID'] == case, 'LeftVsRight'].values[0]
            except IndexError:
                print(f"Unable to find entry for {case} as {self.include_only} != 1")
        return labels

    def get_labels_by_case_ap_ant_post(self):
        labels = {}
        excelfile = pd.read_excel(self.caselabelspath, skiprows=self.skiprows)
        for case in self.cases:
            try:
                labels[case] = excelfile.loc[excelfile['Case ID'] == case, 'PosteriorAnterior'].values[0]
            except IndexError:
                    print(f"Unable to find entry for {case}")
        return labels
    
    def get_labels_by_case_ap_ant_post_lsr(self):
        labels = {}
        excelfile = pd.read_excel(self.caselabelspath, skiprows=self.skiprows)
        for case in self.cases:
            try:
                labels[case] = excelfile.loc[excelfile['Case ID'] == case, 'CombinedLSRAP'].values[0]
            except IndexError:
                    print(f"Unable to find entry for {case}")
        return labels
    
    def get_train_test_data(self, train_test_ratio=0.75, reverse=True, n_folds=4, fold_num=0, downsample_ratio=None, crop=True,
                            axis_align=0, twoD=False, noise_level=0):
        train_cases, test_cases = self.split_cases(train_test_ratio, n_folds=n_folds, fold_num=fold_num)
        if crop:
            crop = self.get_longest_npy_file()
        if self.regression:
            train_x, train_y, train_n, train_caseids = self.data_and_labels_from_cases_regression(train_cases,crop,
                                                                                                   downsample_ratio)
            test_x, test_y, test_n, test_caseids = self.data_and_labels_from_cases_regression(test_cases,crop,
                                                                                               downsample_ratio)
        else:
            train_x, train_y, train_n, train_caseids = self.data_and_labels_from_cases_categorical(train_cases,crop,
                                                                                                   downsample_ratio)
            test_x, test_y, test_n, test_caseids = self.data_and_labels_from_cases_categorical(test_cases,crop,
                                                                                               downsample_ratio)
        if reverse:
            train_x = np.flip(train_x, axis=1)
            test_x = np.flip(test_x, axis=1)
            
        axalignmode = 0
        if axis_align == 1:
            train_x[:,:,3], test_x[:,:,3] = np.negative(train_x[:,:,3]), np.negative(test_x[:,:,3])
            limb_to_precordial = [4,0,3,1,5,2,6,7,8,9,10,11] #aVL,Lead1,-aVR,Lead2,aVF,Lead3,V1,V2,V3,V4,V5,V6
            print('Heart axis - ', limb_to_precordial)
            train_x, test_x = train_x[:,:,limb_to_precordial], test_x[:,:,limb_to_precordial]
        elif axis_align == 2:
            train_x[:,:,3], test_x[:,:,3] = np.negative(train_x[:,:,3]), np.negative(test_x[:,:,3])
            precordial_to_limb = [6,7,8,9,10,11,4,0,3,1,5,2] #V1,V2,V3,V4,V5,V6, aVL,Lead1,-aVR,Lead2,aVF,Lead3
            print('Heart axis - ', precordial_to_limb)
            train_x, test_x = train_x[:,:,precordial_to_limb], test_x[:,:,precordial_to_limb]
            
        if noise_level!=0 :
            train_x = add_noise(train_x)
            
        if twoD:
            train_x, test_x = np.expand_dims(train_x, 3), np.expand_dims(test_x, 3)
            
        return (train_x, train_y, train_n, train_caseids), (test_x, test_y, test_n, test_caseids)
    
    def data_and_labels_from_cases_regression(self, cases, croplength, downsample_ratio=None):
        x = []
        y = []
        caseids = []
        n_cases = 0
        for case in cases:
            n_cases += 1
            try:
                caselabel = self.caselabels[case]
                npyfiles = glob.glob(os.path.join(self.path, case, "*.npy"))
                if not len(npyfiles):
                    print(f"Unable to find any npyfile for case {case}")
                for npyfile in npyfiles:
                    data = np.load(npyfile)
                    if croplength:
                        result = np.full((croplength, data.shape[1]), np.nan) #Create nans for where we don't have data
                        result[:data.shape[0], :data.shape[1]] = data
                    else:
                        result = data

                    if downsample_ratio:
                        result = self.downsample_x(result, ratio=downsample_ratio)
                    caseids.append(case)
                    x.append(result)
                    y.append(caselabel)
            except KeyError:
                print(f"Unable to find label for {case}; skipping")
                pass
        x = np.stack(x)
        x = self.normalise_array(x)
        y = np.stack(y)

        return x, y, n_cases, caseids

    def data_and_labels_from_cases_categorical(self, cases, croplength, downsample_ratio=None, print_sample=0):
        warning_fired = False
        x = []
        y = []
        caseids = []
        n_cases = 0
        for case in cases:
            n_cases += 1
            try:
                caselabel = self.caselabels[case]
                npyfiles = glob.glob(os.path.join(self.path, case, "*.npy"))
                if not len(npyfiles):
                    #print(f"Unable to find any npyfile for case {case}")
                    pass
                for npyfile in npyfiles:
                    data = np.load(npyfile)
                    result = np.full((croplength, data.shape[1]), np.nan) #Create nans for where we don't have data
                    result[:data.shape[0], :data.shape[1]] = data

                    if caselabel and isinstance(caselabel, str):
                        if downsample_ratio:
                            result = self.downsample_x(result, ratio=downsample_ratio)
                        caseids.append(case)
                        x.append(result)
                        y.append(caselabel)
                    else:
                        #print(f"Case {case} not used for some reason - likely missing a label - further warnings supressed (likely more)")
                        pass
            except KeyError:
                print(f"Unable to find label for {case}; skipping")
                pass
        x = np.stack(x)
        x = self.normalise_array(x)
        y = self.list_of_strings_to_onehot(y, self.n_classes)

        return x, y, n_cases, caseids

    def list_of_strings_to_onehot(self, list_of_strings, n_classes):
        list_of_integers = [self.classes.index(string) for string in list_of_strings]
        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(max(list_of_integers) + 1))
        # if n_classes == 2: # Binary problem, labels are ints
        #     onehot = [value[0] for value in label_binarizer.transform(list_of_integers)]
        # else: # Categorical problem, labels are lists
        #     onehot = to_categorical(list_of_integers, n_classes)
        onehot = to_categorical(list_of_integers, n_classes)
        onehot = np.stack(onehot)
        return onehot
    
    @staticmethod
    def normalise_array(array):
        ''' NB THIS NORMALISES ALL COLUMNS EQUALLY - GOOD FOR ECGS, BUT USUALLY BAD
        np.nan REPLACED BY 0s AT END'''
        array -= np.nanmean(array)
        array /= np.nanstd(array)
        array = np.nan_to_num(array)
        return array

    def is_case_valid(self, case):
        excelfile = pd.read_excel(self.caselabelspath, skiprows=self.skiprows)
        if (not self.include_only) or (
                excelfile.loc[excelfile['Case ID'] == case, self.include_only[0]].values[0] == self.include_only[1]):
            return True

    @staticmethod
    def downsample_x(x, ratio):
        x_len = x.shape[0]
        x_interp = interp.interp1d(np.arange(x_len), x, axis=0)
        x_compress = x_interp(np.linspace(0, x_len - 1, int(x_len*ratio)))
        return x_compress

    def get_longest_npy_file(self):
        longestfilelength = 0
        npyfile_list = glob.glob(os.path.join(self.path, "**/*.npy"), recursive=True)
        for npyfile in npyfile_list:
            tmp = np.load(npyfile)
            if tmp.shape[0] > longestfilelength:
                longestfilelength = tmp.shape[0]
        return longestfilelength

from keras.models import load_model
from keras.callbacks import TensorBoard

class Network():
    def train(self, x, y, epochs, batch_size, tensorboard_modelname, validation_data=None, verbose=1,
              class_weight=None):

        tbCallBack = TensorBoard(log_dir='./models/tensorboard_logs/{}/'.format(tensorboard_modelname),
                                 histogram_freq=0, write_graph=True, write_images=True)

        return self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=validation_data,
                              callbacks=[tbCallBack], verbose=verbose, class_weight=class_weight)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)

    def predict(self, x):
        return self.model.predict(x)

    def eval(self, x, y, verbose=1):
        return self.model.evaluate(x=x, y=y, verbose=verbose)
    
def get_distribution_of_labels_categorical_from_caselabels(caselabels):
    labels = []
    label_counts = {}
    label_percentages = {}
    for label in caselabels.keys():
        if str(caselabels[label]) != 'nan':
            labels.append(caselabels[label])
    for category in set(labels):
        label_counts[category] = labels.count(category)
        label_percentages[category] = (labels.count(category) / len(labels)) * 100
    return label_counts, label_percentages

def plot_by_fold(valaccuracy, valloss, n_folds=4):
    for i in range(n_folds):
        plt.plot(valaccuracy[i], "m+-")
        plt.plot(valloss[i], "co-")
    plt.plot(np.mean(valaccuracy, axis=0), "r+-")
    plt.plot(np.mean(valloss, axis=0), "bo-")
    plt.show()
    print('Max of average fold validation accuracy', np.max(np.mean(valaccuracy, axis=0)))
    
def plot_patient(ecg_set, patient_index, lead):
        x_axis = []
        y_axis = []
        for i in range(ecg_set.shape[1]):
            x_axis.append(i)
            y_axis.append(ecg_set[patient_index,i,lead])
        plt.plot(x_axis,y_axis)
        plt.show()
        
def add_noise(dataset,  augmentation_factor=1):  #dataset_labels,
    augmented_image = []
#     augmented_image_labels = []
    
    dirty_data=np.copy(dataset)
    
    for case in range (dataset.shape[0]):
        for x in range(len(dataset[case][:,0])):
            for y in range(len(dataset[case][0,:])):
                new_val = dataset[case][x,y] + np.random.normal(0,noise_level,1)
                dirty_data[case][x,y]= new_val
        augmented_image.append(dirty_data[case])

    return np.array(augmented_image) #, np.array(augmented_image_labels)

class aCNN1D_categorical(Network):

    def __init__(self, sequence_length, input_channels, n_classes):
        self.sess = tf.compat.v1.Session()

        self.model = tf.keras.Sequential()

        self.model.add(Conv1D(64, 5, dilation_rate=2, input_shape=(sequence_length, input_channels)))
        self.model.add(Conv1D(32, 3, dilation_rate=2))
        self.model.add(Conv1D(32, 3, dilation_rate=2))

        self.model.add(LSTM(64, dropout=0.2, return_sequences=False))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(n_classes, activation='sigmoid'))
        self.model.compile(loss="categorical_crossentropy", optimizer='adamax', metrics=['accuracy'])
def plot_by_fold(valaccuracy, valloss, n_folds=4):
    for i in range(n_folds):
        plt.plot(valaccuracy[i], "m+-")
        plt.plot(valloss[i], "co-")
    plt.plot(np.mean(valaccuracy, axis=0), "r+-")
    plt.plot(np.mean(valloss, axis=0), "bo-")
    plt.show()
    print('Max of average fold validation accuracy', np.max(np.mean(valaccuracy, axis=0)))
    
def plot_patient(ecg_set, patient_index, lead):
        x_axis = []
        y_axis = []
        
        for i in range(ecg_set.shape[1]):
            x_axis.append(i)
            y_axis.append(ecg_set[patient_index,i,lead])
        
        plt.plot(x_axis,y_axis)
        plt.show()
    
def get_distribution_of_labels_categorical_from_caselabels(caselabels):
    labels = []
    label_counts = {}
    label_percentages = {}
    for label in caselabels.keys():
        if str(caselabels[label]) != 'nan':
            labels.append(caselabels[label])
    for category in set(labels):
        label_counts[category] = labels.count(category)
        label_percentages[category] = (labels.count(category) / len(labels)) * 100
    return label_counts, label_percentages
dataseries = DataSeries(#path="../input/exportps/exports_ps",
                        path="../input/imperial-barts-data/extracts",
                        labelfunction=DataSeries.get_labels_by_case_ap_left_right_septal,
                        caselabels="../input/withlsrap-labels/james_labels_cvers.xlsx",
                        classes=['L','S','R'],
                        include_only=("C_LSR",1))

modeltype = aCNN1D_categorical
downsample_ratio = 0.01
axis_align = 0
noise_level = 0
twoD = False
reverse=True
n_classes = len(dataseries.classes)
valaccuracy_by_fold = []
valloss_by_fold = []
prop_by_fold = []
verbose = 2
results_dict = {}

print(get_distribution_of_labels_categorical_from_caselabels(dataseries.caselabels))

for fold in range(4):
    print(f"FOLD {fold}")
    (train_x, train_y, train_n, train_caseids),\
    (test_x, test_y, test_n, test_caseids) = dataseries.get_train_test_data(reverse=False, fold_num=fold,downsample_ratio=downsample_ratio,
                                                                            axis_align=axis_align,twoD=twoD, noise_level=noise_level)

    seq_length = train_x.shape[1]
    input_channels = train_x.shape[2]

    if fold == 0:
        print(f"Train: {train_x.shape} from {train_n} cases")
        print(f"Test: {test_x.shape} from {test_n} cases")

    model = modeltype(sequence_length=seq_length, input_channels=input_channels, n_classes=n_classes)

    results = model.train(train_x, train_y, epochs=10, batch_size=32, tensorboard_modelname=f"cnn_lstm_newcat_{fold}",
                          validation_data=(test_x, test_y), verbose=verbose)

    print(f"Categories are {list(enumerate(dataseries.classes))}")
    predictions_by_case = {caseid:[] for caseid in set(test_caseids)}
    answers_by_case = {}
    for case, x, y in zip(test_caseids, test_x, test_y):
        x = x.reshape(1, x.shape[0], x.shape[1])
        predictions_by_case[case].append(model.predict(x)[0])
        answers_by_case[case] = y
    for case in predictions_by_case.keys():
        result = np.mean(np.stack(predictions_by_case[case]),axis=0)
        if np.argmax(result) == np.argmax(answers_by_case[case]):
            correct = "CORRECT"
        else:
            correct = "WRONG"
        print(f"CASE {case}: {result} - {correct}")
        if n_classes == 2:
            results_dict[case] = [result[0], result[1], np.argmax(result), np.argmax(answers_by_case[case]), correct]
        else:
            results_dict[case] = [result[0], result[1], result[2], np.argmax(result), np.argmax(answers_by_case[case]), correct]

    valaccuracy_by_fold.append(results.history['val_accuracy'])
    valloss_by_fold.append(results.history['val_loss'])

valaccuracy_by_fold = np.stack(valaccuracy_by_fold)
valloss_by_fold = np.stack(valloss_by_fold)

pd.DataFrame.from_dict(results_dict,orient='index').to_csv("ap_results.csv")

print(get_distribution_of_labels_categorical_from_caselabels(dataseries.caselabels))

plot_by_fold(valaccuracy_by_fold, valloss_by_fold)

print(model.model.summary())
grads = visualize_saliency(model.model, layer_idx=6, filter_indices=0, 
                                          seed_input=test_x[0])
print(grads)
x = []
y = []
for i in range(30):
    x.append(i)
    y.append(grads[i])
    
plt.plot(x,y)
plt.show()
class visualise:
    def __init__(self, data_name):
        self.data = np.copy(data_name)
        
    def locate_blank(self, data_per_patient):
        for j in range(len(data_per_patient[:,0])):
            if data_per_patient[j,0] != 0:
                if saliency:
                    return j #j is location for cropping
                break
                
    def process_image(self, crop, transpose, index):
        data_2D = self.data[index,:,:]
        if crop:
            data_2D = data_2D[:self.locate_blank(data_2D),:]
        if transpose:
            self.data_final = np.transpose(data_2D)
        else:
            self.data_final = data_2D
    
    def plot_hotmap(self, array, color='viridis'):
        plt.imshow(array, interpolation='nearest', aspect='auto', cmap=color)
        plt.show()
        
    def show_ecg(self, axis_mode, ecg_index=0):
        if axis_mode == 0:        #default
            lead_arr = ['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        elif axis_mode == 1:
            lead_arr = ['aVL','Lead1','-aVR','Lead2','aVF','Lead3','V1','V2','V3','V4','V5','V6']
        elif axis_mode == 2:
            lead_arr = ['V1','V2','V3','V4','V5','V6', 'aVL','Lead1','-aVR','Lead2','aVF','Lead3']
    
        fig1, axs = plt.subplots(12, sharex=True)
        fig1.set_size_inches(10, 10)
        
        print(test_caseids[ecg_index])
        for lead in range(len(lead_arr)):
            x_axis = []
            y_axis = []
            for timestep in range(self.data.shape[1]):
                x_axis.append(timestep)
                y_axis.append(self.data[ecg_index,timestep,lead])
            axs[lead].plot(x_axis,y_axis)
            axs[lead].set_ylabel(lead_arr[lead])
            
    #print out ecg segments as hotmap image (optional: cropping, transposing)
    def display_ecgs(self, start=0, interval=10, crop=False, transpose=False, num_im=1):
        for i in range(start*interval,num_im*interval,interval):
            self.process_image(crop, transpose, index=i)
            self.plot(self.data_final)
            
    #same as above but only one
    def display_ecg(self, pt_index=0, crop=False, transpose=False):
        self.process_image(crop,transpose,index=pt_index*10)
        plt.imshow(self.data_final, interpolation='nearest', aspect='auto')
        plt.show()
        
#     def display_saliency(self, pt_index=0, layer=8, class_index=0, crop=False):
#         list_index=pt_index*10
        
#         grads = visualize_saliency(model.model, layer, filter_indices=class_index, seed_input=self.original[list_index])
        
#         if crop:
#             blank_location = self.locate_blank(self.data[list_index], saliency=True)
#             final = grads[blank_location:,:]
#         else:
#             final = grads
        
#         plt.imshow(final, interpolation='nearest', aspect='auto')
#         plt.show()
        
    def display_saliency(self, class_idx=0, num_im=10, interval=1, layer='all',color='viridis'):
        indices = np.where(test_y[:, class_idx] == 1.)[0]
        for i in range(num_im):
            idx = indices[i*interval]
            if layer != 'all':
                print(test_caseids[idx], ' - Layer ', layer, ' - ', model.model.layers[layer])
                grads = visualize_saliency(model.model, layer_idx=layer, filter_indices=class_idx, 
                                      seed_input=test_x[idx])
                plt.imshow(grads, cmap=color,interpolation='nearest', aspect='auto')
                plt.title(test_caseids[idx] + ' - Layer ' + str(layer))
                plt.show()
            else:
                for i in range(len(model.model.layers)):
                    print(test_caseids[idx], ' - Layer ', i, ' - ', model.model.layers[i])
                    grads = visualize_saliency(model.model, layer_idx=i, filter_indices=class_idx, 
                                          seed_input=test_x[idx])
                    plt.imshow(grads, cmap=color,interpolation='nearest', aspect='auto')
                    plt.title(test_caseids[1] + ' - Layer ' + str(layer))
                    plt.show()

        
    def display_saliency_sbs(self,class_idx=0,num_im=10,interval=1,layer='all',color='viridis'): #sbs stands for side by side; layer can be 'all' or int
        indices = np.where(test_y[:, class_idx] == 1.)[0]
        for i in range(num_im):
            idx = indices[i*interval]
            if layer == 'all':
                for j in range(len(model.model.layers)):
                    print(test_caseids[idx], ' - Layer ',j, ' - ', model.model.layers[j])
                    f, ax = plt.subplots(1, 2)
                    ax[0].imshow(test_x[idx][..., 0],cmap=color,interpolation='nearest', aspect='auto')
                    grads = visualize_saliency(model.model, layer_idx=j, filter_indices=class_idx, 
                                          seed_input=test_x[idx])        
                    ax[1].set_title('saliency map')    
                    ax[1].imshow(grads,cmap=color, interpolation='nearest', aspect='auto') #cmap='jet',
                    plt.show()
            else:
                    print(test_caseids[idx], ' - Layer ', layer, ' - ', model.model.layers[layer])
                    f, ax = plt.subplots(1, 2)
                    ax[0].imshow(test_x[idx][..., 0],cmap=color,interpolation='nearest', aspect='auto')
                    grads = visualize_saliency(model.model, layer_idx=layer, filter_indices=class_idx, 
                                          seed_input=test_x[idx])        
                    ax[1].set_title('saliency map')    
                    ax[1].imshow(grads, cmap=color,interpolation='nearest', aspect='auto') #cmap='jet',
                    plt.show()
                    
        
    def ecg_saliency_overlaid(self,axis_mode, ecg_index=0, layer=5, class_idx=0):
        indices = np.where(test_y[:, class_idx] == 1.)[0]      
        idx = indices[ecg_index]

        if axis_mode == 1:
            lead_arr = ['aVL','Lead1','-aVR','Lead2','aVF','Lead3','V1','V2','V3','V4','V5','V6']
        elif axis_mode == 2:
            lead_arr = ['V1','V2','V3','V4','V5','V6', 'aVL','Lead1','-aVR','Lead2','aVF','Lead3']
        elif axis_mode == 0:        #default
            lead_arr = ['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        
        grads = visualize_saliency(model.model, layer_idx=layer, filter_indices=class_idx, 
                                          seed_input=test_x[idx])
        
        print(test_caseids[idx])
        for i in range(len(lead_arr)):
            y = self.data[idx,:,i]
            y = [value-(i*12+10) for value in y]
            x = list(range(len(y)))
            label = lead_arr[i]
            plt.rcParams['figure.figsize'] = [25, 25]
            plt.plot(x,y,label=label)
        plt.imshow(np.transpose(np.grads), interpolation='nearest',origin='low',extent=[0,len(self.data[0,:,0]),-148,-3],alpha=0.6, cmap='GnBu')
        plt.rcParams['lines.linewidth'] = 1
        plt.legend(loc='lower left')
        plt.show()
        
#     def ecg_saliency_overlaid_cropped(self,axis_mode, crop=False, ecg_index=0, layer=11, class_idx=0):
#         indices = np.where(test_y[:, class_idx] == 1.)[0]      
#         idx = indices[ecg_index]

#         if axis_mode == 1:
#             lead_arr = ['aVL','Lead1','-aVR','Lead2','aVF','Lead3','V1','V2','V3','V4','V5','V6']
#         elif axis_mode == 2:
#             lead_arr = ['V1','V2','V3','V4','V5','V6', 'aVL','Lead1','-aVR','Lead2','aVF','Lead3']
#         elif axis_mode == 0:        #default
#             lead_arr = ['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        
#         grads = visualize_saliency(model.model, layer_idx=layer, filter_indices=class_idx, 
#                                           seed_input=test_x[idx])
        
#         location = 0
#         if crop == True:
#             location = self.locate_blank(data_per_patient=self.data[idx,:,:], saliency=True)
           
#         print(test_caseids[idx])
#         for i in range(len(lead_arr)):
#             y = self.data[idx,location:,i]
#             y = [value+(i*12+10) for value in y]
#             x = list(range(len(y)))
#             label = lead_arr[i]
#             plt.rcParams['figure.figsize'] = [25, 25]
#             plt.plot(x,y,label=label)
#         plt.imshow(np.transpose(grads), interpolation='nearest',origin='low',extent=[0,len(self.data[0,:,0]),3,150],alpha=0.6, cmap='GnBu')
#         plt.rcParams['lines.linewidth'] = 1
#         plt.legend(loc='lower left')
#         plt.show()
        
new = visualise(test_x)
for i in range(100):
    new.ecg_saliency_overlaid(axis_mode=axis_align, ecg_index=i, class_idx=0)
