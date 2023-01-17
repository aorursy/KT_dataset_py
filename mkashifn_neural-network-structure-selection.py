# Imports



########################################################################

# Python Standard Libraries

import os

import multiprocessing

from timeit import default_timer as timer

import random

import math



########################################################################

# Numpy Library

import numpy as np # linear algebra



########################################################################

# Pandas Library

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



########################################################################

# MATPLOT Library

import matplotlib.cm as cm

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from matplotlib.ticker import MaxNLocator

%matplotlib inline



########################################################################

# SKLearn Library

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM

from sklearn import metrics

from sklearn.metrics import accuracy_score, precision_recall_curve, classification_report, confusion_matrix, average_precision_score, roc_curve, auc, multilabel_confusion_matrix



########################################################################

# SCIPY Library

from scipy.stats import gaussian_kde

import scipy.stats as st
# Utility functions

########################################################################

# Print system information

def print_system_info():

    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448

    mem_gib = mem_bytes/(1024.**3)  # e.g. 3.74

    print("{:<23}{:f} GB".format('RAM:', mem_gib))

    print("{:<23}{:d}".format('CORES:', multiprocessing.cpu_count()))

    !lscpu



########################################################################

# Walk through input files

def print_input_files():

    # Input data files are available in the "../input/" directory.

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



########################################################################

# Dump text files

def dump_text_file(fname):

    with open(fname, 'r') as f:

        print(f.read())



########################################################################

# Dump CSV files

def dump_csv_file(fname, count=5):

    # count: 0 - column names only, -1 - all rows, default = 5 rows max

    df = pd.read_csv(fname)

    if count < 0:

        count = df.shape[0]

    return df.head(count)



########################################################################

# Dataset related functions

ds_nbaiot = '/kaggle/input/nbaiot-dataset'

dn_nbaiot = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor', 'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam', 'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']



def fname(ds, f):

    if '.csv' not in f:

        f = f'{f}.csv'

    return os.path.join(ds, f)



def fname_nbaiot(f):

    return fname(ds_nbaiot, f)



def get_nbaiot_device_files():

    nbaiot_all_files = dump_csv_file(fname_nbaiot('data_summary'), -1)

    nbaiot_all_files = nbaiot_all_files.iloc[:,0:1].values

    device_id = 1

    indices = []

    for j in range(len(nbaiot_all_files)):

        if str(device_id) not in str(nbaiot_all_files[j]):

            indices.append(j)

            device_id += 1

    nbaiot_device_files = np.split(nbaiot_all_files, indices)

    return nbaiot_device_files



def get_nbaiot_device_data(device_id, count_norm=-1, count_anom=-1):

    if device_id < 1 or device_id > 9:

        assert False, "Please provide a valid device ID 1-9, both inclusive"

    if count_anom == -1:

        count_anom = count_norm

    device_index = device_id -1

    device_files = get_nbaiot_device_files()

    device_file = device_files[device_index]

    df = pd.DataFrame()

    y = []

    for i in range(len(device_file)):

        fname = str(device_file[i][0])

        df_c = pd.read_csv(fname_nbaiot(fname))

        count = count_anom

        if 'benign' in fname:

            count = count_norm

        rows = count if count >=0 else df_c.shape[0]

        print("processing", fname, "rows =", rows)

        y_np = np.ones(rows) if 'benign' in fname else np.zeros(rows)

        y.extend(y_np.tolist())

        df = pd.concat([df.iloc[:,:].reset_index(drop=True),

                      df_c.iloc[:rows,:].reset_index(drop=True)], axis=0)

    X = df.iloc[:,:].values

    y = np.array(y)

    Xdf = df

    return (X, y, Xdf)



def get_nbaiot_devices_data():

    devices_data = []

    for i in range(9):

        device_id = i + 1

        (X, y) = get_nbaiot_device_data(device_id)

        devices_data.append((X, y))

    return devices_data

#print_input_files()

print_system_info()
def select_network_structure(method, layer_count_min, layer_count_max, node_count_min, node_count_max, samples_count, output_neuron_count):

    structure = []

    if method == 'random':

        layer_count = random.randint(layer_count_min, layer_count_max)

        for i in range(layer_count):

            node_count = random.randint(node_count_min, node_count_max)

            structure.append(node_count)

    if method == 'heuristics':

        N = samples_count

        m = output_neuron_count

        node_count_layer_1 = int(math.sqrt((m + 2) * N) + 2 * math.sqrt(N / (m + 2)))

        node_count_layer_2 = int(m * math.sqrt(N / (m + 2)))

        structure.append(node_count_layer_1)

        structure.append(node_count_layer_2)



    if method == 'genetic':

        l = 10

        chromosome = ''

        for i in range(l):

            x = random.randint(0, 1)

            chromosome += '{}'.format(x)

        chromosome_left = chromosome[0:6]

        chromosome_right = chromosome[6:]

        print('chromosome: {}'.format(chromosome))

        print('split: {} {}'.format(chromosome_left, chromosome_right))

        print('chromosome_left: {}'.format(chromosome_left))

        print('chromosome_right: {}'.format(chromosome_right))

        node_count_layer_1 = int(chromosome_left, 2) + random.randint(1, 10)

        node_count_layer_2 = int(chromosome_right, 2) + random.randint(1, 10)

        structure.append(node_count_layer_1)

        structure.append(node_count_layer_2)

    return structure
print(select_network_structure('random', 5, 10, 10, 20, 100, 2))
print(select_network_structure('heuristics', 5, 10, 10, 20, 100, 2))
print(select_network_structure('genetic', 5, 10, 10, 20, 100, 2))