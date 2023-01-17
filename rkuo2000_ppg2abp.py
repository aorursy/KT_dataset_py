!git clone https://github.com/nibtehaz/PPG2ABP
import os
os.chdir('PPG2ABP/codes')
os.mkdir('data')
#!cp '/kaggle/input/ppg2abp-data/data.hdf5' data
!ln -s '/kaggle/input/ppg2abp-data/data.hdf5' data/data.hdf5
!ls data
from data_handling import fold_data


fold_data()
from helper_functions import *
from models import *
from train_models import train_approximate_network

train_approximate_network()
from train_models import train_refinement_network

train_refinement_network()
from predict_test import predict_test_data

predict_test_data()
from evaluate import predicting_ABP_waveform

predicting_ABP_waveform()
from evaluate import evaluate_BHS_Standard

evaluate_BHS_Standard()
from evaluate import evaluate_AAMI_Standard

evaluate_AAMI_Standard()
from evaluate import evaluate_BP_Classification

evaluate_BP_Classification()
from evaluate import bland_altman_plot

bland_altman_plot()