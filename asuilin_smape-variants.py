import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Approximated differentiable SMAPE

def differentiable_smape(true, predicted):

    epsilon = 0.1

    true_o = true

    pred_o = predicted

    summ = np.maximum(np.abs(true_o) + np.abs(pred_o) + epsilon, 0.5 + epsilon)

    smape = np.abs(pred_o - true_o) / summ

    return smape



# SMAPE, rounded up to the closest integet

def rounded_smape(true, predicted):

    true_o = np.int(np.round(true))

    pred_o = np.round(predicted).astype(np.int32)

    summ = np.abs(true_o) + np.abs(pred_o)

    smape = np.where(summ==0, 0, np.abs(pred_o - true_o) / summ)

    return smape



# SMAPE as Kaggle calculates it

def kaggle_smape(true, predicted):

    true_o = true

    pred_o = predicted

    summ = np.abs(true_o) + np.abs(pred_o)

    smape = np.where(summ==0, 0, np.abs(pred_o - true_o) / summ)

    return smape





# MAE on log1p

def mae(true, predicted):

    true_o = np.log1p(true)

    pred_o = np.log1p(predicted)

    error = np.abs(true_o - pred_o)/2

    return error
def plot_smape(true_y, x_start, x_end):

    x = np.linspace(x_start,x_end, num=100)

    plt.plot(x, differentiable_smape(true_y, x), label='differentiable')

    plt.plot(x, rounded_smape(true_y, x), label='rounded')

    plt.plot(x, mae(true_y, x), label='mae')

    plt.plot(x, kaggle_smape(true_y, x), label='kaggle')

    plt.xlabel('predicted value')

    plt.ylabel('Loss')

    plt.title('True value=' + str(true_y))

    plt.legend()
plot_smape(0, 0, 2)
plot_smape(1, 0, 2)