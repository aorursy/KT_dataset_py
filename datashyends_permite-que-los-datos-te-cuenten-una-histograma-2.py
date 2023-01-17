import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re



import cv2





# Function to plot an histogram

def histogram_plot(example_arr, bins = 25, title = "", label = "", axis = None, show_density=False):

    

    axis.set_title(title, fontsize = 20)

    #sns.distplot(example_arr, kde=False, rug=False, bins=bins, norm_hist=False, ax=axis)

    sns.distplot(example_arr, kde=False, bins=bins, ax=axis)

    

    if show_density:

        second_ax = axis.twinx()

        sns.distplot(example_arr, kde=True, hist=False, ax=second_ax)

        second_ax.set_ylabel("Densidad", fontsize = 20)

    

    axis.set_xlabel(label, fontsize = 20)

    axis.set_ylabel("Frecuencia", fontsize = 20)

    

    axis.xaxis.set_major_locator(plt.MaxNLocator(bins/3))

    

    mean = np.mean(example_arr)

    median = np.median(example_arr)

        

    formatted_mean = "{:.4f}".format(mean)

    formatted_median = "{:.4f}".format(median)



    axis.axvline(mean, color='green', linewidth=2, linestyle='-', label = "Media = " + formatted_mean)

    axis.axvline(median, color='grey', linewidth=2, linestyle='-', label = "Mediana = " + formatted_median)

    

    plt.legend()



def prepare_image_dataset(file_name):



    # Read file and return it as matrix pixel

    image_matrix = cv2.imread(file_name)

    

    # calculate mean value from RGB channels and flatten to 1D array

    val_arr = image_matrix.mean(axis=2).flatten()



    # Return the processed variable

    return val_arr



# Display preparation

figure = plt.figure(figsize=(20, 8))

ax_histogram = figure.add_subplot(1, 1, 1)

plt.rcParams["patch.force_edgecolor"] = True



# File name

file_name = "../input/dataset/Visualizaciones/Histograma/aaron-burden-gF_umQbT5tM-unsplash.jpg"



X = prepare_image_dataset(file_name)



histogram_plot(X, bins = 100, title = "Histograma de una imagen", 

               label = "Escala de grises", axis = ax_histogram, show_density=False)