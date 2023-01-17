import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re



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



def prepare_netflix_dataset(file_name):

    # Read the file

    netflix_catalogue_df = pd.read_csv(file_name)



    # Print unique values for the type of content

    print(netflix_catalogue_df['type'].unique())



    # Keep only the movies

    netflix_catalogue_movies_df = netflix_catalogue_df[netflix_catalogue_df.type == 'Movie']



    # Keep only the 'duration' variable

    netflix_catalogue_movies_duration = netflix_catalogue_movies_df.duration

    

    # The duration variable follows the pattern 'XX min'. Keep only the numeric value and convert it to integer

    netflix_catalogue_movies_duration = netflix_catalogue_movies_duration.str.extract('(\d+)', expand=False)

    netflix_catalogue_movies_duration = netflix_catalogue_movies_duration.astype(int)

    

    print("Rango de duración: [{}-{}]".format(netflix_catalogue_movies_duration.min(), 

                                              netflix_catalogue_movies_duration.max()))





    # Return the processed variable

    return netflix_catalogue_movies_duration



# Display preparation

figure = plt.figure(figsize=(20, 8))

ax_histogram = figure.add_subplot(1, 1, 1)

plt.rcParams["patch.force_edgecolor"] = True



# File name

file_name = "../input/netflix-shows/netflix_titles.csv"



X = prepare_netflix_dataset(file_name)



histogram_plot(X, bins = 100, title = "Histograma de la duración de las películas", 

               label = "Duración (min)", axis = ax_histogram, show_density=True)