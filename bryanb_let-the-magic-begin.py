import numpy as np

import pandas as pd

from scipy.stats import rankdata

import os

import re
def Stacking(input_folder, 

             best_base, 

             output_path, 

             column_names, 

             cutoff_lo, 

             cutoff_hi):

    

    '''

    To be tried on:

        - a same model that is not deterministic (with randomness)

        - a same model with folds (will need to define a meta model)

    '''

    

    sub_base = pd.read_csv(best_base)

    all_files = os.listdir(input_folder)

    nb_files = len(all_files)

    

    # Test compliancy of arguments

    assert type(input_folder) == str, "Wrong type"

    assert type(best_base) == str, "Wrong type"

    assert type(output_path) == str, "Wrong type"

    assert type(cutoff_lo) in [float, int], "Wrong type"

    assert type(cutoff_hi) in [float, int], "Wrong type"

    assert (cutoff_lo >= 0) & (cutoff_lo <= 1) & (cutoff_hi >= 0) & (cutoff_hi <= 1), "cutoff_lo and cutoff_hi must be between 0 and 1"

    assert len(column_names) == 2, "Only two columns must be in column_names"

    assert type(column_names[0]) == str, "Wrong type"

    assert type(column_names[1]) == str, "Wrong type"



    

    # Read and concatenate submissions

    concat_sub = pd.DataFrame()

    concat_sub[column_names[0]] = sub_base[column_names[0]]

    for index, f in enumerate(all_files):

        concat_sub[column_names[1]+str(index)] = pd.read_csv(input_folder + f)[column_names[1]]

    print(" ***** 1/4 Read and concatenate submissions SUCCESSFUL *****")

        

    # Get the data fields ready for stacking

    concat_sub['target_max'] = concat_sub.iloc[:, 1:].max(axis=1)

    concat_sub['target_min'] = concat_sub.iloc[:, 1:].min(axis=1)

    concat_sub['target_mean'] = concat_sub.iloc[:, 1:].mean(axis=1) # Not used but available if needed

    concat_sub['target_median'] = concat_sub.iloc[:, 1:].median(axis=1) # Not used but available if needed

    print(" ***** 2/4 Get the data fields ready for stacking SUCCESSFUL *****")

    

    # Set up cutoff threshold for lower and upper bounds

    concat_sub['target_base'] = sub_base[column_names[1]]

    concat_sub[column_names[1]] = np.where(np.all(concat_sub.iloc[:, 1:] > cutoff_lo, axis=1),

                                    concat_sub['target_max'],

                                    np.where(np.all(concat_sub.iloc[:, 1:] < cutoff_hi, axis=1),

                                             concat_sub['target_min'],

                                             concat_sub['target_base']))

    print(" ***** 3/4 Set up cutoff threshold for lower and upper bounds SUCCESSFUL *****")

    

    # Generating Stacked dataframe

    concat_sub[column_names].to_csv(output_path, index=False, float_format='%.12f')

    print(" ***** 4/4 Generating Stacked dataframe SUCCESSFUL *****")

    print(" ***** COMPLETED *****")
Stacking(input_folder = '../input/siim-isic-baseline-models/',

         best_base = '../input/siim-isic-baseline-models/RESNET_0946.csv',

         output_path = 'stacking.csv',

         column_names = ['image_name', 'target'],

         cutoff_lo = 0.85,

         cutoff_hi = 0.17)
def Ensemble(input_folder, 

             output_path, 

             method, 

             column_names, 

             sorted_files, 

             reverse = False):

    

    '''

    To be tried on:

        - different weak learners (models)

        - several models for manual weightings

    '''

    

    all_files = os.listdir(input_folder)

    nb_files = len(all_files)

    

    # Warning

    print("***** WARNING *****\n")

    print("Your files must be written this way: model_score.csv:")

    print("    - Model without underscore, for example if you use EfficientNet do not write Eff_Net_075.csv but rather EffNet_075.csv")

    print("    - Score without comma, for example if you score 0.95 on XGB, the name can be XGB_095.csv\n")

    print("About the score:")

    print("    - If the score has to be the lowest as possible, set reverse=True as argument\n")

    

    if (sorted_files == False) & (method in ['sum_of_integers', 'sum_of_squares']):

        print("Arguments 'sum_of_integers' and 'sum_of_squares' might perform poorly as your files are not sorted")

        print("     - To sort them, change 'sorted_files' argument to 'True'\n")

        

    # Test compliancy of arguments

    assert type(input_folder) == str, "Wrong type"

    assert type(output_path) == str, "Wrong type"

    assert len(column_names) == 2, "Only two columns must be in column_names"

    assert type(column_names[0]) == str, "Wrong type"

    assert type(column_names[1]) == str, "Wrong type"

    assert method in ['mean', 'geometric_mean', 'sum_of_integers', 'sum_of_squares', 'weights'], 'Select a method among : mean, geometric_mean, sum_of_integers, sum_of_squares, weights.'

    assert type(sorted_files) == bool, "Wrong type"

    assert type(reverse) == bool, "Wrong type"

    assert nb_files >= 1, 'Need at least two models for ensembling.'

    

    # Sorting models by performance

    if sorted_files == True:

        

        # Sort files based on performance

        ranking = [int(re.findall(r'\_(\d*)', file)[0]) for file in all_files]

        dict_files = dict(zip(all_files, ranking))

        sorted_dict = sorted(dict_files.items(), key=lambda x: x[1], reverse = reverse)

        

        assert len(all_files) == len([file[0] for file in sorted_dict]), "Something went wrong with regex filtering"

        all_files = [file[0] for file in sorted_dict]

        print(" ***** Sorting models by performance SUCCESSFUL *****")



    # Create list of dataframes

    DATAFRAMES = [pd.read_csv(input_folder + file) for file in all_files]

    print(" ***** 1/4 Create list of dataframes SUCCESSFUL *****")



    # Create the submission datdaframe initialized with first column

    sub = pd.DataFrame()

    sub[column_names[0]] = DATAFRAMES[0][column_names[0]]

    print(" ***** 2/4 Create the submission datdaframe SUCCESSFUL *****")

    

    # Apply ensembling according to the method

    if method == 'mean':

        sub[column_names[1]] = np.mean([rankdata(df[column_names[1]], method='min') for df in DATAFRAMES], axis = 0)

        

    elif method == 'geometric_mean':

        sub[column_names[1]] = np.exp(np.mean([rankdata(df[column_names[1]].apply(lambda x: np.log2(x)), method='min') for df in DATAFRAMES], axis = 0))

        

    elif method == 'sum_of_integers':        

        constant = 1/(nb_files*(nb_files+1)/2)

        sub[column_names[1]] = np.sum([(i+1)*rankdata(DATAFRAMES[i][column_names[1]], method='min') for i in range(nb_files)], axis = 0) * constant

    

    elif method == 'sum_of_squares':

        constant = 1/((nb_files*(nb_files+1)*(2*nb_files+1))/6)

        sub[column_names[1]] = np.sum([(i+1)*(i+1)*rankdata(DATAFRAMES[i][column_names[1]], method='min') for i in range(nb_files)], axis = 0) * constant

    

    elif method == 'weights':

        # Type manually here your own weights

        #print(all_files)

        weights = [0.2, 0.35, 0.45]

        assert len(weights) == nb_files, "Length of weights doesn't fit with number of models to be ensembled"

        assert sum(weights) == 1, 'Sum of weights must be equal to 1'

        sub[column_names[1]] = np.sum([weights[i]*rankdata(DATAFRAMES[i][column_names[1]], method='min') for i in range(nb_files)], axis = 0)

        print('\n')

        for i in range(len(weights)):

            print(f'    - Applied weight {weights[i]} to file {all_files[i]}')

        print('\n')

            

        

    print(" ***** 3/4 Apply ensembling according to the method SUCCESSFUL *****")

    sub.to_csv(output_path, index=False, float_format='%.12f')

    print(" ***** 4/4 Generating Ensembled dataframe SUCCESSFUL *****")

    print(" ***** COMPLETED *****")
Ensemble(input_folder = '../input/siim-isic-baseline-models/', 

         output_path = 'ensemble.csv',

         method = 'weights',

         column_names = ['image_name', 'target'], 

         sorted_files = True,

         reverse = False)