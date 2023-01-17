import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pickle

import time

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import glob 



import os

print('-------------------------')

print('all files:')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print('-------------------------')



data_foldername = '/kaggle/input/bigger-is-better/'

all_filenames = glob.glob(os.path.join(data_foldername, '*.pickle'))



print('-------------------------')

print('files inside folder "%s":' %(data_foldername))

for k, filename in enumerate(all_filenames):

    print('%3d: %s' %(k + 1, filename.split('/')[-1]))

print('-------------------------')

teachers_folder = '../input/bigger-is-better/'

filename = 'FCN_LReLU_05_gauss_IO_norm_init__I-DxW-O__10-3x8-1__n_models_42__n_inits_30_fanin__n_iter_1200__seed_1234.pickle'



teacher_filename = os.path.join(teachers_folder, filename)



with open(teacher_filename, "rb") as f:

    teacher_training_results = pickle.load(f)

    

GT_model_name = teacher_filename.split('/')[-1].split('__n_models')[0]



print('-----------------------------')

print('GT model name: "%s"' %(GT_model_name))

print('-----------------------------')



print('-----------------------------')

all_model_names = list(teacher_training_results.keys())

print('all student model names:')

for model_name in all_model_names:

    print('  ' + model_name)

print('-----------------------------')


def get_learning_curves(teacher_filename, student_model_name, valid_index=1, percent_var_remaining=False):

    with open(teacher_filename, "rb") as f:

        teacher_training_results = pickle.load(f)

        

    student_model_results = teacher_training_results[student_model_name]



    valid_loss = 'valid_%d_loss' %(valid_index)

    valid_key = 'valid_%d' %(valid_index)

    

    # collect all learning curves for the same model in the same matrix

    learning_curves = np.zeros((len(student_model_results['all learning curves']), student_model_results['all learning curves'][0][valid_loss].shape[0]))

    for k, curr_learning_curves in enumerate(student_model_results['all learning curves']):

        learning_curves[k, :] = curr_learning_curves[valid_loss]



    training_steps = student_model_results['all learning curves'][0]['num_batches']



    if percent_var_remaining:

        baseline_mse = student_model_results['model_hyperparams_dict']['y_GT_stats_dict'][valid_key]['mse_0']

        learning_curves = 100 * learning_curves / baseline_mse

        

    return training_steps, learning_curves





def get_baseline_mse(teacher_filename, student_model_name, valid_index=1):

    with open(teacher_filename, "rb") as f:

        teacher_training_results = pickle.load(f)

        

    student_model_results = teacher_training_results[student_model_name]

    valid_key = 'valid_%d' %(valid_index)

    baseline_mse = student_model_results['model_hyperparams_dict']['y_GT_stats_dict'][valid_key]['mse_0']



    return baseline_mse



def get_student_depth_x_width(student_model_name):

    

    nn_depth = int(student_model_name.split('_stu')[0].split('DxW_')[-1].split('x')[0])

    nn_width = int(student_model_name.split('_stu')[0].split('DxW_')[-1].split('x')[-1])

    

    return nn_depth, nn_width





def teacher_filename_from_depth_x_width_seed(depth, width, seed, teachers_folder='../input/bigger-is-better/glorot_uniform_init/'):

    teacher_filename = 'FCN_LReLU_05_gauss_IO_norm_init__I-DxW-O__10-%dx%d-1__n_models_42__n_inits_30_fanin__n_iter_1200__seed_%d.pickle' %(depth, width, seed)

    return os.path.join(teachers_folder, teacher_filename)





def student_name_from_depth_x_width(depth, width):

    return 'FCN_DxW_%dx%d_student' %(depth, width)

teacher_depths = [1,3,5]

teacher_widths = [16,16,16]

teacher_seeds  = [1234, 1234, 1234]

teacher_colors = ['green', 'orange', 'red']



requested_valid_index = 1

requested_train_step = 1200

show_percent_remaining = True

depth_lims = [1,32]

width_lims = [1,256]



show_best = True

show_error_bars = False



all_student_model_names = list(teacher_training_results.keys())

all_short_student_model_names = [x.split('_stu')[0] for x in all_student_model_names]



# filter the student models to display

inds_to_keep = []

for k, student_model_name in enumerate(all_student_model_names):

    nn_depth, nn_width = get_student_depth_x_width(student_model_name)

    depth_OK = nn_depth >= depth_lims[0] and nn_depth <= depth_lims[1]

    width_OK = nn_width >= width_lims[0] and nn_width <= width_lims[1]



    if depth_OK and width_OK:

        inds_to_keep.append(k)



all_student_model_names       = [all_student_model_names[k] for k in inds_to_keep]

all_short_student_model_names = [all_short_student_model_names[k] for k in inds_to_keep]



bar_plot_x_axis = 1.0 * np.arange(len(all_short_student_model_names))

bar_widths = 0.85 / len(teacher_depths)





teacher_filename = teacher_filename_from_depth_x_width_seed(teacher_depths[0], teacher_widths[0], teacher_seeds[0], teachers_folder=teachers_folder)

student_model_name = all_student_model_names[0]



training_steps, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

training_step_ind = np.argmin(np.abs(training_steps - requested_train_step))

requested_train_step_corrected = training_steps[training_step_ind]



# extract all necessary results (for all teachers)

values_dict = {}

for (t_depth, t_width, t_seed) in zip(teacher_depths, teacher_widths, teacher_seeds):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    depth_list = []

    width_list = []



    result_mean = []

    result_std = []

    result_best = []

    result_80th_percentile = []



    for student_model_name in all_student_model_names:

        _, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

        result_vec = learning_curves[:,training_step_ind]



        depth_list.append(nn_depth)

        width_list.append(nn_width)

        result_mean.append(result_vec.mean())

        result_std.append(result_vec.std())

        result_best.append(result_vec.min())

        result_80th_percentile.append(np.percentile(result_vec, 80))



    result_mean = np.array(result_mean)

    result_std = np.array(result_std)

    result_best = np.array(result_best)

    result_80th_percentile = np.array(result_80th_percentile)



    values_dict[teacher_filename] = {}

    values_dict[teacher_filename]['result_mean'] = result_mean

    values_dict[teacher_filename]['result_std'] = result_std    

    values_dict[teacher_filename]['result_best'] = result_best    

    values_dict[teacher_filename]['result_80th_percentile'] = result_80th_percentile    

    values_dict[teacher_filename]['baseline_mse'] = get_baseline_mse(teacher_filename, student_model_name, valid_index=requested_valid_index)    





# display the figure

plt.figure(figsize=(25,24));

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.96, hspace=0.33, wspace=0.1)

plt.suptitle('%s "valid_%d" fitting patterns for various teachers' %('best' if show_best else 'mean', requested_valid_index), fontsize=24)



plt.subplot(2,1,1);

for k, (t_depth, t_width, t_seed) in enumerate(zip(teacher_depths, teacher_widths, teacher_seeds)):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    curr_result_mean            = values_dict[teacher_filename]['result_mean']    

    curr_result_std             = values_dict[teacher_filename]['result_std']

    curr_result_best            = values_dict[teacher_filename]['result_best']    

    curr_result_80th_percentile = values_dict[teacher_filename]['result_80th_percentile']

    t_baseline_mse              = values_dict[teacher_filename]['baseline_mse']

    

    curr_label = 'FCN_DxW__%dx%d__seed_%d (baseline_mse = %.4f)' %(t_depth, t_width, t_seed, t_baseline_mse)



    if show_best:

        result_zeros = np.zeros(curr_result_80th_percentile.shape)

        y = curr_result_best

        y_err = [result_zeros, curr_result_80th_percentile - curr_result_best]

    else:

        y = curr_result_best

        y_err = curr_result_std

    

    if show_error_bars:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, yerr=y_err, color=teacher_colors[k], label=curr_label)

    else:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, color=teacher_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis + bar_widths, all_short_student_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%)', fontsize=24)

plt.legend(fontsize=24)





plt.subplot(2,1,2);

for k, (t_depth, t_width, t_seed) in enumerate(zip(teacher_depths, teacher_widths, teacher_seeds)):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    curr_result_mean            = values_dict[teacher_filename]['result_mean']    

    curr_result_std             = values_dict[teacher_filename]['result_std']

    curr_result_best            = values_dict[teacher_filename]['result_best']    

    curr_result_80th_percentile = values_dict[teacher_filename]['result_80th_percentile']

    t_baseline_mse              = values_dict[teacher_filename]['baseline_mse']

    

    curr_label = 'FCN_DxW__%dx%d__seed_%d (baseline_mse = %.4f)' %(t_depth, t_width, t_seed, t_baseline_mse)



    if show_best:

        result_zeros = np.zeros(curr_result_80th_percentile.shape)

        y = curr_result_best

        y_err = [result_zeros, curr_result_80th_percentile - curr_result_best]

    else:

        y = curr_result_best

        y_err = curr_result_std

    

    if show_error_bars:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, yerr=y_err, color=teacher_colors[k], label=curr_label)

    else:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, color=teacher_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis + bar_widths, all_short_student_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%) (log scale)', fontsize=24)

plt.yscale('log')

plt.legend(fontsize=24);

teacher_depths = [4,4,4]

teacher_widths = [4,10,16]

teacher_seeds  = [1234, 1234, 1234]

teacher_colors = ['green', 'orange', 'red']



requested_valid_index = 1

requested_train_step = 1200

show_percent_remaining = True

depth_lims = [1,32]

width_lims = [1,256]



show_best = False

show_error_bars = False



all_student_model_names = list(teacher_training_results.keys())

all_short_student_model_names = [x.split('_stu')[0] for x in all_student_model_names]



# filter the student models to display

inds_to_keep = []

for k, student_model_name in enumerate(all_student_model_names):

    nn_depth, nn_width = get_student_depth_x_width(student_model_name)

    depth_OK = nn_depth >= depth_lims[0] and nn_depth <= depth_lims[1]

    width_OK = nn_width >= width_lims[0] and nn_width <= width_lims[1]



    if depth_OK and width_OK:

        inds_to_keep.append(k)



all_student_model_names       = [all_student_model_names[k] for k in inds_to_keep]

all_short_student_model_names = [all_short_student_model_names[k] for k in inds_to_keep]



bar_plot_x_axis = 1.0 * np.arange(len(all_short_student_model_names))

bar_widths = 0.85 / len(teacher_depths)





teacher_filename = teacher_filename_from_depth_x_width_seed(teacher_depths[0], teacher_widths[0], teacher_seeds[0], teachers_folder=teachers_folder)

student_model_name = all_student_model_names[0]



training_steps, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

training_step_ind = np.argmin(np.abs(training_steps - requested_train_step))

requested_train_step_corrected = training_steps[training_step_ind]



# extract all necessary results (for all teachers)

values_dict = {}

for (t_depth, t_width, t_seed) in zip(teacher_depths, teacher_widths, teacher_seeds):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    depth_list = []

    width_list = []



    result_mean = []

    result_std = []

    result_best = []

    result_80th_percentile = []



    for student_model_name in all_student_model_names:

        _, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

        result_vec = learning_curves[:,training_step_ind]



        depth_list.append(nn_depth)

        width_list.append(nn_width)

        result_mean.append(result_vec.mean())

        result_std.append(result_vec.std())

        result_best.append(result_vec.min())

        result_80th_percentile.append(np.percentile(result_vec, 80))



    result_mean = np.array(result_mean)

    result_std = np.array(result_std)

    result_best = np.array(result_best)

    result_80th_percentile = np.array(result_80th_percentile)



    values_dict[teacher_filename] = {}

    values_dict[teacher_filename]['result_mean'] = result_mean

    values_dict[teacher_filename]['result_std'] = result_std    

    values_dict[teacher_filename]['result_best'] = result_best    

    values_dict[teacher_filename]['result_80th_percentile'] = result_80th_percentile    

    values_dict[teacher_filename]['baseline_mse'] = get_baseline_mse(teacher_filename, student_model_name, valid_index=requested_valid_index)    





# display the figure

plt.figure(figsize=(25,24));

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.96, hspace=0.33, wspace=0.1)

plt.suptitle('%s "valid_%d" fitting patterns for various teachers' %('best' if show_best else 'mean', requested_valid_index), fontsize=24)



plt.subplot(2,1,1);

for k, (t_depth, t_width, t_seed) in enumerate(zip(teacher_depths, teacher_widths, teacher_seeds)):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    curr_result_mean            = values_dict[teacher_filename]['result_mean']    

    curr_result_std             = values_dict[teacher_filename]['result_std']

    curr_result_best            = values_dict[teacher_filename]['result_best']    

    curr_result_80th_percentile = values_dict[teacher_filename]['result_80th_percentile']

    t_baseline_mse              = values_dict[teacher_filename]['baseline_mse']

    

    curr_label = 'FCN_DxW__%dx%d__seed_%d (baseline_mse = %.4f)' %(t_depth, t_width, t_seed, t_baseline_mse)



    if show_best:

        result_zeros = np.zeros(curr_result_90th_percentile.shape)

        y = curr_result_best

        y_err = [result_zeros, curr_result_80th_percentile - curr_result_best]

    else:

        y = curr_result_best

        y_err = curr_result_std

    

    if show_error_bars:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, yerr=y_err, color=teacher_colors[k], label=curr_label)

    else:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, color=teacher_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis + bar_widths, all_short_student_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%)', fontsize=24)

plt.legend(fontsize=24)





plt.subplot(2,1,2);

for k, (t_depth, t_width, t_seed) in enumerate(zip(teacher_depths, teacher_widths, teacher_seeds)):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    curr_result_mean            = values_dict[teacher_filename]['result_mean']    

    curr_result_std             = values_dict[teacher_filename]['result_std']

    curr_result_best            = values_dict[teacher_filename]['result_best']    

    curr_result_80th_percentile = values_dict[teacher_filename]['result_80th_percentile']

    t_baseline_mse              = values_dict[teacher_filename]['baseline_mse']

    

    curr_label = 'FCN_DxW__%dx%d__seed_%d (baseline_mse = %.4f)' %(t_depth, t_width, t_seed, t_baseline_mse)



    if show_best:

        result_zeros = np.zeros(curr_result_90th_percentile.shape)

        y = curr_result_best

        y_err = [result_zeros, curr_result_80th_percentile - curr_result_best]

    else:

        y = curr_result_best

        y_err = curr_result_std

    

    if show_error_bars:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, yerr=y_err, color=teacher_colors[k], label=curr_label)

    else:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, color=teacher_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis + bar_widths, all_short_student_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%) (log scale)', fontsize=24)

plt.yscale('log')

plt.legend(fontsize=24);

teacher_depths = [1,3,8]

teacher_widths = [4,8,16]

teacher_seeds  = [1234, 1234, 1234]

teacher_colors = ['green', 'orange', 'red']





requested_valid_index = 1

requested_train_step = 1200

show_percent_remaining = True

depth_lims = [1,32]

width_lims = [1,256]



show_best = False

show_error_bars = False



all_student_model_names = list(teacher_training_results.keys())

all_short_student_model_names = [x.split('_stu')[0] for x in all_student_model_names]



# filter the student models to display

inds_to_keep = []

for k, student_model_name in enumerate(all_student_model_names):

    nn_depth, nn_width = get_student_depth_x_width(student_model_name)

    depth_OK = nn_depth >= depth_lims[0] and nn_depth <= depth_lims[1]

    width_OK = nn_width >= width_lims[0] and nn_width <= width_lims[1]



    if depth_OK and width_OK:

        inds_to_keep.append(k)



all_student_model_names       = [all_student_model_names[k] for k in inds_to_keep]

all_short_student_model_names = [all_short_student_model_names[k] for k in inds_to_keep]



bar_plot_x_axis = 1.0 * np.arange(len(all_short_student_model_names))

bar_widths = 0.85 / len(teacher_depths)





teacher_filename = teacher_filename_from_depth_x_width_seed(teacher_depths[0], teacher_widths[0], teacher_seeds[0], teachers_folder=teachers_folder)

student_model_name = all_student_model_names[0]



training_steps, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

training_step_ind = np.argmin(np.abs(training_steps - requested_train_step))

requested_train_step_corrected = training_steps[training_step_ind]



# extract all necessary results (for all teachers)

values_dict = {}

for (t_depth, t_width, t_seed) in zip(teacher_depths, teacher_widths, teacher_seeds):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    depth_list = []

    width_list = []



    result_mean = []

    result_std = []

    result_best = []

    result_80th_percentile = []



    for student_model_name in all_student_model_names:

        _, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

        result_vec = learning_curves[:,training_step_ind]



        depth_list.append(nn_depth)

        width_list.append(nn_width)

        result_mean.append(result_vec.mean())

        result_std.append(result_vec.std())

        result_best.append(result_vec.min())

        result_80th_percentile.append(np.percentile(result_vec, 80))



    result_mean = np.array(result_mean)

    result_std = np.array(result_std)

    result_best = np.array(result_best)

    result_80th_percentile = np.array(result_80th_percentile)



    values_dict[teacher_filename] = {}

    values_dict[teacher_filename]['result_mean'] = result_mean

    values_dict[teacher_filename]['result_std'] = result_std    

    values_dict[teacher_filename]['result_best'] = result_best    

    values_dict[teacher_filename]['result_80th_percentile'] = result_80th_percentile    

    values_dict[teacher_filename]['baseline_mse'] = get_baseline_mse(teacher_filename, student_model_name, valid_index=requested_valid_index)    





# display the figure

plt.figure(figsize=(25,24));

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.96, hspace=0.33, wspace=0.1)

plt.suptitle('%s "valid_%d" fitting patterns for various teachers' %('best' if show_best else 'mean', requested_valid_index), fontsize=24)



plt.subplot(2,1,1);

for k, (t_depth, t_width, t_seed) in enumerate(zip(teacher_depths, teacher_widths, teacher_seeds)):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    curr_result_mean            = values_dict[teacher_filename]['result_mean']    

    curr_result_std             = values_dict[teacher_filename]['result_std']

    curr_result_best            = values_dict[teacher_filename]['result_best']    

    curr_result_80th_percentile = values_dict[teacher_filename]['result_80th_percentile']

    t_baseline_mse              = values_dict[teacher_filename]['baseline_mse']

    

    curr_label = 'FCN_DxW__%dx%d__seed_%d (baseline_mse = %.4f)' %(t_depth, t_width, t_seed, t_baseline_mse)



    if show_best:

        result_zeros = np.zeros(curr_result_90th_percentile.shape)

        y = curr_result_best

        y_err = [result_zeros, curr_result_80th_percentile - curr_result_best]

    else:

        y = curr_result_best

        y_err = curr_result_std

    

    if show_error_bars:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, yerr=y_err, color=teacher_colors[k], label=curr_label)

    else:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, color=teacher_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis + bar_widths, all_short_student_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%)', fontsize=24)

plt.legend(fontsize=24)





plt.subplot(2,1,2);

for k, (t_depth, t_width, t_seed) in enumerate(zip(teacher_depths, teacher_widths, teacher_seeds)):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    curr_result_mean            = values_dict[teacher_filename]['result_mean']    

    curr_result_std             = values_dict[teacher_filename]['result_std']

    curr_result_best            = values_dict[teacher_filename]['result_best']    

    curr_result_80th_percentile = values_dict[teacher_filename]['result_80th_percentile']

    t_baseline_mse              = values_dict[teacher_filename]['baseline_mse']

    

    curr_label = 'FCN_DxW__%dx%d__seed_%d (baseline_mse = %.4f)' %(t_depth, t_width, t_seed, t_baseline_mse)



    if show_best:

        result_zeros = np.zeros(curr_result_90th_percentile.shape)

        y = curr_result_best

        y_err = [result_zeros, curr_result_80th_percentile - curr_result_best]

    else:

        y = curr_result_best

        y_err = curr_result_std

    

    if show_error_bars:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, yerr=y_err, color=teacher_colors[k], label=curr_label)

    else:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, color=teacher_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis + bar_widths, all_short_student_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%) (log scale)', fontsize=24)

plt.yscale('log')

plt.legend(fontsize=24);

#teacher_depths = [3, 3, 3, 3]

#teacher_widths = [8, 8, 8, 8]



#teacher_depths = [3, 3, 3, 3]

#teacher_widths = [12, 12, 12, 12]



#teacher_depths = [5, 5, 5, 5]

#teacher_widths = [8, 8, 8, 8]



#teacher_depths = [5, 5, 5, 5]

#teacher_widths = [16, 16, 16, 16]



teacher_depths = [8, 8, 8, 8]

teacher_widths = [16, 16, 16, 16]



teacher_seeds  = [1234, 4321, 1111, 1357]

teacher_colors = ['green', 'limegreen', 'deepskyblue', 'orange', 'peru', 'red']





requested_valid_index = 1

requested_train_step = 1200

show_percent_remaining = True

depth_lims = [1,32]

width_lims = [1,256]



show_best = False

show_error_bars = False



all_student_model_names = list(teacher_training_results.keys())

all_short_student_model_names = [x.split('_stu')[0] for x in all_student_model_names]



# filter the student models to display

inds_to_keep = []

for k, student_model_name in enumerate(all_student_model_names):

    nn_depth, nn_width = get_student_depth_x_width(student_model_name)

    depth_OK = nn_depth >= depth_lims[0] and nn_depth <= depth_lims[1]

    width_OK = nn_width >= width_lims[0] and nn_width <= width_lims[1]



    if depth_OK and width_OK:

        inds_to_keep.append(k)



all_student_model_names       = [all_student_model_names[k] for k in inds_to_keep]

all_short_student_model_names = [all_short_student_model_names[k] for k in inds_to_keep]



bar_plot_x_axis = 1.0 * np.arange(len(all_short_student_model_names))

bar_widths = 0.85 / len(teacher_depths)





teacher_filename = teacher_filename_from_depth_x_width_seed(teacher_depths[0], teacher_widths[0], teacher_seeds[0], teachers_folder=teachers_folder)

student_model_name = all_student_model_names[0]



training_steps, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

training_step_ind = np.argmin(np.abs(training_steps - requested_train_step))

requested_train_step_corrected = training_steps[training_step_ind]



# extract all necessary results (for all teachers)

values_dict = {}

for (t_depth, t_width, t_seed) in zip(teacher_depths, teacher_widths, teacher_seeds):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    depth_list = []

    width_list = []



    result_mean = []

    result_std = []

    result_best = []

    result_80th_percentile = []



    for student_model_name in all_student_model_names:

        _, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

        result_vec = learning_curves[:,training_step_ind]



        depth_list.append(nn_depth)

        width_list.append(nn_width)

        result_mean.append(result_vec.mean())

        result_std.append(result_vec.std())

        result_best.append(result_vec.min())

        result_80th_percentile.append(np.percentile(result_vec, 80))



    result_mean = np.array(result_mean)

    result_std = np.array(result_std)

    result_best = np.array(result_best)

    result_80th_percentile = np.array(result_80th_percentile)



    values_dict[teacher_filename] = {}

    values_dict[teacher_filename]['result_mean'] = result_mean

    values_dict[teacher_filename]['result_std'] = result_std    

    values_dict[teacher_filename]['result_best'] = result_best    

    values_dict[teacher_filename]['result_80th_percentile'] = result_80th_percentile    

    values_dict[teacher_filename]['baseline_mse'] = get_baseline_mse(teacher_filename, student_model_name, valid_index=requested_valid_index)    





# display the figure

plt.figure(figsize=(25,24));

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.96, hspace=0.33, wspace=0.1)

plt.suptitle('%s "valid_%d" fitting patterns for various teachers' %('best' if show_best else 'mean', requested_valid_index), fontsize=24)



plt.subplot(2,1,1);

for k, (t_depth, t_width, t_seed) in enumerate(zip(teacher_depths, teacher_widths, teacher_seeds)):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    curr_result_mean            = values_dict[teacher_filename]['result_mean']    

    curr_result_std             = values_dict[teacher_filename]['result_std']

    curr_result_best            = values_dict[teacher_filename]['result_best']    

    curr_result_80th_percentile = values_dict[teacher_filename]['result_80th_percentile']

    t_baseline_mse              = values_dict[teacher_filename]['baseline_mse']

    

    curr_label = 'FCN_DxW__%dx%d__seed_%d (baseline_mse = %.4f)' %(t_depth, t_width, t_seed, t_baseline_mse)



    if show_best:

        result_zeros = np.zeros(curr_result_90th_percentile.shape)

        y = curr_result_best

        y_err = [result_zeros, curr_result_80th_percentile - curr_result_best]

    else:

        y = curr_result_best

        y_err = curr_result_std

    

    if show_error_bars:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, yerr=y_err, color=teacher_colors[k], label=curr_label)

    else:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, color=teacher_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis + bar_widths, all_short_student_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%)', fontsize=24)

plt.legend(fontsize=24)





plt.subplot(2,1,2);

for k, (t_depth, t_width, t_seed) in enumerate(zip(teacher_depths, teacher_widths, teacher_seeds)):

    teacher_filename = teacher_filename_from_depth_x_width_seed(t_depth, t_width, t_seed, teachers_folder=teachers_folder)



    curr_result_mean            = values_dict[teacher_filename]['result_mean']    

    curr_result_std             = values_dict[teacher_filename]['result_std']

    curr_result_best            = values_dict[teacher_filename]['result_best']    

    curr_result_80th_percentile = values_dict[teacher_filename]['result_80th_percentile']

    t_baseline_mse              = values_dict[teacher_filename]['baseline_mse']

    

    curr_label = 'FCN_DxW__%dx%d__seed_%d (baseline_mse = %.4f)' %(t_depth, t_width, t_seed, t_baseline_mse)



    if show_best:

        result_zeros = np.zeros(curr_result_90th_percentile.shape)

        y = curr_result_best

        y_err = [result_zeros, curr_result_80th_percentile - curr_result_best]

    else:

        y = curr_result_best

        y_err = curr_result_std

    

    if show_error_bars:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, yerr=y_err, color=teacher_colors[k], label=curr_label)

    else:

        plt.bar(bar_plot_x_axis + k * bar_widths, y, bar_widths, color=teacher_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis + bar_widths, all_short_student_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%) (log scale)', fontsize=24)

plt.yscale('log')

plt.legend(fontsize=24);



def get_all_fitting_patterns(data_foldername, return_mean=True, requested_train_step=1200):

    

    all_full_filenames = glob.glob(os.path.join(data_foldername, '*.pickle'))

    all_filenames = sorted([filename.split('/')[-1].split('.p')[0] for filename in all_full_filenames])

    num_rows = len(all_full_filenames)

    

    # build teacher_df

    teacher_df = pd.DataFrame(index=range(num_rows), columns=['depth','width','init_method','seed'])

    for k, filename in enumerate(all_filenames):

        

        curr_depth = int(filename.split('-O__')[-1].split('__n_models')[0].split('-')[1].split('x')[0])

        curr_width = int(filename.split('-O__')[-1].split('__n_models')[0].split('-')[1].split('x')[1])

        curr_seed  = int(filename.split('seed_')[-1])

        curr_init  = filename.split('_init__')[0].split('LReLU_05_')[-1]

        

        teacher_df.loc[k,'depth'] = curr_depth

        teacher_df.loc[k,'width'] = curr_width

        teacher_df.loc[k,'seed'] = curr_seed

        teacher_df.loc[k,'init_method'] = curr_init

        

    

    # build all results dfs

    with open(all_full_filenames[0], "rb") as f:

        teacher_training_results = pickle.load(f)



    all_student_model_names = list(teacher_training_results.keys())

    all_short_student_model_names = [x.split('_stu')[0] for x in all_student_model_names]

        

    fitting_pattern_mean_df = pd.DataFrame(index=range(num_rows), columns=all_short_student_model_names)

    fitting_pattern_std_df  = pd.DataFrame(index=range(num_rows), columns=all_short_student_model_names)

    

    fitting_pattern_best_df            = pd.DataFrame(index=range(num_rows), columns=all_short_student_model_names)

    fitting_pattern_median_df          = pd.DataFrame(index=range(num_rows), columns=all_short_student_model_names)

    fitting_pattern_80th_percentile_df = pd.DataFrame(index=range(num_rows), columns=all_short_student_model_names)

    

    training_steps, learning_curves = get_learning_curves(all_full_filenames[0], all_student_model_names[0], valid_index=1, percent_var_remaining=True)

    training_step_ind = np.argmin(np.abs(training_steps - requested_train_step))

    for k, teacher_filename in enumerate(all_full_filenames):

        result_mean = []

        result_std  = []

        result_best            = []

        result_median          = []

        result_80th_percentile = []



        for student_model_name in all_student_model_names:

            _, learning_curves = get_learning_curves(teacher_filename, student_model_name, valid_index=1, percent_var_remaining=True)

            result_vec = learning_curves[:,training_step_ind]



            result_mean.append(result_vec.mean())

            result_std.append(result_vec.std())

            result_best.append(result_vec.min())

            result_median.append(np.percentile(result_vec, 50))

            result_80th_percentile.append(np.percentile(result_vec, 80))



        fitting_pattern_mean_df.loc[k,:] = np.array(result_mean)

        fitting_pattern_std_df.loc[k,:]  = np.array(result_std)

        fitting_pattern_best_df.loc[k,:]            = np.array(result_best)

        fitting_pattern_median_df.loc[k,:]          = np.array(result_median)

        fitting_pattern_80th_percentile_df.loc[k,:] = np.array(result_80th_percentile)



    if return_mean:

        return teacher_df, fitting_pattern_mean_df, fitting_pattern_std_df

    else:

        return teacher_df, fitting_pattern_best_df, fitting_pattern_median_df, fitting_pattern_80th_percentile_df

teacher_df, fitting_pattern_mean_df, fitting_pattern_std_df = get_all_fitting_patterns(data_foldername, return_mean=True)



teacher_df, fitting_pattern_best_df, fitting_pattern_median_df, fitting_pattern_80th_percentile_df = get_all_fitting_patterns(data_foldername, return_mean=False)
plt.figure(figsize=(25,30));

plt.subplots_adjust(left=0.06, bottom=0.14, right=0.94, top=0.96, hspace=0.12, wspace=0.1)



all_short_student_model_names = fitting_pattern_mean_df.columns.tolist()

x_axis = range(len(all_short_student_model_names))



max_depth_log2 = np.log2(teacher_df['depth'].max())

max_width_log2 = np.log2(teacher_df['width'].max())



curve_colors = []

curve_names = []

for k in range(teacher_df.shape[0]):

    

    nn_depth = teacher_df.loc[k,'depth']

    nn_width = teacher_df.loc[k,'width']

    

    curr_curve_color = (np.log2(nn_depth) / max_depth_log2, 0.2, np.log2(nn_width) / max_width_log2)

    curve_colors.append(curr_curve_color)

    curve_names.append('FCN_DxW_%dx%d' %(nn_depth, nn_width))



plt.subplot(5,1,1);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.plot(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('mean', fontsize=20)

plt.xticks([], []); plt.legend(curve_names, ncol=8)



plt.subplot(5,1,2);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.plot(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('std', fontsize=20)

plt.xticks([], []);



plt.subplot(5,1,3);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.plot(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('best', fontsize=20)

plt.xticks([], [])



plt.subplot(5,1,4);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.plot(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('median', fontsize=20)

plt.xticks([], [])



plt.subplot(5,1,5);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.plot(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('80th percentile', fontsize=20)



plt.xticks(x_axis, all_short_student_model_names, rotation=90, fontsize=18);

plt.figure(figsize=(25,30));

plt.subplots_adjust(left=0.06, bottom=0.14, right=0.94, top=0.96, hspace=0.12, wspace=0.1)



all_short_student_model_names = fitting_pattern_mean_df.columns.tolist()

x_axis = range(len(all_short_student_model_names))



max_depth_log2 = np.log2(teacher_df['depth'].max())

max_width_log2 = np.log2(teacher_df['width'].max())



curve_colors = []

curve_names = []

for k in range(teacher_df.shape[0]):

    

    nn_depth = teacher_df.loc[k,'depth']

    nn_width = teacher_df.loc[k,'width']

    

    curr_curve_color = (np.log2(nn_depth) / max_depth_log2, 0.2, np.log2(nn_width) / max_width_log2)

    curve_colors.append(curr_curve_color)

    curve_names.append('FCN_DxW_%dx%d' %(nn_depth, nn_width))



plt.subplot(5,1,1);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.semilogy(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('mean', fontsize=20)

plt.xticks([], []); plt.legend(curve_names, ncol=8)



plt.subplot(5,1,2);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.semilogy(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('std', fontsize=20)

plt.xticks([], []);



plt.subplot(5,1,3);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.semilogy(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('best', fontsize=20)

plt.xticks([], [])



plt.subplot(5,1,4);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.semilogy(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('median', fontsize=20)

plt.xticks([], [])



plt.subplot(5,1,5);

for k in range(fitting_pattern_mean_df.shape[0]):

    plt.semilogy(fitting_pattern_mean_df.loc[k,:], c=curve_colors[k]); 

plt.title('80th percentile', fontsize=20)



plt.xticks(x_axis, all_short_student_model_names, rotation=90, fontsize=18);

X = np.array(fitting_pattern_mean_df).astype(float)



num_components = 6

meanPCA_model = decomposition.PCA(n_components=num_components, whiten=True)

meanPCA_model.fit(X)

pattern_features = meanPCA_model.transform(np.log10(X))



depth = np.array(teacher_df['depth'])

width = np.array(teacher_df['width'])



fig, axs = plt.subplots(nrows=4, ncols=num_components, figsize=(25,13))

fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.08, wspace=0.08)

for k in range(num_components):

    

    axs[0,k].scatter(pattern_features[:,k], depth + 0.5 * np.random.randn(depth.shape[0]))

    axs[1,k].scatter(pattern_features[:,k], width + 0.5 * np.random.randn(width.shape[0]))

    axs[2,k].scatter(pattern_features[:,k], width + 2 * depth + 0.5 * np.random.randn(width.shape[0]))

    axs[3,k].scatter(pattern_features[:,k], width * depth)

    

    if k == 0:

        axs[0,k].set_ylabel('depth', fontsize=20)

        axs[1,k].set_ylabel('width', fontsize=20)

        axs[2,k].set_ylabel('width + 2*depth', fontsize=20)

        axs[3,k].set_ylabel('width * depth', fontsize=20)

    axs[2,k].set_xlabel('PC %d' %(k+1), fontsize=20)



fig.suptitle('PCA representation of fitting pattern vs network size', fontsize=20);