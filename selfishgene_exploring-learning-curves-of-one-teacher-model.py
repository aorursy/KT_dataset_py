import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pickle

import time

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



import os

print('-------------------------')

print('all files:')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

print('-------------------------')
#teachers_folder = '../input/bigger-is-better/orthogonal_normlized_init/orthogonal_normlized_init/'

#filename = 'FCN_LReLU_04__I-DxW-O__10-2x2-1__n_models_63__n_inits_20__n_iter_1200__seed_1111.pickle'



#teachers_folder = '../input/bigger-is-better/with_weights/with_weights/'

#filename = 'FCN_LReLU_05_tnorm_init__I-DxW-O__10-4x4-1__n_models_63__n_inits_20_fanin__n_iter_1200__seed_1234.pickle'



#teachers_folder = '../input/bigger-is-better/with_weights_2/with_weights_2/'

#filename = 'FCN_LReLU_05_tnorm_init__I-DxW-O__10-6x6-1__n_models_42__n_inits_20_fanin__n_iter_1200__seed_1234.pickle'



teachers_folder = '../input/bigger-is-better/'

filename = 'FCN_LReLU_05_gauss_IO_norm_init__I-DxW-O__10-8x8-1__n_models_42__n_inits_30_fanin__n_iter_1200__seed_1234.pickle'

filename = 'FCN_LReLU_05_gauss_IO_norm_init__I-DxW-O__10-2x4-1__n_models_42__n_inits_30_fanin__n_iter_1200__seed_1234.pickle'

filename = 'FCN_LReLU_05_gauss_IO_norm_init__I-DxW-O__10-5x8-1__n_models_42__n_inits_30_fanin__n_iter_1200__seed_1234.pickle'



teacher_filename = os.path.join(teachers_folder, filename)



results_filename = teacher_filename

with open(results_filename, "rb") as f:

    training_results = pickle.load(f)

    

GT_model_name = results_filename.split('/')[-1].split('__n_models')[0]



print('-----------------------------')

print('GT model name: "%s"' %(GT_model_name))

print('-----------------------------')



print('-----------------------------')

all_model_names = list(training_results.keys())

print('all model names:')

for model_name in all_model_names:

    print('  ' + model_name)

print('-----------------------------')



single_model_results = training_results[all_model_names[8]]

all_results_dict_keys = list(single_model_results.keys())

print('single model results dict keys = %s' %(all_results_dict_keys))





print('single model hyperparams = %s' %(single_model_results['model_hyperparams_dict']))



print('single model y_GT_stats = ' %(single_model_results['model_hyperparams_dict']['y_GT_stats_dict']))

[print(x, single_model_results['model_hyperparams_dict']['y_GT_stats_dict'][x]) for x in single_model_results['model_hyperparams_dict']['y_GT_stats_dict'].keys()]



print('-----------------------------')



# collect all learning curves for the same model in the same matrix

learning_curve_matrix = np.zeros((len(single_model_results['all learning curves']), single_model_results['all learning curves'][0]['valid_1_loss'].shape[0]))

for k, curr_learning_curves in enumerate(single_model_results['all learning curves']):

    learning_curve_matrix[k, :] = curr_learning_curves['valid_1_loss']

    

num_batches_vec = single_model_results['all learning curves'][0]['num_batches']

num_samples_vec = single_model_results['all learning curves'][0]['num_samples']



print('single model learning_curve_matrix.shape = %s' %(str(learning_curve_matrix.shape)))

print('max number of batches (training steps/iterations) is %d' %(num_batches_vec[-1]))

print('max number training samples is %d' %(num_samples_vec[-1]))



print('-----------------------------')

print('single model all final losses = %s' %(single_model_results['all final losses']))



print('single model key outcomes = %s' %(single_model_results['key outcomes']))

print('-----------------------------')
plt.figure(figsize=(15,7))

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.94, hspace=0.1, wspace=0.1)



plt.plot(num_batches_vec, learning_curve_matrix.T, color='b', alpha=0.6)

plt.plot(num_batches_vec, learning_curve_matrix.mean(axis=0), color='k', label='average')

plt.plot(num_batches_vec, learning_curve_matrix.max(axis=0), color='r', label='worst')

plt.plot(num_batches_vec, learning_curve_matrix.min(axis=0), color='g', label='best')

plt.xlabel('batch index', fontsize=20)

plt.ylabel('MSE', fontsize=20)

plt.legend(fontsize=20)
def get_learning_curve_matrix(model_name, valid_index=1):

    single_model_results = training_results[model_name]



    valid_loss = 'valid_%d_loss' %(valid_index)

    

    # collect all learning curves for the same model in the same matrix

    learning_curve_matrix = np.zeros((len(single_model_results['all learning curves']), single_model_results['all learning curves'][0][valid_loss].shape[0]))

    for k, curr_learning_curves in enumerate(single_model_results['all learning curves']):

        learning_curve_matrix[k, :] = curr_learning_curves[valid_loss]



    return learning_curve_matrix





def get_learning_curves(model_name, valid_index=1, percent_var_remaining=False):

    single_model_results = training_results[model_name]



    valid_loss = 'valid_%d_loss' %(valid_index)

    valid_key = 'valid_%d' %(valid_index)

    

    # collect all learning curves for the same model in the same matrix

    learning_curves = np.zeros((len(single_model_results['all learning curves']), single_model_results['all learning curves'][0][valid_loss].shape[0]))

    for k, curr_learning_curves in enumerate(single_model_results['all learning curves']):

        learning_curves[k, :] = curr_learning_curves[valid_loss]



    training_steps = single_model_results['all learning curves'][0]['num_batches']

    

    if percent_var_remaining:

        baseline_mse = single_model_results['model_hyperparams_dict']['y_GT_stats_dict'][valid_key]['mse_0']

        learning_curves = 100 * learning_curves / baseline_mse

        

    return training_steps, learning_curves





def get_depth_x_width(model_name):

    single_model_results = training_results[model_name]

    

    nn_depth = single_model_results['model_hyperparams_dict']['student_nn_depth']

    nn_width = single_model_results['model_hyperparams_dict']['student_nn_width']

    

    return nn_depth, nn_width





def get_key_outcomes(model_name):

    single_model_results = training_results[model_name]



    return single_model_results['key outcomes']

plt.figure(figsize=(16,12))

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.93, hspace=0.1, wspace=0.1)

plt.suptitle('Average Learning Curves', fontsize=24)



max_depth_log2 = 5

max_width_log2 = 8



all_model_names = list(training_results.keys())

ax0 = plt.subplot(2,1,1)

ax1 = plt.subplot(2,1,2)



for model_name in all_model_names:

    model_label = model_name.split('_stu')[0]

    learning_curve_matrix = get_learning_curve_matrix(model_name)

    nn_depth, nn_width = get_depth_x_width(model_name)

    curve_color = (np.log2(nn_depth) / max_depth_log2, 0.2, np.log2(nn_width) / max_width_log2)

    ax0.plot(num_batches_vec, learning_curve_matrix.mean(axis=0), color=curve_color, label=model_label)

    ax1.semilogy(num_batches_vec, learning_curve_matrix.mean(axis=0), color=curve_color, label=model_label)



ax0.set_ylabel('MSE', fontsize=20)

ax1.set_ylabel('MSE (log scale)', fontsize=20)

ax0.legend(fontsize=11, ncol=6)

plt.xlabel('training steps', fontsize=20);
plt.figure(figsize=(16,12))

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.93, hspace=0.1, wspace=0.1)

plt.suptitle('Best Learning Curves', fontsize=24)



all_model_names = list(training_results.keys())

ax0 = plt.subplot(2,1,1)

ax1 = plt.subplot(2,1,2)



for model_name in all_model_names:

    model_label = model_name.split('_stu')[0]

    learning_curve_matrix = get_learning_curve_matrix(model_name)

    nn_depth, nn_width = get_depth_x_width(model_name)

    curve_color = (np.log2(nn_depth) / max_depth_log2, 0.2, np.log2(nn_width) / max_width_log2)

    ax0.plot(num_batches_vec, learning_curve_matrix.min(axis=0), color=curve_color, label=model_label)

    ax1.semilogy(num_batches_vec, learning_curve_matrix.min(axis=0), color=curve_color, label=model_label)



ax0.set_ylabel('MSE', fontsize=20)

ax1.set_ylabel('MSE (log scale)', fontsize=20)

ax0.legend(fontsize=11, ncol=6)

plt.xlabel('training steps', fontsize=20);
requested_valid_index = 1

var_remaining_threshold = 5



not_reached_value = 1390



depth_lims = [1,32]

width_lims = [1,256]



all_model_names = list(training_results.keys())

short_model_names = [x.split('_stu')[0] for x in all_model_names]



depth_list = []

width_list = []



num_iterations_to_reach = []

fraction_reached = []



all_short_model_names = []



for model_name in all_model_names:

    nn_depth, nn_width = get_depth_x_width(model_name)

    depth_OK = nn_depth >= depth_lims[0] and nn_depth <= depth_lims[1]

    width_OK = nn_width >= width_lims[0] and nn_width <= width_lims[1]

    

    if depth_OK and width_OK:

        training_steps, learning_curves = get_learning_curves(model_name, valid_index=requested_valid_index, percent_var_remaining=True)

        

        has_reached_matrix = learning_curves < var_remaining_threshold

        

        if has_reached_matrix.any():

            fraction_reached_vec = has_reached_matrix.mean(axis=0)

            first_ind = np.argmax(fraction_reached_vec > 0)

            

            num_iterations_to_reach.append(training_steps[first_ind])

            fraction_reached.append(fraction_reached_vec[first_ind])

        else:

            num_iterations_to_reach.append(not_reached_value)

            fraction_reached.append(0)



        depth_list.append(nn_depth)

        width_list.append(nn_width)

        all_short_model_names.append(model_name.split('_stu')[0])



num_iterations_to_reach = np.array(num_iterations_to_reach)

fraction_reached = np.array(fraction_reached)



bar_plot_x_axis = range(len(all_short_model_names))



plt.figure(figsize=(18,20));

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.96, hspace=0.33, wspace=0.1)

plt.suptitle('learning speed for "valid_%d", GT model "%s"' %(requested_valid_index, GT_model_name), fontsize=24)

plt.subplot(2,1,1);

plt.bar(bar_plot_x_axis, num_iterations_to_reach)

plt.xticks(bar_plot_x_axis, all_short_model_names, rotation=90, fontsize=18);

plt.ylabel('num iterations to reach %d%s var remaining' %(var_remaining_threshold, '%'), fontsize=20)

plt.ylim([0, 1.1 * num_iterations_to_reach.max()])

plt.subplot(2,1,2);

plt.bar(bar_plot_x_axis, num_iterations_to_reach)

plt.xticks(bar_plot_x_axis, all_short_model_names, rotation=90, fontsize=18);

plt.ylabel('num iterations to reach %d%s var remaining (log scale)' %(var_remaining_threshold, '%'), fontsize=20)

plt.yscale('log');
requested_valid_index = 1

requested_train_steps = [30, 140, 1200]

train_step_colors = ['red', 'blue', 'green']

train_step_colors.reverse()

show_percent_remaining = True

depth_lims = [1,32]

width_lims = [4,256]



training_steps, learning_curves = get_learning_curves(model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)





all_model_names = list(training_results.keys())

short_model_names = [x.split('_stu')[0] for x in all_model_names]

bar_plot_x_axis = range(len(all_short_model_names))





# extract all necessary results

values_dict = {}

for requested_train_step in requested_train_steps:



    training_steps, _ = get_learning_curves(model_name)

    training_step_ind = np.argmin(np.abs(training_steps - requested_train_step))

    requested_train_step_corrected = training_steps[training_step_ind]



    depth_list = []

    width_list = []



    result_mean = []

    result_std = []

    result_best = []

    result_90th_percentile = []



    all_short_model_names = []



    for model_name in all_model_names:

        nn_depth, nn_width = get_depth_x_width(model_name)

        depth_OK = nn_depth >= depth_lims[0] and nn_depth <= depth_lims[1]

        width_OK = nn_width >= width_lims[0] and nn_width <= width_lims[1]



        if depth_OK and width_OK:

            _, learning_curves = get_learning_curves(model_name, valid_index=requested_valid_index, percent_var_remaining=show_percent_remaining)

            result_vec = learning_curves[:,training_step_ind]



            depth_list.append(nn_depth)

            width_list.append(nn_width)

            result_mean.append(result_vec.mean())

            result_std.append(result_vec.std())

            all_short_model_names.append(model_name.split('_stu')[0])



    result_mean = np.array(result_mean)

    result_std = np.array(result_std)



    values_dict[requested_train_step_corrected] = {}

    values_dict[requested_train_step_corrected]['result_mean'] = result_mean

    values_dict[requested_train_step_corrected]['result_std'] = result_std    





    

# display the figure    

sorted_train_steps = sorted(list(values_dict.keys()), reverse=True)



plt.figure(figsize=(18,20));

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.96, hspace=0.33, wspace=0.1)

plt.suptitle('"valid_%d" training dynamics for GT model "%s"' %(requested_valid_index, GT_model_name), fontsize=24)





plt.subplot(2,1,1);

for k, train_step in enumerate(sorted_train_steps):

    curr_result_mean = values_dict[train_step]['result_mean']    

    curr_result_std = values_dict[train_step]['result_std']

    curr_label = 'after_%d_iterations' %(train_step)

    

    if k == 0:

        plt.bar(bar_plot_x_axis, curr_result_mean, yerr=curr_result_std, color=train_step_colors[-k], label=curr_label)

    else:

        prev_result_mean = values_dict[sorted_train_steps[k-1]]['result_mean']

        result_mean_diff = curr_result_mean - prev_result_mean

        plt.bar(bar_plot_x_axis, result_mean_diff, yerr=curr_result_std, bottom=prev_result_mean, color=train_step_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis, all_short_model_names, rotation=90, fontsize=18);

if show_percent_remaining:

    plt.ylabel('variance remaining (%)', fontsize=20)

else:

    plt.ylabel('MSE', fontsize=20)

plt.ylim([0, 1.3 * values_dict[sorted_train_steps[-1]]['result_mean'].max()])

plt.legend(fontsize=20)





plt.subplot(2,1,2);

for k, train_step in enumerate(sorted_train_steps):

    curr_result_mean = values_dict[train_step]['result_mean']    

    curr_result_std = values_dict[train_step]['result_std']

    curr_label = 'after_%d_iterations' %(train_step)

    

    if k == 0:

        plt.bar(bar_plot_x_axis, curr_result_mean, yerr=curr_result_std, color=train_step_colors[-k], label=curr_label)

    else:

        prev_result_mean = values_dict[sorted_train_steps[k-1]]['result_mean']

        result_mean_diff = curr_result_mean - prev_result_mean

        plt.bar(bar_plot_x_axis, result_mean_diff, yerr=curr_result_std, bottom=prev_result_mean, color=train_step_colors[k], label=curr_label)

        

plt.xticks(bar_plot_x_axis, all_short_model_names, rotation=90, fontsize=18);

if show_percent_remaining:

    plt.ylabel('variance remaining (%)', fontsize=20)

else:

    plt.ylabel('MSE', fontsize=20)



plt.xticks(bar_plot_x_axis, all_short_model_names, rotation=90, fontsize=18);

if show_percent_remaining:

    plt.ylabel('variance remaining (%) (log scale)', fontsize=20)

else:

    plt.ylabel('MSE (log scale)', fontsize=20)

plt.yscale('log')

plt.legend(fontsize=20);
all_model_names = list(training_results.keys())



final_mean = []

final_std = []

depth = []

width = []



for model_name in all_model_names:

    key_outcomes = get_key_outcomes(model_name)

    nn_depth, nn_width = get_depth_x_width(model_name)



    final_mean.append(key_outcomes['mean final point'])

    final_std.append(key_outcomes['std final point'])

    

    depth.append(nn_depth)

    width.append(nn_width)



short_model_names = [x.split('_stu')[0] for x in all_model_names]



final_mean = np.array(final_mean)

final_std = np.array(final_std)

    

x_axis = range(len(short_model_names))



mse_0 = single_model_results['model_hyperparams_dict']['y_GT_stats_dict']['valid_1']['mse_0']

final_mean = 100 * final_mean / mse_0

final_std  = 100 * final_std / mse_0



plt.figure(figsize=(18,20));

plt.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.93, hspace=0.33, wspace=0.1)

plt.subplot(2,1,1);

plt.bar(x_axis, final_mean, yerr=np.minimum(final_mean, final_std))

plt.xticks(x_axis, short_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%)', fontsize=20)

plt.subplot(2,1,2);

plt.bar(x_axis, final_mean, yerr=np.minimum(final_mean, final_std))

plt.xticks(x_axis, short_model_names, rotation=90, fontsize=18);

plt.ylabel('variance remaining (%) (log scale)', fontsize=20)

plt.yscale('log')
# build a dataframe

dataframe_cols = ['full_model_name', 'short_name', 'depth', 'width', 'final_MSE_mean', 'final_MSE_std']

results_summary_df = pd.DataFrame(index=range(len(all_model_names)), columns=dataframe_cols)



results_summary_df.loc[:,'full_model_name'] = all_model_names

results_summary_df.loc[:,'short_name'] = short_model_names

results_summary_df.loc[:,'depth'] = depth

results_summary_df.loc[:,'width'] = width

results_summary_df.loc[:,'final_MSE_mean'] = final_mean

results_summary_df.loc[:,'final_MSE_std'] = final_std

results_summary_df
plt.figure(figsize=(15,10))

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.97, hspace=0.1)



max_depth = 32

unique_depth = sorted(results_summary_df['depth'].unique())

unique_depth = [x for x in unique_depth if x <= max_depth]



plt.subplot(2,1,1);

for depth in unique_depth:

    const_depth_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'width']

    const_depth_curve     = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_mean']

    const_depth_curve_std = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_std']

    const_depth_curve_std = np.minimum(const_depth_curve_std, const_depth_curve)

    plt.errorbar(const_depth_curve_x, const_depth_curve, yerr=const_depth_curve_std, label='%d_layers' %(depth))



plt.ylabel('MSE', fontsize=20)

plt.legend(fontsize=16, ncol=1)





plt.subplot(2,1,2);

for depth in unique_depth:

    const_depth_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'width']

    const_depth_curve     = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_mean']

    const_depth_curve_std = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_std']

    const_depth_curve_std = np.minimum(const_depth_curve_std, const_depth_curve)

    plt.errorbar(const_depth_curve_x, const_depth_curve, yerr=const_depth_curve_std, label='%d_layers' %(depth))

    

plt.xlabel('Width', fontsize=20)

plt.ylabel('MSE (log scale)', fontsize=20)

plt.yscale('log')
plt.figure(figsize=(15,10))

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.97, hspace=0.1)



max_depth = 32

unique_depth = sorted(results_summary_df['depth'].unique())

unique_depth = [x for x in unique_depth if x <= max_depth]



plt.subplot(2,1,1);

for depth in unique_depth:

    const_depth_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'width']

    const_depth_curve     = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_mean']

    const_depth_curve_std = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_std']

    const_depth_curve_std = np.minimum(const_depth_curve_std, const_depth_curve)

    plt.errorbar(const_depth_curve_x, const_depth_curve, yerr=const_depth_curve_std, label='%d_layers' %(depth))



plt.ylabel('MSE', fontsize=20)

plt.legend(fontsize=16, ncol=1)

plt.xscale('log')



plt.subplot(2,1,2);

for depth in unique_depth:

    const_depth_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'width']

    const_depth_curve     = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_mean']

    const_depth_curve_std = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_std']

    const_depth_curve_std = np.minimum(const_depth_curve_std, const_depth_curve)

    plt.errorbar(const_depth_curve_x, const_depth_curve, yerr=const_depth_curve_std, label='%d_layers' %(depth))

    

plt.xlabel('Width (log scale)', fontsize=20)

plt.ylabel('MSE (log scale)', fontsize=20)

plt.yscale('log')

plt.xscale('log');
plt.figure(figsize=(15,10))

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.97, hspace=0.1)



max_depth = 16

unique_depth = sorted(results_summary_df['depth'].unique())

unique_depth = [x for x in unique_depth if x <= max_depth]



plt.subplot(2,1,1);

for depth in unique_depth:

    const_depth_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'width']

    const_depth_curve     = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_mean']

    const_depth_curve_std = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_std']

    const_depth_curve_std = np.minimum(const_depth_curve_std, const_depth_curve)

    plt.errorbar(const_depth_curve_x, const_depth_curve, label='%d_layers' %(depth))



plt.ylabel('MSE', fontsize=20)

plt.legend(fontsize=16, ncol=1)

plt.xscale('log')



plt.subplot(2,1,2);

for depth in unique_depth:

    const_depth_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'width']

    const_depth_curve     = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_mean']

    const_depth_curve_std = results_summary_df.loc[results_summary_df.loc[:,'depth'] == depth, 'final_MSE_std']

    const_depth_curve_std = np.minimum(const_depth_curve_std, const_depth_curve)

    plt.errorbar(const_depth_curve_x, const_depth_curve, label='%d_layers' %(depth))

    

plt.xlabel('Width (log scale)', fontsize=20)

plt.ylabel('MSE (log scale)', fontsize=20)

plt.yscale('log')

plt.xscale('log');
plt.figure(figsize=(15,10))

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.97, hspace=0.1)

max_depth = 32



unique_width = sorted(results_summary_df['width'].unique())



plt.subplot(2,1,1);

for width in unique_width:

    const_width_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'depth']

    const_width_curve     = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_mean']

    const_width_curve_std = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_std']

    const_width_curve_std = np.minimum(const_width_curve_std, const_width_curve)

    

    selected_inds = const_width_curve_x <= max_depth

    const_width_curve_x   = const_width_curve_x[selected_inds]

    const_width_curve     = const_width_curve[selected_inds]

    const_width_curve_std = const_width_curve_std[selected_inds]

    

    plt.errorbar(const_width_curve_x, const_width_curve, yerr=const_width_curve_std, label='%d_units' %(width))



plt.ylabel('MSE', fontsize=20)

plt.legend(fontsize=16, ncol=4)





plt.subplot(2,1,2);

for width in unique_width:

    const_width_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'depth']

    const_width_curve     = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_mean']

    const_width_curve_std = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_std']

    const_width_curve_std = np.minimum(const_width_curve_std, const_width_curve)

    

    selected_inds = const_width_curve_x <= max_depth

    const_width_curve_x   = const_width_curve_x[selected_inds]

    const_width_curve     = const_width_curve[selected_inds]

    const_width_curve_std = const_width_curve_std[selected_inds]



    plt.errorbar(const_width_curve_x, const_width_curve, yerr=const_width_curve_std, label='%d_units' %(width))

    

plt.xlabel('Depth', fontsize=20)

plt.ylabel('MSE (log scale)', fontsize=20)

plt.yscale('log')
plt.figure(figsize=(15,10))

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.97, hspace=0.1)

max_depth = 32



unique_width = sorted(results_summary_df['width'].unique())



plt.subplot(2,1,1);

for width in unique_width:

    const_width_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'depth']

    const_width_curve     = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_mean']

    const_width_curve_std = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_std']

    const_width_curve_std = np.minimum(const_width_curve_std, const_width_curve)

    

    selected_inds = const_width_curve_x <= max_depth

    const_width_curve_x   = const_width_curve_x[selected_inds]

    const_width_curve     = const_width_curve[selected_inds]

    const_width_curve_std = const_width_curve_std[selected_inds]

    

    plt.errorbar(const_width_curve_x, const_width_curve, label='%d_units' %(width))



plt.ylabel('MSE', fontsize=20)

plt.legend(fontsize=16, ncol=4)





plt.subplot(2,1,2);

for width in unique_width:

    const_width_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'depth']

    const_width_curve     = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_mean']

    const_width_curve_std = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_std']

    const_width_curve_std = np.minimum(const_width_curve_std, const_width_curve)

    

    selected_inds = const_width_curve_x <= max_depth

    const_width_curve_x   = const_width_curve_x[selected_inds]

    const_width_curve     = const_width_curve[selected_inds]

    const_width_curve_std = const_width_curve_std[selected_inds]



    plt.errorbar(const_width_curve_x, const_width_curve, label='%d_units' %(width))

    

plt.xlabel('Depth', fontsize=20)

plt.ylabel('MSE (log scale)', fontsize=20)

plt.yscale('log')
plt.figure(figsize=(15,10))

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.97, hspace=0.1)

max_depth = 32



unique_width = sorted(results_summary_df['width'].unique())



plt.subplot(2,1,1);

for width in unique_width:

    const_width_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'depth']

    const_width_curve     = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_mean']

    const_width_curve_std = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_std']

    const_width_curve_std = np.minimum(const_width_curve_std, const_width_curve)

    

    selected_inds = const_width_curve_x <= max_depth

    const_width_curve_x   = const_width_curve_x[selected_inds]

    const_width_curve     = const_width_curve[selected_inds]

    const_width_curve_std = const_width_curve_std[selected_inds]

    

    plt.errorbar(const_width_curve_x, const_width_curve, yerr=const_width_curve_std, label='%d_units' %(width))



plt.ylabel('MSE', fontsize=20)

plt.legend(fontsize=16, ncol=4)

plt.xscale('log')



plt.subplot(2,1,2);

for width in unique_width:

    const_width_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'depth']

    const_width_curve     = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_mean']

    const_width_curve_std = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_std']

    const_width_curve_std = np.minimum(const_width_curve_std, const_width_curve)

    

    selected_inds = const_width_curve_x <= max_depth

    const_width_curve_x   = const_width_curve_x[selected_inds]

    const_width_curve     = const_width_curve[selected_inds]

    const_width_curve_std = const_width_curve_std[selected_inds]



    plt.errorbar(const_width_curve_x, const_width_curve, yerr=const_width_curve_std, label='%d_units' %(width))

    

plt.xlabel('Depth (log scale)', fontsize=20)

plt.ylabel('MSE (log scale)', fontsize=20)

plt.xscale('log')

plt.yscale('log')
plt.figure(figsize=(15,10))

plt.subplots_adjust(left=0.08, right=0.97, bottom=0.05, top=0.97, hspace=0.1)

max_depth = 16



unique_width = sorted(results_summary_df['width'].unique())



plt.subplot(2,1,1);

for width in unique_width:

    const_width_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'depth']

    const_width_curve     = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_mean']

    const_width_curve_std = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_std']

    const_width_curve_std = np.minimum(const_width_curve_std, const_width_curve)

    

    selected_inds = const_width_curve_x <= max_depth

    const_width_curve_x   = const_width_curve_x[selected_inds]

    const_width_curve     = const_width_curve[selected_inds]

    const_width_curve_std = const_width_curve_std[selected_inds]

    

    plt.errorbar(const_width_curve_x, const_width_curve, label='%d_units' %(width))



plt.ylabel('MSE', fontsize=20)

plt.legend(fontsize=16, ncol=4)

plt.xscale('log')



plt.subplot(2,1,2);

for width in unique_width:

    const_width_curve_x   = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'depth']

    const_width_curve     = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_mean']

    const_width_curve_std = results_summary_df.loc[results_summary_df.loc[:,'width'] == width, 'final_MSE_std']

    const_width_curve_std = np.minimum(const_width_curve_std, const_width_curve)

    

    selected_inds = const_width_curve_x <= max_depth

    const_width_curve_x   = const_width_curve_x[selected_inds]

    const_width_curve     = const_width_curve[selected_inds]

    const_width_curve_std = const_width_curve_std[selected_inds]



    plt.errorbar(const_width_curve_x, const_width_curve, label='%d_units' %(width))

    

plt.xlabel('Depth (log scale)', fontsize=20)

plt.ylabel('MSE (log scale)', fontsize=20)

plt.xscale('log')

plt.yscale('log')