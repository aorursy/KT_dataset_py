from glob import glob

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib import colors

import json
data_path = '/kaggle/input/abstraction-and-reasoning-challenge/'

train_set_json_files = glob(f'{data_path}training/*.json')

train_set_json_files.sort()
def show_task(task_file_name, grid=True):

    """

    Plots train and test pairs of a specified task,

    using same color scheme as the ARC app

    """

    def show_pairs(pairs, sublot):

        for i, pair in enumerate(pairs):

            for j, key in enumerate(('input', 'output')):

                width, height = len(pair[key][0]), len(pair[key])

                axs = plt.Subplot(fig, subplot[i*2 + j])

                axs.imshow(pair[key], cmap=cmap, norm=norm,

                           extent=(0, width, 0, height))

                axs.set_title(f'{key}, {len(pair[key])}X{len(pair[key][0])}')

                if grid:

                    axs.set_xticks(range(0, width))

                    axs.set_yticks(range(0, height))

                    axs.set_xticklabels([])

                    axs.set_yticklabels([])

                    axs.tick_params(length=0)

                    axs.grid(True)

                    for axis in ['top','bottom','left','right']:

                        axs.spines[axis].set_linewidth(0)

                else:

                    axs.axis('off')

                fig.add_subplot(axs)



    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    

    with open(task_file_name) as f:

        task = json.load(f)

    

    train_pairs = task['train']

    test_pairs = task['test']

    n_rows = max(len(train_pairs), len(test_pairs))

    

    fig = plt.figure(figsize=(12, 3 * n_rows))

    outer = gridspec.GridSpec(1, 2, wspace=0.3)

    for i, pairs in enumerate((train_pairs, test_pairs)):

        subplot = gridspec.GridSpecFromSubplotSpec(n_rows, 2,

                      subplot_spec=outer[i], wspace=0.1, hspace=0.15)

        show_pairs(pairs, subplot)

    fig.show()
print(f'File {train_set_json_files[0]}')

show_task(train_set_json_files[0])
print(f'File {train_set_json_files[1]}')

show_task(train_set_json_files[1])
print(f'File {train_set_json_files[2]}')

show_task(train_set_json_files[2])
print(f'File {train_set_json_files[3]}')

show_task(train_set_json_files[3])
print(f'File {train_set_json_files[4]}')

show_task(train_set_json_files[4])
print(f'File {train_set_json_files[5]}')

show_task(train_set_json_files[5])
print(f'File {train_set_json_files[6]}')

show_task(train_set_json_files[6])
print(f'File {train_set_json_files[7]}')

show_task(train_set_json_files[7])
print(f'File {train_set_json_files[8]}')

show_task(train_set_json_files[8])
print(f'File {train_set_json_files[9]}')

show_task(train_set_json_files[9])