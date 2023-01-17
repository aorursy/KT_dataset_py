import numpy as np

import pandas as pd



import os

import json

from pathlib import Path



import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np

    

from pathlib import Path
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

evaluation_path = data_path / 'evaluation'

test_path = data_path / 'test'



training_tasks = sorted(os.listdir(training_path))

evaluation_tasks = sorted(os.listdir(evaluation_path))

test_tasks = sorted(os.listdir(test_path))
cmap = colors.ListedColormap(

    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)





def plot_task(task, swap=False):

    plt_index = (0,1) if not swap else (1,0)

    n = len(task["train"]) + len(task["test"])

    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_num = 0

    for i, t in enumerate(task["train"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[plt_index[0]][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[plt_index[0]][fig_num].set_title(f'Train-{i} in')

        axs[plt_index[0]][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[plt_index[0]][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[plt_index[1]][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[plt_index[1]][fig_num].set_title(f'Train-{i} out')

        axs[plt_index[1]][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[plt_index[1]][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    for i, t in enumerate(task["test"]):

        t_in, t_out = np.array(t["input"]), np.array(t["output"])

        axs[plt_index[0]][fig_num].imshow(t_in, cmap=cmap, norm=norm)

        axs[plt_index[0]][fig_num].set_title(f'Test-{i} in')

        axs[plt_index[0]][fig_num].set_yticks(list(range(t_in.shape[0])))

        axs[plt_index[0]][fig_num].set_xticks(list(range(t_in.shape[1])))

        axs[plt_index[1]][fig_num].imshow(t_out, cmap=cmap, norm=norm)

        axs[plt_index[1]][fig_num].set_title(f'Test-{i} out')

        axs[plt_index[1]][fig_num].set_yticks(list(range(t_out.shape[0])))

        axs[plt_index[1]][fig_num].set_xticks(list(range(t_out.shape[1])))

        fig_num += 1

    

    plt.tight_layout()

    plt.show()
def open_plot_task(task_name, swap=False):

    task_file = str(training_path / task_name)

    

    with open(task_file, 'r') as f:

        task = json.load(f)



    plot_task(task, swap=swap)
reversible = pd.DataFrame([["e509e548.json",True,True],["11852cab.json",False,False],["ecdecbb3.json",False,False],["1b60fb0c.json",True,True],["9d9215db.json",False,False],["6455b5f5.json",True,True],["dae9d2b5.json",False,False],["f9012d9b.json",False,False],["10fcaaa3.json",True,False],["39a8645d.json",False,False],["e76a88a6.json",True,False],["6c434453.json",True,False],["a1570a43.json",False,False],["d43fd935.json",True,False],["88a10436.json",False,False],["855e0971.json",False,False],["9aec4887.json",False,False],["e9614598.json",True,True],["98cf29f8.json",False,False],["f5b8619d.json",True,False],["48d8fb45.json",False,False],["f25ffba3.json",True,False],["8a004b2b.json",False,False],["7447852a.json",True,True],["9ecd008a.json",False,False],["d06dbe63.json",True,True],["fafffa47.json",False,False],["952a094c.json",True,False],["1fad071e.json",False,False],["746b3537.json",False,False],["d687bc17.json",False,False],["99b1bc43.json",False,False],["90f3ed37.json",True,True],["b7249182.json",True,False],["93b581b8.json",True,False],["4c5c2cf0.json",False,False],["d23f8c26.json",False,False],["7b7f7511.json",True,False],["6e19193c.json",True,False],["3eda0437.json",True,True],["beb8660c.json",False,False],["af902bf9.json",True,True],["dc1df850.json",True,True],["264363fd.json",False,False],["27a28665.json",False,False],["c59eb873.json",True,False],["9172f3a0.json",True,False],["5521c0d9.json",False,False],["0e206a2e.json",False,False],["3c9b0459.json",True,False]],

                         columns=["task_name", "reversible", "trivial"])

reversible.head()
reversible.groupby(["reversible", "trivial"]).reversible.count() / reversible.reversible.count()
open_plot_task(reversible.query("reversible==False").task_name.iloc[0], swap=True)
open_plot_task(reversible.query("reversible==True and trivial==False").task_name.iloc[0], swap=True)
open_plot_task(reversible.query("reversible==True and trivial==True").task_name.iloc[0], swap=True)