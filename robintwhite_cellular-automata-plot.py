import numpy as np

import matplotlib.pyplot as plt

from IPython.display import clear_output

import pandas as pd

from PIL import Image

import time
size = 25
# from http://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/

def life_step_1(X):

    """Game of life step using generator expressions"""

    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)

                     for i in (-1, 0, 1) for j in (-1, 0, 1)

                     if (i != 0 or j != 0))

    return (nbrs_count == 3) | (X & (nbrs_count == 2))
def draw_image(img):

    img = Image.fromarray(np.uint8(img) * 255)

    return img
def plot_animate(arr):

    clear_output(wait=True)

    plt.imshow(draw_image(arr))

    plt.show()
arr = np.random.choice([0,1], (size, size), p=[0.5, 0.5])

for x in range(50):

    arr = life_step_1(arr)

    plot_animate(arr)

    if sum(arr.ravel()) == 0:

        print(x)

        break
arr = np.random.choice([0,1], (size, size), p=[0.85, 0.15])

#warm-up, based on Kaggle desccription

for i in range(5):

    arr = life_step_1(arr)

# 1 interation

new_arr = life_step_1(arr)
fig, ax = plt.subplots(1,2, figsize=(12,12))

ax[0].imshow(draw_image(arr))

ax[0].set_title('start')

ax[1].imshow(draw_image(new_arr))

ax[1].set_title('stop after 1 iteration')

plt.show()
# WORKING WITH THE DATA

# parts based on https://www.kaggle.com/candaceng/understanding-the-problem-and-eda
train_df = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/train.csv')

test_df = pd.read_csv('/kaggle/input/conways-reverse-game-of-life-2020/test.csv')

print(train_df.shape)

print(test_df.shape)
train_df.head()
train_df.groupby(['delta']).size()
# SELECTING ONE SAMPLE TO VISUALIZE
train_sample = train_df.sample()
sample_start = train_sample.loc[:, train_sample.columns.str.startswith('start')]

sample_stop = train_sample.loc[:, train_sample.columns.str.startswith('stop')]
start_arr = np.asarray(sample_start).reshape(25, 25)

stop_arr = np.asarray(sample_stop).reshape(25, 25)

# time step 

time_step = train_sample['delta'].values[0]

print(time_step)
def plot_comp(arr1, arr2, step):

    fig, ax = plt.subplots(1,2, figsize=(12,12))

    ax[0].imshow(draw_image(arr1))

    ax[0].set_title('start')

    ax[0].axis('off')

    ax[1].imshow(draw_image(arr2))

    ax[1].set_title(f'stop after: {step}')

    ax[1].axis('off')

    plt.show()
plot_comp(start_arr, stop_arr, time_step)
updated_arr = np.copy(start_arr)

steps = []

steps.append(updated_arr)

for x in range(time_step):

    updated_arr = life_step_1(updated_arr)

    steps.append(updated_arr)

    plot_animate(updated_arr)

    time.sleep(0.2)
fig, m_axs = plt.subplots(1, len(steps), figsize = (10,20))

for c_ax, c_row in zip(m_axs.flatten(), steps):

    c_ax.imshow(c_row, cmap='gray')

    c_ax.axis('off')
# CREATING SINGLE STEP DATASET
single_step_df = pd.DataFrame(columns=[train_df.columns])
start_key = ['start_' + str(i) for i in range(625)]

stop_key = ['stop_' + str(i) for i in range(625)]
arr = np.random.choice([0,1], (size, size))

#warm-up, based on Kaggle desccription

for i in range(5):

    arr = life_step_1(arr)

# 1 interation

update_arr = life_step_1(arr)
new_row = np.concatenate((np.array([0]).reshape(1,-1), np.array([1]).reshape(1,-1), arr.reshape(-1, 625).round(0).astype('uint8'), update_arr.reshape(-1, 625).round(0).astype('uint8')), axis=1)
single_step_df = single_step_df.append(pd.DataFrame(new_row, columns=list(single_step_df)), ignore_index=True)
single_step_df.head()
plot_comp(np.asarray(single_step_df.loc[0, train_sample.columns.str.startswith('start')]).reshape(25,25),

          np.asarray(single_step_df.loc[0, train_sample.columns.str.startswith('stop')]).reshape(25,25), 1)
r = list(np.arange(0,1,0.01))

p = np.around(list(zip(r,np.subtract(1.0,r))), 2)

ind = list(np.arange(len(p)))
for i in range(10):

    arr = np.random.choice([0,1], (size, size), p = p[np.random.choice(ind)])

    #warm-up, based on Kaggle desccription

    for i in range(5):

        arr = life_step_1(arr)

    # 1 interation

    update_arr = life_step_1(arr)



    new_row = np.concatenate((np.array([len(single_step_df)]).reshape(1,-1), np.array([1]).reshape(1,-1), arr.reshape(-1, 625).round(0).astype('uint8'), update_arr.reshape(-1, 625).round(0).astype('uint8')), axis=1)

    single_step_df = single_step_df.append(pd.DataFrame(new_row, columns=list(single_step_df)), ignore_index=True)
single_step_df.head()
fig, m_axs = plt.subplots(5, 2, figsize=(12,12))

for i, (c_ax, c_row) in enumerate(zip(m_axs.flatten(), single_step_df.sample(5).iterrows())):

    

    m_axs[i,0].imshow(np.asarray(c_row[1][627:]).reshape(25,25).astype('uint8'))

    m_axs[i,0].set_title(c_row[0])

    m_axs[i,0].axis('off')

    

    m_axs[i,1].imshow(np.asarray(c_row[1][2:627]).reshape(25,25).astype('uint8'))

    m_axs[i,1].set_title(c_row[0])

    m_axs[i,1].axis('off')