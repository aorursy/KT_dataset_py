# Download the logging script

!wget https://raw.githubusercontent.com/scottclowe/cpu-gpu-utilisation-logging-python/master/log_gpu_cpu_stats.py
# Start the logger running in a background process. It will keep running until you tell it to stop.

# We will save the CPU and GPU utilisation stats to a CSV file every 0.2 seconds.

import subprocess

!rm -f log_compute.csv

logger_fname = 'log_compute.csv'

logger_pid = subprocess.Popen(

    ['python', 'log_gpu_cpu_stats.py',

     logger_fname,

     '--loop',  '0.2',  # Interval between measurements, in seconds (optional, default=1)

    ])

print('Started logging compute utilisation')
import os

import time



import numpy as np

import torch
t_per_exp = 2

t_sleep = 2



tgen_list = [

    ('ones_float', lambda s, d: torch.ones(s, dtype=torch.float, device=d)),

    #('rand_float', lambda s, d: torch.rand(s, dtype=torch.float, device=d)),

]

op_list = [

    ('ADD', lambda x, y: x + y),

    ('MUL', lambda x, y: x * y),

    ('MATMUL', lambda x, y: torch.matmul(x, y)),

]



# Do some compute on CPU and GPU

for regen_tensors in [False, True]:

    for tgen_name, tgen_fn in tgen_list:

        print("\n{} tensors ({})...".format(tgen_name, 'regenerate inputs' if regen_tensors else 'static inputs'))

        for op_name, op_fun in op_list:

            print("\n  {} operations...".format(op_name))

            time.sleep(5)

            for device in ['cpu', 'cuda']:

                for shp in [(8, 8), (64, 64), (512, 512), (4096, 4096)]: #[(10, 10), (100, 100), (1000, 1000), (10000, 10000)]:

                    print(

                        '    Beginning {:<12} {} {:<6} operations on {:<4} for {}s ({})'

                        ''.format(str(shp), tgen_name, op_name, device.upper(), t_per_exp,

                                 'regenerate inputs' if regen_tensors else 'static inputs')

                    )

                    i = 0

                    t_start = time.time()

                    t_gen = 0

                    t_op = 0

                    if not regen_tensors:

                        x = tgen_fn(shp, device)

                        y = tgen_fn(shp, device)

                    while time.time() - t_start < t_per_exp:

                        t0 = time.time()

                        if regen_tensors:

                            x = tgen_fn(shp, device)

                            y = tgen_fn(shp, device)

                        t1 = time.time()

                        t_gen += t1 - t0

                        z = op_fun(x, y)

                        t_op += time.time() - t1

                        i += 1

                    dur = time.time() - t_start

                    print(

                        '      Completed {:>7} iterations in {:.1f}s ({:10.3f}it/s);'

                        ' {:.1f}% was generating tensors'

                        ''.format(i, dur, i / dur, 100 * t_gen / (t_gen + t_op + 0.001))

                    )

                    time.sleep(t_sleep)
!head log_compute.csv
!tail log_compute.csv
time.sleep(60)

!tail log_compute.csv
import pandas as pd

from matplotlib import pyplot as plt
logger_df = pd.read_csv(logger_fname)
logger_df
t = pd.to_datetime(logger_df['Timestamp (s)'], unit='s')

cols = [col for col in logger_df.columns

        if 'time' not in col.lower() and 'temp' not in col.lower()]

plt.figure(figsize=(15, 9))

plt.plot(t, logger_df[cols])

plt.legend(cols)

plt.xlabel('Time')

plt.ylabel('Utilisation (%)')

plt.show()
for col in logger_df.columns:

    if 'time' in col.lower(): continue

    plt.figure(figsize=(15, 9))

    plt.plot(t, logger_df[col])

    plt.xlabel('Time')

    plt.ylabel(col)

    plt.show()
# End the background process logging the CPU and GPU utilisation.

logger_pid.terminate()

print('Terminated the compute utilisation logger background process')