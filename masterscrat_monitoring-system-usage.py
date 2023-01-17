import psutil

import numpy as np

import matplotlib.pyplot as plt

import os

import time

import multiprocessing

from IPython.display import clear_output

from collections import deque
class SystemMonitorProcess:

    def __init__(self, start_timestamp, update_interval=0.1):

        self.update_interval = update_interval

        self.cpu_nums = psutil.cpu_count()

        self.max_mem = psutil.virtual_memory().total

        self.sysCpuLogs = deque()

        self.sysMemLogs = deque()

        self.timeLogs = deque()

        self.start_time = start_timestamp



    def get_system_info(self):

        cpu_percent = psutil.cpu_percent(interval=0.0, percpu=False)

        mem_percent = float(psutil.virtual_memory().used) / self.max_mem * 100

        return cpu_percent, mem_percent

        

    def monitor(self):

        while True:

            time.sleep(self.update_interval)

            sCpu, sMem = self.get_system_info()  

            self.sysCpuLogs.append(sCpu)

            self.sysMemLogs.append(sMem)

            self.timeLogs.append(time.time() - self.start_time)

            logs.update({

                'sysCpuLogs': self.sysCpuLogs,

                'sysMemLogs': self.sysMemLogs,

                'time': self.timeLogs

            })
class SystemMonitor:

    def __init__(self, update_interval=0.1):

        self.graph = None

        self.update_interval = update_interval

        self.start_timestamp = time.time()

        self.msgs = []

    

    def monitor(self):

        self.graph = SystemMonitorProcess(self.start_timestamp, self.update_interval)

        self.graph.monitor()

        

    def annotate(self, msg):

        self.msgs.append([time.time() - self.start_timestamp, msg])

        

    def plot(self):

        if not 'sysCpuLogs' in logs:

            print('No data yet.')

            return



        fig = plt.figure(figsize=(20,3))

        plt.ylabel('usage (%)')

        

        # FIXME display running time on primary X axis!

        ax = plt.axes()

        #plt.xlabel('running time (s)')

        

        ax2 = ax.twiny()

        ax2.plot(list(logs['time']), logs['sysCpuLogs'], label="cpu")

        ax2.plot(list(logs['time']), logs['sysMemLogs'], label="mem")

        ax2.set_xticks([msg[0] for msg in self.msgs])

        ax2.set_xticklabels([msg[1] for msg in self.msgs], rotation=90)



        ax2.legend(loc='best')

        plt.show()
# example usage:

sm = SystemMonitor(0.1) # polling frequency

logs = multiprocessing.Manager().dict()

smp = multiprocessing.Process(target=sm.monitor)

smp.start()



### YOUR ACTUAL CODE GOES HERE: ###

for i in range(5):

    sm.annotate('step '+str(i))

    clear_output()

    sm.plot()

    time.sleep(1.0)

### END OF YOUR CODE ###



print('Final plot:')

sm.plot() # shows plot

smp.terminate() # kills the system monitor