import time
import numpy
%load_ext memory_profiler
%load_ext line_profiler
import memory_profiler
import line_profiler
%time
%timeit int((numpy.sqrt(2)**2) - 2)
%prun
%lprun
%memit
%mprun
int((numpy.sqrt(2)**2) - 2)
