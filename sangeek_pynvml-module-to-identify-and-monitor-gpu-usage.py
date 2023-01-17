!conda install pynvml -c conda-forge --yes
import pynvml

pynvml.nvmlInit()
print(f"NVIDIA Driver version - {pynvml.nvmlSystemGetDriverVersion().decode()}")
deviceCount = pynvml.nvmlDeviceGetCount()

for i in range(deviceCount):

    handle = pynvml.nvmlDeviceGetHandleByIndex(i)

    print(f"Device {i} {pynvml.nvmlDeviceGetName(handle).decode()}")
#Does not work!

!nvidia-smi
#But wait!

!which nvidia-smi
#Oh ho .. lo and behold there is another nvidia-smi in /opt/bin 

!cd /; find . -name nvidia-smi
#And that one works ;-)

!/opt/bin/nvidia-smi
from pynvml.smi import nvidia_smi

print(f"Free GPU memory - {nvidia_smi.getInstance().DeviceQuery('memory.total')}")

print(f"Used GPU memory - {nvidia_smi.getInstance().DeviceQuery('memory.used')}")
import cupy as cp

X = cp.linspace(0,100, num=50)

y = 3*X + 4*cp.random.randn(50)
%matplotlib inline

#Plot the values of the array to get an view of the distribution of values

import matplotlib.pyplot as plt

#Get the numpy representation of CuPy arrays X and y for plotting

plt.plot(cp.asnumpy(X),cp.asnumpy(y),'o')
from pynvml.smi import nvidia_smi

print(f"Free GPU memory - {nvidia_smi.getInstance().DeviceQuery('memory.total')}")

print(f"Used GPU memory - {nvidia_smi.getInstance().DeviceQuery('memory.used')}")
#Verifying here using nvidia-smi that it is same as nvidia-smi reports

!/opt/bin/nvidia-smi