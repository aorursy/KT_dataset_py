print('# CPU')

!cat /proc/cpuinfo | egrep -m 1 "^model name"

!cat /proc/cpuinfo | egrep -m 1 "^cpu MHz"

!cat /proc/cpuinfo | egrep -m 1 "^cpu cores"



print('\n# OS')

!cat /etc/*-release



print('\n# Kernel')

!uname -a



print('\n# RAM')

!cat /proc/meminfo | egrep "^MemTotal"



print('\n# Nvidia driver')

!nvidia-smi



print('\n# Cuda compiler driver')

!nvcc --version