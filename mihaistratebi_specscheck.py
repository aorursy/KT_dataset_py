!nvidia-smi -L

!lscpu | grep 'Model name'

!lscpu | grep 'Core(s) per socket'

!lscpu | grep 'Thread(s) per core'

!lscpu | grep 'L3 cache'

!lscpu | grep 'MHz'

!cat /proc/meminfo | grep 'MemAvailable'