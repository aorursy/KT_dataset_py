#GPU count and name

!nvidia-smi -L
#use this command to see GPU activity while doing Deep Learning tasks, for this command 'nvidia-smi' and for above one to work, go to 'Runtime > change runtime type > Hardware Accelerator > GPU'

!nvidia-smi
!lscpu |grep 'Model name'
#no.of sockets i.e available slots for physical processors

!lscpu | grep 'Socket(s):'
#no.of cores each processor is having

!lscpu | grep 'Core(s) per socket'
#no.of threads each core is having

!lscpu | grep 'Thread(s) per core'
!lscpu | grep 'L3 cache'
!lscpu | grep MHz
!cat /proc/meminfo | grep 'MemAvailable'
!df -h / | awk '{print $4}'