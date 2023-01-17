!nvidia-smi
!cat /usr/local/cuda/version.txt
!cat /proc/cpuinfo
import multiprocessing

multiprocessing.cpu_count()
!cat /proc/meminfo
!df -h 