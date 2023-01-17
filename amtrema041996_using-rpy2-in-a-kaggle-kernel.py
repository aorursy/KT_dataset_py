!pip install rpy2
import subprocess

subprocess.run('conda install -c conda-forge r-base', shell=True)
!pip install rpy2
import rpy2 