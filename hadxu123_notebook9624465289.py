# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!wget https://www.tbi.univie.ac.at/RNA/download/ubuntu/ubuntu_18_04/viennarna_2.4.15-1_amd64.deb
!apt-get install ./viennarna_2.4.15-1_amd64.deb -y
!git clone https://github.com/DasLab/arnie

!/opt/conda/bin/python3.7 -m pip install --upgrade pip
!git clone https://www.github.com/DasLab/draw_rna draw_rna_pkg
!cd draw_rna_pkg && python setup.py install

!yes '' | cpan -i Graph
!git clone https://github.com/hendrixlab/bpRNA
import os
import sys

!echo "vienna_2: /usr/bin" > arnie.conf
!echo "TMP: /kaggle/working/tmp" >> arnie.conf
!mkdir -p /kaggle/working/tmp
os.environ["ARNIEFILE"] = f"/kaggle/working/arnie.conf"
sys.path.append('/kaggle/working/draw_rna_pkg/')
sys.path.append('/kaggle/working/draw_rna_pkg/ipynb/')
pkg = 'vienna_2'
import os
import sys

!echo "vienna_2: /usr/bin" > arnie.conf
!echo "TMP: /kaggle/working/tmp" >> arnie.conf
!mkdir -p /kaggle/working/tmp
os.environ["ARNIEFILE"] = f"/kaggle/working/arnie.conf"
sys.path.append('/kaggle/working/draw_rna_pkg/')
sys.path.append('/kaggle/working/draw_rna_pkg/ipynb/')
pkg = 'vienna_2'
import numpy as np
import pandas as pd
from multiprocessing import Pool
from arnie.pfunc import pfunc
from arnie.mea.mea import MEA
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils
from tqdm.notebook import tqdm as tqdm

n_candidates = 2
# turn off for all data
debug = True
bp_matrix = bpps('AATATCATCTCTCAGAGCAGAATGCATGAATGATGCAGGAGGGTTATAGGATGAATTCACTGA', package=pkg)
bp_matrix.shape
df = pd.read_csv('../input/k562-ss/k562-filter-210-10_28mer_SS.csv')
os.mkdir('/bpps/')
df.head()
for index, s in enumerate(df.seq):
    bp_matrix = bpps('AATATCATCTCTCAGAGCAGAATGCATGAATGATGCAGGAGGGTTATAGGATGAATTCACTGA', package=pkg)
    np.save(f'bpps-{index}', bp_matrix)
