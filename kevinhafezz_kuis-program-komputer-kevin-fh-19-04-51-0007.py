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
i = 0

nama = [""] * (10)

uts = [0] * (10)

uas = [0] * (10)

nilaiakhir = [0] * (10)



for i in range(1, 10 + 1, 1):

    print("masukkan uts=")

    uts[1] = float(input())

    print("masukkan uas=")

    uas[i] = float(input())

    nilaiakhir[i] = 4.0 * uts[i] + 6.0 * uas[i]

    print("Hasil akhir=")

    print(nilaiakhir[i])

    if nilaiakhir[i] >= 70:

        print(" 7 Mahasiswa lulus  ")

        print("Kevin,fery,niko,intan,indra,wahyu,vero LULUS")

    else:

        if nilaiakhir[i] < 70:

            print(" 3 Mahasiswa tidak lulus ")

            print("MAAF agus,arnold,hagus TIDAK LULUS")

            print(" Mahasiswa yang lulus : Kevin,fery,niko,intan,indra,wahyu,vero")
