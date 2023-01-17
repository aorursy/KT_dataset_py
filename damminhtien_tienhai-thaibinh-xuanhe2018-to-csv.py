# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/"))



# Any results you write to the current directory are saved as output.
from skimage import io

import numpy as np

import csv

import pandas as pd
imvh1 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180204.tif").reshape(5002044)

imvh2 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180216.tif").reshape(5002044)

imvh3 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180228.tif").reshape(5002044)

imvh4 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180312.tif").reshape(5002044)

imvh5 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180324.tif").reshape(5002044)

imvh6 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180405.tif").reshape(5002044)

imvh7 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180417.tif").reshape(5002044)

imvh8 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180429.tif").reshape(5002044)

imvh9 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180511.tif").reshape(5002044)

imvh10 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180514.tif").reshape(5002044)

imvh11 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180523.tif").reshape(5002044)

imvh12 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180604.tif").reshape(5002044)

imvh13 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180616.tif").reshape(5002044)

imvh14 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180628.tif").reshape(5002044)

imvv1 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180204.tif").reshape(5002044)

imvv2 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180216.tif").reshape(5002044)

imvv3 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180228.tif").reshape(5002044)

imvv4 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180312.tif").reshape(5002044)

imvv5 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180324.tif").reshape(5002044)

imvv6 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180405.tif").reshape(5002044)

imvv7 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180417.tif").reshape(5002044)

imvv8 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180429.tif").reshape(5002044)

imvv9 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180511.tif").reshape(5002044)

imvv10 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180514.tif").reshape(5002044)

imvv11 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vh_20180523.tif").reshape(5002044)

imvv12 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180604.tif").reshape(5002044)

imvv13 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180616.tif").reshape(5002044)

imvv14 = io.imread("../input/tienhai_thaibinh_xuanhe_2018/TienHai_ThaiBinh_XuanHe_2018/vv_20180628.tif").reshape(5002044)
with open('tienhai_thaibinh1.csv', mode='w+') as file1:

    writer = csv.writer(file1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["vh_20180204","vh_20180216","vh_20180228","vh_20180312","vh_20180324","vh_20180405","vh_20180417"

                    ,"vh_20180429","vh_20180511","vh_20180514","vh_20180523","vh_20180604","vh_20180616","vh_20180628"

                    ,"vv_20180204","vv_20180216","vv_20180228","vv_20180312","vv_20180324","vv_20180405","vv_20180417"

                     ,"vv_20180429","vv_20180511","vv_20180514","vh_20180523","vv_20180604","vv_20180616","vv_20180628"])

    for i in range(0,1000000):

        writer.writerow([imvh1[i],imvh2[i],imvh3[i],imvh4[i],imvh5[i],imvh6[i],imvh7[i],imvh8[i],imvh9[i],

                        imvh10[i],imvh11[i],imvh12[i],imvh13[i],imvh14[i],imvv1[i],imvv2[i],imvv3[i],imvv4[i],

                        imvv5[i],imvv6[i],imvv7[i],imvv8[i],imvv9[i],imvv10[i],imvv11[i],imvv12[i],imvv13[i],imvv14[i],])           

file1.close()
df1 = pd.read_csv('tienhai_thaibinh1.csv')

df1
df1.describe()
with open('tienhai_thaibinh2.csv', mode='w+') as file2:

    writer = csv.writer(file2, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["vh_20180204","vh_20180216","vh_20180228","vh_20180312","vh_20180324","vh_20180405","vh_20180417"

                    ,"vh_20180429","vh_20180511","vh_20180514","vh_20180523","vh_20180604","vh_20180616","vh_20180628"

                    ,"vv_20180204","vv_20180216","vv_20180228","vv_20180312","vv_20180324","vv_20180405","vv_20180417"

                     ,"vv_20180429","vv_20180511","vv_20180514","vh_20180523","vv_20180604","vv_20180616","vv_20180628"])

    for i in range(1000000,2000000):

        writer.writerow([imvh1[i],imvh2[i],imvh3[i],imvh4[i],imvh5[i],imvh6[i],imvh7[i],imvh8[i],imvh9[i],

                        imvh10[i],imvh11[i],imvh12[i],imvh13[i],imvh14[i],imvv1[i],imvv2[i],imvv3[i],imvv4[i],

                        imvv5[i],imvv6[i],imvv7[i],imvv8[i],imvv9[i],imvv10[i],imvv11[i],imvv12[i],imvv13[i],imvv14[i],])           

file2.close()
df2 = pd.read_csv('tienhai_thaibinh2.csv')

df2
df2.describe()
with open('tienhai_thaibinh3.csv', mode='w+') as file3:

    writer = csv.writer(file3, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["vh_20180204","vh_20180216","vh_20180228","vh_20180312","vh_20180324","vh_20180405","vh_20180417"

                    ,"vh_20180429","vh_20180511","vh_20180514","vh_20180523","vh_20180604","vh_20180616","vh_20180628"

                    ,"vv_20180204","vv_20180216","vv_20180228","vv_20180312","vv_20180324","vv_20180405","vv_20180417"

                     ,"vv_20180429","vv_20180511","vv_20180514","vh_20180523","vv_20180604","vv_20180616","vv_20180628"])

    for i in range(2000000,3000000):

        writer.writerow([imvh1[i],imvh2[i],imvh3[i],imvh4[i],imvh5[i],imvh6[i],imvh7[i],imvh8[i],imvh9[i],

                        imvh10[i],imvh11[i],imvh12[i],imvh13[i],imvh14[i],imvv1[i],imvv2[i],imvv3[i],imvv4[i],

                        imvv5[i],imvv6[i],imvv7[i],imvv8[i],imvv9[i],imvv10[i],imvv11[i],imvv12[i],imvv13[i],imvv14[i],])           

file3.close()
df3 = pd.read_csv('tienhai_thaibinh3.csv')

df3
df3.describe()
with open('tienhai_thaibinh4.csv', mode='w+') as file4:

    writer = csv.writer(file4, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["vh_20180204","vh_20180216","vh_20180228","vh_20180312","vh_20180324","vh_20180405","vh_20180417"

                    ,"vh_20180429","vh_20180511","vh_20180514","vh_20180523","vh_20180604","vh_20180616","vh_20180628"

                    ,"vv_20180204","vv_20180216","vv_20180228","vv_20180312","vv_20180324","vv_20180405","vv_20180417"

                     ,"vv_20180429","vv_20180511","vv_20180514","vh_20180523","vv_20180604","vv_20180616","vv_20180628"])

    for i in range(3000000,4000000):

        writer.writerow([imvh1[i],imvh2[i],imvh3[i],imvh4[i],imvh5[i],imvh6[i],imvh7[i],imvh8[i],imvh9[i],

                        imvh10[i],imvh11[i],imvh12[i],imvh13[i],imvh14[i],imvv1[i],imvv2[i],imvv3[i],imvv4[i],

                        imvv5[i],imvv6[i],imvv7[i],imvv8[i],imvv9[i],imvv10[i],imvv11[i],imvv12[i],imvv13[i],imvv14[i],])           

file4.close()
df4 = pd.read_csv('tienhai_thaibinh4.csv')

df4
df4.describe()
with open('tienhai_thaibinh5.csv', mode='w+') as file5:

    writer = csv.writer(file5, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["vh_20180204","vh_20180216","vh_20180228","vh_20180312","vh_20180324","vh_20180405","vh_20180417"

                    ,"vh_20180429","vh_20180511","vh_20180514","vh_20180523","vh_20180604","vh_20180616","vh_20180628"

                    ,"vv_20180204","vv_20180216","vv_20180228","vv_20180312","vv_20180324","vv_20180405","vv_20180417"

                     ,"vv_20180429","vv_20180511","vv_20180514","vh_20180523","vv_20180604","vv_20180616","vv_20180628"])

    for i in range(4000000,5002044):

        writer.writerow([imvh1[i],imvh2[i],imvh3[i],imvh4[i],imvh5[i],imvh6[i],imvh7[i],imvh8[i],imvh9[i],

                        imvh10[i],imvh11[i],imvh12[i],imvh13[i],imvh14[i],imvv1[i],imvv2[i],imvv3[i],imvv4[i],

                        imvv5[i],imvv6[i],imvv7[i],imvv8[i],imvv9[i],imvv10[i],imvv11[i],imvv12[i],imvv13[i],imvv14[i],])           

file5.close()
df5 = pd.read_csv('tienhai_thaibinh5.csv')

df5
df5.describe()