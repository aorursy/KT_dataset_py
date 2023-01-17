# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
'''

You can get my program in the output section of this kernel as the program took too long than 6 hours on the kernel, I can not run it here. So you can
run it locally on your computer. You need to just submit the output file after uploading it to any cloud like the code shown below just replace my url
to your submission file download url, In my case, my URL is for downloading this program which gives me a score of 0.0233.

'''

from urllib.request import urlretrieve

url = "https://tlrpnq.bl.files.1drv.com/y4mEEd6L9fgtckxdWIHm1FGQUOM3vEONm5KHOGP23w6TIsDkw4iNRHDKb2ElBLXqhbteJRLpfX_S_bZIxl4pJgrUPqq7VALF1_mtAMjnkcUa80NKsf6kVVtxaPoc1uWIp_Ko8c0kBBxpJNrQ3MzoRcj6IC5UdKgHncOsSbBDsGvTLK3CJb2mPzjVUmj5t2VydeO/Program.py?download&psid=1"

urlretrieve(url, 'program.py')
'''

Submission File Download Url (I suggest you to not to directly copy this kernel and run the program cuz program will give some different result all the time so it may give better in your case.)

'''

url = "https://tlrpnq.bl.files.1drv.com/y4mKLzjKXuHrRkFjdHhR0P70ybcSGUJsP-DRHWDCLggxcvxw4zuljM3zRVmC0rTBm4H8S-iwKPNPq-4IC6LLV7mYM0HTtEBiTtoK6KFQ4BGTdcB-32VXvdpX2-LULnYU8JSwdOJ4cUcG4rM19xw3c1_4prObi_r3dyHh9jmuKSPUWFKiIWEqxUbYWJj40-wFNCy/submission_adjusted_456.csv?download&psid=1"

urlretrieve(url, 'submission.csv')

'''

Plz comment down below your ideas and If you are having any problem to run the program I will be there to help you.

Thank You

'''
