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
# List /kaggle/input để xem các files và folders dữ liệu được liên kết với Notebook

!ls /kaggle/input
# Import data (giống như Jupyter Notebook, bạn có thể sử dụng tab để auto-complete đường dẫn)

data = pd.read_csv("/kaggle/input/dataisbeautiful/r_dataisbeautiful_posts.csv")

data.head(3)
# Tạo ổ nhớ tạm

os.makedirs("/kaggle/temp",exist_ok=True)

# Lưu data vào ổ nhớ tạm, file này sẽ không được lưu xuất hiện sau khi bạn lưu và commit notebook

data[:3].to_csv("/kaggle/temp/temp.csv",index=False)
# Đọc data đã lưu từ ổ nhớ tạm

pd.read_csv("/kaggle/temp/temp.csv")
# Lưu data để sử dụng cho những version hay notebook sau, file `submission.csv` sẽ lưu giữ sau khi Notebook được commit

data[:3].to_csv("/kaggle/working/submission.csv",index=False)