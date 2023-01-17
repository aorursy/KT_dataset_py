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
# ดึงข้อมูลจากไฟล์มาเก็บไว้ในตัวแปร data
Ydata = pd.read_csv("../input/youtube-new/USvideos.csv")

# ก็อปปี้ข้อมูลไว้
original_data = Ydata.copy()
Ydata.head()
# ตรวจสอบรูปแบบของข้อมูล
Ydata.info()
Ydata.apply(lambda x: sum(x.isnull()))

# ดูข้อมูล column 'trending_date'
Ydata['trending_date'].head()
column_list = ['views', 'likes', 'dislikes', 'comment_count']
corr_matrix = Ydata[column_list].corr()
corr_matrix
plt.figure(figsize = (16,8))

#Let's verify the correlation of each value
ax = sb.heatmap(Ydata[['views', 'likes', 'dislikes', 'comment_count']].corr(), \
            annot=True, annot_kws={"size": 20}, cmap=cm.coolwarm, linewidths=0.5, linecolor='black')
plt.yticks(rotation=30, fontsize=20) 
plt.xticks(rotation=30, fontsize=20) 
plt.title("\nCorrelation between views, likes, dislikes & comments\n", fontsize=25)
plt.show()
sns.countplot(x="likes", data=Ydata);
colors = ["#FF6600", "#FFCCCC"]
labels ="likes", "dislikes"

plt.suptitle('Information on data_split', fontsize=20)

data["Ydata"].value_counts().plot.pie(autopct='%1.2f%%',  shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=70)
