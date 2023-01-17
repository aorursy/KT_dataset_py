# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import warnings
import warnings
#ignore warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/column_2C_weka.csv")
print(plt.style.available)
plt.style.use("ggplot")
data.head()
data.info()
data.describe()
color_list=["red" if i=="Abnormal" else "green" for i in data.loc[:,"class"]]
pd.plotting.scatter_matrix(data.loc[:,data.columns!="class"],
                            c=color_list,
                            figsize=[15,15],
                            diagonal="hist",
                            alpha=0.5,
                            s=200,
                            marker="*",
                            edgecolor="black")
plt.show()
sns.countplot(x="class",data=data)
data.loc[:,"class"].value_counts()
data1=data[data["class"]=="Abnormal"]
x = np.array(data1.loc[:,"pelvic_incidence"]).reshape(-1,1)
y= np.array(data1.loc[:,"sacral_slope"]).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
predict_space=np.linspace(min(x),max(x)).reshape(-1,1)
reg.fit(x,y)
predicted=reg.predict(predict_space)
print("R^2 score: ",reg.score(x,y))
plt.plot(predict_space, predicted,color="black",linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()
