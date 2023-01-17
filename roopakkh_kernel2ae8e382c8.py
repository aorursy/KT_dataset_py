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
perf = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
perf.describe()
perf.columns = ['gender', 'ethnicity', 'parents_education', 'lunch', 'test_prep_course', 'math_score', 'reading_score', 'writing_score']
import matplotlib.pyplot as plt

score_plt,score_ax = plt.subplots(nrows=1,ncols=3)
score_ax[0].set_title("Math scores")
score_ax[0].violinplot(perf["math_score"])
score_ax[1].set_title("Reading scores")
score_ax[1].violinplot(perf["reading_score"])
score_ax[2].set_title("Writing scores")
score_ax[2].violinplot(perf["writing_score"])
grp = pd.DataFrame(perf["math_score"].groupby(perf["parents_education"]).mean())
grp = grp.reset_index() #This converts the index into a separate column
plt.bar(grp["parents_education"],grp["math_score"])
alltests = pd.DataFrame(perf[["math_score"]+["reading_score"]+["writing_score"]].groupby(perf["parents_education"]).mean())
alltests = alltests.reset_index()
alltests
bar_parented = plt.figure()
plt.rcParams["figure.figsize"] = (12,4) #Set (width,height) in inches
allscore,ax = plt.subplots()
x1 = range(0,len(alltests["parents_education"])*3,3)
x2 = [x+0.4 for x in x1]
x3 = [x+0.8 for x in x1]
a1 = ax.bar(x1,alltests["math_score"],width=0.4)
a2 = ax.bar(x2,alltests["reading_score"],width=0.4)
a3 = ax.bar(x3,alltests["writing_score"],width=0.4)
plt.xticks(range(0,len(alltests["parents_education"])*3,3),alltests["parents_education"],wrap=True)
plt.legend(["math_score","reading_score","writing_score"],loc="upper center")
plt.show()
eth = pd.DataFrame(perf[["math_score"]+["reading_score"]+["writing_score"]].groupby(perf["ethnicity"]).mean())
eth = eth.reset_index()
p_eth,ax_eth = plt.subplots()
x1 = range(0,len(eth["ethnicity"])*3,3)
x2 = [x+0.4 for x in x1]
x3 = [x+0.8 for x in x1]
a1 = ax_eth.bar(x1,eth["math_score"],width=0.4)
a2 = ax_eth.bar(x2,eth["reading_score"],width=0.4)
a3 = ax_eth.bar(x3,eth["writing_score"],width=0.4)
plt.xticks(range(0,len(eth["ethnicity"])*3,3),eth["ethnicity"],wrap=True)
plt.legend(["math_score","reading_score","writing_score"])
plt.show()
gend = pd.concat([pd.DataFrame(perf[["math_score"]+["reading_score"]+["writing_score"]].loc[perf["gender"]=="male"].mean()),
           pd.DataFrame(perf[["math_score"]+["reading_score"]+["writing_score"]].loc[perf["gender"]=="female"].mean())],
          axis=1).reset_index()
gend.columns = ["test_type","male","female"]
gend
from scipy.stats import pearsonr
print(pearsonr(perf["reading_score"],perf["writing_score"]))
m,c = np.polyfit(perf["reading_score"],perf["writing_score"],deg=1) #Slope and intercept for best fit line
yfit = m*perf["reading_score"]+c #creating best fit line
plt.scatter(perf["reading_score"],perf["writing_score"]) #scatter plot of individual points
plt.plot(perf["reading_score"],yfit,color="#111111") #overlay with best fit line
plt.text(x=80,y=30,s=("Pearson score = \n"+str(pearsonr(perf["reading_score"],perf["writing_score"]))))
eth = pd.crosstab(perf["ethnicity"],perf["parents_education"],normalize="index")
eth
eth_heatmap = plt.imshow(eth.transpose(),cmap="Blues")
plt.rcParams["figure.figsize"] = (5,6)
for y in range(0,len(eth.columns)):
    for x in range(0,len(eth.index)):
        plt.text(x,y,'%.4f' % eth[eth.columns[y]][eth.index[x]],
                 horizontalalignment="center",verticalalignment="top")
plt.yticks(np.arange(0,len(eth.columns)),eth.columns)
plt.xticks(np.arange(0,len(eth.index)),eth.index)

lun1 = pd.concat([pd.DataFrame(perf[["lunch"]+["math_score"]+["reading_score"]+["writing_score"]])],axis=1).reset_index()
lplot = plt.figure(figsize=(15,20)) #Size of each plot
#Starting here, this segment consists of all things that will be present in one subplot
plt.subplot(3,1,1) #(rows, columns, serial no.)
plt.title("Math scores") #Title of subplot
pd.DataFrame(lun1.loc[lun1["lunch"]=="standard"])["math_score"].plot(kind="density",label="standard lunch")
pd.DataFrame(lun1.loc[lun1["lunch"]!="standard"])["math_score"].plot(kind="density",label="free/reduced lunch")
plt.legend(loc="upper left") #Plot legend location
#Elements of first subplot end here
plt.subplot(3,1,2)
plt.title("Reading scores")
pd.DataFrame(lun1.loc[lun1["lunch"]=="standard"])["reading_score"].plot(kind="density",label="standard lunch")
pd.DataFrame(lun1.loc[lun1["lunch"]!="standard"])["reading_score"].plot(kind="density",label="free/reduced lunch")
plt.legend(loc="upper left")
plt.subplot(3,1,3)
plt.title("Writing scores")
pd.DataFrame(lun1.loc[lun1["lunch"]=="standard"])["writing_score"].plot(kind="density",label="standard lunch")
pd.DataFrame(lun1.loc[lun1["lunch"]!="standard"])["writing_score"].plot(kind="density",label="free/reduced lunch")
plt.legend(loc="upper left")