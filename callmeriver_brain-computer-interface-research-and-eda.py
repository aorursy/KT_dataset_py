# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# 
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# these are the results of all subjects. included in them are those who were classified later on 
# as learner/responders and non-learners/non-responders
data = pd.read_csv('../input/BCI_input.csv')

print(data)
plt.show()
sns.pairplot(data)
plt.show()
t = data[data['HRV_HF-test']>0.001]
t
data.rename(index=str, columns={'HRV-Vagal-Practice':'Vagal',
                                'HRV-HF-Practice':'HF',
                                'Average of NF-BB':'BLpractice',
                                'Average of NF_GL':'GLpractice',
                                "HRV_Vagal-test": "Vagaltest", 
                                "HRV_HF-test": "HFtest",
                                "NF(GL)-Test": "GLtest", 
                                "NF(BB)-Test": "BLtest"}
            ,inplace=True)
nonLearnersLabels = [102,103,104,112,107,108,109,115,121]
# learnersLabels = [101,105,106,110,111,113,114,116,117,118,119,120,122]
learnersLabels = [ x for x in range(101,123) if x not in nonLearnersLabels]

# isin function here helps us filter the participants based on their subject ID
learners = data[data.subject.isin(learnersLabels)]
nonLearners = data[data.subject.isin(nonLearnersLabels)]

# let's show the learners dataframe as an example
learners
from scipy.stats import pearsonr

plt.figure(num=None, figsize=(10, 4))
ax = plt.subplot(1,2,1)
ax.set_title("Learners")
sns.regplot(x=learners.GLpractice,y=learners.GLtest)
ax = plt.subplot(1,2,2)
ax.set_title("Non-Learners")
sns.regplot(x=nonLearners.GLpractice,y=nonLearners.GLtest)

r1,p1 = pearsonr(learners.GLpractice,learners.GLtest)
r2,p2 = pearsonr(nonLearners.GLpractice,nonLearners.GLtest)
# corrLearn = np.corrcoef(learners.GLpractice,learners.GLtest)[1][0]
# corrNonLearn = np.corrcoef(nonLearners.GLpractice,nonLearners.GLtest)[1][0]
print("learners: r={}, p-val={}, \nnon-learners: r={}, p-val={}".format(r1,p1,r2,p2))

from scipy.stats import pearsonr

plt.figure(num=None, figsize=(10, 4))
ax = plt.subplot(1,2,1)
ax.set_title("Learners")
sns.regplot(x=learners.BLpractice,y=learners.BLtest)
ax = plt.subplot(1,2,2)
ax.set_title("Non-Learners")
sns.regplot(x=nonLearners.BLpractice,y=nonLearners.BLtest)

r1,p1 = pearsonr(learners.BLpractice,learners.BLtest)
r2,p2 = pearsonr(nonLearners.BLpractice,nonLearners.BLtest)
# corrLearn = np.corrcoef(learners.GLpractice,learners.GLtest)[1][0]
# corrNonLearn = np.corrcoef(nonLearners.GLpractice,nonLearners.GLtest)[1][0]
print("learners: r={}, p-val={}, \nnon-learners: r={}, p-val={}".format(r1,p1,r2,p2))

print(learners[['HF','HFtest']])
plt.scatter(learners.HF,learners.HFtest)
learners = learners.assign(logHFpractice=np.log(learners.HF))
learners = learners.assign(logHFtest=np.log(learners.HFtest))

nonLearners = nonLearners.assign(logHFpractice=np.log(nonLearners.HF))
nonLearners = nonLearners.assign(logHFtest=np.log(nonLearners.HFtest))

# display the learners to check the data
learners
print(learners[['logHFpractice','logHFtest']])
plt.scatter(learners.logHFpractice,learners.logHFtest)
from scipy.stats import pearsonr

plt.figure(num=None, figsize=(10, 4))
ax = plt.subplot(1,2,1)
ax.set_title("Learners")
sns.regplot(x=learners.logHFpractice,y=learners.logHFtest)
ax = plt.subplot(1,2,2)
ax.set_title("Non-Learners")
sns.regplot(x=nonLearners.logHFpractice,y=nonLearners.logHFtest)

r1,p1 = pearsonr(learners.logHFpractice,learners.logHFtest)
r2,p2 = pearsonr(nonLearners.logHFpractice,nonLearners.logHFtest)
# corrLearn = np.corrcoef(learners.GLpractice,learners.GLtest)[1][0]
# corrNonLearn = np.corrcoef(nonLearners.GLpractice,nonLearners.GLtest)[1][0]
print("learners: r={}, p-val={}, \nnon-learners: r={}, p-val={}".format(r1,p1,r2,p2))
from scipy.stats import pearsonr

plt.figure(num=None, figsize=(10, 4))
ax = plt.subplot(1,2,1)
ax.set_title("Learners")
sns.regplot(x=learners.Vagal,y=learners.Vagaltest)
ax = plt.subplot(1,2,2)
ax.set_title("Non-Learners")
sns.regplot(x=nonLearners.Vagal,y=nonLearners.Vagaltest)

r1,p1 = pearsonr(learners.Vagal,learners.Vagaltest)
r2,p2 = pearsonr(nonLearners.Vagal,nonLearners.Vagaltest)
# corrLearn = np.corrcoef(learners.GLpractice,learners.GLtest)[1][0]
# corrNonLearn = np.corrcoef(nonLearners.GLpractice,nonLearners.GLtest)[1][0]
print("learners: r={}, p-val={}, \nnon-learners: r={}, p-val={}".format(r1,p1,r2,p2))

