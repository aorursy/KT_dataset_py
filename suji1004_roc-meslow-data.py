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
data =pd.read_csv('/kaggle/input/roc-malslow/roc_maslow.csv')
print('There are {} rows and {} columns in train'.format(data.shape[0],data.shape[1]))
data.head(3)
import matplotlib.pyplot as plt
import seaborn as sns
x = data.GoalAchieved.value_counts()
label = ["0:Not-Achieved","1:Achieved"]
ax = sns.barplot(label,x)
    
plt.gca().set_ylabel('GoalAchieved', fontsize=20)
plt.xticks(fontsize=20)

for p in ax.patches:
    h = p.get_height()
    ax.annotate("%d (%d%%)" % (p.get_height(),h/sum(z)*100), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, -20),
                 textcoords='offset points')
plt.figure(figsize=(15,8))

y = data.MaslowLevel.value_counts()
label = ["0:Not-Certain","1:Pysiological","2:Safety","3:Love&Belong","4:Achievement","5:Self-Actualization"]
ax = sns.barplot(label,y)

plt.gca().set_ylabel('MaslowLevel', fontsize=20)
plt.xticks(fontsize=15)

for p in ax.patches:
    h = p.get_height()
    ax.annotate("%d (%d%%)" % (p.get_height(),h/sum(z)*100), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, -20),
                 textcoords='offset points')
plt.figure(figsize=(15,8))

z = data.FinalEmotion.value_counts()
label = ["0:Not-Certain","1:Happy","2:Sad","3:Angry","4:Fear","5:Surprised","6:Disgust","7:Anticipation","8:Trust"]
ax = sns.barplot(label,z)

plt.gca().set_ylabel('FinalEmotion', fontsize=20)
plt.xticks(fontsize=15)

for p in ax.patches:
    h = p.get_height()
    ax.annotate("%d (%d%%)" % (p.get_height(),h/sum(z)*100), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=15, color='black', xytext=(0, -5),
                 textcoords='offset points')
#특정 GoalAchieved 스토리 찾기

sample = data[data['GoalAchieved'] == 1]
story = sample.sample()
print('Title : {}'.format(story['storytitle'].values[0]))
print()
print('GoalAchieved : {}'.format(story['GoalAchieved'].values[0]))
print('MaslowLevel : {}'.format(story['MaslowLevel'].values[0]))
print('FinalEmotion : {}'.format(story['FinalEmotion'].values[0]))
print()
print('{}'.format(story['sentence1'].values[0]))
print('{}'.format(story['sentence2'].values[0]))
print('{}'.format(story['sentence3'].values[0]))
print('{}'.format(story['sentence4'].values[0]))
print('{}'.format(story['sentence5'].values[0]))
#특정 MaslowLevel 스토리 찾기

sample = data[data['MaslowLevel'] == 5]
story = sample.sample()
print('Title : {}'.format(story['storytitle'].values[0]))
print()
print('GoalAchieved : {}'.format(story['GoalAchieved'].values[0]))
print('MaslowLevel : {}'.format(story['MaslowLevel'].values[0]))
print('FinalEmotion : {}'.format(story['FinalEmotion'].values[0]))
print()
print('{}'.format(story['sentence1'].values[0]))
print('{}'.format(story['sentence2'].values[0]))
print('{}'.format(story['sentence3'].values[0]))
print('{}'.format(story['sentence4'].values[0]))
print('{}'.format(story['sentence5'].values[0]))
#특정 FinalEmotion 스토리 찾기

sample = data[data['FinalEmotion'] == 8]
story = sample.sample()
print('Title : {}'.format(story['storytitle'].values[0]))
print()
print('GoalAchieved : {}'.format(story['GoalAchieved'].values[0]))
print('MaslowLevel : {}'.format(story['MaslowLevel'].values[0]))
print('FinalEmotion : {}'.format(story['FinalEmotion'].values[0]))
print()
print('{}'.format(story['sentence1'].values[0]))
print('{}'.format(story['sentence2'].values[0]))
print('{}'.format(story['sentence3'].values[0]))
print('{}'.format(story['sentence4'].values[0]))
print('{}'.format(story['sentence5'].values[0]))
#복합 조건 스토리 찾기

sample = data[data['GoalAchieved'] == 0]
sample = sample[sample['MaslowLevel'] == 0]
sample = sample[sample['FinalEmotion'] == 0]

story = sample.sample()
print('Title : {}'.format(story['storytitle'].values[0]))
print()
print('GoalAchieved : {}'.format(story['GoalAchieved'].values[0]))
print('MaslowLevel : {}'.format(story['MaslowLevel'].values[0]))
print('FinalEmotion : {}'.format(story['FinalEmotion'].values[0]))
print()
print('{}'.format(story['sentence1'].values[0]))
print('{}'.format(story['sentence2'].values[0]))
print('{}'.format(story['sentence3'].values[0]))
print('{}'.format(story['sentence4'].values[0]))
print('{}'.format(story['sentence5'].values[0]))

#sample.head()
print()
print("total : {}".format(sum(sample.value_counts())))