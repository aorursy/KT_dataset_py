
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#load the data

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
 
#for plot a bar chart
ax = train['Name'].value_counts().head(20).plot(kind='bar',fontsize=8)
fig = ax.get_figure()
#save to png for display below
fig.savefig('animal_name_train.png')
#for plot a bar chart
ax = test['Name'].value_counts().head(20).plot(kind='bar',fontsize=8)
fig = ax.get_figure()
fig.savefig('animal_name_test.png')