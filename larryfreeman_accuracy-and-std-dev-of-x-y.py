# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Let's get top 10 most checked-in place ids:
train = pd.read_csv("../input/train.csv")
print(train.place_id.value_counts().nlargest(10))

# Now, let's look at the accuracy distribution
place_data = train[train.place_id==8772469670]
print(place_data.accuracy.min(),place_data.accuracy.max())
place_data.accuracy.plot(kind='hist',title='accuracy distribution',bins=100)



place_data.accuracy.plot(kind='hist',title='deeper look',xlim=(0,100),bins=100)
def category(acc):
    if acc <= 10:
        return '0-10'
    elif acc <= 20:
        return '10-20'
    elif acc <= 30:
        return '20-30'
    elif acc <= 40:
        return '30-40'
    elif acc <= 50:
        return '40-50'
    elif acc <= 60:
        return '50-60'
    elif acc <= 70:
        return '60-70'
    elif acc <= 80:
        return '70-80'
    elif acc <= 90:
        return '80-90'
    else:
        return '90-1000'
    
pd.options.mode.chained_assignment=None
place_data['accuracy_cat'] = place_data.accuracy.map(lambda x: category(x))

# plot standard deviation vs. accuracy category
print(place_data.groupby('accuracy_cat').count()[['x','y']])
print(place_data.groupby('accuracy_cat').std()[['x','y']])





# Now, let's look at the accuracy distribution
print(place_data.accuracy.min(),place_data.accuracy.max())
place_data.accuracy.plot(kind='hist',title='accuracy distribution',bins=100)
place_data.accuracy.plot(kind='hist',title='deeper look',xlim=(0,100),bins=100)
def category(acc):
    if acc <= 10:
        return '0-10'
    elif acc <= 20:
        return '10-20'
    elif acc <= 30:
        return '20-30'
    elif acc <= 40:
        return '30-40'
    elif acc <= 50:
        return '40-50'
    elif acc <= 60:
        return '50-60'
    elif acc <= 70:
        return '60-70'
    elif acc <= 80:
        return '70-80'
    elif acc <= 90:
        return '80-90'
    else:
        return '90-1000'
    
pd.options.mode.chained_assignment=None
place_data['accuracy_cat'] = place_data.accuracy.map(lambda x: category(x))

# plot standard deviation vs. accuracy category
print(place_data.groupby('accuracy_cat').count()[['x','y']])
print(place_data.groupby('accuracy_cat').std()[['x','y']])
# Now, let's look at the accuracy distribution
place_data = train[train.place_id==1308450003]
print(place_data.accuracy.min(),place_data.accuracy.max())
place_data.accuracy.plot(kind='hist',title='accuracy distribution',bins=100)
place_data.accuracy.plot(kind='hist',title='deeper look',xlim=(0,100),bins=100)
def category(acc):
    if acc <= 10:
        return '0-10'
    elif acc <= 20:
        return '10-20'
    elif acc <= 30:
        return '20-30'
    elif acc <= 40:
        return '30-40'
    elif acc <= 50:
        return '40-50'
    elif acc <= 60:
        return '50-60'
    elif acc <= 70:
        return '60-70'
    elif acc <= 80:
        return '70-80'
    elif acc <= 90:
        return '80-90'
    else:
        return '90-1000'
    
pd.options.mode.chained_assignment=None
place_data['accuracy_cat'] = place_data.accuracy.map(lambda x: category(x))

# plot standard deviation vs. accuracy category
print(place_data.groupby('accuracy_cat').count()[['x','y']])
print(place_data.groupby('accuracy_cat').std()[['x','y']])
