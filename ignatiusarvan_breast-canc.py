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
!pip install sweetviz

!pip install plotly 

#import data





import sweetviz as sv



import plotly.express as px

import plotly.graph_objs as go

from scipy import stats



import numpy as np 

import pandas as pd 



print("===IMPORT -OK-===")
#masukkan train data

train_data = pd.read_csv("../input/breast/train.csv")

train_data.head()
#test data

test_data = pd.read_csv("../input/breast/test.csv")

test_data.head()
svrep = sv.analyze(train_data)



svrep.show_html()
datatemp = train_data.astype({'diagnosis':str})

svcompare = sv.compare_intra(datatemp, datatemp["diagnosis"]=='M', ['M','B'])

svcompare.show_html("comparediagnosis.html")
datamalignant = train_data[train_data["diagnosis"]=='M']

databenign = train_data[train_data["diagnosis"]=='B']

datamalignant.head()

#databenign.head()
datamalignant = train_data[train_data["diagnosis"]=='M']

databenign = train_data[train_data["diagnosis"]=='B']

#datamalignant.head()

databenign.head()
bpdiagnosis = px.box(train_data,x="diagnosis",y="fractal_dimension_worst",points="all")

bpdiagnosis.show()
m_malignant = datamalignant['fractal_dimension_worst'].dropna()

m_benign = databenign['fractal_dimension_worst'].dropna()



tscore, pval = stats.ttest_ind(m_malignant,m_benign)

print("----- T Test Results -----")

print("T stat. = " + str(tscore))

print("P value = " + str(pval))
pcmalignant =datamalignant["diagnosis"].value_counts()

pcmalignant = pd.DataFrame({'diagnosis':pcmalignant.index,'count':pcmalignant.values})





pcbenign = databenign["diagnosis"].value_counts()

pcbenign = pd.DataFrame({'diagnosis':pcbenign.index,'count':pcbenign.values})



piemalignant = px.pie(pcmalignant, values='count', names='diagnosis',title='Kanker Ganas')

piemalignant.show()



piebenign = px.pie(pcbenign, values='count', names='diagnosis',title='Kanker Jinak')

piebenign.show()



labels = ['Malignant','Benign']

values = [170,285]





pietotal = go.Figure(data=[go.Pie(labels=labels, values=values,title='Total')])

pietotal.show()

import seaborn as sn

import matplotlib.pyplot as plt



corr = train_data.corr()



snhet = sn.heatmap(corr, cmap='Blues')

plt.show()

train_data.head()
test_data.head()
from sklearn.ensemble import RandomForestClassifier

y = train_data["diagnosis"]



features = ["radius_mean", "concave points_mean", "perimeter_worst", "compactness_worst"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X,y)

prediction = model.predict(X_test)
output = pd.DataFrame({'id': test_data.id, 'diagnosis': prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")