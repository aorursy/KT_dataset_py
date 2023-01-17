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
import sweetviz as sv

import plotly.express as px
import plotly.graph_objs as go
from scipy import stats

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print("===IMPORT -OK-===")
train_data = pd.read_csv("../input/stmm-challenge-1/train.csv")
train_data.head()
test_data = pd.read_csv("../input/stmm-challenge-1/test.csv")
test_data.head()
svrep = sv.analyze(train_data)

svrep.show_html()
datatemp = train_data.astype({'diagnosis':str})
svcompare = sv.compare_intra(datatemp, datatemp["diagnosis"]=='M', ['M','B'])
svcompare.show_html("comparediagnosis.html")
datamalignant = train_data[train_data["diagnosis"]=="M"]
databenign = train_data[train_data["diagnosis"]=="B"]

datamalignant.head()
#databenign.head()
datamalignant = train_data[train_data["diagnosis"]=="M"]
databenign = train_data[train_data["diagnosis"]=="B"]

#datamalignant.head()
databenign.head()
#box plot sebaran malignant vs benign

bpsurvivor = px.box(train_data,x="diagnosis",y="radius_mean",points="all")
bpsurvivor.show()
m_mal = datamalignant['radius_mean'].dropna()
m_ben = databenign['radius_mean'].dropna()

tscore, pval = stats.ttest_ind(m_mal,m_ben)
print("----- T Test Results -----")
print("T stat. = " + str(tscore))
print("P value = " + str(pval))

pcmalignant = datamalignant["diagnosis"].value_counts()
pcmalignant = pd.DataFrame({'diagnosis':pcmalignant.index,'count':pcmalignant.values})


pcbenign = databenign["diagnosis"].value_counts()
pcbenign = pd.DataFrame({'diagnosis':pcbenign.index,'count':pcbenign.values})

piemalign = px.pie(pcmalignant, values='count', names='diagnosis',title='Kanker Ganas')
piemalign.show()
pieben = px.pie(pcbenign, values='count', names='diagnosis',title='Kanler Jinak')
pieben.show()
import plotly.graph_objects as go

labels = ['Malignant','Benign']
values = [170 , 285]

fig = go.Figure(data=[go.Pie(labels=labels , values=values)])
fig.show()
import seaborn as sn
import matplotlib.pyplot as plt

corr = train_data.corr()

snhet = sn.heatmap(corr, cmap="Blues")
plt.show()
#Feature Selection
train_data = train_data[['radius_mean','texture_mean','perimeter_mean','area_mean','radius_se','texture_se','perimeter_se','area_se','radius_worst','texture_worst','perimeter_worst','area_worst']]
y = train_data["diagnosis"]
features = ["radius_mean","texture_mean","perimeter_mean","area_mean","radius_se","texture_se","perimeter_se","area_se","radius_worst","texture_worst","perimeter_worst","area_worst"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
train_data.head()
test_data.head()
#create model
from sklearn.ensemble import RandomForestClassifier
def create_forest(n_pohon,max_lv):
    

    model = RandomForestClassifier(n_estimators=n_pohon, max_depth=max_lv, random_state=1)
    return model
    
model.fit(X, y)
#Evaluating using Kfold cross validation

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
n_folds = 5
pohon = 100
lev = 5
kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=13)
kfold = kfold.split(X, y)



#mean_fpr = np.linspace(0, 1, 100)

i = 0
totalscore = []
for i, (train, test) in enumerate(kfold):
    
    traindata = X.iloc[train]
    testdata = X.iloc[test]
    ytrain = y.iloc[train]
    ytest = y.iloc[test]

  
    print("Running Fold", i+1, "/", n_folds)
  
    evalmodel = create_forest(pohon,lev)

    
    evalmodel.fit(traindata, ytrain)
    logits = evalmodel.predict(testdata)
  
    predicts=(logits > 0.5).astype("int32")
    #scores = evalmodel.evaluate(testdata, ytest, verbose=0)

    cm = confusion_matrix(ytest, predicts)
    creport = classification_report(ytest, predicts)
    print('Confusion matrix')
    print(cm)
    print(creport)
    score = (evalmodel.score(testdata,ytest))*100
    totalscore.append(score)
    print('Tingkat akurasi model adalah: ', str(score), '%' )
    #print("%s: %.2f%%" % (evalmodel.metrics_names[1], scores[1]*100))
meanscore = sum(totalscore)/len(totalscore)
print('Rata-rata tingkat Akurasi model anda dari Kfold Crossval adalah: ',str(meanscore))
jml_pohon = int(input('Masukkan jumlah Pohon: '))
level = int(input('Masukkan Level Terdalam: '))

finalmodel = create_forest(jml_pohon, level)
finalmodel.fit(X,y)
predictions = finalmodel.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")