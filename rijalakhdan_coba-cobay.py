!pip install sweetviz
import sweetviz as sv

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)

import plotly.express as px
import plotly.graph_objs as go
from scipy import stats

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


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
datamalignant = train_data[train_data["diagnosis"]=='M']
databenign = train_data[train_data["diagnosis"]=='B']

datamalignant.head()
datamalignant = train_data[train_data["diagnosis"]=='M']
databenign = train_data[train_data["diagnosis"]=='B']

databenign.head()
train_data.describe()

# checking the different values contained in the diagnosis column

diagnosis = train_data['diagnosis'].value_counts()

diagnosis_label = diagnosis.index
diagnosis_size = diagnosis.values

colors = ['red', 'blue']

trace = go.Pie(labels = diagnosis_label,
              values = diagnosis_size,
               marker = dict(colors = colors),
               name = 'Breast Cancer',
               hole = 0.3
              )
df = [trace]
layout = go.Layout(title = 'Distribution of Patients')

fig = go.Figure(data = df, layout = layout)
py.iplot(fig)
y = train_data['diagnosis']
x = train_data.drop('diagnosis', axis = 1)

x = (x - x.mean()) / (x.std()) 
df = pd.concat([y, x.iloc[:,0:10]], axis=1)
df = pd.melt(df, id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(15, 10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=df,split=True, inner="quart", palette = 'cool')
plt.title('Mean Features vs Diagnosis', fontsize = 20)
plt.xticks(rotation=90)
plt.show()
plt.rcParams['figure.figsize'] = (18, 15)

sns.heatmap(train_data.corr(), cmap = 'Purples_r', annot = True, linewidths = 0.5, fmt = '.1f')
plt.title('Heat Map for Correlations', fontsize = 20)
plt.show()
list_to_delete = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',
                  'radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst',
                  'concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']
x = x.drop(list_to_delete, axis = 1)

plt.rcParams['figure.figsize'] = (18, 15)
sns.heatmap(x.corr(), annot = True, cmap = 'autumn')
plt.title('Heat Map for the Reduced Data', fontsize = 20)
plt.show()
from sklearn.preprocessing import LabelEncoder

# performing label encoding
le = LabelEncoder()
y= le.fit_transform(y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 16)

print("Shape of x_train_data :", x_train.shape)
print("Shape of y_train_data :", y_train.shape)
print("Shape of x_test_data :", x_test.shape)
print("Shape of y_test_data :", y_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# creating a model
model = RandomForestClassifier(n_estimators = 400, max_depth = 10)

# feeding the training set into the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# Calculating the accuracies
print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuarcy :", model.score(x_test, y_test))

# classification report
cr = classification_report(y_test, y_pred)
print(cr)
# confusion matrix 
cm = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, cmap = 'winter')
plt.title('Confusion Matrix', fontsize = 20)
plt.show()
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
model = RandomForestClassifier() 
rfecv = RFECV(estimator = model, step = 1, cv = 5, scoring = 'accuracy')
rfecv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])
y_pred = rfecv.predict(x_test)

print("Training Accuracy :", rfecv.score(x_train, y_train))
print("Testing Accuracy :", rfecv.score(x_test, y_test))

cm = confusion_matrix(y_pred, y_test)
plt.rcParams['figure.figsize'] = (5, 5)
sns.heatmap(cm, annot = True, cmap = 'copper')
plt.show()
diagnosis_map={'M':1,'B':0}
train_data['diagnosis'] = train_data['diagnosis'].map(diagnosis_map)
train_data
#Feature Selection
train = train_data[['radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst','diagnosis']]

#Hapus missing values
train = train.dropna()

y = train_data["diagnosis"]
#one hot encoding -- mengubah data kategorikal menjadi numerik
features = ['radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
X = pd.get_dummies(train[features])
X_test = pd.get_dummies(test_data[features])

from sklearn.ensemble import RandomForestClassifier
def create_forest(n_pohon,max_lv):
    

    model = RandomForestClassifier(n_estimators=n_pohon, max_depth=max_lv, random_state=1)
    return model
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

output = pd.DataFrame({'id': test_data.id, 'diagnosis': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
submission=pd.read_csv('./my_submission.csv')
submission
diagnosis_map={1:'M',0:'B'}
submission['diagnosis'] = submission['diagnosis'].map(diagnosis_map)
submission
output = pd.DataFrame(submission)
output.to_csv('submissionku_CobaCobay.csv', index=False)
print("Your submission was successfully saved!")