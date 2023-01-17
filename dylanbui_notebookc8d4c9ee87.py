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
import numpy as np

import pandas as pd

import matplotlib as pl

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_auc_score

import sklearn.model_selection

import sklearn.metrics

import sklearn.preprocessing

import torch

import os

import seaborn as sns

from tqdm import tqdm

from pylab import rcParams

import torch.nn.functional as F

from torch import nn, optim

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve

import xgboost as xgb
data_presc = pd.read_csv('/kaggle/input/us-opiate-prescriptions/prescriber-info.csv')

data_presc = data_presc.drop(columns = ['NPI','Credentials'])
count = data_presc ['Specialty'].value_counts()
plt.figure(figsize=(20,25))

count.plot.bar()
opioid_prescribers = data_presc[data_presc["Opioid.Prescriber"]==1]
data_presc.loc['Total',:]= data_presc.sum(axis=0)

opioid_prescribers.loc['Total', :] = opioid_prescribers.sum(axis=0)
total_presc = data_presc.iloc[25000].tolist()[3:253]
total_opioid_presc = opioid_prescribers.iloc[14688].tolist()[3:253]
res = [i / j for i, j in zip(total_opioid_presc,total_presc )] 
res_labeled = pd.DataFrame(res, columns=["Correlation"])

res_labeled.insert(0, "Drug",data_presc.columns[3:253].tolist(),True)
res_labeled.nlargest(25,'Correlation',)
total_males = data_presc.iloc[25000,0].count('M') 
total_females = data_presc.iloc[25000,0].count('F')
total_opioid_males = opioid_prescribers.iloc[14688,0].count('M')
total_opioid_females = opioid_prescribers.iloc[14688,0].count('F')
corr_males = total_opioid_males/total_males
corr_females = total_opioid_females/total_females
state_all = data_presc.iloc[0:25000,1].value_counts()
state_opioid = opioid_prescribers.iloc[0:14688,1].value_counts()
corr_state = state_opioid/state_all
spec_all = data_presc.iloc[0:25000,2].value_counts()
spec_opioid = opioid_prescribers.iloc[0:14688,2].value_counts()
corr_spec= spec_opioid/spec_all
res_labeled.nlargest(25,'Correlation',).plot.bar(x='Drug',ylim = (0.95,1.0))
corr_spec.sort_values(ascending=False) [0:25].plot.bar(ylim = (0.85,1.0))
corr_gender = {'Name': ['corr_males','corr_females'],

                           'Ratio': [corr_males, corr_females]}
corr_gender_series = pd.DataFrame(corr_gender, index =['Male','Female'])
plot_gender = corr_gender_series.plot.pie(y='Ratio')
corr_state.sort_values(ascending=False) [0:10].plot.bar(ylim = (0.65,1.0))
corr_spec_comp = 1 - corr_spec
pd.set_option('display.max_rows', 10)

corr_raw= corr_spec.copy()

corr_raw_comp = corr_spec_comp.copy()

def entropy_calc (a,b):

    result = 0

    prob_1 = np.log2(a**a)

    prob_2 = np.log2(b**b)

    result = -(prob_1+prob_2)

    return result

entropy_spec = entropy_calc(corr_raw, corr_raw_comp)

entropy_spec_df = entropy_spec.to_frame()

entropy_spec_df ["Correlation"]= corr_spec

clean_entropy_spec_df= entropy_spec_df.dropna()

clean_entropy_spec_df = clean_entropy_spec_df [clean_entropy_spec_df['Specialty']!=0]

clean_entropy_spec_df = clean_entropy_spec_df.drop(columns= 'Specialty')

clean_entropy_spec_df.sort_values(by=['Correlation'],ascending=False) [0:25].plot.bar(ylim = (0.79,1.0))
corr_state_comp = 1- corr_state

raw_corr_state = corr_state.copy()

raw_corr_state_comp = corr_state_comp.copy()

entropy_calc(raw_corr_state,raw_corr_state_comp)

entropy_state =entropy_calc(raw_corr_state,raw_corr_state_comp)

entropy_state_df = entropy_state.to_frame()

entropy_state_df["Correlation"] = corr_state

clean_entropy_state_df = entropy_state_df.dropna()

clean_entropy_state_df = clean_entropy_state_df [clean_entropy_state_df['State'] != 0]

clean_entropy_state_df = clean_entropy_state_df.drop ('State',axis =1)

clean_entropy_state_df.sort_values(by='Correlation',ascending=False) [0:10].plot.bar(ylim = (0.65,1.0))

clean_entropy_state_df.sort_values(by='Correlation',ascending=False) [0:10].plot.bar(ylim = (0.65,0.74))

plt.show()
corr_drug= res_labeled['Correlation']
corr_drug_comp = 1- corr_drug

raw_corr_drug = corr_drug.copy()

raw_corr_drug_comp = corr_drug_comp.copy()

entropy_calc(raw_corr_drug,raw_corr_drug_comp)

entropy_drug =entropy_calc(raw_corr_drug,raw_corr_drug_comp)

entropy_drug_df = entropy_drug.to_frame()

entropy_drug_df['Drug'] = res_labeled['Drug']

entropy_drug_df["Correlation"] = corr_drug

clean_entropy_drug_df = entropy_drug_df.dropna()

clean_entropy_drug_df.sort_values(by='Correlation',ascending=False) [0:25].plot.bar(x='Drug', ylim = (0.95,1.0))
dataXY = data_presc.iloc[:25000]
dataXY.columns
val =['Behavioral Analyst', 'Chiropractic', 'Clinical Pharmacology', 

      'Community Health Worker', 'Counselor', 'Hand Surgery', 'Health Maintenance Organization', 

      'Homeopath', 'Hospital (Dmercs Only)', 'Licensed Clinical Social Worker', 'Medical Genetics', 

      'Medical Genetics, Ph.D. Medical Genetics', 'Midwife', 'Military Health Care Provider', 

      'Pharmacy Technician', 'Physical Therapist', 'Preferred Provider Organization', 'Slide Preparation Facility', 

      'Specialist/Technologist', 'Surgical Oncology', 'Thoracic Surgery (Cardiothoracic Vascular Surgery)', 

      'Unknown Physician Specialty Code', 'Unknown Supplier/Provider']
dataXY = dataXY.loc[~dataXY.Specialty.isin(val)]
def odds_ratio (a,b):

  x = len(dataXY[(dataXY[a]== b)& (dataXY['Opioid.Prescriber'] == 1)]) + 0.5

  y = len(dataXY[(dataXY[a]== b)& (dataXY['Opioid.Prescriber'] == 0)]) + 0.5

  w = len(dataXY[(dataXY[a]!= b)& (dataXY['Opioid.Prescriber'] == 1)]) + 0.5

  z = len(dataXY[(dataXY[a]!= b)& (dataXY['Opioid.Prescriber'] == 0)]) + 0.5

  return (x*z)/(w*y)
def odds_ratio2 (a):

  x = len(dataXY[(dataXY[a] > 0)& (dataXY['Opioid.Prescriber'] == 1)]) + 0.5

  y = len(dataXY[(dataXY[a] > 0)& (dataXY['Opioid.Prescriber'] == 0)]) + 0.5

  w = len(dataXY[(dataXY[a] == 0)& (dataXY['Opioid.Prescriber'] == 1)]) + 0.5

  z = len(dataXY[(dataXY[a] == 0)& (dataXY['Opioid.Prescriber'] == 0)])+ 0.5

  ans = (x*z)/(w*y)

  return ans 
values_spec = dataXY.Specialty.unique()

values_drug = dataXY.columns[3:253].tolist()

values_gender = ["M","F"]

values_state = dataXY.State.unique()
list_state = []

for elt in values_state:

  odds_state = odds_ratio("State",elt)

  list_state.append(odds_state)

data = {"State":values_state,"Odds Ratio":list_state}

state_OR = pd.DataFrame(data,columns = ["State","Odds Ratio"])

state_OR.sort_values(by="Odds Ratio",ascending=False) [0:15].plot.bar(x='State',ylim = (0,5.001))



plt.title('States vs Odds Ratio')

plt.ylabel('Odds Ratio')



state_OR.sort_values(by="Odds Ratio",ascending=True) [0:15].plot.bar(x='State',ylim = (0,5.001))

plt.title('States vs Odds Ratio')

plt.ylabel('Odds Ratio')

plt.show()
list_spec = []

for elt in values_spec:

  odds_spec = odds_ratio("Specialty",elt)

  list_spec.append(odds_spec)

data = {"Specialty":values_spec,"Odds Ratio":list_spec}

spec_OR = pd.DataFrame(data,columns = ["Specialty","Odds Ratio"])

spec_OR.sort_values(by="Odds Ratio",ascending=False) [0:10].plot.bar(x='Specialty',ylim = (0,50))

plt.title('Specialty vs Odds Ratio')
list_gend = []

for elt in values_gender:

  odds_gend = odds_ratio("Gender",elt)

  list_gend.append(odds_gend)

data = {"Gender":values_gender,"Odds Ratio":list_gend}

gend_OR = pd.DataFrame(data,columns = ["Gender","Odds Ratio"])

gend_OR.sort_values(by="Odds Ratio",ascending=False) [0:2].plot.bar(x='Gender',ylim = (0,1.5))

plt.title('Gender vs Odds Ratio')
list_drug = []

for elt in values_drug:

  odds_drug = odds_ratio2(elt) 

  list_drug.append(odds_drug)

data = {"Drug":values_drug,"Odds Ratio":list_drug}

drug_OR = pd.DataFrame(data,columns = ["Drug","Odds Ratio"])

drug_OR.sort_values(by="Odds Ratio",ascending=False) [0:10].plot.bar(x='Drug',ylim = (0,45000))

plt.title('Drugs vs Odds Ratio')
X = dataXY.iloc[:,0:253]
Y = dataXY.iloc[:,253]
le = preprocessing.LabelEncoder()

X['Gender'] = le.fit_transform(X['Gender'])

X['State'] = le.fit_transform(X['State'])

X['Specialty'] = le.fit_transform(X['Specialty'])
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.1, random_state = 42)
model = RandomForestClassifier(n_estimators=100, bootstrap = True, max_features = 'sqrt')
model.fit(trainX, trainY)
rf_predictions = model.predict(testX)

rf_probs = model.predict_proba(testX)[:, 1]
roc_value = roc_auc_score(testY, rf_probs)
roc_value
classes = ['No Opioid', 'Opioid']

print(classification_report(testY, rf_predictions, target_names=classes))
cm = confusion_matrix(testY, rf_predictions)

df_cm = pd.DataFrame(cm, index=classes, columns=classes)

hmap = sns.heatmap(df_cm, annot=True, fmt="d")

hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')

hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

plt.ylabel('True label')

plt.xlabel('Predicted label');
logisticRegressor = LogisticRegression()

logisticRegressor.fit(trainX, trainY)

logisticRegressor.score(testX, testY)

lr_pred = logisticRegressor.predict(testX)
classes = ['No Opioid', 'Opioid']

print(classification_report(testY, lr_pred, target_names=classes))
cm = confusion_matrix(testY, lr_pred)

df_cm = pd.DataFrame(cm, index=classes, columns=classes)

hmap = sns.heatmap(df_cm, annot=True, fmt="d")

hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')

hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

plt.ylabel('True label')

plt.xlabel('Predicted label');
mnb = MultinomialNB()

mnb.fit(trainX, trainY)

mnb.score(testX, testY)

mnb_pred =mnb.predict(testX)
classes = ['No Opioid', 'Opioid']

print(classification_report(testY, mnb_pred, target_names=classes))
cm = confusion_matrix(testY, mnb_pred)

df_cm = pd.DataFrame(cm, index=classes, columns=classes)

hmap = sns.heatmap(df_cm, annot=True, fmt="d")

hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')

hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

plt.ylabel('True label')

plt.xlabel('Predicted label');
classifier = xgb.XGBClassifier()

classifier.fit(trainX, trainY)

XGB_pred = classifier.predict(testX)
classifier.score(testX,testY)
classes = ['No Opioid', 'Opioid']

print(classification_report(testY, XGB_pred, target_names=classes))
cm = confusion_matrix(testY, XGB_pred)

df_cm = pd.DataFrame(cm, index=classes, columns=classes)

hmap = sns.heatmap(df_cm, annot=True, fmt="d")

hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')

hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

plt.ylabel('True label')

plt.xlabel('Predicted label');
trainX = torch.from_numpy(trainX.to_numpy()).float()

trainY = torch.squeeze(torch.from_numpy(trainY.to_numpy()).float())

testX = torch.from_numpy(testX.to_numpy()).float()

testY = torch.squeeze(torch.from_numpy(testY.to_numpy()).float())
class Net(nn.Module):

  def __init__(self, n_features):

    super(Net, self).__init__()

    self.fc1 = nn.Linear(n_features, 20)

    self.fc2 = nn.Linear(20, 15)

    self.fc3 = nn.Linear(15, 10)

    self.fc4 = nn.Linear(10, 5)

    self.fc5 = nn.Linear(5, 1)

  def forward(self, x):

    x = F.relu(self.fc1(x))

    x = F.relu(self.fc2(x))

    x = F.relu(self.fc3(x))

    x = F.relu(self.fc4(x))

    return torch.sigmoid(self.fc5(x))
net = Net(trainX.shape[1])

criterion = nn.BCELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainX = trainX.to(device)

trainY = trainY.to(device)

testX = testX.to(device)

testY = testY.to(device)
net = net.to(device)

criterion = criterion.to(device)
def calculate_accuracy(y_true, y_pred):

  predicted = y_pred.ge(.5).view(-1)

  return (y_true == predicted).sum().float() / len(y_true)
def round_tensor(t, decimal_places=3):

  return round(t.item(), decimal_places)
for epoch in range(1000):

    y_pred = net(trainX)

    y_pred = torch.squeeze(y_pred)

    train_loss = criterion(y_pred, trainY)

    if epoch % 100 == 0:

      train_acc = calculate_accuracy(trainY, y_pred)

      y_test_pred = net(testX)

      y_test_pred = torch.squeeze(y_test_pred)

      test_loss = criterion(y_test_pred, testY)

      test_acc = calculate_accuracy(testY, y_test_pred)

      print(

f'''epoch {epoch}

Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}

Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}

''')

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()
MODEL_PATH = 'model_opioid.pth'

torch.save(net, MODEL_PATH)
net = torch.load(MODEL_PATH)
classes = ['No Opioid', 'Opioid']

y_pred = net(testX)

y_pred = y_pred.ge(.5).view(-1).cpu()

testY = testY.cpu()

print(classification_report(testY, y_pred, target_names=classes))
cm = confusion_matrix(testY, y_pred)

df_cm = pd.DataFrame(cm, index=classes, columns=classes)

hmap = sns.heatmap(df_cm, annot=True, fmt="d")

hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')

hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')

plt.ylabel('True label')

plt.xlabel('Predicted label');