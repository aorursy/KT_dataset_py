import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.listdir("../input/osic-pulmonary-fibrosis-progression")
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
train.head()
test.head()
submission = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
submission.head()
print(f"Train info {train.shape}")
print(f"test info {test.shape}")
print(f"submission info {submission.shape}")
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
print(f"Total Patient Id {train['Patient'].count()}")
print(f"NUmber of Unique Id {train['Patient'].value_counts().shape[0]}")
train["SmokingStatus"].value_counts().plot(kind="bar")
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
ds = pydicom.dcmread(img)
plt.figure(figsize = (5,5))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
import random


def get_random(smokes):
    smoke_pat = train[train["SmokingStatus"]==smokes] 
    patientz = [i for i in smoke_pat["Patient"]] # patient id list
    r_st = random.choice(patientz) # random choice
    print(r_st)
    image_dir = f"../input/osic-pulmonary-fibrosis-progression/train/{r_st}" # image directory
    image_list = os.listdir(image_dir) # list of images
    c = []
    for t in image_list:
        first, exts = os.path.splitext(t) # split text
        first = int(first) # int
        c.append(first) # append
    d = [num for num in range(1, 31)] # num from 1 to 30
    gh = []
    for x in c:
        if x in d:
            gh.append(x) # if number is in list then append
    fig = plt.figure(figsize=(10, 10)) # figure
    columns = 5
    row = 6
    for ab in gh:
        files = image_dir + "/" + str(ab) + ".dcm" # file directory
        ds = pydicom.dcmread(files) # read dcm file
        fig.add_subplot(row, columns, ab) # add plot
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone) # show images
    plt.suptitle(smokes) # title
get_random("Ex-smoker")
get_random("Never smoked")
get_random("Currently smokes")
submission["Patient"] = submission["Patient_Week"].apply(lambda x:x.split("_")[0])
submission["Weeks"] = submission["Patient_Week"].apply(lambda x:x.split("_")[1])

submission =  submission[['Patient','Weeks', 'Confidence','Patient_Week']]
submission = submission.merge(test.drop('Weeks', axis=1), on="Patient")
submission.tail()
submission.shape
submission["Patient"].unique()
train["Dataset"] = "train"
test["Dataset"] = "test"
submission["Dataset"] = "submission"
dataset = train.append([test, submission])
dataset = dataset.reset_index()
dataset = dataset.drop(columns=['index'])
dataset.head()
dataset["Weeks"] = dataset["Weeks"].astype("int64")
dataset.info()
dataset["First_week"] = dataset["Weeks"]
dataset.loc[dataset.Dataset=='submission','First_week'] = np.nan
dataset["First_week"] = dataset.groupby('Patient')['First_week'].transform('min')
dataset.head()
dataset = dataset.merge(dataset[dataset["Weeks"] == dataset["First_week"]][["Patient", "FVC"]].rename({"FVC": "First_FVC"}, axis=1).groupby("Patient").first().reset_index(), on="Patient", how="left")
dataset["Week_diff"] = dataset["Weeks"] - dataset["First_week"]
# dataset["FVC_diff"] = dataset["FVC"] - dataset["First_FVC"]

dataset = pd.concat([dataset,pd.get_dummies(dataset.Sex),pd.get_dummies(dataset.SmokingStatus)], axis=1)

dataset = dataset.drop(columns=['Sex', 'SmokingStatus'])
dataset.head()
dataset.info()
train = dataset[dataset["Dataset"]=="train"]
test = dataset[dataset["Dataset"]=="test"]
submission = dataset[dataset["Dataset"]=="submission"]
from sklearn.preprocessing import StandardScaler

col = ['Weeks', 'Percent', 'Age', 'First_week', 'First_FVC', 'Week_diff',
       'Female', 'Male', 'Currently smokes', 'Ex-smoker', 'Never smoked']

train_data = train[col]
train_data.isnull().any()
plt.subplots(figsize=(14,10))
g = train.corr()
sns.heatmap(g, annot=True, fmt='.2', cmap="Dark2_r")
g["FVC"].sort_values(ascending=False)
stdscale = StandardScaler()
train_data[col] = stdscale.fit_transform(train_data[col])
train_data[col]
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
model_params = {
    "svr": {
        "model": SVR(gamma="auto"),
        "params": {
            "C": [1, 5, 10, 15, 20],
            "kernel":['linear', 'poly', 'rbf', 'sigmoid']
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators":[100, 200],
        }
    },
    "LR": {
        "model": LinearRegression(),
        "params": {
            
        }
    },
    "Decision Tree": {
        "model": DecisionTreeRegressor(),
        "params": {
            "splitter": ["best", "random"],
            "criterion": ["mse", "mae"],
            "max_depth": [5, 10, 15],
        }
    }
}
from sklearn.model_selection import GridSearchCV
scores = []
for model_name, param in model_params.items():
    clf = GridSearchCV(param["model"], param["params"], cv=10, return_train_score=False)
    clf.fit(train_data[col], train["FVC"])
    scores.append({
        "model": model_name,
        "best_score": clf.best_score_,
        "best_params": clf.best_params_,
    })

df = pd.DataFrame(scores, columns=["model", "best_score", "best_params"])
df
model = LinearRegression()

model.fit(train_data[col], train["FVC"])
pred = model.predict(train_data)
pred
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(train["FVC"], pred, squared=False)

print(mse)

mae = mean_absolute_error(train["FVC"], pred)
print(mae)
a = list(train["FVC"])
b = list(pred)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(15, 5))
plt.plot(b[:10], c='green', label= 'predictions')
plt.plot(a[:10], c='red', label= 'actual')
plt.legend()
submission[col].isnull().any()
sub_data = submission[col]
sub_data = stdscale.fit_transform(sub_data[col])
pred_2 = model.predict(sub_data)
a = list(submission["FVC"])
b = list(pred_2)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(15, 5))
plt.plot(b, c='green', label= 'predictions')
plt.plot(a, c='red', label= 'actual')
plt.legend()
submission["FVC_1"] = pred_2

confidence_dict={}
for id in submission['Patient'].unique():
    real=float(test[test['Patient']==id]['FVC'])
    predicted=float(submission[(submission['Patient']==id) & (submission['Weeks'].astype(int)==int(test[test['Patient']==id]['Weeks']))]['FVC_1'])
    confidence_dict[id]=abs(real-predicted)
    
    
confidence=[]
for i in range(len(submission)):
    confidence.append(confidence_dict[submission.iloc[i,0]])
submission['Confidence']=confidence
new = submission[["Patient_Week", "FVC_1", "Confidence"]]
new.rename(columns={"FVC_1":"FVC"}, inplace=True)
new.to_csv("submission.csv", index=False)
