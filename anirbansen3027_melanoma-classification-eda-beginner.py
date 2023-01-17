import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import cv2



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler
train_data = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")

test_data = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")

TRAIN_IMAGES_DIR = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"

TEST_IMAGES_DIR = "/kaggle/input/siim-isic-melanoma-classification/jpeg/test/"

submission_file = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")
train_data.shape,test_data.shape
train_data.head()
test_data.head()
train_data.isna().sum()/train_data.shape[0]
test_data.isna().sum()
train_data.nunique()
test_data.nunique()
test_data[test_data.patient_id.isin([train_data.patient_id.unique])].shape
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

sns.distplot(train_data.groupby("patient_id")["image_name"].nunique(),kde = False,ax=ax1)

sns.distplot(test_data.groupby("patient_id")["image_name"].nunique(),kde = False,ax=ax2)

ax1.set_title("Train data")

ax2.set_title("Test data")

plt.suptitle("",fontweight = "bold")

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

pd.value_counts(train_data['sex']).plot(kind = 'pie', ax=ax1,autopct='%1.1f%%')

pd.value_counts(test_data['sex']).plot(kind ='pie', ax=ax2,autopct='%1.1f%%')

ax1.set_title("Train Data")

ax2.set_title("Test Data")

plt.suptitle("",fontweight = "bold")

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

sns.kdeplot(train_data[train_data.sex=="male"].age_approx, shade=True,color = "g", ax= ax1)

sns.kdeplot(test_data[test_data.sex=="female"].age_approx, shade=True,color = "r", ax= ax1)

sns.kdeplot(train_data[train_data.sex=="male"].age_approx, shade=True,color = "g", ax= ax2)

sns.kdeplot(test_data[test_data.sex=="female"].age_approx, shade=True,color = "r", ax= ax2)

ax1.set_title("train_data")

ax2.set_title("test_data")

ax1.legend(['male','female'])

ax2.legend(['male','female'])

plt.suptitle("",fontweight = "bold")

plt.show()
# train_data.anatom_site_general_challenge.value_counts()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

pd.value_counts(train_data.anatom_site_general_challenge).plot(kind = 'bar', ax=ax1)

pd.value_counts(test_data.anatom_site_general_challenge).plot(kind ='bar', ax=ax2)

ax1.set_title("Train Data")

ax2.set_title("Test Data")

plt.suptitle("",fontweight = "bold")

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

pd.value_counts(train_data.benign_malignant).plot(kind = 'bar', ax=ax1)

pd.value_counts(train_data.target).plot(kind ='bar', ax=ax2)

ax1.set_title("Train Data")

ax2.set_title("Test Data")

plt.suptitle("",fontweight = "bold")

plt.show()
train_data[(train_data.benign_malignant == "benign") & (train_data.target != 0)]

train_data[(train_data.benign_malignant == "malignant") & (train_data.target != 1)]
pd.value_counts(train_data['diagnosis']).plot(kind = 'pie',autopct='%1.1f%%')

ax1.set_title("Test Data")

plt.show()
train_data.corr()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

pd.value_counts(train_data.loc[train_data['sex']=='male',['target','image_name']]["target"]).plot(kind = 'pie', ax=ax1,autopct='%1.1f%%')

pd.value_counts(train_data.loc[train_data['sex']=='female',['target','image_name']]["target"]).plot(kind ='pie', ax=ax2,autopct='%1.1f%%')

ax1.set_title("Male")

ax2.set_title("Female")

plt.suptitle("",fontweight = "bold")

plt.show()
train_data.anatom_site_general_challenge.value_counts(dropna=False)
# anatom_site_general_challenge target

fig, ax = plt.subplots(2,3 , figsize=(16, 8))

pd.value_counts(train_data.loc[train_data['anatom_site_general_challenge']=='torso',['target','image_name']]["target"]).plot(kind = 'pie', ax=ax[0][0],autopct='%1.1f%%')

pd.value_counts(train_data.loc[train_data['anatom_site_general_challenge']=='lower extremity',['target','image_name']]["target"]).plot(kind ='pie', ax=ax[0][1],autopct='%1.1f%%')

pd.value_counts(train_data.loc[train_data['anatom_site_general_challenge']=='upper extremity',['target','image_name']]["target"]).plot(kind = 'pie', ax=ax[0][2],autopct='%1.1f%%')

pd.value_counts(train_data.loc[train_data['anatom_site_general_challenge']=='head/neck',['target','image_name']]["target"]).plot(kind ='pie', ax=ax[1][0],autopct='%1.1f%%')

pd.value_counts(train_data.loc[train_data['anatom_site_general_challenge']=='palms/soles',['target','image_name']]["target"]).plot(kind = 'pie', ax=ax[1][1],autopct='%1.1f%%')

pd.value_counts(train_data.loc[train_data['anatom_site_general_challenge']=='oral/genital',['target','image_name']]["target"]).plot(kind ='pie', ax=ax[1][2],autopct='%1.1f%%')

ax[0][0].set_title("torso")

ax[0][1].set_title("lower extremity")

ax[0][2].set_title("upper extremity")

ax[1][0].set_title("head/neck")

ax[1][1].set_title("palms/soles")

ax[1][2].set_title("oral/genital")

plt.suptitle("",fontweight = "bold")

plt.show()

# train_data.loc[train_data['target']==0,['sex']]
train_data_baseline = train_data.copy()

test_data_baseline = test_data.copy()

train_data_baseline = train_data_baseline.fillna(train_data_baseline.mode().iloc[0])

test_data_baseline = test_data_baseline.fillna(train_data_baseline.mode().iloc[0])

train_data_baseline["male"] = np.where(train_data_baseline["sex"] == "male", 1,0)

test_data_baseline["male"] = np.where(test_data_baseline["sex"] == "male", 1,0)

train_data_baseline = train_data_baseline.join(pd.get_dummies(train_data_baseline["anatom_site_general_challenge"],drop_first = True))

test_data_baseline = test_data_baseline.join(pd.get_dummies(test_data_baseline["anatom_site_general_challenge"],drop_first = True))

scaler = MinMaxScaler()

train_data_baseline[['age_approx']] = scaler.fit_transform(train_data_baseline[['age_approx']])

test_data_baseline[['age_approx']] = scaler.fit_transform(test_data_baseline[['age_approx']])

X = train_data_baseline.drop(['image_name','patient_id','sex','anatom_site_general_challenge','diagnosis','benign_malignant','target'],axis = 1)

X_pred = test_data_baseline.drop(['image_name','patient_id','sex','anatom_site_general_challenge'],axis = 1)

y = train_data_baseline[['target']]
X.head()
#Baseline Model

model=LogisticRegression()

skf = StratifiedKFold(shuffle=True,random_state =42)

error = []

for train_index, test_index in skf.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train.values.ravel())

    error.append(metrics.roc_auc_score(y_test, model.predict(X_test)))

    # printing the score 

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
submission_file['target']= model.predict(X_pred)

submission_file.to_csv('submission_file.csv',index = False)
def plot_images(image_list,rows,cols,title):

    fig,ax = plt.subplots(rows,cols,figsize = (25,5))

    ax = ax.flatten()

    for i, image_id in enumerate(image_list):

        image = cv2.imread(TRAIN_IMAGES_DIR+'{}.jpg'.format(image_id))

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        ax[i].imshow(image)

        ax[i].set_axis_off()

        ax[i].set_title(image_id)

    plt.suptitle(title)
plot_images(train_data[train_data.target == 0].sample(5)["image_name"].values,1,5,"Benign")
plot_images(train_data[train_data.target == 1].sample(5)["image_name"].values,1,5,"Malignant")