import pandas as pd

import numpy as np

import os

import cv2

from collections import Counter, defaultdict

import random

from datetime import date



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.figure_factory as ff



from kaggle_datasets import KaggleDatasets



import tensorflow as tf

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder



SEED = 2020

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

os.environ['PYTHONHASHSEED'] = str(SEED)



# from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"
DATA_PATH = '/kaggle/input/siim-isic-melanoma-classification/'

os.listdir(DATA_PATH)
trainMeta = pd.read_csv(DATA_PATH + 'train.csv')

testMeta = pd.read_csv(DATA_PATH + 'test.csv')

sampleSubmission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
trainMeta.head()
trainMeta.tail()
print("Train data shape: ",trainMeta.shape)
trainMeta.describe()
# missing values in the train-dataset

trainMeta.info()
print("Number of unique patients in train-data: ",trainMeta.patient_id.nunique())

print("Average number of images per patient in train-data: ",trainMeta.image_name.nunique()/trainMeta.patient_id.nunique())
testMeta.head()
testMeta.tail()
print("Test data shape: ",testMeta.shape)
# missing values in the test-dataset

testMeta.info()
print("Number of unique patients in test-data: ",testMeta.patient_id.nunique())

print("Average number of images per patient in test-data: ",testMeta.image_name.nunique()/testMeta.patient_id.nunique())
def getPiechartDistribution(feature):

    fig = go.Figure(data=[go.Pie(labels=feature.value_counts().index.values,

                             values=feature.value_counts().values)])



    fig = fig.update_traces(hoverinfo='label+percent',

                      textinfo='value',

                      textfont_size=20,

                      marker=dict(line=dict(color='#000000', width=1)))



    return fig
# target distribution



getPiechartDistribution(trainMeta.benign_malignant).update_layout(title_text="Target Distribution of the Train-data")
print("Patients with missing value of sex: ")

for id in trainMeta[trainMeta.sex.isna()].patient_id.unique():

    print(id)

#     print(id in trainMeta[trainMeta.sex.notna()].patient_id.unique())

    

# patients with missing values are not in patients with not-null sex value
# lets check if there is any patient with more than one sex



if len(np.unique(list(map(len,trainMeta.groupby(['patient_id'])['sex'].unique().values)))) == 1:

    print("There are no patients with more than one sex")

else:

    print("There are patient with more than one sex")
getPiechartDistribution(trainMeta.groupby(['patient_id'])['sex'].first().fillna("NA")).update_layout(title_text="Distribution of sex feature in train data")
getPiechartDistribution(testMeta.groupby(['patient_id'])['sex'].first().fillna("NA")).update_layout(title_text="Distribution of sex feature in test data")
fig = px.histogram(trainMeta.fillna("NA"), x="sex", y="target",color='benign_malignant',barmode="group",title="Distribution of sex wrt to target")

fig.show()
trainMeta.fillna("NA").groupby(['sex','anatom_site_general_challenge'])['target'].aggregate(['sum','count','mean']).reset_index().style.background_gradient(cmap='Reds')
hist_data = [trainMeta.age_approx.fillna(0).values, testMeta.age_approx.fillna(0).values]

group_labels = ['train-age','test-age']



fig = ff.create_distplot(hist_data, group_labels, bin_size=5.).update_layout(title='Train & Test Age distribution')

fig.show()
hist_data = [trainMeta[trainMeta.target==1].age_approx.fillna(0).values, trainMeta[trainMeta.target==0].age_approx.fillna(0).values]

group_labels = ['Malignant','Benign']



fig = ff.create_distplot(hist_data, group_labels, bin_size=5.,colors=['rgb(200,0,0)','rgb(0,200,0)']).update_layout(title='Distribution of age wrt target')

fig.show()
fig = px.box(trainMeta.fillna(-1),x='sex',y='age_approx',color='target',title="Distribution of age wrt sex")



fig.show()
trainMeta[trainMeta.age_approx>=60].fillna("NA").groupby(['age_approx','anatom_site_general_challenge'])['target'].aggregate(['sum','count','mean']).sort_values(by='mean',ascending=False).reset_index().style.background_gradient(cmap='Reds') 
fig = px.histogram(trainMeta.fillna("NA"), x="anatom_site_general_challenge", y="benign_malignant",color='benign_malignant',barmode="group",title="Distribution of Anatom-site wrt to target")

fig.show()
fig = px.histogram(testMeta.fillna("NA"), x="anatom_site_general_challenge", y="anatom_site_general_challenge",barmode="group",title="Distribution of Anatom-site in the Test Data")

fig.show()
fig = px.histogram(trainMeta.fillna("NA"), x="diagnosis", y="target",color='benign_malignant',barmode="group",title="Distribution of diagnosis wrt to target")

fig.show()
# there is no overlapping of patients between train & test set



set(trainMeta.patient_id.unique()).intersection(set(testMeta.patient_id.unique()))
hist_data = [trainMeta[trainMeta.target==1].groupby('patient_id')['image_name'].count().values, trainMeta[trainMeta.target==0].groupby('patient_id')['image_name'].count().values]

group_labels = ['Malignant','Benign']



fig = ff.create_distplot(hist_data, group_labels, bin_size=1.,colors=['rgb(200,0,0)','rgb(0,200,0)']).update_layout(title='Distribution of images/patient in Train-data wrt target')

fig.show()
hist_data = [testMeta.groupby('patient_id')['image_name'].count().values]



fig = ff.create_distplot(hist_data, bin_size=1.,group_labels=['test-data']).update_layout(title='Distribution of images/patient in Test-data')

fig.show()
hist_data = [trainMeta[trainMeta.target==1].fillna(0).groupby(['patient_id'])['age_approx'].max().values - trainMeta[trainMeta.target==1].fillna(0).groupby(['patient_id'])['age_approx'].min().values, trainMeta[trainMeta.target==0].fillna(0).groupby(['patient_id'])['age_approx'].max().values - trainMeta[trainMeta.target==0].fillna(0).groupby(['patient_id'])['age_approx'].min().values]

group_labels = ['Malignant','Benign']



fig = ff.create_distplot(hist_data, group_labels, bin_size=1.,colors=['rgb(200,0,0)','rgb(0,200,0)']).update_layout(title='Distribution of age-diff of patients wrt to target in Train-data')

fig.show()
hist_data = [testMeta.fillna(0).groupby(['patient_id'])['age_approx'].max().values - testMeta.fillna(0).groupby(['patient_id'])['age_approx'].min().values]



fig = ff.create_distplot(hist_data, bin_size=1.,group_labels=['test-data']).update_layout(title='Distribution of age-diff of patients in Test-data')

fig.show()
for patient_id in np.random.choice(trainMeta[trainMeta.target==1].fillna("NA").patient_id.unique(),size=1,replace=False):

    x = trainMeta[trainMeta.patient_id == patient_id].sort_values(['age_approx','image_name'])

    

    r, c = int(np.ceil(x.shape[0]/5)), 5

    

    fig, ax = plt.subplots(r,c, figsize=(20,4*r))

    

    fig = fig.suptitle(f'{patient_id} Sex: {x.sex.values[0]}',fontsize=20)

    

    for i, image_name in enumerate(x.image_name.values):

        img = cv2.imread(DATA_PATH + f'jpeg/train/{image_name}.jpg')

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        

        if x.benign_malignant.values[i] == "malignant":

            color = "red"

        else:

            color = "black"

        

        if r>1:

            ax[i//5,i%5].imshow(img)

            ax[i//5,i%5].set_title(f"{x.age_approx.values[i]} {x.benign_malignant.values[i]} {x.anatom_site_general_challenge.values[i]}",color=color)

        else:

            ax[i%5].imshow(img)

            ax[i%5].set_title(f"{x.age_approx.values[i]} {x.benign_malignant.values[i]} {x.anatom_site_general_challenge.values[i]}",color=color)

    

    plt.savefig(f'{x.target.sum()}_{patient_id}_{x.sex.values[0]}.png')



plt.show()
for patient_id in np.random.choice(trainMeta[trainMeta.target==0].fillna("NA").patient_id.unique(),size=1,replace=False):

    x = trainMeta[trainMeta.patient_id == patient_id].sort_values(['age_approx','image_name'])

    

    r, c = int(np.ceil(x.shape[0]/5)), 5

    

    fig, ax = plt.subplots(r,c, figsize=(20,4*r))

    

    fig = fig.suptitle(f'{patient_id} Sex: {x.sex.values[0]}',fontsize=20)

    

    for i, image_name in enumerate(x.image_name.values):

        img = cv2.imread(DATA_PATH + f'jpeg/train/{image_name}.jpg')

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        

        if x.benign_malignant.values[i] == "malignant":

            color = "red"

        else:

            color = "black"

        

        if r>1:

            ax[i//5,i%5].imshow(img)

            ax[i//5,i%5].set_title(f"{x.age_approx.values[i]} {x.benign_malignant.values[i]} {x.anatom_site_general_challenge.values[i]}",color=color)

        else:

            ax[i%5].imshow(img)

            ax[i%5].set_title(f"{x.age_approx.values[i]} {x.benign_malignant.values[i]} {x.anatom_site_general_challenge.values[i]}",color=color)

    

    plt.savefig(f'{x.target.sum()}_{patient_id}_{x.sex.values[0]}.png')



plt.show()
for patient_id in np.random.choice(testMeta.fillna("NA").patient_id.unique(),size=1,replace=False):

    x = testMeta[testMeta.patient_id == patient_id].sort_values(['age_approx','image_name'])

    

    r, c = int(np.ceil(x.shape[0]/5)), 5

    

    fig, ax = plt.subplots(r,c, figsize=(20,4*r))

    

    fig = fig.suptitle(f'{patient_id} Sex: {x.sex.values[0]}',fontsize=20)

    

    for i, image_name in enumerate(x.image_name.values):

        img = cv2.imread(DATA_PATH + f'jpeg/test/{image_name}.jpg')

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        

        if r>1:

            ax[i//5,i%5].imshow(img)

            ax[i//5,i%5].set_title(f"{x.age_approx.values[i]} {x.anatom_site_general_challenge.values[i]} {image_name}")

        else:

            ax[i%5].imshow(img)

            ax[i%5].set_title(f"{x.age_approx.values[i]} {x.anatom_site_general_challenge.values[i]} {image_name}")

    

#     plt.savefig(f'{x.target.sum()}_{patient_id}_{x.sex.values[0]}.png')



plt.show()
image_folder_path = '../input/siim-isic-melanoma-classification/jpeg/train/'

imageName = np.random.choice(trainMeta[trainMeta.target==1].image_name.values)

sampleImage = cv2.imread(os.path.join(image_folder_path, f'{imageName}.jpg'))[:,:,::-1]

plt.title(f'{imageName} - {trainMeta[trainMeta.image_name==imageName].target.values}')

plt.imshow(sampleImage)
transforms = ['Identity','RandomBrightness','RandomContrast','Crop','FlipLeftRight','FlipUpDown','RandomSaturation','Rot90','Rot180','Rot270']



def randAugment(image=sampleImage,N=3):

    

    augmentations = np.random.choice(transforms,N,replace=False)

    

    image = tf.cast(image, tf.float32) / 255.0

    

    for transform in augmentations:



        if transform=='Identity':

            continue



        elif transform=='RandomBrightness':

            image = tf.image.random_brightness(image,max_delta=0.2)

        

        elif transform=='RandomContrast':

            image = tf.image.random_contrast(image,1.0,3.0)

        

        elif transform=='Crop':

#             image = tf.image.random_crop(image,[512,512,3])

            image = tf.image.central_crop(image,0.5)

    

        elif transform=='FlipLeftRight':

            image = tf.image.flip_left_right(image)

            

        elif transform=='FlipUpDown':

            image = tf.image.flip_up_down(image)

        

        elif transform=='RandomSaturation':

            image = tf.image.random_saturation(image,0.6,1.5)

        

        elif transform=='Rot90':

            image = tf.image.rot90(image,k=1)

        

        elif transform=='Rot180':

            image = tf.image.rot90(image,k=2)

            

        elif transform=='Rot270':

            image = tf.image.rot90(image,k=3)

        

    image = tf.image.resize(image,(450,600))

#     image = cv2.resize(image.numpy(),(600,450))

#     print(np.all(image1.numpy()==image1))

        

    return image, augmentations
r, c = 3, 5



fig, ax = plt.subplots(r,c, figsize=(25,5*r))



fig = fig.suptitle(f"Image Augmentation of {imageName} - {trainMeta[trainMeta.image_name==imageName].target.values}",fontsize=20)



for i in range(r*c):

    img, augmentations = randAugment(sampleImage,5)



    ax[i//5,i%5].imshow(img)

    ax[i//5,i%5].set_title("-".join(augmentations[:2]) + "\n" + "-".join(augmentations[2:]))



#     plt.savefig(f'{x.target.sum()}_{patient_id}_{x.sex.values[0]}.png')



plt.show()
def stratified_group_k_fold(X, y, groups, k, seed=None):

    labels_num = np.max(y) + 1

    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))

    y_distr = Counter()

    for label, g in zip(y, groups):

        y_counts_per_group[g][label] += 1

        y_distr[label] += 1



    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))

    groups_per_fold = defaultdict(set)



    def eval_y_counts_per_fold(y_counts, fold):

        y_counts_per_fold[fold] += y_counts

        std_per_label = []

        for label in range(labels_num):

            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])

            std_per_label.append(label_std)

        y_counts_per_fold[fold] -= y_counts

        return np.mean(std_per_label)

    

    groups_and_y_counts = list(y_counts_per_group.items())

    random.Random(seed).shuffle(groups_and_y_counts)



    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):

        best_fold = None

        min_eval = None

        for i in range(k):

            fold_eval = eval_y_counts_per_fold(y_counts, i)

            if min_eval is None or fold_eval < min_eval:

                min_eval = fold_eval

                best_fold = i

        y_counts_per_fold[best_fold] += y_counts

        groups_per_fold[best_fold].add(g)



    all_groups = set(groups)

    

    for i in range(k):

        train_groups = all_groups - groups_per_fold[i]

        test_groups = groups_per_fold[i]



        train_indices = [i for i, g in enumerate(groups) if g in train_groups]

        test_indices = [i for i, g in enumerate(groups) if g in test_groups]



        yield train_indices, test_indices
def get_stratify_group(row):

    stratify_group = row['sex']

#     stratify_group += f'_{row["age_approx"]}'

    stratify_group += f'_{row["anatom_site_general_challenge"]}'

    stratify_group += f'_{row["target"]}'

    return stratify_group
%%time



train = trainMeta.copy()

train['stratify_group'] = train.fillna("NA").apply(get_stratify_group, axis=1)

train['stratify_group'] = train['stratify_group'].astype('category').cat.codes



train['fold'] = 0



k = 5

for fold_ind, (train_ind, val_ind) in enumerate(stratified_group_k_fold(trainMeta, train.stratify_group.values, trainMeta.patient_id.values, k=k, seed=SEED)):

    train.loc[val_ind,'fold'] = fold_ind



train.fold.value_counts(normalize=True)
for i in range(5):

    for j in range(i+1,5):

        print(f"fold_{i} intersection fold_{j}: {set(train[train.fold==i].patient_id).intersection(set(train[train.fold==j].patient_id))}")
fig = px.histogram(train.fillna("NA"), x="benign_malignant", y="target",color='fold',barmode="group",title="Distribution of Targets wrt to Folds")

fig.show()
fig = px.histogram(train.fillna("NA"), x="anatom_site_general_challenge", y="anatom_site_general_challenge",color='fold',barmode="group",title="Distribution of Anatom-site wrt folds")

fig.show()
fig = px.histogram(train.fillna("NA"), x="sex", y="sex",color='fold',barmode="group",title="Distribution of Sex wrt folds")

fig.show()
fig = px.histogram(train.fillna("NA"), x="age_approx", y="age_approx",color='fold',barmode="group",title="Distribution of Age wrt folds")

fig.show()
train.head()
# train.to_csv('train_StratifiedGroupK(5)Fold(SEED2020)(Group_sex_anatomsite_target).csv',index=False)
y = train.target

folds = train.fold

X = train.drop(['target','benign_malignant','diagnosis','stratify_group','fold','image_name'],axis=1)

X.head()
X.sex = X.sex.fillna("unknown")

X.anatom_site_general_challenge = X.anatom_site_general_challenge.fillna("unknown")

X.age_approx = X.age_approx.fillna(0)



X_test = testMeta.copy()

X_test.anatom_site_general_challenge = X_test.anatom_site_general_challenge.fillna("unknown")
def labelEncoder(train,val,test,columns):

    for col in columns:

        le = LabelEncoder()    

        train[f'le_{col}'] = le.fit_transform(train[col])

        val[f'le_{col}'] = le.transform(val[col])

        test[f'le_{col}'] = le.transform(test[col])

    

    return train,val,test



def oneHotEncode(train,val,test,cols):

    train['temp'] = 0

    val['temp'] = 1

    test['temp'] = 2

    

    temp = pd.get_dummies(pd.concat([train,val,test],axis=0),columns=cols,drop_first=True)

    

    train = temp[temp.temp==0]

    val = temp[temp.temp==1]

    test = temp[temp.temp==2]

    

    train.drop(['temp'],inplace=True,axis=1)

    val.drop(['temp'],inplace=True,axis=1)

    test.drop(['temp'],inplace=True,axis=1)

    

    return train, val, test





def standardScale(X_train,X_val,test,cols):

    

    for col in cols:

        ss = StandardScaler()

        X_train[f'std_{col}'] = ss.fit_transform(X_train[col].values.reshape(-1,1))

        X_val[f'std_{col}'] = ss.transform(X_val[col].values.reshape(-1,1))

        test[f'std_{col}'] = ss.transform(test[col].values.reshape(-1,1))



    return X_train, X_val, test





def targetEncode(X_train,y_train,X_val,X_test,cols):

    

    X = pd.concat([X_train,y_train],axis=1)

    

    alpha = 15

    global_mean = y_train.mean()

    

    for col in cols:

        encodings = dict((X.groupby([col])['target'].sum() + alpha*global_mean)/(alpha + X.groupby([col])['target'].count()))

        X_train[f'te_{col}'] = X_train[col].map(encodings).fillna(global_mean)

        X_val[f'te_{col}'] = X_val[col].map(encodings).fillna(global_mean)

        X_test[f'te_{col}'] = X_test[col].map(encodings).fillna(global_mean)

    

    return X_train, X_val, X_test





def featureInteractions(X_train,X_val,X_test,cols):

    

    for i in range(len(cols)):

        for j in range(i+1,len(cols)):

            X_train[f'{cols[i]}_{cols[j]}'] = f'{X_train[cols[i]]}_{X_train[cols[j]]}'

            X_val[f'{cols[i]}_{cols[j]}'] = f'{X_val[cols[i]]}_{X_val[cols[j]]}'

            X_test[f'{cols[i]}_{cols[j]}'] = f'{X_test[cols[i]]}_{X_test[cols[j]]}'

    

    return X_train, X_val, X_test

    

    

    

def preprocessData(X_train,y_train,X_val,X_test):

    

    data = [X_train,X_val,X_test]

    

    # Sun-Exposed or not feature

#     for X in data:

#         X['sun_exposed'] = X.anatom_site_general_challenge.map({'torso':1,'lower extremity':2,'upper extremity':2,'head/neck':3,'unknown':0,'palms/soles':0,'oral/genital':0})

    

    # Feature Interactions

#     X_train, X_val, X_test = featureInteractions(X_train,X_val,X_test,['sex','age_approx','anatom_site_general_challenge'])

#     X_train["_".join(['sex','age_approx','anatom_site_general_challenge'])] = f'{X_train["sex"]}_{X_train["age_approx"]}_{X_train["anatom_site_general_challenge"]}'

#     X_val["_".join(['sex','age_approx','anatom_site_general_challenge'])] = f'{X_val["sex"]}_{X_val["age_approx"]}_{X_val["anatom_site_general_challenge"]}'

#     X_test["_".join(['sex','age_approx','anatom_site_general_challenge'])] = f'{X_test["sex"]}_{X_test["age_approx"]}_{X_test["anatom_site_general_challenge"]}'



    

    

    X_train, X_val, X_test = standardScale(X_train,X_val,X_test,['age_approx'])

    

    

#     X_train,X_val,X_test = labelEncoder(X_train,X_val,X_test,['sex','age_approx','anatom_site_general_challenge'])

#     X_train, X_val, X_test = targetEncode(X_train,y_train,X_val,X_test,

#                                           [col for col in ['sex','age_approx','anatom_site_general_challenge','sex_age_approx','sex_anatom_site_general_challenge','age_approx_anatom_site_general_challenge',"_".join(['sex','age_approx','anatom_site_general_challenge'])] if col in X_train.columns])

    X_train, X_val, X_test = oneHotEncode(X_train,X_val,X_test,cols=[col for col in ['anatom_site_general_challenge','sex','sex_age_approx','sex_anatom_site_general_challenge','age_approx_anatom_site_general_challenge'] if col in X_train.columns])

    

    

    # Drop unwanted columns

    dropCols = list(X_train.dtypes[X_train.dtypes=='object'].index.values) + ['age_approx']

    X_train = X_train.drop(dropCols,axis=1)

    X_val = X_val.drop(dropCols,axis=1)

    X_test = X_test.drop(dropCols,axis=1)

        

#     print(X_train.columns)

    

    return X_train, X_val, X_test
def trainOnMetaData(X,y,X_test,folds):

    

    CVScores = []

    

    valPred = y.copy()

    

    testPred = {}

    

    for fold in range(k):

        val_idx = X[folds==fold].index

        train_idx = X[folds!=fold].index

        

        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

        test = X_test[X_train.columns].copy()

        

        # Preprocessing

        X_train, X_val, test = preprocessData(X_train,y_train,X_val,test)

            

        model = LogisticRegression(n_jobs=-1,random_state=SEED,max_iter=100)

        

        model.fit(X_train,y_train)

        

        print(f'\nFold {fold}: ')

        print('----------------')

        

        valPred.iloc[val_idx] = model.predict_proba(X_val)[:,1]

        

        valScore = roc_auc_score(y_val,valPred.iloc[val_idx])

        print("Validation Score: ",valScore)

        CVScores.append(valScore)

        

        print("\nCoeff: ",dict(zip(X_train.columns,model.coef_[0])))

#         print("\nFeature Importance: ",dict(zip(X_train.columns,model.feature_importances_)))

        

        testPred[f'fold_{fold}'] = model.predict_proba(test)[:,1]

    

    print(f"\nMean CV Score: {np.mean(CVScores)} +/- {np.std(CVScores)}")

    

    return CVScores, valPred, pd.DataFrame(testPred)
CVScores, valPred, testPred = trainOnMetaData(X,y,X_test,folds)
def visualizeResults(CVScores,valPred,testPred):

    

    fig, ax = plt.subplots(1,3,figsize=(18,5))

    fig.suptitle("Results",fontsize=20)

    

    sns.barplot(x=list(range(k)),y=CVScores, ax=ax[0])

    ax[0].set_title(f"CV-Scores of {k}-Folds")

    

    sns.kdeplot(testPred.mean(axis=1),shade=True,ax=ax[1])

    ax[1].set_title("Distribution of Testset Predictions")

    

    sns.kdeplot(valPred[y==0],label='benign',shade=True,ax=ax[2])

    sns.kdeplot(valPred[y==1],label='malignant',shade=True,ax=ax[2])

    ax[2].set_title('Distribution of Cross-Validation set Predictions')
visualizeResults(CVScores,valPred,testPred)
hist_data = [valPred[folds==fold] for fold in range(k)]

group_labels = [f'fold_{fold}' for fold in range(k)]



fig = ff.create_distplot(hist_data, group_labels, show_hist=False).update_layout(title='Distribution of Cross Validation Set Predictions wrt of folds')

fig.show()
def saveResults(valPred,testPred,CVScores,modelName):

    

    sampleSubmission.iloc[:,1] = testPred.mean(axis=1)

    sampleSubmission.to_csv(f'{date.today()}_Test_{modelName}_{np.mean(CVScores)}.csv',index=False)

    

    val = pd.DataFrame(train['image_name']) 

    val['target'] = valPred



    val.to_csv(f'{date.today()}_Val_{modelName}_{np.mean(CVScores)}.csv',index=False)
saveResults(valPred,testPred,CVScores,'LogisticRegression')

os.listdir('/kaggle/working/')