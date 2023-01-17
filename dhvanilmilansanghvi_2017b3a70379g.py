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
df = pd.read_csv("../input/minor-project-2020/train.csv")

# print(df.head())

# print("\n###############\n")

# print(df.describe())

# print("\n###############\n")

# print(df.info())
colList = []

for c in df.columns:

    if c not in ["target", "id"]:

        colList.append(c)
df["target"].value_counts()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from scipy import stats



X = df.drop(["target", "id"], axis=1)

for col in X.columns:

    X[col] = np.cbrt(X[col])

y = df['target']
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler, PowerTransformer

import imblearn

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.over_sampling import BorderlineSMOTE

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import LinearSVC

'''

Steps to be performed



1. K-Fold split of the dataset

1. Execute models for all the datasets one by one.

2. Go over all the splits one by one.

3. OverSample the train data.

4. Train the model.

5. Predict preds

6. Get ROC_AUC 

7. Compare all scores

'''
'''

Step 1

'''



skf = StratifiedKFold(n_splits=5, random_state = 100, shuffle = True)

skf.get_n_splits(X, y)

print(skf)
'''

The models

'''

modelList = []



# model1 = LogisticRegression(solver='lbfgs', random_state = 42, class_weight = {0:10, 1:1}, fit_intercept = False, penalty = 'none')

model1 = LogisticRegression(solver='lbfgs', random_state = 42, class_weight = {0:10, 1:1})

# model1 = LinearSVC(random_state=0, tol=1e-5)

#‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’

modelList.append(model1)



# model2 = LogisticRegression(solver='newton-cg', random_state = 100)

# #‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’

# modelList.append(model2)



# model3 = LogisticRegression(solver='newton-cg', class_weight='balanced', random_state = 100)

# #‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’

# modelList.append(model3)



# model4 = LogisticRegression(solver='lbfgs', class_weight='balanced', random_state = 100)

# # #‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’

# modelList.append(model4)





# model5 = LogisticRegression(solver='newton-cg', class_weight='balanced', random_state = 100, penalty = 'l2')

# #‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’

# modelList.append(model5)



# model6 = LogisticRegression(solver='lbfgs', class_weight='balanced', penalty = 'l2', random_state = 100)

# #‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’

# modelList.append(model6)

df_test = pd.read_csv("../input/minor-project-2020/test.csv")



X_val = df_test.drop(["id"], axis=1)

for col in X_val.columns:

    X_val[col] = np.cbrt(X_val[col])

    #X[col] = np.cbrt(X[col])
i=0

# from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD, KernelPCA

# pca = PCA(n_components=70, whiten = True, random_state = 100)



skf = StratifiedKFold(n_splits=5, random_state = 100, shuffle = True)

skf.get_n_splits(X, y)

print(skf)



for model in modelList:

    scores = []

    j=0

    for train_index, test_index in skf.split(X, y):

        print("TRAIN:", train_index, "TEST:", test_index)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         X_train_pca = pca.fit_transform(X_train, y=None)

#         X_test_pca = pca.transform(X_test)

        print("Normalizing...\n")

        scalar = StandardScaler()

        scaled_X_train = scalar.fit_transform(X_train)

        scaled_X_test = scalar.transform(X_test)

        print("Oversampling...\n")

        oversample = SMOTE(sampling_strategy=1,random_state = 100)

        X_train_os = scaled_X_train.copy()

        lb_make = LabelEncoder()

        y_train_os = y_train.copy()

        y_train_os = lb_make.fit_transform(y_train_os)

        X_train_os, y_train_os = oversample.fit_resample(X_train_os, y_train_os)

        print("Training Classifier...\n")

        currModel = model

        currModel.fit(X_train_os, y_train_os)

        #preds = currModel.predict(scaled_X_test)

#         preds = currModel.predict_proba(scaled_X_test)[:,1]

        preds = currModel.predict_proba(scaled_X_test)[:,1]

#         preds2 = model_temp.predict_proba(scaled_X_val)[:,1]

#         for k in range(len(preds)):

#             preds[k] = (4*preds[k] + preds2[k])/5

#         preds = (preds + preds2)/2

        print("Classification Report: ")

        plt.style.use('seaborn-pastel')

        FPR, TPR, _ = roc_curve(y_test, preds)

        ROC_AUC = auc(FPR, TPR)

        scores.append(ROC_AUC)

        print (f"ROC_AUC : {ROC_AUC}")

        plt.figure(figsize =[11,9])

        plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

        plt.plot([0,1],[0,1], 'k--', linewidth = 4)

        plt.xlim([0.0,1.0])

        plt.ylim([0.0,1.05])

        plt.xlabel('False Positive Rate', fontsize = 18)

        plt.ylabel('True Positive Rate', fontsize = 18)

        plt.title('ROC for target', fontsize= 18)

        plt.show()

#         plot_confusion_matrix(currModel, scaled_X_test, y_test, cmap = plt.cm.Blues)

#         X_pca_val = pca.transform(X_val)

        scaled_X_val = scalar.transform(X_val)

#         preds = model.predict_proba(scaled_X_val)

        preds = model.predict_proba(scaled_X_val)[:,1]

#         preds = preds[:,1]

#         preds2 = model_temp.predict_proba(scaled_X_val)[:,1]

#         for k in range(len(preds)):

#             preds[k] = (4*preds[k] + preds2[k])/5

        idx = df_test["id"]

        data = [pd.Series(idx), pd.Series(preds)]

        headers = ["id", "target"]

        df_result = pd.concat(data, axis=1, keys=headers)

        df_result.to_csv(f"./predictions-{i}_{j}.csv", index=False)

        j+=1

    print(scores)

    scores = pd.Series(scores)

    print(scores.min(), scores.mean(), scores.max())

    i+=1
# plt.figure(figsize=(25,25))

# sns.boxplot(x="col_2", y="target", data=df)



# 0.05 : 0.95

# [0.6613236798163222, 0.6960234919640993, 0.6750790231684408, 0.6439353997077855, 0.6658012304252026]

# 0.6439353997077855 0.66843256501637 0.6960234919640993



# 0.1: 0.9

# [0.6620932477562096, 0.698391035274473, 0.6777978501356712, 0.6449743477353371, 0.6675578157322083]

# 0.6449743477353371 0.6701628593267799 0.698391035274473



# 0.5:0.5

# [0.6616248278021291, 0.7041256627008975, 0.6827345230640784, 0.6488771655186809, 0.6735225460172698]

# 0.6488771655186809 0.6741769450206112 0.7041256627008975



# 0.4:0.6

# [0.6619261218952203, 0.7035324879983302, 0.6824610102275099, 0.6481095178459612, 0.6724256527525962]

# 0.6481095178459612 0.6736909581439235 0.7035324879983302



# 0.45:0.55

# [0.6617930807764559, 0.7038764036735546, 0.6826243164266332, 0.6484890628261323, 0.672984053742798]

# 0.6484890628261323 0.6739533834891148 0.7038764036735546



# 0.6:0.4

# [0.6612845230640785, 0.7044927468169485, 0.6828358171571698, 0.6495610728449176, 0.6744637269128329]

# 0.6495610728449176 0.6745275773591894 0.7044927468169485



# 10:1

# [0.6603732936756419, 0.704759705698184, 0.6823980797328325, 0.6513299311208516, 0.6760655984070031]

# 0.6513299311208516 0.6749853217269026 0.704759705698184

    

# 100:1

# [0.6602146420371531, 0.7045417345021917, 0.6822365685660614, 0.6514959089960343, 0.6755472020775629]

# 0.6514959089960343 0.6748072112358007 0.7045417345021917





# 10:1, no intercept

# [0.6634941452723857, 0.7080656543519099, 0.686460634523064, 0.6540771237737425, 0.6790208673089506]

# 0.6540771237737425 0.6782236850460105 0.7080656543519099



# 10:1, no intercept, no penalty

# [0.6634941035274473, 0.7080657587142558, 0.6864605092882488, 0.6540769359215195, 0.679020908915462]

# 0.6540769359215195 0.6782236432733867 0.7080657587142558



# 10:1, no intercept, l1+l1/2, elasticnet

# [0.6634954602379461, 0.7080649029430182, 0.6864684825714882, 0.654086975579211, 0.6790229684377789]

# 0.654086975579211 0.6782277579538885 0.7080649029430182



# n = 20

# [0.6477858589021082, 0.6948456585264037, 0.6809311834690044, 0.666731642663327, 0.6552290574716776]

# 0.6477858589021082 0.6691046802065042 0.6948456585264037



# n=10

# [0.6431698288457525, 0.6846401899394698, 0.6834154873721561, 0.6680781047797955, 0.6573206584088975]

# 0.6431698288457525 0.6673248538692143 0.6846401899394698



# n=30

# [0.6527734815278649, 0.691587716551868, 0.6832365894385306, 0.6584526194948862, 0.6501126423486178]

# 0.6501126423486178 0.6672326098723534 0.691587716551868



# n=40

# [0.6579813504487582, 0.6894414214151533, 0.6816323940722187, 0.6531559799624296, 0.6564387667921541]

# 0.6531559799624296 0.6677299825381426 0.6894414214151533



# n=50

# [0.6623810164892507, 0.6892539866416197, 0.6873237737424338, 0.6425079941557087, 0.664561980481595]

# 0.6425079941557087 0.6692057503021216 0.6892539866416197



# n=60

# [0.6645245982049676, 0.6896285013567105, 0.6900300563556669, 0.6494378835316218, 0.6691865858360326]

# 0.6494378835316218 0.672561525057 0.6900300563556669



# n=70

# whiten = True, cbrt

# [0.6625747338760176, 0.6936005948653725, 0.6873951575871425, 0.6482078897933625, 0.6674485778363902]

# 0.6482078897933625 0.671845390791657 0.6936005948653725



# PCA, sigmoid, n=70

# [0.6530995303694427, 0.6820610415362137, 0.6779387810477979, 0.6518936965142976, 0.6728058530542477]

# 0.6518936965142976 0.6675597805043999 0.6820610415362137



# PCA, sqrt, n=70

# [0.6635144333124607, 0.6942108641202255, 0.6887774368607806, 0.6507877269881027, 0.6727892728594339]

# 0.6507877269881027 0.6740159468282008 0.6942108641202255



# sqrt

# [0.6657186704237111, 0.7029307764558548, 0.6826874347735337, 0.6498226048841578, 0.6768169287908092]

# 0.6498226048841578 0.6755952830656133 0.7029307764558548



# cbrt

# [0.6616240346482989, 0.704125704445836, 0.682734773533709, 0.6488767480692966, 0.6735222547716896]

# 0.6488767480692966 0.6741767030937661 0.704125704445836



# n=80

# [0.6599063974118139, 0.6950815800459195, 0.6837060112711333, 0.6426025255687747, 0.6669188229293234]

# 0.6426025255687747 0.6696430674453929 0.6950815800459195



# incrementalPCA, n=70

# [0.6634094239198497, 0.6933213838447088, 0.6875576915049051, 0.6460140680442497, 0.6672855843277757]

# 0.6460140680442497 0.6715176303282978 0.6933213838447088



# TSVD, n = 70

# [0.6631785535378836, 0.6930922667501566, 0.6872866624921729, 0.6483543727823001, 0.6673756832283257]

# 0.6483543727823001 0.6718575077581678 0.6930922667501566







df_l1 = df[df["target"]==1]

df_l0 = df[df["target"]==0].sample(frac = 1).iloc[0:1500,:]

df_l = df_l1.append(df_l0)

df_l.shape



X = df_l.drop(["target", "id"], axis=1)

for col in X.columns:

    X[col] = np.cbrt(X[col])

y = df_l['target'] 
df_l["target"].value_counts()
print("Looking for bad data")

for col in df.columns:

    print(f"{col} has {sum(df[col].isnull())} nan cells")

    

# NO NAN CELLS WERE FOUND
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import power_transform

# #Plotting scatter charts of the columns one by one

# for col in df.columns:

#     df[col].plot.hist()

#     plt.xlabel(col)

#     plt.show()



# fig, axs = plt.subplots(ncols=3, nrows=40, figsize=(200, 100))

# index = 0

# axs = axs.flatten()

# for k,v in df.items():

#     sns.distplot(v, ax=axs[index])

#     index += 1

#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

#X = power_transform(X, method='yeo-johnson')

for col in pd.DataFrame(X).columns:

    fig, ax = plt.subplots()

    num_bins = 25

    series = df[col]

    n, bins, patches = ax.hist(series, num_bins, density=True)



    sigma = series.std()

    mu = series.mean()

    

    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *

         np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

    ax.plot(bins, y, '--')

    ax.set_xlabel('Smarts')

    ax.set_ylabel('Probability density')

    ax.set_title(f'{col}')



    # Tweak spacing to prevent clipping of ylabel

    fig.tight_layout()

    plt.show()
Xtemp = df.drop(["target", "id"], axis=1)

X_pca = pd.DataFrame(X_pca, columns = Xtemp.columns)
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components = 3, verbose = 1, random_state = 100).fit_transform(X)

X_embedded.shape
# using Random Forest Classfier

from sklearn.ensemble import RandomForestClassifier



X_train, X_test, y_train, y_test = train_test_split(X_changed, y_changed, test_size=0.2, random_state=1, stratify = y_changed)



rf = RandomForestClassifier(random_state=12)

model = rf.fit(X_train, y_train)
from sklearn.decomposition import PCA

X_pca = PCA(n_components=20).fit_transform(X, y=None)