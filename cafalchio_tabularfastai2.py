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
!pip install fastai2 fastdot -q

# from fast_tabnet.core import *

# from fast_tabnet.core import TabNetNoEmbeddings

from fastai2.tabular.all import *

# import pandas as pd

# import numpy as np
path = '../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx'

df = pd.read_excel(path)
dfnum = df.select_dtypes(include=['float64']).astype('float32')
df.head(15).T
cat_name = ['WINDOW', 'AGE_ABOVE65', 'GENDER', 'AGE_PERCENTIL', 'DISEASE GROUPING 1', 'DISEASE GROUPING 2', 'DISEASE GROUPING 3',

             'DISEASE GROUPING 4', 'DISEASE GROUPING 5', 'DISEASE GROUPING 6', 'IMMUNOCOMPROMISED', 'HTN']

cont_name = [ "ALBUMIN_MEDIAN", "ALBUMIN_MEAN", "ALBUMIN_MIN",

               "ALBUMIN_MAX", "BE_ARTERIAL_MEDIAN", "BE_ARTERIAL_MEAN",

               "BE_ARTERIAL_MIN", "BE_ARTERIAL_MAX", "BE_VENOUS_MEDIAN",

               "BE_VENOUS_MEAN", "BE_VENOUS_MIN", "BE_VENOUS_MAX",

               "HEMATOCRITE_MEDIAN", "HEMATOCRITE_MEAN", "HEMATOCRITE_MIN",

               "HEMATOCRITE_MAX", "HEMOGLOBIN_MEDIAN", "HEMOGLOBIN_MEAN",

               "HEMOGLOBIN_MIN", "HEMOGLOBIN_MAX", "LACTATE_MEDIAN", "LACTATE_MEAN",

               "LACTATE_MIN", "LACTATE_MAX", "LEUKOCYTES_MEDIAN", "LEUKOCYTES_MEAN",

               "LEUKOCYTES_MIN", "LEUKOCYTES_MAX", "NEUTROPHILES_MEDIAN",

               "NEUTROPHILES_MEAN", "NEUTROPHILES_MIN", "NEUTROPHILES_MAX",

               "UREA_MEDIAN", "UREA_MEAN", "UREA_MIN", "UREA_MAX",

               "BLOODPRESSURE_DIASTOLIC_MEAN", "RESPIRATORY_RATE_MEAN",

               "BLOODPRESSURE_DIASTOLIC_MEDIAN", "RESPIRATORY_RATE_MEDIAN",

               "BLOODPRESSURE_DIASTOLIC_MIN", "HEART_RATE_MIN", "TEMPERATURE_MIN",

               "OXYGEN_SATURATION_MIN", "BLOODPRESSURE_SISTOLIC_MAX", "HEART_RATE_MAX",

               "RESPIRATORY_RATE_MAX", "OXYGEN_SATURATION_MAX",

               "BLOODPRESSURE_DIASTOLIC_DIFF", "BLOODPRESSURE_SISTOLIC_DIFF",

               "HEART_RATE_DIFF", "RESPIRATORY_RATE_DIFF", "TEMPERATURE_DIFF",

               "OXYGEN_SATURATION_DIFF", "BLOODPRESSURE_DIASTOLIC_DIFF_REL",

               "BLOODPRESSURE_SISTOLIC_DIFF_REL", "HEART_RATE_DIFF_REL",

               "RESPIRATORY_RATE_DIFF_REL", "TEMPERATURE_DIFF_REL",

               "OXYGEN_SATURATION_DIFF_REL"]



procs = [FillMissing, Categorify, Normalize] 





msk = np.random.rand(len(df)) < 0.7



train = df[msk] # Train df 70%

test = df[~msk] # Test df  30%

ln = len(train)

idxs = []

for i in range(2):

    np.random.seed(i+4)

    valid_idx = np.random.choice(ln, int(ln*0.2), replace=False)

    idxs.append(valid_idx)
preds = []

for idx in idxs:

    cat_name = ['WINDOW', 'AGE_ABOVE65', 'GENDER', 'AGE_PERCENTIL']#, 'DISEASE GROUPING 1', 'DISEASE GROUPING 2', 'DISEASE GROUPING 3',

#              'DISEASE GROUPING 4', 'DISEASE GROUPING 5', 'DISEASE GROUPING 6', 'IMMUNOCOMPROMISED', 'HTN']

    cont_name = [ "ALBUMIN_MEDIAN", "ALBUMIN_MEAN", "ALBUMIN_MIN",

               "ALBUMIN_MAX", "BE_ARTERIAL_MEDIAN", "BE_ARTERIAL_MEAN",

               "BE_ARTERIAL_MIN", "BE_ARTERIAL_MAX", "BE_VENOUS_MEDIAN",

               "BE_VENOUS_MEAN", "BE_VENOUS_MIN", "BE_VENOUS_MAX",

               "HEMATOCRITE_MEDIAN", "HEMATOCRITE_MEAN", "HEMATOCRITE_MIN",

               "HEMATOCRITE_MAX", "HEMOGLOBIN_MEDIAN", "HEMOGLOBIN_MEAN",

               "HEMOGLOBIN_MIN", "HEMOGLOBIN_MAX", "LACTATE_MEDIAN", "LACTATE_MEAN",

               "LACTATE_MIN", "LACTATE_MAX", "LEUKOCYTES_MEDIAN", "LEUKOCYTES_MEAN",

               "LEUKOCYTES_MIN", "LEUKOCYTES_MAX", "NEUTROPHILES_MEDIAN",

               "NEUTROPHILES_MEAN", "NEUTROPHILES_MIN", "NEUTROPHILES_MAX",

               "UREA_MEDIAN", "UREA_MEAN", "UREA_MIN", "UREA_MAX",

               "BLOODPRESSURE_DIASTOLIC_MEAN", "RESPIRATORY_RATE_MEAN",

               "BLOODPRESSURE_DIASTOLIC_MEDIAN", "RESPIRATORY_RATE_MEDIAN",

               "BLOODPRESSURE_DIASTOLIC_MIN", "HEART_RATE_MIN", "TEMPERATURE_MIN",

               "OXYGEN_SATURATION_MIN", "BLOODPRESSURE_SISTOLIC_MAX", "HEART_RATE_MAX",

               "RESPIRATORY_RATE_MAX", "OXYGEN_SATURATION_MAX",

               "BLOODPRESSURE_DIASTOLIC_DIFF", "BLOODPRESSURE_SISTOLIC_DIFF",

               "HEART_RATE_DIFF", "RESPIRATORY_RATE_DIFF", "TEMPERATURE_DIFF",

               "OXYGEN_SATURATION_DIFF", "BLOODPRESSURE_DIASTOLIC_DIFF_REL",

               "BLOODPRESSURE_SISTOLIC_DIFF_REL", "HEART_RATE_DIFF_REL",

               "RESPIRATORY_RATE_DIFF_REL", "TEMPERATURE_DIFF_REL",

               "OXYGEN_SATURATION_DIFF_REL"]



    procs = [FillMissing, Categorify, Normalize] 

    to = TabularPandas(train, procs, cat_names=cat_name, cont_names=cont_name, y_names="ICU", splits = IndexSplitter(idx)(range_of(train)))

    dls = to.dataloaders(bs=512)

    learn = tabular_learner(dls, layers=[400,100], loss_func=MSELossFlat(), emb_drop=0.5)

    learn.clip =.25

    learn.fit_one_cycle(8, 5e-4)

    dl = learn.dls.test_dl(test)

    pred, _ = learn.get_preds()

    preds.append(pred.numpy())

len(preds)
for i in range(len(preds)-1):

    predict = preds[i]+preds[i+1]

pre = predict/len(preds)

_, icu = learn.get_preds()

icus = icu.numpy()
def decode(res, th):

    result = []

    for p in res:

        if p > th:

            result.append(1)

        else:

            result.append(0)

    return result
res = decode(pre, 0.01)
def precision(res, icus):

    tp = 0

    fp = 0

    tn = 0

    fn = 0

    for i, re in enumerate(res):

        if (icus[i]==1):

            if re==1:

                tp=tp+1

            else:

                fn=fn+1

        else:

            if re==1:

                fp=fp+1

            else:

                tn=tn+1

    precision = (tp/(tp+fp))

    recall = (tp/(tp+fn))

    

    print('Accuracy = {:.2f}'.format(((tp+tn)/(tp+fp+tn+fn))))

    print('Precision = {:.2f}'.format(precision))

    print('Recall = {:.2f}'.format(recall))

    print('F1 = {:.2f}'.format(2*((precision*recall)/(precision+recall))))



precision(res, icus)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import auc

fpr, tpr, thresholds = roc_curve(icus, res)
auc = roc_auc_score(icus, res)

print('AUC: %.3f' % auc)

