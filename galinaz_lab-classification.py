from os.path import join as pjoin

from collections import Counter



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

PATH_TO_DATA = '../input/lab12-classification-problem'
data = pd.read_csv(pjoin(PATH_TO_DATA, 'train.csv'))

data.shape





dataTest = pd.read_csv(pjoin(PATH_TO_DATA, 'test.csv'))

labels = data['Label']

words = data['Word']





wordsTest = dataTest['Word']
# Первая буква большая



isUpperFirst = words.str.slice(0, 1).str.isupper()

data['is_upper_first'] = isUpperFirst



isUpperFirstTest = wordsTest.str.slice(0, 1).str.isupper()

dataTest['is_upper_first'] = isUpperFirstTest





pd.crosstab(isUpperFirst, labels).plot(kind='bar')
# все буквы большие

isAllUpper = words.str.isupper()

data['is_all_upper'] = isAllUpper







isAllUpperTest = wordsTest.str.isupper()

dataTest['is_all_upper'] = isAllUpperTest





pd.crosstab(isAllUpper, labels).plot(kind='bar')
# все буквы маленькие

isAllLower = words.str.islower()

data['is_all_lower'] = isAllLower







isAllLowerTest = wordsTest.str.islower()

dataTest['is_all_lower'] = isAllLowerTest





pd.crosstab(isAllLower, labels).plot(kind='bar')
#по последней букве

lastLetter = words.str.slice(-1, None).str.lower()

data['last_letter'] = lastLetter



lastLetterTest = wordsTest.str.slice(-1, None).str.lower()

dataTest['last_letter'] = lastLetterTest





pd.crosstab(lastLetter, labels).plot(kind='bar')
# по предпоследней букве

prelastLetter = words.str.slice(-1, None).str.lower()

data['prelast_letter'] = prelastLetter



prelastLetterTest = wordsTest.str.slice(-1, None).str.lower()

dataTest['prelast_letter'] = prelastLetterTest





pd.crosstab(prelastLetter, labels).plot(kind='bar')
# по пред предпоследней букве

preprelastLetter = words.str.slice(-1, None).str.lower()

data['preprelast_letter'] = preprelastLetter



preprelastLetterTest = wordsTest.str.slice(-1, None).str.lower()

dataTest['preprelast_letter'] = preprelastLetterTest





pd.crosstab(preprelastLetter, labels).plot(kind='bar')
# по известным окончаниям фамилий разных народностей

t = '''Абхазы: ба уа ипа 

Азербайджанцы: заде ли лы оглу кызы 

Армяне: ян янц уни 

Белорусы: ич чик ка ко онак ёнак ук ик ски

Болгары: ев ов 

Гагаузы: огло 

Греки: пулос кос иди 

Грузины: швили дзе ури иа уа ава ли си ни те

Итальянцы: ини ино елло илло етти етто ито

Литовцы: те ис не онис унас утис айтис ена ювен увен ут полуют айт

Латыши: ис

Молдоване: ску у ул ан

Мордва: ын ин шкин кин

Немцы: ман ер

Осетины: ти

Поляки: ск цк ий 

Португальцы: ез ес аз

Русские: ан ын ин ских ов ев ской цкой их ых ова ева ская ина ана ына

Румыны: ску ул ан

Татары: ов ев ин

Турки: оглу джи заде

Шведы: ссон берг стед стром'''





endings = []

for line in t.split('\n'):

    endings.extend([end for end in line.split(':')[1].strip().split(' ')])

endings = set(endings)
endsWithKnownSuffix = np.zeros_like(words.values, dtype=bool)

endsWithKnownSuffixTest = np.zeros_like(wordsTest.values, dtype=bool)



for ending in endings:

    endsWithKnownSuffix  |= words.str.lower().str.endswith(ending)

    endsWithKnownSuffixTest |= wordsTest.str.lower().str.endswith(ending)

    

data['ends_with_a_known_suffix'] = endsWithKnownSuffix 

dataTest['ends_with_a_known_suffix'] = endsWithKnownSuffixTest



pd.crosstab(endsWithKnownSuffix , labels).plot(kind='bar')


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve

from catboost import CatBoostClassifier
W = data.drop(columns = ['Word', 'Label'])

l = data['Label']

WTest = dataTest.drop(columns = ['Word'])

WTrain, WTrainTest, lTrain, lTrainTest = train_test_split(W, l, test_size=0.3, random_state=1)

WTrain = W

lTrain = l





def getAcuracy(lTrue, predProba, threshold=0.5):

    pred = np.zeros_like(predProba)

    pred[predProba > threshold] = 1

    acuracy = accuracy_score(lTrue, pred)

    return acuracy


gb = CatBoostClassifier(

    cat_features= WTrain,

    eval_metric='AUC',

    random_seed=1,

    nan_mode='Forbidden',

    task_type='CPU',

    verbose=True,

    n_estimators=150,

    max_depth=6,

)

gb.fit(WTrain, lTrain)
# Quality on train

predProbaTrain = gb.predict_proba(WTrain)[:, 1]

acc = getAcuracy(lTrain, predProbaTrain)

print("Acuracy = ", acc)

# Quality on test

predProbaTest = gb.predict_proba(WTrainTest)[:, 1]

acc = getAcuracy(lTrainTest, predProbaTest)

print("Acuracy = ", acc)
pr, rec, thr = precision_recall_curve(lTrain, predProbaTrain)

f1 = 2 * (pr * rec) / (pr + rec)

best_thr = thr[f1.argmax() - 1]

best_thr, f1.max()
predTest = gb.predict_proba(WTest)[:, 1]



answers = np.zeros(len(predTest), dtype=bool)





for i in range(len(predTest)):

    if predTest[i] >= best_thr :

        answers[i] = True

        



res = pd.DataFrame({'Id': WTest.index, 'Prediction': predTest})

res.to_csv('result.csv', index=False)