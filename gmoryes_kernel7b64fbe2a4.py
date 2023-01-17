import pandas as pd

import numpy as np

%pylab inline

plt.style.use('seaborn-dark')

import warnings

warnings.filterwarnings("ignore") # отключение варнингов

pd.set_option('display.max_columns', None) # pd.options.display.max_columns = None 

# pd.set_option('display.max_rows', None) # не прятать столбцы при выводе дата-фреймов

import matplotlib.pyplot as plt

import matplotlib as mpl

plt.rc('font', size=14)

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import AdaBoostClassifier 

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('./train.csv')

test = pd.read_csv('./solutionex.csv')
train.head()
test.head()
visists_column = train['visits'].values

visists_column = visists_column.astype(str)

visist_splitted = np.char.split(visists_column)
visist_splitted
visist_splitted_cnt = []

for i, l in enumerate(visist_splitted):

    visist_splitted_cnt.append(len(l))
train['count'] = visist_splitted_cnt
train.head()
bincount = np.bincount(train['count'])

bincount
bincount.argmax()
def draw_array(arr, start_from=0, title=""):

    plt.figure(figsize=(20, 10))

    x_plt = np.arange(start_from, len(arr) + start_from, 1)

    y_plt = arr

    plt.scatter(x_plt, y_plt)

    plt.grid()

    plt.title(title)

    

def draw_arrays(arrs):

    n = len(arrs)

    plt.figure(figsize=(20, 80))

    

    for index, (arr, title) in enumerate(arrs):

        plt.subplot(n // 2 + 1, 2, index + 1)

        x_plt = np.arange(0, len(arr), 1)

        y_plt = arr  

        plt.scatter(x_plt, y_plt)

        plt.grid()

        plt.title(str(title))

        

def save_solution(results):

    f = open("result.csv","w")

    f.write("id,nextvisit\n")

    for i in np.arange(1, len(visist_splitted_int) + 1):

        f.write("%d, %d\n" % (i, results[i - 1]))

    f.close()
def find_shift(arr):

    result = 1

    last_shift = arr[-1] - arr[-2]

    #print("-1 =", arr[-1], "-2 =", arr[-2], "shift =", last_shift)

    for i in range(2, len(arr) + 1):

        #print("arr[", -(i - 1), "] =", arr[-(i - 1)])

        #print("arr[", -i, "] =", arr[-i])

        #print("diff =", arr[-(i - 1)] - arr[-i])

        if arr[-(i - 1)] - arr[-i] <= last_shift:

            result += 1

            #print("result =", result)

        else:

            break

    return result
draw_array(bincount, title="Кол-во посещений, гистрограмма")

print("max =", bincount.max(), "in", bincount.argmax())

print("% =", bincount[60:150].sum() / 300000 * 100)
visist_splitted_int = []

for l in visist_splitted:

    visist_splitted_int.append(np.array([int(i) for i in l]))
arr = visist_splitted_int[10000]

data = np.array(arr)

fit = np.polyfit(np.arange(len(arr)), arr, 1)

fit[0] * (len(arr)) + fit[1]
draw_array(visist_splitted_int[150000])
mapping = [-1, 0, 1]

p = [0.33, 0.34, 0.33]

def get_res_plus(rand_arr, i):

    global p

    summ = 0

    j = 0

    while (summ + p[j] < rand_arr[i]):

        summ += p[j]

        j += 1

        

    return mapping[j]



def convert_to_weekday(day):

    day %= 7

    if day == 0:

        day = 7

    return day



map_period = {

    'month': 0.1,

    '2month': 0.3,

    '3month': 0.35,

    'year': 365,

    'full':1

}



def analyze_period(visits, period, skip):

    step = 7

    left = 0

    right = left + step

    

    result = np.zeros((8, 8))

    

    last_days = map_period[period]

    data = visits[len(visits) - last_days: ]

    if skip != 'no':

        to_skip = int(len(visits) * map_period[skip])

        data = data[:len(data) - to_skip]

        

    n = len(data)

    for i, d in enumerate(data):

        day = convert_to_weekday(d)

        if i + 1 != n:

            result[day][convert_to_weekday(data[i + 1])] += 1

      

    if False:

        summ = np.zeros(8)

        for i, v in enumerate(result.sum(1)):

            if v:

                summ[i] = v

            else:

                summ[i] = 1



        result = result / summ

    

    return result



year_cnt = 0

month_cnt = 0

month_year_cnt = 0

once = 0

skip_enable=True

debug=False

skipped = 0



year_no_return_err=0

month_no_return_err=0



year_return_err=0

month_return_err=0



year_no_info=0

month_no_info=0



def analyze(visits, t, true=-1):

    global year_cnt

    global month_cnt

    global once

    global month_year_cnt

    global skipped

    global year_no_return_err

    global month_no_return_err

    global year_no_info

    global month_no_info

    global year_return_err

    global month_return_err

    

    prev = visits[-1]

    prev_day = convert_to_weekday(prev)

    

    n = len(visits)

    

    first_p = 'full'

    second_p = '2month'

    

    result_year = analyze_period(visits, first_p, skip='no')

    result_month = result_year

    #result_year = analyze_period(visits, first_p, skip=second_p)

    #result_month = analyze_period(visits, second_p, skip='no')

    

    week_year = result_year[prev_day]

    week_month = result_month[prev_day]

    

    if debug and False:

        print("year")

        print(week_year)

        print("month")

        print(week_month)

    

    if week_year.argmax() == week_month.argmax() or week_month.argmax() == 0 and week_month.max() >= t:

        year_cnt += 1

        return week_year.argmax()

    else:

        if debug and true != week_month.argmax():

            print("prev_day =", prev_day)

            print("year")

            print(result_year)

            print("month")

            print(result_month)

            print("return =", week_month.argmax(), "true =", true)

            print('==============')

        

        month_cnt += 1

        return week_month.argmax()
last_day = 0

last_days = []

for arr in visist_splitted_int:

    if arr[-1] > last_day:

        last_day = arr[-1]

    last_days.append(arr[-1])



last_days = sorted(last_days)
last_days[:-498]
year_cnt = 0

month_cnt = 0



debug = False



ans = []

score = 0

for i, arr in enumerate(visist_splitted_int):

    if i % 50000 == 0:

        print(i, '/', len(visist_splitted_int))

    i += 1

    

    result = np.zeros(8)

    for j in arr[np.where(arr > arr[-1] - 350)]:

        result[convert_to_weekday(j)] += 1

    

    ans.append(result.argmax())

    #prev_day = analyze(arr[:-1], 0, convert_to_weekday(arr[-1]))

    #if convert_to_weekday(arr[-1]) == prev_day:

    #    score += 1



print("year_cnt =", year_cnt)

print("month_cnt =", month_cnt)

print("score =", score / len(visist_splitted_int))

save_solution(ans)

print(np.bincount(ans))
arr = visist_splitted_int[100001]

arr[np.where(arr > 700)]
q
q = np.array([[1,2],[3,4]])

q[:,-1].max()
def get_vector(arr):

    data = []

    

    res_analyze = analyze_period(arr_data_train, 'year', skip='no')

    res_analyze = res_analyze.flatten()[8:]

    data.extend(res_analyze)

        

    return np.array(data)



validate = False

ans = []

n = len(visist_splitted_int)



to_predict_0 = []

to_predict_1 = []



x_0 = []

y_0 = []



x_1 = []

y_1 = []



idx = []



for i, arr in enumerate(visist_splitted_int):

    

    #if len(arr) > 60:

    #    continue

        

    idx.append(i)

    

    if i % 50000 == 0:

        print(i, '/', n)

    i += 1

    

    len_arr = len(arr)

    

    arr_data_train = arr[:-1]

    arr_data_predict = arr

    

    if i % 2 == 1:

        to_predict_1.append(get_vector(arr_data_predict))

        x_0.append(get_vector(arr_data_train))

        y_0.append(convert_to_weekday(arr[-1]))

    else:

        to_predict_0.append(get_vector(arr_data_predict))

        x_1.append(get_vector(arr_data_train))

        y_1.append(convert_to_weekday(arr[-1]))



x_0 = np.array(x_0)

y_0 = np.array(y_0)



x_1 = np.array(x_1)

y_1 = np.array(y_1)



to_predict_0 = np.array(to_predict_0)

to_predict_1 = np.array(to_predict_1)



#x = x / (x.max(0) - x.min(0) + 1)

    

print("END", "x.shape ", x.shape)
x[1000]
q = np.array([1,2,3,4,5])

d = np.where(q > 3)[0]
boost = XGBClassifier(silent=False, 

                    scale_pos_weight=1,

                    learning_rate=0.2,  

                    colsample_bytree = 0.4,

                    subsample = 0.8,

                    objective='binary:logistic', 

                    n_estimators=10, 

                    reg_alpha = 0.3,

                    max_depth=3, 

                    gamma=10,

                    n_jobs=8)
ans = boost.predict(to_predict)

print(np.bincount(ans))

save_solution(ans)
eachN = 100

train_x_ = x[::eachN]

train_y_ = y[::eachN]



ids = np.arange(train_x_.shape[0])

np.random.shuffle(ids)



train_x = []

train_y = []



for i in ids:

    train_x.append(train_x_[i])

    train_y.append(train_y_[i])

    

train_x = np.array(train_x)

train_y = np.array(train_y)



print(train_x.shape)

print(train_y.shape)
## cv = ShuffleSplit(n_splits=5, test_size=0.1, train_size=None, random_state=1)



models = {'xgb_3': boost_n3}



#for model in boosts:

    #model = models[model_name]



cvs = cross_val_score(boost, train_x, train_y, cv=cv, scoring='accuracy')

print (model_name, f"accuracy={np.round(np.mean(cvs), 3)}", f"std={np.round(np.std(cvs), 3)}")
print('XGBoost with grid search')

# play with these params

params={

    'max_depth': [2, 3, 4, 5], # 5 is good but takes too long in kaggle env

    #'subsample': [0.4,0.5,0.6],

    #'colsample_bytree': [0.5,0.6],

    #'n_estimators': [20,30,40,50],

    #'reg_alpha': [0.01, 0.03]

}



np.random.shuffle(arr)



xgb_clf = XGBClassifier(missing=9999999999)

rs = GridSearchCV(xgb_clf,

                  params,

                  cv=5,

                  scoring="accuracy",

                  n_jobs=8,

                  verbose=2)

rs.fit(train_x, train_y)

best_est = rs.best_estimator_

print(best_est)

res = rs.cv_results_
res
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                    colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

                    max_depth=1, min_child_weight=1, missing=9999999999,

                    n_estimators=100, n_jobs=8, nthread=None,

                    objective='multi:softprob', random_state=0, reg_alpha=0,

                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,

                    subsample=1)
cv = ShuffleSplit(n_splits=5, test_size=0.1, train_size=None, random_state=1)



models = {'xgb_3': xgb, 

          'knn':KNeighborsClassifier(n_neighbors=20)}



for model_name in models:

    model = models[model_name]

    cvs = cross_val_score(model, train_x, train_y, cv=cv, scoring='accuracy')

    print (model_name, f"accuracy={np.round(np.mean(cvs), 3)}", f"std={np.round(np.std(cvs), 3)}")
xgb.fit(x_0, y_0)

ans_1 = xgb.predict(to_predict_1)
xgb.fit(x_1, y_1)

ans_0 = xgb.predict(to_predict_0)
ans = []

for i in range(len(ans1)):

    ans.append(ans_0[i])

    ans.append(ans_1[i])

print(np.bincount(ans))

save_solution(ans)
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(train_x)
#        ["rock", "hiphop", "classical", "metall"]

colors = ['red',  'green',    'blue',    'black', 'yellow', 'purple', 'pink']

for i, pair in enumerate(X_embedded):

    plt.scatter(pair[0], pair[1], s=2, color=colors[y[i] - 1])