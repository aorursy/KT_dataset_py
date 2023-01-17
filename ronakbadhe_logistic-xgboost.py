import numpy as np

import pandas as pd

from random import shuffle

from itertools import count

from copy import deepcopy



import matplotlib.pyplot as plt

import math



from sklearn.metrics import accuracy_score

import sklearn.preprocessing as preprocessing

import sklearn.utils as utils

from sklearn.linear_model import LogisticRegression



from xgboost import XGBRegressor, plot_importance

import xgboost as xgb



import time



from colorama import Fore, Style 
def dateToInt(date):

    days = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]

    month, day = date.split('-')[1:]

    return sum(days[:int(month)]) + int(day) - 22

dateToInt("2020-01-22")
# def dateToInt2(date):

#     days = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]

#     month, day = date.split('/')[:-1]

#     return sum(days[:int(month)]) + int(day) - 22
# ustempname = "/kaggle/input/ustemperature/usTemp.csv"

# pustemp = pd.read_csv(ustempname)

# def getTemp(state):

#     if state == "District of Columbia":

#         return getTemp("Maryland")

#     return pustemp[pustemp["State"] == state].to_numpy()[0][1]

# getTemp("Alabama")

# pustemp
# otherinfoname = "/kaggle/input/countryinfo/covid19countryinfo.csv"

# pinfo = pd.read_csv(otherinfoname)

# # pinfo = pinfo.drop(pinfo.index[182])

# pinfo = pinfo[["region", "country", "density", "quarantine", "pop", "avgtemp"]][pinfo["country"] == "US"]

# # pinfo.columns

# # len(pinfo[pinfo["country"] == "Italy"])

# pinfo

states = ["AL - Alabama", "AK - Alaska", "AZ - Arizona", "AR - Arkansas", "CA - California", "CO - Colorado",

"CT - Connecticut", "DE - Delaware", "FL - Florida", "GA - Georgia",

"HI - Hawaii", "ID - Idaho", "IL - Illinois", "IN - Indiana", "IA - Iowa",

"KS - Kansas", "KY - Kentucky", "LA - Louisiana", "ME - Maine", "MD - Maryland",

"MA - Massachusetts", "MI - Michigan", "MN - Minnesota", "MS - Mississippi",

"MO - Missouri", "MT - Montana", "NE - Nebraska", "NV - Nevada", "NH - New Hampshire",

"NJ - New Jersey", "NM - New Mexico", "NY - New York", "NC - North Carolina",

"ND - North Dakota", "OH - Ohio", "OK - Oklahoma", "OR - Oregon", "PA - Pennsylvania",

"RI - Rhode Island", "SC - South Carolina", "SD - South Dakota", "TN - Tennessee",

"TX - Texas", "UT - Utah", "VT - Vermont", "VA - Virginia", "WA - Washington", "WV - West Virginia",

"WI - Wisconsin", "WY - Wyoming", "DC - District of Columbia"]

states = tuple(i[5:] for i in states)
testname = "/kaggle/input/covid19-global-forecasting-week-4/test.csv"

ptest = pd.read_csv(testname)

testnames = [p[1] if type(p[1]) is str else p[2] for p in ptest.to_numpy()]

ptest



# Change location of training path

trainname = "/kaggle/input/covid19-global-forecasting-week-4/train.csv"

ptrain = pd.read_csv(trainname)

nptrain = ptrain.to_numpy()

names = set()

provinces = set()

pdatas = dict()

for data in nptrain:

    name = data[2]

    names.add(name)

for name in names:

    pdatas.update({name: ptrain[ptrain["Country_Region"] == name].to_numpy()})

for name, data in list(pdatas.items()):

    for d in data:

        state = d[1]

        if type(state) is float or state in provinces:

            continue

        try:

            if name not in ("Canada",) and name not in testnames:

                del pdatas[name]

                names.remove(name)

        except:

            pass

        names.add(state)

        provinces.add(state)

        pdatas.update({state: ptrain[ptrain["Province_State"] == state].to_numpy()})

counter = 0

stuff = [0 for i in range(72)]

for i in pdatas["Illinois"]:

    stuff[counter % 72] += i[-2]

    counter += 1
oldxbycountry = dict()

oldybycountry = dict()

xbycountry = dict()

ybycountry = dict()

shufflexbycountry = dict()

shuffleybycountry = dict()



for name in names:

    data = pdatas[name]

    countryx = [dateToInt(p[3]) for p in data]

    countryy = [p[4:] for p in data]

    oldxbycountry.update({name: countryx})

    oldybycountry.update({name: countryy})

l = len(oldxbycountry["Italy"])

for name in names:

    scheme = list(range(l))

    shuffle(scheme)

    newx = list(0 for i in range(l))

    newy = list([0, 0] for i in range(l))

    shufflex = list(0 for i in range(l))

    shuffley = list([0,0] for i in range(l))

    for i, x, y in zip(count(), oldxbycountry[name], oldybycountry[name]):

        newx[i%l] = x

        newy[i%l][0] += y[0]

        newy[i%l][1] += y[1]

        shufflex[scheme[i%l]] = x

        shuffley[scheme[i%l]][0] += y[0]

        shuffley[scheme[i%l]][1] += y[1]

    xbycountry.update({name: np.array(newx)})

    ybycountry.update({name: np.array(newy)})

    shufflexbycountry.update({name: np.array(shufflex)})

    shuffleybycountry.update({name: np.array(shuffley)})

len(ybycountry["California"])

shufflexbycountry["California"]
#TODO Implement binary search

def optimize(objective, possiblerange, x, y, xtest, ytest, params, name=''):

    beg = time.time()

    goodmodel = None

    score = 0

    n = 0

    l = len(possiblerange)

    try:

        for i, norm in enumerate(possiblerange):

            print(progressbar(i/l), end = '\r', flush=True)

            model = XGBRegressor(**params)

            model.fit(x,y/norm)

            y_pred = model.predict(xtest)

            predictions = np.array([round(value*norm) for value in y_pred])

            try:

                mscore = objective(ytest, predictions)

            except AssertionError:

                mscore = 0

            if mscore >= score:

                score = mscore

                goodmodel = deepcopy(model)

                n = norm

    except KeyboardInterrupt:

        pass

    print(f"\nMax Score: {score}")

    print(f"{name} trained in {time.time()-beg} seconds")

    return [n, goodmodel]
def accuracy(y_actual, y_pred):

    assert np.mean(y_actual) != 0

    try:

        percentincorrect =  np.mean(abs(y_actual-y_pred))/np.mean(y_actual)

    except:

        return 0

    return 1-percentincorrect
def ncat(*arrs):

    new = []

    for arr in arrs:

        new += list(arr)

    return np.array(new)
def progressbar(percent):

    numberpound = round(percent*20)

    numberdash = 20 - numberpound

    prog =  '[' + '#'*numberpound + '-'*numberdash + ']'

    if numberpound == 20:

        prog = Style.BRIGHT + Fore.GREEN + prog + Style.RESET_ALL

    else:

        prog = Fore.RED + prog + Style.RESET_ALL

    return prog
eu = ("Sweden", "Austria", "Belgium", "Bulgaria", "Croatia", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain")
cnames = ("Alaska", "New Jersey", "Arizona", "Colorado", "Florida", "Hawaii", "Idaho", "Kentucky", "Maine", "Minnesota", "Montana", "New Jersey", "Oregon", "Tennessee", "Wyoming")
blacklist = ("Andorra", "Austria", "British Columbia", "French Polynesia", "Iceland", "Italy", "Jordan", "Latvia", "Lebanon", "Luxembourg", "Mongolia", "New Brunswick", "New South", "Newfoundland and Labrador", "Norway", "Reunion", "Queensland", "Rwanda", "Saint Kitts and Nevis", "Saskatchewan", "Senegal", "Slovenia", "Slovakia", "South Africa", "Sri Lanka", "Switzerland", "Syria", "Uruguay", "Venezuela", "Victoria", "Western Australia", "Zimbabwe")

cnames = list(names)

cnames.sort()



casemodels = dict()

fatmodels = dict()

for cname in cnames:

    try:

        portion = slice(20,None)

        x = xbycountry[cname]

        y = ybycountry[cname]

        

#         x = ncat(*[xbycountry[name] for name in names])

#         y = ncat(*[ybycountry[name] for name in names])



        mnorm = max([i[0] for i in y])

        mnormfat = max([i[1] for i in y]) + .01

        

        blacklisted = cname in blacklist

        





        x_train = np.array([[float(i)] for i in x])

        y_train_case = np.array([i[0] for i in y])

        y_train_fat = np.array([i[1] for i in y])

        x_test = np.array([[float(i)] for i in ncat(x[-20:-18], x[-4:], x[-4:], x[-2:], x[-2:], x[-2:], x[-2:])])

        y_test_case = np.array([i[0] for i in ncat(y[-20:-18], y[-4:], y[-4:], y[-2:], y[-2:], y[-2:], y[-2:])])

        y_test_fat = np.array([i[1] for i in ncat(y[-20:-18], y[-4:], y[-4:], y[-2:], y[-2:], y[-2:], y[-2:])])



        params = {

            "objective": "reg:logistic",

            "booster": "gblinear",

            "learning_rate": .1,

            "n_estimators": 1000 if blacklisted else 2500,#10000,

            "n_jobs": 4

        }



        casemodels[cname] = optimize(accuracy, mnorm * np.linspace(1, 10 if cname != "Japan" else 15), x_train, y_train_case, x_test, y_test_case, params, cname + " cases")

        fatmodels[cname] = optimize(accuracy, mnormfat * np.linspace(1, 10 if cname != "Japan" else 15), x_train, y_train_fat, x_test, y_test_fat, params, cname + " fatalities")

#         print(mnorm, mnormfat)

#         model = XGBRegressor(**params)

#         model.fit(x_train, y_train_case/mnorm)

#         casemodels["New Jersey"] = mnorm, model

        print()

    except KeyError:

        print(f"{cname} is not a country")
# y_pred = model.predict(x_test)

# predictions = np.array([round(value*norm) for value in y_pred])

# # predictions = lmodel.predict(np.array([[i] for i in y_pred]))

# print("Predictions:", *predictions, sep='\t')

# print("Results:", *y_test, sep='\t')

# print(f"Accuracy: {100*accuracy(y_test, predictions)}%")
# Cases

for cname in cnames:

    try:

        norm, model = casemodels[cname]

        x = list(xbycountry[cname])

        del plt



        import matplotlib.pyplot as plt



        plotx = np.array([[float(i)] for i in xbycountry[cname]])

        ploty_actual = np.array([i[0] for i in ybycountry[cname]])

#         ploty_model = model.predict(np.array([[float(i)] for i in xbycountry[cname]]))*norm



        futureplotx = np.array(list([i] for i in range(150)))

#         futurex = np.array([[float(i)] + x[1:] for i in range(150)])

        futurey = model.predict(futureplotx)*norm



        plt.plot(futureplotx, futurey, label="Future")

        plt.plot(plotx, ploty_actual, label="Actual")

        # plt.plot(plotx, ploty_model, label="Model")



        plt.legend()

        plt.show()

        print(f"{cname} hopefully good")

    except KeyError:

        print(f"{cname} is not a country")
# Fatalities

for cname in cnames:

    try:

        norm, model = fatmodels[cname]

        x = list(xbycountry[cname])

        del plt



        import matplotlib.pyplot as plt



        plotx = np.array([[float(i)] for i in xbycountry[cname]])

        ploty_actual = np.array([i[1] for i in ybycountry[cname]])

#         ploty_model = model.predict(np.array([[float(i)] for i in xbycountry[cname]]))*norm



        futureplotx = np.array(list([i] for i in range(150)))

#         futurex = np.array([[float(i/70)] + x[1:] for i in range(150)])

        futurey = model.predict(futureplotx)*norm



        plt.plot(futureplotx, futurey, label="Future")

        plt.plot(plotx, ploty_actual, label="Actual")

        # plt.plot(plotx, ploty_model, label="Model")



        plt.legend()

        plt.show()

        print(f"{cname} hopefully good")

    except KeyError:

        print(f"{cname} is not a country")
!rm /kaggle/working/submission.csv

stuff = []

with open("/kaggle/working/submission.csv", 'a+') as fout:

    fout.write("ForecastId,ConfirmedCases,Fatalities\n")

    for i, p in enumerate(ptest.to_numpy()):

        cname = p[1] if type(p[1]) is str else p[2]

        casenorm, casemodel = deepcopy(casemodels[cname])

        fatnorm, fatmodel = deepcopy(fatmodels[cname])

        date = dateToInt(p[3])

        cases = casemodel.predict(np.array([[date]]))[0]*casenorm

        fatalities = fatmodel.predict(np.array([[date]]))[0]*fatnorm

        print(i + 1, round(cases), round(fatalities), sep=',', file=fout)

#         stuff.append((i+1, predictions[0]))

# print(len(stuff))