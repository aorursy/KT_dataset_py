import pandas, numpy

from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import Lasso

from sklearn.ensemble import GradientBoostingRegressor

from datetime import datetime

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

train = pandas.read_csv("../input/train.csv")

test = pandas.read_csv("../input/test.csv")
def regCheck(data, propTrain, models, features, outcome, regNames = None):

    ind = data.index.values

    size = int(numpy.round(len(ind)*propTrain))

    use = numpy.random.choice(ind, size, replace = False)

    train = data.loc[use]

    test = data.loc[set(ind) - set(use)]

    regmeas = []

    if regNames == None:

        names = []

    for m in models:

        if regNames == None:

            names.append(str(m).split("(")[0])

        trained = m.fit(train[features], train[outcome])

        test["prediction"] = trained.predict(test[features])

        out = r2_score(test[outcome], test["prediction"])

        regmeas.append(out)

    regmeas = pandas.DataFrame(regmeas)

    regmeas = regmeas.transpose()

    if regNames == None:

        regmeas.columns = names

    else:

        regmeas.columns = regNames

    return(regmeas)
def simmer(data, models, features, outcome, nsamples = 100, propTrain = .8, regNames = None, maxTime = 1440):

    tstart = datetime.now()

    sd = dict()

    for i in range(0, nsamples):

        sd[i] = regCheck(data, propTrain, models, features, outcome, regNames)

        if (datetime.now() - tstart).seconds/60 > maxTime:

            print("Stopped at " + str(i + 1) + " replications to keep things under " + str(maxTime) + " minutes")

            break

    output = pandas.concat(sd)

    output = output.reset_index(drop = True)

    return(output)
def paramTester(param, model, values, features, outcome, data, nsamples = 100, propTrain = .5, maxTime = 10):

    models = []

    names = []

    for v in values:

        models.append(eval(model + "(" + param + "=" + str(v) + ")"))

        names.append(param + " = " + str(v))

    out = simmer(data, models, features, outcome, nsamples, propTrain, regNames = names, maxTime = maxTime)

    return(out)
train = pandas.get_dummies(train, drop_first = True)

test = pandas.get_dummies(test, drop_first = True)



usevars = list(set(train.columns).intersection(test.columns))



usevars = numpy.sort(usevars)  



preds = usevars[1:]

values = [.5, .25, .1, .01]

model = "Lasso"

param = "alpha"

        

output = paramTester("alpha", "Lasso", values, preds, "y", train, propTrain = .8, maxTime = .5)

output
values = [.05, .025, .01, .001]

model = "Lasso"

param = "alpha"

        

output = paramTester("alpha", "Lasso", values, preds, "y", train, propTrain = .8, maxTime = .5)

output