import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier



X_test = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_test/ib_base_inicial_test.csv")

sunat = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_sunat/ib_base_sunat.csv")

train = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_train/ib_base_inicial_train.csv")

rcc = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_rcc/ib_base_rcc.csv")

reniec = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_reniec/ib_base_reniec.csv")

digital = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_digital/ib_base_digital.csv")

vehicular = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_vehicular/ib_base_vehicular.csv")

campanias = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_campanias/ib_base_campanias.csv")
mes = 201904







mesesTrain =  {201904: 201901}

mesesDigital = {201904: (201811, 201901)}

mesesMora = mesesDigital = {201904: (201806, 201812)}

y_train = train[train.codmes == mes].set_index("id_persona")

y_train = (y_train["margen"] >0).astype(int)



train = train[train.codmes == mes].set_index("id_persona")

for variable in ["cem", "ingreso_neto", "linea_ofrecida"]:

    train = train.join(pd.qcut(train[variable], 10, duplicates = "drop", labels=False).to_frame().rename(columns = {variable : "decil_{0}".format(variable)}))

    train = train.join(pd.qcut(train[variable], 4, duplicates = "drop", labels=False).to_frame().rename(columns = {variable : "quartil_{0}".format(variable)}))

    train = train.join(pd.qcut(train[variable], 5, duplicates = "drop", labels=False).to_frame().rename(columns = {variable : "quintil_{0}".format(variable)}))

    train = train.join((train[variable] / train[variable].mean()).to_frame().rename(columns = {variable: "ratioMedian_{0}".format(variable)}))

    train = train.join((train[variable] / train[variable].mean()).to_frame().rename(columns = {variable: "rationMean_{0}".format(variable)}))



train = train.drop(columns = ["cem", "ingreso_neto", "linea_ofrecida", "codmes", "codtarget", "margen"])
digital["codmes"] = digital.apply(lambda x: x.codday // 100, axis = 1).astype(int)

digital = digital[digital.codmes.between(mesesDigital[mes][0], mesesDigital[mes][1])].fillna(0)



agregatedColumns = ['simu_prestamo', 'benefit', 'email', 'facebook',

       'goog', 'youtb', 'compb', 'movil', 'desktop', 'lima_dig', 'provincia_dig', 'extranjero_dig', 'n_sesion',

       'busqtc', 'busqvisa', 'busqamex', 'busqmc', 'busqcsimp', 'busqmill',

       'busqcsld', 'busq', 'n_pag', 'android', 'iphone']



aggFunctions = dict()

for column in agregatedColumns:

    aggFunctions[column] = "sum"



digital = digital.groupby(["id_persona"]).agg(aggFunctions)



train = train.join(digital, how = "left").fillna(0)
rcc = rcc[rcc.codmes.between (mesesMora[mes][0], mesesMora[mes][1])]

rcc["rango_mora"] = rcc["rango_mora"].fillna(0).astype(int)

rcc1 = rcc.groupby("id_persona").agg({"rango_mora": "max", "mto_saldo": "mean"}).join(rcc.groupby("id_persona").cod_banco.nunique())

rcc1 = rcc1.join(rcc.groupby("id_persona").producto.nunique())

rcc_producto = rcc.groupby([ "id_persona", "producto"]).mto_saldo.sum().unstack(level=1, fill_value=0).reset_index().set_index("id_persona").sort_index().astype("int32")

rcc1 = rcc1.join(rcc_producto)



train = train.join(rcc1, how = "left").fillna(0)


reniecColumns = reniec.columns[2:-1]

reniecDummies = pd.concat([pd.get_dummies(reniec.set_index("id_persona")[col]) for col in reniecColumns], axis=1, keys=reniecColumns)

reniecDummies.columns = [col[0]+"_"+str(col[1]) for col in reniecDummies.columns]

reniecDummies

reniec1 = reniec.set_index("id_persona").join(reniecDummies).drop(columns = reniecColumns)



train = train.join(reniec1, how = "left").fillna(0)
vehicular1 = vehicular.groupby(["id_persona", "marca"]).veh_var1.sum().unstack(level=1, fill_value=0).astype("float32")

vehicular1.columns = [c + "_v1" for c in vehicular1.columns]





vehicular2 = vehicular.groupby(["id_persona", "marca"]).veh_var1.sum().unstack(level=1, fill_value=0).astype("float32")

vehicular2.columns = [c + "_v2" for c in vehicular2.columns]



train = train.join(vehicular1, how = "left").fillna(0)

train = train.join(vehicular2, how = "left").fillna(0)







train = train.join(y_train.to_frame().rename(columns = {"margen": "target"}))

xTotal = train.drop("target", axis = 1)

yTotal = train["target"]

X, X_validation, y, y_validation = train_test_split(xTotal, yTotal, test_size=0.20, random_state=42)
baseFinal = pd.DataFrame()

kfolds = StratifiedKFold(4, shuffle=True)

for train_index, test_index in kfolds.split(X, y):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index] 

#    for i, c in enumerate(X_train.columns[[not all(ord(c) < 128 for c in s) for s in X_train.columns]]):

#        X_train["non_ascii_" + str(i)] = X_train[c]

#        X_train = X_train.drop(c, axis= 1)

#        X_test["non_ascii_" + str(i)] = X_test[c]

#        X_test = X_test.drop(c, axis= 1)

    auxTest = X_test.copy()

    X_train = X_train.values.astype(np.float32, order="C")

    X_test = X_test.values.astype(np.float32, order="C")

    scaler = StandardScaler()

    scaler.fit(np.r_[X_train, X_test])

    X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)  

    learner = LGBMClassifier(n_estimators=10000)

    learner.fit(X_train, y_train,  early_stopping_rounds=10, eval_metric="auc",

                eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)

    probsArray = learner.predict_proba(X_test)[:, 1]

    auxTest["modeloGanador"] = probsArray

    baseFinal = baseFinal.append(auxTest[["modeloGanador"]])
(baseFinal.modeloGanador > 0.5).sum()