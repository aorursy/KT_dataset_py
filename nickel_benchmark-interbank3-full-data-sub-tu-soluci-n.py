import pandas as pd
X_test = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_test/ib_base_inicial_test.csv")

sunat = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_sunat/ib_base_sunat.csv")

train = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_inicial_train/ib_base_inicial_train.csv")

rcc = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_rcc/ib_base_rcc.csv")

reniec = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_reniec/ib_base_reniec.csv")

digital = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_digital/ib_base_digital.csv")

vehicular = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_vehicular/ib_base_vehicular.csv")

campanias = pd.read_csv("/kaggle/input/interbank-internacional-2019/ib_base_campanias/ib_base_campanias.csv")
y_train = train[['codmes', 'id_persona', 'margen']].copy()

y_train["prediction_id"] = y_train["id_persona"].astype(str) + "_" + y_train["codmes"].astype(str)

# y_train["target"] = y_train["margen"].astype("float32")

y_train = y_train.set_index("prediction_id")

X_train = train.drop(["codtarget", "margen"], axis=1)

X_train["prediction_id"] = X_train["id_persona"].astype(str) + "_" + X_train["codmes"].astype(str)

del train
X_train["ratio"] = X_train["linea_ofrecida"] / X_train["ingreso_neto"]

X_test["ratio"] = X_test["linea_ofrecida"] / X_test["ingreso_neto"]
rcc.clasif.fillna(-1, inplace=True)

rcc.rango_mora.fillna(-1, inplace=True)

rcc_clasif = rcc.groupby(["codmes", "id_persona"]).clasif.max().reset_index().set_index("codmes").sort_index().astype("int32")

rcc_mora = rcc.groupby(["codmes", "id_persona", "rango_mora"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

rcc_producto = rcc.groupby(["codmes", "id_persona", "producto"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

rcc_banco = rcc.groupby(["codmes", "id_persona", "cod_banco"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

del rcc
rcc_mora.columns = ["mora_" + str(c) if c != "id_persona" else c for c in rcc_mora.columns ]

rcc_producto.columns = ["producto_" + str(c) if c != "id_persona" else c for c in rcc_producto.columns]

rcc_banco.columns = ["banco_" + str(c) if c != "id_persona" else c for c in rcc_banco.columns]
camp_canal = campanias.groupby(["codmes", "id_persona", "canal_asignado"]).size().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

camp_prod = campanias.groupby(["codmes", "id_persona", "producto"]).size().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

del campanias
camp_canal.columns = ["canal_" + str(c) if c != "id_persona" else c for c in camp_canal.columns]

camp_prod.columns = ["producto_campania_" + str(c) if c != "id_persona" else c for c in camp_prod.columns]
digital["codmes"] = digital.codday.astype(str).str[:-2].astype(int)

digital = digital.drop("codday", axis=1).fillna(0)

digital = digital.groupby(["codmes", "id_persona"]).sum().reset_index().set_index("codmes").sort_index().astype("int32")
sunat = sunat.groupby(["id_persona", "activ_econo"]).meses_alta.sum().unstack(level=1, fill_value=0).astype("int32")

vehicular1 = vehicular.groupby(["id_persona", "marca"]).veh_var1.sum().unstack(level=1, fill_value=0).astype("float32")

vehicular2 = vehicular.groupby(["id_persona", "marca"]).veh_var2.sum().unstack(level=1, fill_value=0).astype("float32")

reniec = reniec.set_index("id_persona").astype("float32")

del vehicular
vehicular1.columns = [c + "_v1" for c in vehicular1.columns]

vehicular2.columns = [c + "_v2" for c in vehicular2.columns]
X_train = X_train.set_index("prediction_id").astype("int32").reset_index().set_index("id_persona").join(vehicular1).join(vehicular2).join(reniec).join(sunat)

X_test = X_test.set_index("prediction_id").astype("int32").reset_index().set_index("id_persona").join(vehicular1).join(vehicular2).join(reniec).join(sunat)

del vehicular1, vehicular2, reniec, sunat
import gc

gc.collect()
meses = {

    201901: slice(201808, 201810),

    201902: slice(201809, 201811),

    201903: slice(201810, 201812),

    201904: slice(201811, 201901),

    201905: slice(201812, 201902),

    201906: slice(201901, 201903),

    201907: slice(201902, 201904)

}



meses_train = X_train.codmes.unique()

meses_test = X_test.codmes.unique()

complementos = []

for mes in meses.keys():

    print("*"*10, mes, "*"*10)

    res = pd.concat([

        rcc_clasif.loc[meses[mes]].groupby("id_persona").sum(),

        rcc_mora.loc[meses[mes]].groupby("id_persona").sum(),

        rcc_producto.loc[meses[mes]].groupby("id_persona").sum(),

        rcc_banco.loc[meses[mes]].groupby("id_persona").sum(),

        camp_canal.loc[meses[mes]].groupby("id_persona").sum(),

        camp_prod.loc[meses[mes]].groupby("id_persona").sum(),

        digital.loc[meses[mes]].groupby("id_persona").sum()

        

    ], axis=1)

    res["codmes"] = mes

    res = res.reset_index().set_index(["id_persona", "codmes"]).astype("float32")

    complementos.append(res)



gc.collect()

print("concatenando complementos")

complementos = pd.concat(complementos)

gc.collect()

print("X_train join")

X_train = X_train.reset_index().join(complementos, on=["id_persona", "codmes"]).set_index("prediction_id")

gc.collect()

print("X_test join")

X_test = X_test.reset_index().join(complementos, on=["id_persona", "codmes"]).set_index("prediction_id")

gc.collect()



del rcc_clasif, rcc_mora, rcc_producto, rcc_banco, camp_canal, camp_prod, digital, complementos,res

gc.collect()
for i, c in enumerate(X_train.columns[[not all(ord(c) < 128 for c in s) for s in X_train.columns]]):

    X_train["non_ascii_" + str(i)] = X_train[c]

    X_train = X_train.drop(c, axis= 1)

    X_test["non_ascii_" + str(i)] = X_test[c]

    X_test = X_test.drop(c, axis= 1)
from lightgbm import LGBMRegressor

gc.collect()
drop_cols = ["codmes"]

test_preds = []

train_preds = []

y_train["target"] = y_train["margen"].astype("float32")

for mes in X_train.codmes.unique():

    print("*"*10, mes, "*"*10)

    Xt = X_train[X_train.codmes != mes]

    yt = y_train.loc[Xt.index, "target"]

    Xt = Xt.drop(drop_cols, axis=1)



    Xv = X_train[X_train.codmes == mes]

    yv = y_train.loc[Xv.index, "target"]

    

    learner = LGBMRegressor(n_estimators=1000)

    learner.fit(Xt, yt,  early_stopping_rounds=10, eval_metric="mae",

                eval_set=[(Xt, yt), (Xv.drop(drop_cols, axis=1), yv)], verbose=50)

    gc.collect()

    test_preds.append(pd.Series(learner.predict(X_test.drop(drop_cols, axis=1)),

                                index=X_test.index, name="fold_" + str(mes)))

    train_preds.append(pd.Series(learner.predict(Xv.drop(drop_cols, axis=1)),

                                index=Xv.index, name="probs"))

    gc.collect()



test_preds = pd.concat(test_preds, axis=1).mean(axis=1)

train_preds = pd.concat(train_preds)
from lightgbm import LGBMClassifier

gc.collect()
drop_cols = ["codmes"]

fi = []

test_probs = []

train_probs = []

y_train["target"] = (y_train["margen"] > 0).astype("int32")

for mes in X_train.codmes.unique():

    print("*"*10, mes, "*"*10)

    Xt = X_train[X_train.codmes != mes]

    yt = y_train.loc[Xt.index, "target"]

    Xt = Xt.drop(drop_cols, axis=1)



    Xv = X_train[X_train.codmes == mes]

    yv = y_train.loc[Xv.index, "target"]

    

    learner = LGBMClassifier(n_estimators=1000)

    learner.fit(Xt, yt,  early_stopping_rounds=10, eval_metric="mae",

                eval_set=[(Xt, yt), (Xv.drop(drop_cols, axis=1), yv)], verbose=50)

    gc.collect()

    test_probs.append(pd.Series(learner.predict_proba(X_test.drop(drop_cols, axis=1))[:, -1],

                                index=X_test.index, name="fold_" + str(mes)))

    train_probs.append(pd.Series(learner.predict_proba(Xv.drop(drop_cols, axis=1))[:, -1],

                                index=Xv.index, name="probs"))

    gc.collect()



test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

train_probs = pd.concat(train_probs)
test = pd.concat([test_probs.rename("probs"), test_preds.rename("preds")], axis=1)

train = pd.concat([train_probs.rename("probs"), train_preds.rename("preds")], axis=1)
from scipy.optimize import differential_evolution



def clasificar(res, c):

    return ((res.probs > c[0]) | (res.preds > c[1])) * c[2] + ((res.probs > c[3]) & (res.preds > c[4])) * c[5] > c[6]



def cost(res, coefs):

    return -((clasificar(res, coefs) * res.margen) / res.margen.sum()).sum()



res = y_train.join(train)

optimization = differential_evolution(lambda x: cost(res, x), [(-100, 100), (0, 1), (0, 1),

                                                               (-100, 100), (0, 1), (0, 1),

                                                               (0, 2)])

optimization
test_preds = clasificar(test, optimization["x"]).astype(int)

test_preds.index.name="prediction_id"

test_preds.name="class"

test_preds.to_csv("benchmark3.csv", header=True)