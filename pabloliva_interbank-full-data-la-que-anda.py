import math

import numpy as np

import pandas as pd

import time

import datetime

import xgboost as xgb

from sklearn.decomposition import PCA
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

#del train
X_train["ratio"] = X_train["linea_ofrecida"] / X_train["ingreso_neto"]

X_test["ratio"] = X_test["linea_ofrecida"] / X_test["ingreso_neto"]
pd.crosstab(pd.qcut(X_train.ratio, 10), train.codtarget).apply(lambda x: x/x.sum(), axis=1)
rcc.clasif.fillna(-1, inplace=True)

rcc.rango_mora.fillna(-1, inplace=True)

rcc_clasif = rcc.groupby(["codmes", "id_persona"]).clasif.max().reset_index().set_index("codmes").sort_index().astype("int32")

rcc_mora = rcc.groupby(["codmes", "id_persona", "rango_mora"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

rcc_producto = rcc.groupby(["codmes", "id_persona", "producto"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

rcc_banco = rcc.groupby(["codmes", "id_persona", "cod_banco"]).mto_saldo.sum().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

rcc_cantidad_prod = rcc.groupby(["codmes", "id_persona", "cod_banco"]).size().unstack(level=2, fill_value=0).reset_index().set_index("codmes").sort_index().astype("int32")

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
# Agregamos canaritos

canarito_amount = 20

print(canarito_amount)



for i in range(canarito_amount):

    identifier = "canarito" + str(i)

    X_train[identifier] = np.random.random_sample(X_train.shape[0])   

    X_test[identifier] = np.random.random_sample(X_test.shape[0])
# Mas FE: 

# Cuantos paquetes tiene

X_train["cantidad_paquetes"] = np.nan_to_num(X_train["producto_campania_Combos Cuenta Sueldo + APP"]) + np.nan_to_num(X_train["producto_campania_ExtraCash Ataque"]) + np.nan_to_num(X_train["producto_campania_Cuenta Millonaria SuperTasa"]) + np.nan_to_num(X_train["producto_campania_Convenios Compra Deuda No Cliente"]) + np.nan_to_num(X_train["producto_campania_Convenios Compra Deuda Cliente"]) + np.nan_to_num(X_train["producto_campania_Convenios Ampliaciones"]) + np.nan_to_num(X_train["producto_campania_Combos TC+CUENTA+APP"]) + np.nan_to_num(X_train["producto_campania_Combos TC + PA"]) + np.nan_to_num(X_train["producto_campania_Combos CUENTA+APP"]) + np.nan_to_num(X_train["producto_campania_Upgrade TC"]) + np.nan_to_num(X_train["producto_campania_Certificado Bancario"]) + np.nan_to_num(X_train["producto_campania_Cartera ABP"]) + np.nan_to_num(X_train["producto_campania_CD-Defensa"]) + np.nan_to_num(X_train["producto_campania_Adelanto de Sueldo"]) + np.nan_to_num(X_train["producto_campania_Hipotecario"]) + np.nan_to_num(X_train["producto_campania_Incremento Linea"]) + np.nan_to_num(X_train["producto_campania_Plazo"]) + np.nan_to_num(X_train["producto_campania_Retencion TC"]) + np.nan_to_num(X_train["producto_campania_Upgrade"]) + np.nan_to_num(X_train["producto_campania_Trading"]) + np.nan_to_num(X_train["producto_campania_TELEVENTAS"]) + np.nan_to_num(X_train["producto_campania_Seguros"]) + np.nan_to_num(X_train["producto_campania_Seguro Vida Retorno"]) + np.nan_to_num(X_train["producto_campania_Seguro Viajes"]) + np.nan_to_num(X_train["producto_campania_Seguro Vehicular"]) + np.nan_to_num(X_train["producto_campania_Seguro Salud"]) + np.nan_to_num(X_train["producto_campania_Seguro Renta Hospitalaria"]) + np.nan_to_num(X_train["producto_campania_Seguro Oncosalud"]) + np.nan_to_num(X_train["producto_campania_Seguro Dental"]) + np.nan_to_num(X_train["producto_campania_Seguro Contra Accidentes"]) + np.nan_to_num(X_train["producto_campania_Seguro Blindado de TC"]) + np.nan_to_num(X_train["producto_campania_Seguro Asistencia Completa"]) + np.nan_to_num(X_train["producto_campania_Seguro Accidentes REMARK"]) + np.nan_to_num(X_train["producto_campania_Cuenta Sueldo Independiente"]) + np.nan_to_num(X_train["producto_campania_Cuenta Sueldo"]) + np.nan_to_num(X_train["producto_campania_Combos"]) + np.nan_to_num(X_train["producto_campania_Cuenta Simple"]) + np.nan_to_num(X_train["producto_campania_CD-Ataque"]) + np.nan_to_num(X_train["producto_SOBREGIRO"]) + np.nan_to_num(X_train["producto_campania_CTS"]) + np.nan_to_num(X_train["producto_TARJETA_EMP SIN DEFINIR"]) + np.nan_to_num(X_train["producto_CREDITOS CASTIGOS"]) + np.nan_to_num(X_train["producto_campania_ExtraCash"]) + np.nan_to_num(X_train["producto_campania_Combos TC+PA"]) + np.nan_to_num(X_train["producto_campania_Cuenta Millonaria"]) + np.nan_to_num(X_train["producto_campania_Pago Automatico"]) + np.nan_to_num(X_train["producto_campania_Membresia"]) 

X_test["cantidad_paquetes"] = np.nan_to_num(X_test["producto_campania_Combos Cuenta Sueldo + APP"]) + np.nan_to_num(X_test["producto_campania_ExtraCash Ataque"]) + np.nan_to_num(X_test["producto_campania_Cuenta Millonaria SuperTasa"]) + np.nan_to_num(X_test["producto_campania_Convenios Compra Deuda No Cliente"]) + np.nan_to_num(X_test["producto_campania_Convenios Compra Deuda Cliente"]) + np.nan_to_num(X_test["producto_campania_Convenios Ampliaciones"]) + np.nan_to_num(X_test["producto_campania_Combos TC+CUENTA+APP"]) + np.nan_to_num(X_test["producto_campania_Combos TC + PA"]) + np.nan_to_num(X_test["producto_campania_Combos CUENTA+APP"]) + np.nan_to_num(X_test["producto_campania_Upgrade TC"]) + np.nan_to_num(X_test["producto_campania_Certificado Bancario"]) + np.nan_to_num(X_test["producto_campania_Cartera ABP"]) + np.nan_to_num(X_test["producto_campania_CD-Defensa"]) + np.nan_to_num(X_test["producto_campania_Adelanto de Sueldo"]) + np.nan_to_num(X_test["producto_campania_Hipotecario"]) + np.nan_to_num(X_test["producto_campania_Incremento Linea"]) + np.nan_to_num(X_test["producto_campania_Plazo"]) + np.nan_to_num(X_test["producto_campania_Retencion TC"]) + np.nan_to_num(X_test["producto_campania_Upgrade"]) + np.nan_to_num(X_test["producto_campania_Trading"]) + np.nan_to_num(X_test["producto_campania_TELEVENTAS"]) + np.nan_to_num(X_test["producto_campania_Seguros"]) + np.nan_to_num(X_test["producto_campania_Seguro Vida Retorno"]) + np.nan_to_num(X_test["producto_campania_Seguro Viajes"]) + np.nan_to_num(X_test["producto_campania_Seguro Vehicular"]) + np.nan_to_num(X_test["producto_campania_Seguro Salud"]) + np.nan_to_num(X_test["producto_campania_Seguro Renta Hospitalaria"]) + np.nan_to_num(X_test["producto_campania_Seguro Oncosalud"]) + np.nan_to_num(X_test["producto_campania_Seguro Dental"]) + np.nan_to_num(X_test["producto_campania_Seguro Contra Accidentes"]) + np.nan_to_num(X_test["producto_campania_Seguro Blindado de TC"]) + np.nan_to_num(X_test["producto_campania_Seguro Asistencia Completa"]) + np.nan_to_num(X_test["producto_campania_Seguro Accidentes REMARK"]) + np.nan_to_num(X_test["producto_campania_Cuenta Sueldo Independiente"]) + np.nan_to_num(X_test["producto_campania_Cuenta Sueldo"]) + np.nan_to_num(X_test["producto_campania_Combos"]) + np.nan_to_num(X_test["producto_campania_Cuenta Simple"]) + np.nan_to_num(X_test["producto_campania_CD-Ataque"]) + np.nan_to_num(X_test["producto_SOBREGIRO"]) + np.nan_to_num(X_test["producto_campania_CTS"]) + np.nan_to_num(X_test["producto_TARJETA_EMP SIN DEFINIR"]) + np.nan_to_num(X_test["producto_CREDITOS CASTIGOS"]) + np.nan_to_num(X_test["producto_campania_ExtraCash"]) + np.nan_to_num(X_test["producto_campania_Combos TC+PA"]) + np.nan_to_num(X_test["producto_campania_Cuenta Millonaria"]) + np.nan_to_num(X_test["producto_campania_Pago Automatico"]) + np.nan_to_num(X_test["producto_campania_Membresia"]) 



X_train["mora_promedio"] = (np.nan_to_num(X_train["mora_-1.0"]) + np.nan_to_num(X_train["mora_1.0"]) + np.nan_to_num(X_train["mora_2.0"]) + np.nan_to_num(X_train["mora_3.0"]) + np.nan_to_num(X_train["mora_4.0"]) + np.nan_to_num(X_train["mora_6.0"]) + np.nan_to_num(X_train["mora_7.0"]) + np.nan_to_num(X_train["mora_5.0"])) / 8.0

X_test["mora_promedio"] = (np.nan_to_num(X_test["mora_-1.0"]) + np.nan_to_num(X_test["mora_1.0"]) + np.nan_to_num(X_test["mora_2.0"]) + np.nan_to_num(X_test["mora_3.0"]) + np.nan_to_num(X_test["mora_4.0"]) + np.nan_to_num(X_test["mora_6.0"]) + np.nan_to_num(X_test["mora_7.0"]) + np.nan_to_num(X_test["mora_5.0"])) / 8.0

# Additional FE (seems to lower the score)

X_train["cantidad_banco"] = X_train["banco_1"] + X_train["banco_2"] + X_train["banco_3"] + X_train["banco_4"] + X_train["banco_5"] + X_train["banco_6"] + X_train["banco_7"] + X_train["banco_8"] + X_train["banco_9"] + X_train["banco_10"] + X_train["banco_11"] + X_train["banco_12"] + X_train["banco_13"] + X_train["banco_14"] + X_train["banco_15"] + X_train["banco_16"] + X_train["banco_17"] + X_train["banco_18"] + X_train["banco_19"] + X_train["banco_20"] + X_train["banco_21"] + X_train["banco_22"] + X_train["banco_23"] + X_train["banco_24"] + X_train["banco_25"] + X_train["banco_26"] + X_train["banco_27"] + X_train["banco_28"] + X_train["banco_29"] + X_train["banco_30"] + X_train["banco_31"] + X_train["banco_32"] + X_train["banco_33"] + X_train["banco_34"] + X_train["banco_35"] + X_train["banco_36"] + X_train["banco_37"] + X_train["banco_38"] + X_train["banco_39"] + X_train["banco_40"] + X_train["banco_41"] + X_train["banco_42"] + X_train["banco_43"] + X_train["banco_44"] + X_train["banco_45"] + X_train["banco_46"] + X_train["banco_47"] + X_train["banco_48"] + X_train["banco_49"] + X_train["banco_50"] + X_train["banco_51"] + X_train["banco_52"] + X_train["banco_53"] + X_train["banco_54"] + X_train["banco_55"] + X_train["banco_56"] + X_train["banco_57"] + X_train["banco_58"] + X_train["banco_59"] + X_train["banco_60"] + X_train["banco_61"] + X_train["banco_62"] + X_train["banco_63"] + X_train["banco_64"] + X_train["banco_65"] + X_train["banco_66"] + X_train["banco_67"] + X_train["banco_68"] + X_train["banco_69"] + X_train["banco_70"] + X_train["banco_71"] + X_train["banco_72"] + X_train["banco_73"] + X_train["banco_74"] + X_train["banco_75"] + X_train["banco_76"] + X_train["banco_77"] + X_train["banco_78"] + X_train["banco_79"] + X_train["banco_80"] + X_train["banco_81"] + X_train["banco_82"] + X_train["banco_83"] + X_train["banco_84"]

X_test["cantidad_banco"] = X_test["banco_1"] + X_test["banco_2"] + X_test["banco_3"] + X_test["banco_4"] + X_test["banco_5"] + X_test["banco_6"] + X_test["banco_7"] + X_test["banco_8"] + X_test["banco_9"] + X_test["banco_10"] + X_test["banco_11"] + X_test["banco_12"] + X_test["banco_13"] + X_test["banco_14"] + X_test["banco_15"] + X_test["banco_16"] + X_test["banco_17"] + X_test["banco_18"] + X_test["banco_19"] + X_test["banco_20"] + X_test["banco_21"] + X_test["banco_22"] + X_test["banco_23"] + X_test["banco_24"] + X_test["banco_25"] + X_test["banco_26"] + X_test["banco_27"] + X_test["banco_28"] + X_test["banco_29"] + X_test["banco_30"] + X_test["banco_31"] + X_test["banco_32"] + X_test["banco_33"] + X_test["banco_34"] + X_test["banco_35"] + X_test["banco_36"] + X_test["banco_37"] + X_test["banco_38"] + X_test["banco_39"] + X_test["banco_40"] + X_test["banco_41"] + X_test["banco_42"] + X_test["banco_43"] + X_test["banco_44"] + X_test["banco_45"] + X_test["banco_46"] + X_test["banco_47"] + X_test["banco_48"] + X_test["banco_49"] + X_test["banco_50"] + X_test["banco_51"] + X_test["banco_52"] + X_test["banco_53"] + X_test["banco_54"] + X_test["banco_55"] + X_test["banco_56"] + X_test["banco_57"] + X_test["banco_58"] + X_test["banco_59"] + X_test["banco_60"] + X_test["banco_61"] + X_test["banco_62"] + X_test["banco_63"] + X_test["banco_64"] + X_test["banco_65"] + X_test["banco_66"] + X_test["banco_67"] + X_test["banco_68"] + X_test["banco_69"] + X_test["banco_70"] + X_test["banco_71"] + X_test["banco_72"] + X_test["banco_73"] + X_test["banco_74"] + X_test["banco_75"] + X_test["banco_76"] + X_test["banco_77"] + X_test["banco_78"] + X_test["banco_79"] + X_test["banco_80"] + X_test["banco_81"] + X_test["banco_82"] + X_test["banco_83"] + X_test["banco_84"]



X_train["auto_alta_gama"] = X_train["ACURA_v1"] + X_train["ALFA ROMEO_v1"] + X_train["ALFA ROMERO_v1"] + X_train["ASTON MARTIN_v1"] + X_train["AUDI_v1"] + X_train["BMW_v1"] + X_train["BUICK_v1"] + X_train["CADILLAC_v1"] + X_train["CHEROKEE_v1"] + X_train["DODGE_v1"] + X_train["DUCATI_v1"] + X_train["FERRARI_v1"] + X_train["HUMMER_v1"] + X_train["JAGUAR_v1"] + X_train["JEEP_v1"] + X_train["KTM_v1"] + X_train["LAMBORGHINI_v1"] + X_train["LAMBRETTA_v1"] + X_train["LAND ROVER_v1"] + X_train["LEXUS_v1"] + X_train["M.G._v1"] + X_train["MERCEDES B_v1"] + X_train["MERCEDES BENZ_v1"] + X_train["PIAGGIO_v1"] + X_train["PLYMOUTH_v1"] + X_train["PONTIAC_v1"] + X_train["PORSCHE_v1"] + X_train["ROVER_v1"] + X_train["TRIUMPH_v1"] + X_train["VESPA_v1"] + X_train["ALFA ROMEO_v2"] + X_train["ALFA ROMERO_v2"] + X_train["ASTON MARTIN_v2"] + X_train["AUDI_v2"] + X_train["BMW_v2"] + X_train["BUICK_v2"] + X_train["CADILLAC_v2"] + X_train["CHEROKEE_v2"] + X_train["DODGE_v2"] + X_train["DUCATI_v2"] + X_train["FERRARI_v2"] + X_train["HUMMER_v2"] + X_test["INFINITI_v1"]  + X_train["JAGUAR_v2"] + X_train["JEEP_v2"] + X_train["KTM_v2"] + X_train["LAMBORGHINI_v2"] + X_train["LAMBRETTA_v2"] + X_train["LAND ROVER_v2"] + X_train["LEXUS_v2"] + X_train["MERCEDES B_v2"] + X_train["MERCEDES BENZ_v2"] + X_train["PIAGGIO_v2"] + X_train["PLYMOUTH_v2"] + X_train["PONTIAC_v2"] + X_train["PORSCHE_v2"] + X_train["ROVER_v2"] + X_train["SUBARU_v2"] + X_train["TRIUMPH_v2"] + X_train["VOLVO_v2"]

X_test["auto_alta_gama"] = X_test["ACURA_v1"] + X_test["ALFA ROMEO_v1"] + X_test["ALFA ROMERO_v1"] + X_test["ASTON MARTIN_v1"] + X_test["AUDI_v1"] + X_test["BMW_v1"] + X_test["BUICK_v1"] + X_test["CADILLAC_v1"] + X_test["CHEROKEE_v1"] + X_test["DODGE_v1"] + X_test["DUCATI_v1"] + X_test["FERRARI_v1"] + X_test["HUMMER_v1"] + X_test["JAGUAR_v1"] + X_test["JEEP_v1"] + X_test["KTM_v1"] + X_test["LAMBORGHINI_v1"] + X_test["LAMBRETTA_v1"] + X_test["LAND ROVER_v1"] + X_test["LEXUS_v1"] + X_test["M.G._v1"] + X_test["MERCEDES B_v1"] + X_test["MERCEDES BENZ_v1"] + X_test["PIAGGIO_v1"] + X_test["PLYMOUTH_v1"] + X_test["PONTIAC_v1"] + X_test["PORSCHE_v1"] + X_test["ROVER_v1"] + X_test["TRIUMPH_v1"] + X_test["VESPA_v1"] + X_test["ALFA ROMEO_v2"] + X_test["ALFA ROMERO_v2"] + X_test["ASTON MARTIN_v2"] + X_test["AUDI_v2"] + X_test["BMW_v2"] + X_test["BUICK_v2"] + X_test["CADILLAC_v2"] + X_test["CHEROKEE_v2"] + X_test["DODGE_v2"] + X_test["DUCATI_v2"] + X_test["FERRARI_v2"] + X_test["HUMMER_v2"] + X_test["JAGUAR_v2"] + X_test["JEEP_v2"] + X_test["KTM_v2"] + X_test["LAMBORGHINI_v2"] + X_test["LAMBRETTA_v2"] + X_test["LAND ROVER_v2"] + X_test["LEXUS_v2"] + X_test["MERCEDES B_v2"] + X_test["MERCEDES BENZ_v2"] + X_test["PIAGGIO_v2"] + X_test["PLYMOUTH_v2"] + X_test["PONTIAC_v2"] + X_test["PORSCHE_v2"] + X_test["ROVER_v2"] + X_test["SUBARU_v2"] + X_test["TRIUMPH_v2"] + X_test["VOLVO_v2"]



X_train["auto_normal"] = X_train["CHEVROLET_v1"] + X_train["CHEVY_v1"] + X_train["CHRYSLER_v1"] + X_train["CITROEN_v1"] + X_train["DAELIM_v1"] + X_train["DAELIM HONDA_v1"] + X_train["FIAT_v1"] + X_train["FORD_v1"] + X_train["FORD INGLES_v1"] + X_train["FORD TAUNUS_v1"] + X_train["FORLAND_v1"] + X_train["HONDA_v1"] + X_train["HONDA DAELIM_v1"] + X_train["HONDA DAILIM_v1"] + X_train["HYUNDAI_v1"] + X_train["KIA_v1"] + X_train["KIA MOTORS_v1"] + X_train["NISSAN_v1"] + X_train["NISSAN DIESEL_v1"] + X_train["PEUGEOT_v1"] + X_train["SEAT_v1"] + X_train["SUSUKI_v1"] + X_train["SUZUKI_v1"] + X_train["TAUNUS_v1"] + X_train["TOYOTA_v1"] + X_train["WY HONDA_v1"] + X_train["ZNA-ZHENGZHOU NISSAN AUTO_v1"] + X_train["CHEVROLET_v2"] + X_train["CHEVY_v2"] + X_train["CHRYSLER_v2"] + X_train["CITROEN_v2"] + X_train["DAELIM_v2"] + X_train["DAELIM HONDA_v2"] + X_train["FIAT_v2"] + X_train["FORD_v2"] + X_train["FORD INGLES_v2"] + X_train["FORD TAUNUS_v2"] + X_train["FORLAND_v2"] + X_train["HONDA_v2"] + X_train["HONDA DAELIM_v2"] + X_train["HONDA DAILIM_v2"] + X_train["HYOSUNG SUZKI_v2"] + X_train["HYOSUNG SUZUKI_v2"] + X_train["HYUNDAI_v2"] + X_train["ISUZU_v2"] + X_train["KIA_v2"] + X_train["KIA MOTORS_v2"] + X_train["MAZDA_v2"] + X_train["MITSUBISHI_v2"] + X_train["MITSUBISHI FUSO_v2"] + X_train["NISSAN_v2"] + X_train["NISSAN DIESEL_v2"] + X_train["OPEL_v2"] + X_train["PEUGEOT_v2"] + X_train["RENAULT_v2"] + X_train["SEAT_v2"] + X_train["SUSUKI_v2"] + X_train["SUZUKI_v2"] + X_train["TAUNUS_v2"] + X_train["TOYOTA_v2"] + X_train["VOLKSWAGEN_v2"] + X_train["VOLSWAGEN_v2"] + X_train["WY HONDA_v2"]

X_test["auto_normal"] = X_test["CHEVROLET_v1"] + X_test["CHEVY_v1"] + X_test["CHRYSLER_v1"] + X_test["CITROEN_v1"] + X_test["DAELIM_v1"] + X_test["DAELIM HONDA_v1"] + X_test["FIAT_v1"] + X_test["FORD_v1"] + X_test["FORD INGLES_v1"] + X_test["FORD TAUNUS_v1"] + X_test["FORLAND_v1"] + X_test["HONDA_v1"] + X_test["HONDA DAELIM_v1"] + X_test["HONDA DAILIM_v1"] + X_test["HYUNDAI_v1"] + X_test["KIA_v1"] + X_test["KIA MOTORS_v1"] + X_test["NISSAN_v1"] + X_test["NISSAN DIESEL_v1"] + X_test["PEUGEOT_v1"] + X_test["SEAT_v1"] + X_test["SUSUKI_v1"] + X_test["SUZUKI_v1"] + X_test["TAUNUS_v1"] + X_test["TOYOTA_v1"] + X_test["WY HONDA_v1"] + X_test["ZNA-ZHENGZHOU NISSAN AUTO_v1"] + X_test["CHEVROLET_v2"] + X_test["CHEVY_v2"] + X_test["CHRYSLER_v2"] + X_test["CITROEN_v2"] + X_test["DAELIM_v2"] + X_test["DAELIM HONDA_v2"] + X_test["FIAT_v2"] + X_test["FORD_v2"] + X_test["FORD INGLES_v2"] + X_test["FORD TAUNUS_v2"] + X_test["FORLAND_v2"] + X_test["HONDA_v2"] + X_test["HONDA DAELIM_v2"] + X_test["HONDA DAILIM_v2"] + X_test["HYOSUNG SUZKI_v2"] + X_test["HYOSUNG SUZUKI_v2"] + X_test["HYUNDAI_v2"] + X_test["ISUZU_v2"] + X_test["KIA_v2"] + X_test["KIA MOTORS_v2"] + X_test["MAZDA_v2"] + X_test["MITSUBISHI_v2"] + X_test["MITSUBISHI FUSO_v2"] + X_test["NISSAN_v2"] + X_test["NISSAN DIESEL_v2"] + X_test["OPEL_v2"] + X_test["PEUGEOT_v2"] + X_test["RENAULT_v2"] + X_test["SEAT_v2"] + X_test["SUSUKI_v2"] + X_test["SUZUKI_v2"] + X_test["TAUNUS_v2"] + X_test["TOYOTA_v2"] + X_test["VOLKSWAGEN_v2"] + X_test["VOLSWAGEN_v2"] + X_test["WY HONDA_v2"]



X_train["auto_economico"] = X_train["BEIJING AUTO WORKS_v1"] + X_train["BEIJING AUTOMOBILE WORKS - BAW_v1"] + X_train["CHANA_v1"] + X_train["CHANG_v1"] + X_train["CHANGAN_v1"] + X_train["CHANGE_v1"] + X_train["CHANGFENG_v1"] + X_train["CHANGHE_v1"] + X_train["CHANGSHU_v1"] + X_train["CHENGLONG_v1"] + X_train["CHERY_v1"] + X_train["DAEWOO_v1"] + X_train["DAIHATSU_v1"] + X_train["DATSUN_v1"] + X_train["DAVEST_v1"] + X_train["DAYUN_v1"] + X_train["DONG FENG_v1"] + X_train["DONGFENG_v1"] + X_train["DONGFENG BRAND_v1"] + X_train["GEELY_v1"] + X_train["GEELY FORTTE_v1"] + X_train["GREAT WALL_v1"] + X_train["HAOJIANG_v1"] + X_train["HAOJIN_v1"] + X_train["HONGYI_v1"] + X_train["HONSAI_v1"] + X_train["HOWO_v1"] + X_train["HUALIN_v1"] + X_train["HUANGHAI_v1"] + X_train["HUARI_v1"] + X_train["HUAWIN_v1"] + X_train["HYOSUNG_v1"] + X_train["HYOSUNG SUZKI_v1"] + X_train["HYOSUNG SUZUKI_v1"] + X_train["ISUZU_v1"] + X_train["JAC_v1"] + X_train["JAPANY MOTORS_v1"] + X_train["JAWA CZ_v1"] + X_train["JIAJUE_v1"] + X_train["JIALING_v1"] + X_train["JIANSHE_v1"] + X_train["JIAPENG_v1"] + X_train["JIEDA_v1"] + X_train["JIMBEI_v1"] + X_train["JIN BEI_v1"] + X_train["JINBEI_v1"] + X_train["JINCHENG_v1"] + X_train["JINCO_v1"] + X_train["JINGUAN_v1"] + X_train["JIREH_v1"] + X_train["JOYLONG_v1"] + X_train["JPANY MOTORS_v1"] + X_train["KOREA MOTOS_v1"] + X_train["KORIAN MOTOS_v1"] + X_train["LIFAN_v1"] + X_train["MITSUBISHI_v1"] + X_train["MITSUBISHI FUSO_v1"] + X_train["SAIC_v1"] + X_train["SAIC WULING_v1"] + X_train["SAN YANG_v1"] + X_train["SSANG YONG_v1"] + X_train["SSANGYONG_v1"] + X_train["SSANYONG_v1"] + X_train["TIANHONG_v1"] + X_train["TIANJIN FAW_v1"] + X_train["YANG ZU_v1"] + X_train["ASIA_v2"] + X_train["ASIA MOTORS_v2"] + X_train["BAIC_v2"] + X_train["BAIC YINXI_v2"] + X_train["BAIC YINXIANG_v2"] + X_train["BEIJING AUTO WORKS_v2"] + X_train["BEIJING AUTOMOBILE WORKS - BAW_v2"] + X_train["CHERY_v2"] + X_train["DAEWOO_v2"] + X_train["DAIHATSU_v2"] + X_train["DATSUN_v2"] + X_train["GEELY_v2"] + X_train["GEELY FORTTE_v2"] + X_train["HAOJIANG_v2"] + X_train["HAOJIN_v2"] + X_train["HONGYI_v2"] + X_train["HONSAI_v2"] + X_train["HOWO_v2"] + X_train["HUALIN_v2"] + X_train["HUANGHAI_v2"] + X_train["HYOSUNG_v2"] + X_train["JAC_v2"] + X_train["JAPANY MOTORS_v2"] + X_train["JAWA CZ_v2"] + X_train["JPANY MOTORS_v2"] + X_train["LIFAN_v2"] + X_train["SAAB_v2"] + X_train["SAIC_v2"] + X_train["SAIC WULING_v2"] + X_train["SSANG YONG_v2"] + X_train["SSANGYONG_v2"] + X_train["SSANYONG_v2"] + X_train["SUKIDA_v2"] + X_train["TAO YAN_v2"] + X_train["TAO-YAN_v2"] + X_train["WUYANG_v2"] + X_train["XINGFU_v2"] + X_train["XINKAI_v2"] + X_train["YAKUMA_v2"] + X_train["YANG ZU_v2"] + X_train["YANSUMI_v2"] + X_train["YINGANG_v2"] + X_train["YINHE_v2"] + X_train["YINXIANG_v2"] + X_train["YOUYI_v2"] + X_train["YUEJIN_v2"] + X_train["YUMBO_v2"] + X_train["ZHONGXING_v2"] + X_train["ZHONGYU_v2"] + X_train["ZNA-ZHENGZHOU NISSAN AUTO_v2"]

X_test["auto_economico"] = X_test["BEIJING AUTO WORKS_v1"] + X_test["BEIJING AUTOMOBILE WORKS - BAW_v1"] + X_test["CHANA_v1"] + X_test["CHANG_v1"] + X_test["CHANGAN_v1"] + X_test["CHANGE_v1"] + X_test["CHANGFENG_v1"] + X_test["CHANGHE_v1"] + X_test["CHANGSHU_v1"] + X_test["CHENGLONG_v1"] + X_test["CHERY_v1"] + X_test["DAEWOO_v1"] + X_test["DAIHATSU_v1"] + X_test["DATSUN_v1"] + X_test["DAVEST_v1"] + X_test["DAYUN_v1"] + X_test["DONG FENG_v1"] + X_test["DONGFENG_v1"] + X_test["DONGFENG BRAND_v1"] + X_test["GEELY_v1"] + X_test["GEELY FORTTE_v1"] + X_test["GREAT WALL_v1"] + X_test["HAOJIANG_v1"] + X_test["HAOJIN_v1"] + X_test["HONGYI_v1"] + X_test["HONSAI_v1"] + X_test["HOWO_v1"] + X_test["HUALIN_v1"] + X_test["HUANGHAI_v1"] + X_test["HUARI_v1"] + X_test["HUAWIN_v1"] + X_test["HYOSUNG_v1"] + X_test["HYOSUNG SUZKI_v1"] + X_test["HYOSUNG SUZUKI_v1"] + X_test["ISUZU_v1"] + X_test["JAC_v1"] + X_test["JAPANY MOTORS_v1"] + X_test["JAWA CZ_v1"] + X_test["JIAJUE_v1"] + X_test["JIALING_v1"] + X_test["JIANSHE_v1"] + X_test["JIAPENG_v1"] + X_test["JIEDA_v1"] + X_test["JIMBEI_v1"] + X_test["JIN BEI_v1"] + X_test["JINBEI_v1"] + X_test["JINCHENG_v1"] + X_test["JINCO_v1"] + X_test["JINGUAN_v1"] + X_test["JIREH_v1"] + X_test["JOYLONG_v1"] + X_test["JPANY MOTORS_v1"] + X_test["KOREA MOTOS_v1"] + X_test["KORIAN MOTOS_v1"] + X_test["LIFAN_v1"] + X_test["MITSUBISHI_v1"] + X_test["MITSUBISHI FUSO_v1"] + X_test["SAIC_v1"] + X_test["SAIC WULING_v1"] + X_test["SAN YANG_v1"] + X_test["SSANG YONG_v1"] + X_test["SSANGYONG_v1"] + X_test["SSANYONG_v1"] + X_test["TIANHONG_v1"] + X_test["TIANJIN FAW_v1"] + X_test["YANG ZU_v1"] + X_test["ASIA_v2"] + X_test["ASIA MOTORS_v2"] + X_test["BAIC_v2"] + X_test["BAIC YINXI_v2"] + X_test["BAIC YINXIANG_v2"] + X_test["BEIJING AUTO WORKS_v2"] + X_test["BEIJING AUTOMOBILE WORKS - BAW_v2"] + X_test["CHERY_v2"] + X_test["DAEWOO_v2"] + X_test["DAIHATSU_v2"] + X_test["DATSUN_v2"] + X_test["GEELY_v2"] + X_test["GEELY FORTTE_v2"] + X_test["HAOJIANG_v2"] + X_test["HAOJIN_v2"] + X_test["HONGYI_v2"] + X_test["HONSAI_v2"] + X_test["HOWO_v2"] + X_test["HUALIN_v2"] + X_test["HUANGHAI_v2"] + X_test["HYOSUNG_v2"] + X_test["JAC_v2"] + X_test["JAPANY MOTORS_v2"] + X_test["JAWA CZ_v2"] + X_test["JPANY MOTORS_v2"] + X_test["LIFAN_v2"] + X_test["SAAB_v2"] + X_test["SAIC_v2"] + X_test["SAIC WULING_v2"] + X_test["SSANG YONG_v2"] + X_test["SSANGYONG_v2"] + X_test["SSANYONG_v2"] + X_test["SUKIDA_v2"] + X_test["TAO YAN_v2"] + X_test["TAO-YAN_v2"] + X_test["WUYANG_v2"] + X_test["XINGFU_v2"] + X_test["XINKAI_v2"] + X_test["YAKUMA_v2"] + X_test["YANG ZU_v2"] + X_test["YANSUMI_v2"] + X_test["YINGANG_v2"] + X_test["YINHE_v2"] + X_test["YINXIANG_v2"] + X_test["YOUYI_v2"] + X_test["YUEJIN_v2"] + X_test["YUMBO_v2"] + X_test["ZHONGXING_v2"] + X_test["ZHONGYU_v2"] + X_test["ZNA-ZHENGZHOU NISSAN AUTO_v2"]



X_train["moto"] = X_train["CF MOTO_v1"] + X_train["GILERA_v1"] + X_train["KAWASAKI_v1"] + X_train["KTM_v1"] + X_train["QUE MOTO_v1"] + X_train["SAKIMOTO_v1"] + X_train["SKYMOTO_v1"] + X_train["SUMOTO_v1"] + X_train["TRIUMPH_v1"] + X_train["VESPA_v1"] + X_train["YAMAHA_v1"] + X_train["DUCATI_v2"] + X_train["GILERA_v2"] + X_train["HARLEY DAVIDSON_v2"] + X_train["KAWASAKI_v2"] + X_train["KOREA MOTOS_v2"] + X_train["KORIAN MOTOS_v2"] + X_train["KTM_v2"] + X_train["LC MOTOS_v2"] + X_train["LC-MOTOS_v2"] + X_train["MOTOCOSMO_v2"] + X_train["MOTODUR_v2"] + X_train["MOTOKALY_v2"] + X_train["MOTOKAR_v2"] + X_train["MOTOS ALELUYA_v2"] + X_train["MOTTO KAZO_v2"] + X_train["QUE MOTO_v2"] + X_train["SAKIMOTO_v2"] + X_train["SAN YANG_v2"] + X_train["SUMOTO_v2"] + X_train["TRIUMPH_v2"] + X_train["YAMAHA_v2"] + X_train["ZANELLA_v2"]

X_test["moto"] = X_test["CF MOTO_v1"] + X_test["GILERA_v1"] + X_test["KAWASAKI_v1"] + X_test["KTM_v1"] + X_test["QUE MOTO_v1"] + X_test["SAKIMOTO_v1"] + X_test["SKYMOTO_v1"] + X_test["SUMOTO_v1"] + X_test["TRIUMPH_v1"] + X_test["VESPA_v1"] + X_test["YAMAHA_v1"] + X_test["DUCATI_v2"] + X_test["GILERA_v2"] + X_test["HARLEY DAVIDSON_v2"] + X_test["KAWASAKI_v2"] + X_test["KOREA MOTOS_v2"] + X_test["KORIAN MOTOS_v2"] + X_test["KTM_v2"] + X_test["LC MOTOS_v2"] + X_test["LC-MOTOS_v2"] + X_test["MOTOCOSMO_v2"] + X_test["MOTODUR_v2"] + X_test["MOTOKALY_v2"] + X_test["MOTOKAR_v2"] + X_test["MOTOS ALELUYA_v2"] + X_test["MOTTO KAZO_v2"] + X_test["QUE MOTO_v2"] + X_test["SAKIMOTO_v2"] + X_test["SAN YANG_v2"] + X_test["SUMOTO_v2"] + X_test["TRIUMPH_v2"] + X_test["YAMAHA_v2"] + X_test["ZANELLA_v2"]



X_train["soc_total"] = X_train["soc_var1"] + X_train["soc_var2"] + X_train["soc_var3"] + X_train["soc_var4"] + X_train["soc_var5"] + X_train["soc_var6"]

X_test["soc_total"] = X_test["soc_var1"] + X_test["soc_var2"] + X_test["soc_var3"] + X_test["soc_var4"] + X_test["soc_var5"] + X_test["soc_var6"]



X_train["soc_avg"] = X_train["soc_total"] / 6

X_test["soc_avg"] = X_test["soc_total"] / 6

# Como quedaron los valores?

"""

print(max(X_train["cantidad_paquetes"]))

print(min(X_train["cantidad_paquetes"]))

print(np.mean(X_train["cantidad_paquetes"]))

print(max(X_test["mora_promedio"]))

print(min(X_test["mora_promedio"]))

print(np.mean(X_test["mora_promedio"]))

"""
#print(len(X_train.columns))

#print(len(X_test.columns))
from lightgbm import LGBMRegressor, LGBMClassifier

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

    learner.fit(Xt, yt,  early_stopping_rounds=10, eval_metric="auc",

                eval_set=[(Xt, yt), (Xv.drop(drop_cols, axis=1), yv)], verbose=50)

    gc.collect()

    test_preds.append(pd.Series(learner.predict(X_test.drop(drop_cols, axis=1)),

                                index=X_test.index, name="fold_" + str(mes)))

    train_preds.append(pd.Series(learner.predict(Xv.drop(drop_cols, axis=1)),

                                index=Xv.index, name="probs"))

    gc.collect()



test_preds = pd.concat(test_preds, axis=1).mean(axis=1)

train_preds = pd.concat(train_preds)
# Vemos las importancias de variables para saber quienes se van.

feature_importances = learner.feature_importances_

#print(len(feature_importances))

x_t_columns_no_month = list(X_train.columns)

x_t_columns_no_month.remove('codmes')

#print(len(x_t_columns_no_month))



feature_imp = pd.DataFrame({'Value':learner.feature_importances_,'Feature':x_t_columns_no_month})

feature_imp = feature_imp.sort_values(by="Value",ascending=False)

feature_imp.to_csv("variables" + str(datetime.datetime.now()).replace(" ", "_").replace(".", "_").replace(":", "_").replace("-", "_") + ".csv", header=True)



numerical_col_names =  X_train.select_dtypes(include=np.number).columns.tolist()



# Hasta cuantos canaritos aguanto antes de eliminar todas las columnas menos importantes (mas alto el limite,

# mas columnas tendra el reducido)

limite_canaritos = 4

valuable_variables = []

numerical_variables_below_limit = []

cantidad_actual_canaritos = 0

# Recorro el listado de variables hasta encontrar limite_canaritos canaritos

for index, row in feature_imp.iterrows():

    if 'canarito' in row['Feature']:

        cantidad_actual_canaritos += 1

        if cantidad_actual_canaritos >= limite_canaritos:

            break

    else:

        valuable_variables.append(row['Feature'])



# We forcibly append codmes to rerun lgbm

valuable_variables.append('codmes')

        

#print(valuable_variables)

"""

# This for can be used (replacing the for in the notebook above) to separate 

# all the numerical variables below canaries, for PCA

# Sadly, there are too many NULL values for PCA, and imputation didn't work.

for index, row in feature_imp.iterrows():

    if 'canarito' in row['Feature']:

        cantidad_actual_canaritos += 1

        if cantidad_actual_canaritos >= limite_canaritos:

            adding = False

    elif adding:

        valuable_variables.append(row['Feature'])

    else:

        if (row['Feature'] in numerical_col_names):

            numerical_variables_below_limit.append(row['Feature'])

"""

#numerical_variables_below_limit
#train_df_for_pca_cols = [col for col in X_train.columns if col in numerical_variables_below_limit]

#train_df_for_pca = X_train[train_df_for_pca_cols]

#test_df_for_pca_cols = [col for col in X_test.columns if col in numerical_variables_below_limit]

#test_df_for_pca = X_test[test_df_for_pca_cols]

#train_df_for_pca.shape



# Imputacion de nan a la media

# Hace saltar el kernel, asi que tiramos columnas con NAN

#train_df_for_pca.fillna(train_df_for_pca.mean(), inplace=True)

#train_df_for_pca = train_df_for_pca.fillna(train_df_for_pca.groupby("prediction_id").transform("mean"))

#test_df_for_pca = test_df_for_pca.dropna(axis = 1)

#train_df_for_pca = train_df_for_pca.dropna(axis = 1)

# Tirar las columnas con NAN nos deja sólo dos columnas, no vale la pena hacer PCA

"""

component_amount = train_df_for_pca.shape[1]

pca_train = PCA(n_components=component_amount)

pca_train.fit(train_df_for_pca)

pca_train_df_vars = pd.DataFrame(pca_train)



pca_test = PCA(n_components=component_amount)

pca_test.fit(test_df_for_pca)

pca_test_df_vars = pd.DataFrame(pca_test)

"""
# train_df_for_pca.shape[1]

#test_df_for_pca
reduced_train_cols = [col for col in X_train.columns if col in valuable_variables]

reduced_train = X_train[reduced_train_cols]

#reduced_train = pd.concat([reduced_train_tmp, pca_train_df_vars], axis=1)



reduced_test_cols = [col for col in X_test.columns if col in valuable_variables]

reduced_test = X_test[reduced_test_cols]

#reduced_test = pd.concat([reduced_test_tmp, pca_test_df_vars], axis=1)

print(reduced_train)
# LGBM sobre el dataset reducido

drop_cols = ["codmes"]

fi = []

test_probs = []

train_probs = []

y_train["target"] = (y_train["margen"] > 0).astype("int32")

for mes in reduced_train.codmes.unique():

    print("*"*10, mes, "*"*10)

    Xt = reduced_train[reduced_train.codmes != mes]

    yt = y_train.loc[Xt.index, "target"]

    Xt = Xt.drop(drop_cols, axis=1)



    Xv = reduced_train[reduced_train.codmes == mes]

    yv = y_train.loc[Xv.index, "target"]

    

    learner = LGBMClassifier(n_estimators=1000)

    learner.fit(Xt, yt,  early_stopping_rounds=10, eval_metric="auc",

                eval_set=[(Xt, yt), (Xv.drop(drop_cols, axis=1), yv)], verbose=50)

    gc.collect()

    test_probs.append(pd.Series(learner.predict_proba(reduced_test.drop(drop_cols, axis=1))[:, -1],

                                index=reduced_test.index, name="fold_" + str(mes)))

    train_probs.append(pd.Series(learner.predict_proba(Xv.drop(drop_cols, axis=1))[:, -1],

                                index=Xv.index, name="probs"))

    gc.collect()



reduced_test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

reduced_train_probs = pd.concat(train_probs)
"""

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

    learner.fit(Xt, yt,  early_stopping_rounds=10, eval_metric="auc",

                eval_set=[(Xt, yt), (Xv.drop(drop_cols, axis=1), yv)], verbose=50)

    gc.collect()

    test_probs.append(pd.Series(learner.predict_proba(X_test.drop(drop_cols, axis=1))[:, -1],

                                index=X_test.index, name="fold_" + str(mes)))

    train_probs.append(pd.Series(learner.predict_proba(Xv.drop(drop_cols, axis=1))[:, -1],

                                index=Xv.index, name="probs"))

    gc.collect()



test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

train_probs = pd.concat(train_probs)

"""
lgb_red_test = pd.concat([reduced_test_probs.rename("probs"), test_preds.rename("preds")], axis=1)

lgb_red_train = pd.concat([reduced_train_probs.rename("probs"), train_preds.rename("preds")], axis=1)

test = pd.concat([reduced_test_probs.rename("probs"), test_preds.rename("preds")], axis=1)

train = pd.concat([reduced_train_probs.rename("probs"), train_preds.rename("preds")], axis=1)

print(max(lgb_red_test["probs"]))

print(min(lgb_red_test["probs"]))

test["probs"] = lgb_red_test["probs"]

train["probs"] = lgb_red_train["probs"]

from scipy.optimize import differential_evolution



def clasificar(res, c):

    return ((res.preds > c[0]) | (res.probs > c[1])) * c[2] + ((res.probs > c[3]) & (res.preds > c[4])) * c[5] > c[6]



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

file_name_output = "benchmark_reducido" + str(datetime.datetime.now()).replace(" ", "_").replace(".", "_").replace(":", "_").replace("-", "_") + ".csv" 

test_preds.to_csv(file_name_output, header=True)