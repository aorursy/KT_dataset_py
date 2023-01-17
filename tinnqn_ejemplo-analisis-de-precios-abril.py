import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
sucursales = pd.read_csv("/kaggle/input/precios-claros-precios-de-argentina/sucursales.csv") 

sucursales
productos = pd.read_csv("/kaggle/input/precios-claros-precios-de-argentina/productos.csv") 

productos
# precios de la semana pasada

precios_last_w = pd.read_csv("/kaggle/input/precios-claros-precios-de-argentina/precios_20200412_20200413.csv")

# precios de esta semana

precios_current_w = pd.read_csv("/kaggle/input/precios-claros-precios-de-argentina/precios_20200419_20200419.csv")



precios_current_w
sucursales_prov = sucursales[["id", "provincia"]]

sucursales_prov["id_prov"] = sucursales["comercioId"].astype(str) + "-" + sucursales["banderaId"].astype(str) +  "-" + sucursales["provincia"]

sucursales_prov
precios_last_w = precios_last_w.join(sucursales_prov.set_index("id"), on="sucursal_id")

precios_current_w = precios_current_w.join(sucursales_prov.set_index("id"), on="sucursal_id")

precios_current_w
precios_current_w.rename(columns={"precio": "precio_19abril"}, inplace=True)

precios_last_w.rename(columns={"precio": "precio_12abril"}, inplace=True)
precios_last_w
precios = pd.merge(precios_last_w, precios_current_w, on=["producto_id", "id_prov", "provincia"],  how="inner").dropna()[["producto_id", "id_prov", "provincia", "precio_12abril", "precio_19abril"]]

precios = pd.merge(precios, productos[["id", "marca", "nombre"]], left_on="producto_id", right_on="id").drop('id', axis=1)

precios["diferencia"] = precios["precio_19abril"] - precios["precio_12abril"]

precios["diferencia_porcentual"] = precios["diferencia"].abs() / precios["precio_12abril"] * 100

precios
precios.to_csv("precios_20200412_vs_20200419.csv", index=False)
precios[precios.provincia == "AR-C"].sort_values("diferencia_porcentual", ascending=False)[:50]