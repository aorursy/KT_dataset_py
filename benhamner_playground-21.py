import pandas as pd
df_train = pd.read_csv('../input/train.csv',
                           dtype  = {'Semana': 'int32',
                                     'Agencia_ID': 'int32',
                                     'Canal_ID': 'int32',
                                     'Ruta_SAK': 'int32',
                                     'Cliente_ID': 'int32',
                                     'Producto_ID':'int32',
                                     'Venta_hoy':'float32',
                                     'Venta_uni_hoy': 'int32',
                                     'Dev_uni_proxima':'int32',
                                     'Dev_proxima':'float32',
                                     'Demanda_uni_equil':'int32'})
print(df_train)
