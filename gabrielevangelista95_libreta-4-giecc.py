# Importar librerias necesarias 
import numpy as np
import pandas as pd
from pandas import read_excel
from pandas import DataFrame
# Indicar donde se encuentra el archivo 
# PATH_TO_FILE = "../Datasets/viajes.xlsx"
# Leer archivo
# raw_data = read_excel(PATH_TO_FILE)
# Explorar contenido del archivo
raw_data = pd.read_csv('../input/viajes-giecc/viajes_GIECC.csv')
# Revisar contenido de las variables por lectura 
raw_data.iloc[0]
# Revisar valor una de variable en particular
raw_data.iloc[0]["Commanded Equivalence Ratio(lambda)"]
# Revisar valor de otra variable en particular
raw_data.iloc[0]["Fuel flow rate/hour(l/s)"]
# Extraer variables de interes
raw_data_ = raw_data[['Viaje',
                      ' Device Time',
                      'Speed (OBD)(km/h)',
                      'Engine Load(%)',
                      'Slope\n[rad]',
                      'Fuel flow rate/hour(l/s)',
                      'Litres Per 100 Kilometer(Instant)(l/100km)']]
# Explorar nuevo conjunto de datos
raw_data_
# Extraer unicamente los datos del viaje M = 16 
M = 16
viaje_M_mask = raw_data_['Viaje'] == M
viaje_M = raw_data_[viaje_M_mask]
# Explorar resultado
viaje_M
from datetime import datetime
import math
n_sample = 10
delta_t = abs(  int(viaje_M.iloc[n_sample    ][' Device Time'].split(":")[-1]) \
              - int(viaje_M.iloc[n_sample - 1][' Device Time'].split(":")[-1]) ) # * 86400
speed = viaje_M.iloc[n_sample]['Speed (OBD)(km/h)'] / 3.6
# Excel equation checks with a mask (+/- 2) to check the values are form the same trip and if the delta_time == 1sec 
accel = ( -       viaje_M.iloc[n_sample + 2]['Speed (OBD)(km/h)'] \
          + ( 8 * viaje_M.iloc[n_sample + 1]['Speed (OBD)(km/h)'] ) \
          - ( 8 * viaje_M.iloc[n_sample - 1]['Speed (OBD)(km/h)'] ) \
          +       viaje_M.iloc[n_sample - 2]['Speed (OBD)(km/h)']  \
        ) / (12 * delta_t * 3.6) # 3.6 added to normalize the units as in speed above

accel1= ( -       viaje_M.iloc[n_sample + 1 + 2]['Speed (OBD)(km/h)'] \
          + ( 8 * viaje_M.iloc[n_sample + 1 + 1]['Speed (OBD)(km/h)'] ) \
          - ( 8 * viaje_M.iloc[n_sample + 1 - 1]['Speed (OBD)(km/h)'] ) \
          +       viaje_M.iloc[n_sample + 1 - 2]['Speed (OBD)(km/h)']  \
        ) / (12 * delta_t * 3.6) # 3.6 added to normalize the units as in speed above

print("Acceleration [n]   : {} \nAccelaration [n+1] : {}".format(accel,accel1))
accel_km = 1 if ((accel1-accel)>0 and (accel<=0)) else 0
pos_accel = accel if (accel > 0.14) else ""
neg_accel = accel if (accel <= -0.14) else ""
accel_sqrd = accel**2
freq = ""
E_pow_2ni = accel_sqrd**2
# Compute or obtain +/- 2 elements mask of speed 
speed_minus_2 = viaje_M.iloc[n_sample - 2]['Speed (OBD)(km/h)'] / 3.6
speed_minus_1 = viaje_M.iloc[n_sample - 1]['Speed (OBD)(km/h)'] / 3.6
speed_plus_1  = viaje_M.iloc[n_sample + 1]['Speed (OBD)(km/h)'] / 3.6
speed_plus_2  = viaje_M.iloc[n_sample + 2]['Speed (OBD)(km/h)'] / 3.6
# Excel equation checks with a mask (+/- 2) to check if values are from the same trip and if 
distance = ( 1 / 8 ) * ( speed_minus_2 + (3 * speed_minus_1 ) + ( 3 + speed_plus_1 ) + speed_plus_2 ) * delta_t
cum_distance = 0 # to initialize counter
cum_distance += distance 
q_viaje = ((speed**2)-(speed_minus_1**2)) if (speed > speed_minus_1) else 0
vsp_approx = speed * (  ( 1.10 * accel ) + ( 9.81 * math.tan(viaje_M.iloc[n_sample]['Slope\n[rad]']) ) +   0.132 ) \
                      + ( 0.000302 * ( speed**2 ) * speed)
rho_aire = 1.165 
Cd       = 0.32
A        = 5.647
m        = 2268
Cr       = 0.015
VSP = ( ( 0.50 * rho_aire * Cd * A * ( speed**3 ) ) / m ) \
      + ( 9.81 * math.sin(viaje_M.iloc[n_sample]['Slope\n[rad]']) * speed ) \
      + ( accel * speed ) \
      + ( 9.81 * Cr * math.cos(viaje_M.iloc[n_sample]['Slope\n[rad]']) * speed )
resultados = DataFrame(
            [[delta_t, speed, accel, accel_km, pos_accel, neg_accel, accel_sqrd, freq, E_pow_2ni, distance, cum_distance, q_viaje, vsp_approx, VSP]],
            columns = ["Delta T [s]", "Speed [m/s]", "Acceleration [m/s^2]", "Accel/km [km^-1]",
                       "Positive acceleration [m/s^2]", "Negative acceleration [m/s^2]", 
                       "Acceleration^2  [(m/s^2)^2]", "Frequency", "E^2*n(i)", 
                       "Distance [m]", "Accumulated distance [m]",
                       "Q de viaje [m^2/a^2]", "VSP approx [kW/t]", 
                       "VSP [kW/t]"])
resultados
V_3 = (   ( speed_plus_1 ** 3 ) \
        + ( ( speed_plus_1 ** 2 ) * speed ) \
        + ( speed_plus_1 * ( speed ** 2 ) ) \
        + ( speed ** 3 ) ) / 4
Ca_plus = max( ( 0.5 * ( ( speed_plus_1 ** 2 ) - ( speed ** 2 ) ) ), 0 )
fuel_flow = viaje_M.iloc[n_sample]['Fuel flow rate/hour(l/s)'] * 1000 
inst_aver_fuel = viaje_M.iloc[n_sample]['Litres Per 100 Kilometer(Instant)(l/100km)']
results = DataFrame(
            [{"Delta T [s]" : delta_t, 
              "Speed [m/s]" : speed, 
              "Acceleration [m/s^2]" : accel, 
              "Accel/km [km^-1]" : accel_km,
              "Positive acceleration [m/s^2]" : pos_accel, 
              "Negative acceleration [m/s^2]" : neg_accel, 
              "Acceleration^2 [(m/s^2)^2]" : accel_sqrd, 
              "Frequency" : freq,
              "E^2*n(i)" : E_pow_2ni, 
              "Distance [m]" : distance,
              "Accumulated distance [m]" : cum_distance,
              "Q de viaje [m^2/a^2]" : q_viaje, 
              "VSP approx [kW/t]" : vsp_approx, 
              "VSP [kW/t]" : VSP,
              "V^3" : V_3,
              "Ca+" : Ca_plus,
              "Fuel flow [ml/s]" : fuel_flow,
              "Instantaneous average fuel" : inst_aver_fuel
            }])
results
def get_trip_values(viaje_M):
    
    data = []
    
    rho_aire = 1.165 
    Cd       = 0.32
    A        = 5.647
    m        = 2268
    Cr       = 0.015
    
    cum_distance = 0 # to initialize counter
    
    for n_sample in range(2,viaje_M.shape[0] - 3):
        
        # print(n_sample) # Uncomment for debugging
        
        delta_t = abs(  int(viaje_M.iloc[n_sample    ][' Device Time'].split(":")[-1]) \
                  - int(viaje_M.iloc[n_sample - 1][' Device Time'].split(":")[-1]) ) # * 86400

        speed = viaje_M.iloc[n_sample]['Speed (OBD)(km/h)'] / 3.6
        
        # Compute or obtain +/- 2 elements mask of speed 
        speed_plus_2  = viaje_M.iloc[n_sample + 2]['Speed (OBD)(km/h)'] / 3.6
        speed_plus_1  = viaje_M.iloc[n_sample + 1]['Speed (OBD)(km/h)'] / 3.6
        speed_minus_1 = viaje_M.iloc[n_sample - 1]['Speed (OBD)(km/h)'] / 3.6
        speed_minus_2 = viaje_M.iloc[n_sample - 2]['Speed (OBD)(km/h)'] / 3.6
        
        # Compute or obtain +/- 2 elements mask of speed 
        speed_plus_2_next  = viaje_M.iloc[n_sample + 2]['Speed (OBD)(km/h)'] / 3.6
        speed_plus_1_next  = viaje_M.iloc[n_sample + 1]['Speed (OBD)(km/h)'] / 3.6
        speed_minus_1_next = viaje_M.iloc[n_sample - 1]['Speed (OBD)(km/h)'] / 3.6
        speed_minus_2_next = viaje_M.iloc[n_sample - 2]['Speed (OBD)(km/h)'] / 3.6

        # Excel equation checks with a mask (+/- 2) to check the values are form the same trip and if the delta_time == 1sec 
        accel = ( -       speed_plus_2    \
                  + ( 8 * speed_plus_1  ) \
                  - ( 8 * speed_minus_1 ) \
                  +       speed_minus_2   \
                ) / (12 * delta_t )

        accel1= ( -       speed_plus_2_next    \
                  + ( 8 * speed_plus_1_next  ) \
                  - ( 8 * speed_minus_1_next ) \
                  +       speed_minus_2_next   \
                ) / (12 * delta_t )

        accel_km = 1 if ((accel1-accel)>0 and (accel<=0)) else 0

        pos_accel = accel if (accel > 0.14) else ""

        neg_accel = accel if (accel <= -0.14) else ""

        accel_sqrd = accel**2

        freq = ""

        E_pow_2ni = accel_sqrd**2

        # Excel equation checks with a mask (+/- 2) to check if values are from the same trip 
        distance = ( 1 / 8 ) * ( speed_minus_2 + (3 * speed_minus_1 ) + ( 3 + speed_plus_1 ) + speed_plus_2 ) * delta_t

        cum_distance += distance 

        q_viaje = ((speed**2)-(speed_minus_1**2)) if (speed > speed_minus_1) else 0

        vsp_approx = speed * (  ( 1.10 * accel ) + ( 9.81 * math.tan(viaje_M.iloc[n_sample]['Slope\n[rad]']) ) +   0.132 ) \
                          + ( 0.000302 * ( speed**2 ) * speed)

        VSP = ( ( 0.50 * rho_aire * Cd * A * ( speed**3 ) ) / m ) \
              + ( 9.81 * math.sin(viaje_M.iloc[n_sample]['Slope\n[rad]']) * speed ) \
              + ( accel * speed ) \
              + ( 9.81 * Cr * math.cos(viaje_M.iloc[n_sample]['Slope\n[rad]']) * speed )

        V_3 = (   ( speed_plus_1 ** 3 ) \
                + ( ( speed_plus_1 ** 2 ) * speed ) \
                + ( speed_plus_1 * ( speed ** 2 ) ) \
                + ( speed ** 3 ) ) / 4

        Ca_plus = max( ( 0.5 * ( ( speed_plus_1 ** 2 ) - ( speed ** 2 ) ) ), 0 )

        fuel_flow = viaje_M.iloc[n_sample]['Fuel flow rate/hour(l/s)'] * 1000 

        inst_aver_fuel = viaje_M.iloc[n_sample]['Litres Per 100 Kilometer(Instant)(l/100km)']
        
        data.append([delta_t, 
                     speed, 
                     accel, 
                     accel_km, 
                     pos_accel, 
                     neg_accel, 
                     accel_sqrd, 
                     freq, E_pow_2ni, 
                     distance, 
                     cum_distance, 
                     q_viaje, 
                     vsp_approx, 
                     VSP,
                     V_3,
                     Ca_plus,
                     fuel_flow,
                     inst_aver_fuel
                    ])
        data_labels = ["Delta T [s]", "Speed [m/s]", "Acceleration [m/s^2]", "Accel/km [km^-1]",
                       "Positive acceleration [m/s^2]", "Negative acceleration [m/s^2]", 
                       "Acceleration^2  [(m/s^2)^2]", "Frequency", "E^2*n(i)", 
                       "Distance [m]", "Accumulated distance [m]",
                       "Q de viaje [m^2/a^2]", "VSP approx [kW/t]", 
                       "VSP [kW/t]", "V^3 [m^3/s^2]", "Ca+ [m^2/s^2]",
                       "Fuel flow [ml/s]", "Instantaneous average fuel [l/100km]"]
    return [data, data_labels]
M_trip_data = get_trip_values(viaje_M)
DataFrame(M_trip_data[0],columns = M_trip_data[1])
# Extraer unicamente los datos del viaje M = 16 
M = 5
viaje_O_mask = raw_data_['Viaje'] == M
viaje_O = raw_data_[viaje_O_mask]

O_trip_data = get_trip_values(viaje_O)
DataFrame(O_trip_data[0],columns = O_trip_data[1])