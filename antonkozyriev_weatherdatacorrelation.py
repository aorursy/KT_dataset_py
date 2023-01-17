import numpy as np

import pandas as pd



def v_norm(x):

    return x / np.linalg.norm(x)



dataframe = pd.read_csv("/kaggle/input/weather-dataset/weatherHistory.csv")

vals = dataframe.values



summaries = [row[1] for row in vals]

temperature = v_norm([row[3] for row in vals])

a_temperature = v_norm([row[4] for row in vals])

humidity = v_norm([row[5] for row in vals])

wind_speed = v_norm([row[6] for row in vals])

pressure = v_norm([row[10] for row in vals])



unique_summaries = sorted(set(summaries))



dict_summaries = {u:i for i, u in enumerate(unique_summaries)}

arr_summaries = np.array(unique_summaries)



summaries_as_int = v_norm(np.array([dict_summaries[s] for s in summaries]))



np.corrcoef([summaries_as_int, temperature, a_temperature, humidity, wind_speed, pressure])