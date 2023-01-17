import pandas as pd

import numpy as np
df_train = pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")

df_train
diff_cases = [df_train['ConfirmedCases'][i+1] - df_train['ConfirmedCases'][i] for i in range(len(df_train)-1)]

diff_fatalities = [df_train['Fatalities'][i+1] - df_train['Fatalities'][i] for i in range(len(df_train)-1)]
diff_cases
diff_fatalities
diff_fatal_case = [f/c if c != 0 else 0 for f, c in zip(diff_fatalities, diff_cases)]
diff_fatal_case
np.mean([x for x in diff_fatal_case if x != 0])
increase = [df_train['ConfirmedCases'][i+1]/diff_cases[i] if diff_cases[i] != 0 else 0 for i in range(len(diff_cases))]
increase
increase_rate = np.mean([x for x in increase if x != 0])

increase_rate
fatality_rate = np.mean([x for x in diff_fatal_case if x != 0])

fatality_rate
def n_step_pred(cases, n):

    new_cases = []

    new_fatalities = []

    for i in range(n):

        if i == 0:

            new_cases.append(cases + int(cases/increase_rate))

        else:

            new_cases.append(new_cases[i-1] + int(new_cases[i-1]/increase_rate))

        new_fatalities.append(int(new_cases[i] * fatality_rate))

    

    pred_df = pd.DataFrame(list(zip(forecast_id ,new_cases, new_fatalities)), columns=["ForecastId", "ConfirmedCases", "Fatalities"])

    pred_df = pred_df.set_index("ForecastId")

    return pred_df
test_df = pd.read_csv("../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")

forecast_id = list(test_df["ForecastId"])

test_df.head()
predicted = n_step_pred(df_train['ConfirmedCases'].iloc[-1], len(test_df))
predicted.to_csv("submission.csv")