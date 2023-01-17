import pandas as pd

pd.set_option('max_rows', 5)
crashes = pd.read_csv("../input/planecrashinfo_20181121001952.csv")
# add year 
crashes['CrashYear'] = pd.DatetimeIndex(pd.to_datetime(crashes['date'])).year
crashes['CrashMonth'] = pd.DatetimeIndex(pd.to_datetime(crashes['date'])).month
crashes['CrashDay'] = pd.DatetimeIndex(pd.to_datetime(crashes['date'])).day
crashes
crashes.groupby("CrashYear").size().plot(figsize=(25, 6))
crashes.groupby("CrashMonth").size().plot.bar(figsize=(25, 6))
#Top 50 operators
operators = crashes.groupby("operator").size()
ordered_operators = operators.sort_values(ascending = False)
ordered_operators = ordered_operators.iloc[:50]
ordered_operators.plot.bar(figsize=(25, 6))
#Top 50 models
models = crashes.groupby("ac_type").size()
ordered_models = models.sort_values(ascending = False)
ordered_models = ordered_models.iloc[:50]
ordered_models.plot.bar(figsize=(25, 6))
aeroflot = crashes[crashes["operator"] == "Aeroflot"]
aeroflot_ordered_models = aeroflot.groupby("ac_type").size().sort_values(ascending = False)
aeroflot_ordered_models.plot.bar(figsize=(25, 6))
not_aeroflot_models = crashes[crashes["operator"] != "Aeroflot"]
not_aeroflot_ordered_models = not_aeroflot_models.groupby("ac_type").size().sort_values(ascending = False)
not_aeroflot_ordered_models.iloc[:50].plot.bar(figsize=(25, 6))
aeroflot.groupby("CrashYear").size().plot(figsize=(25, 6))
douglas_dc3 = crashes[crashes["ac_type"] == "Douglas DC-3"]
douglas_dc3_ordered = douglas_dc3.groupby("operator").size().sort_values(ascending = False)
douglas_dc3_ordered.iloc[:50].plot.bar(figsize=(25, 6))
douglas_dc3.groupby("CrashYear").size().plot(figsize=(25, 6))