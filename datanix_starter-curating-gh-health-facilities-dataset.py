import pandas as pd

df_health_facilities = pd.read_csv("../input/health-facilities-gh.csv")
df_health_facilities.head(5)
# Filter out facilities of interest Type=='Others'
df_hf_others = df_health_facilities[df_health_facilities['Type']=='Others']
df_hf_others.head(15)
df_hf_others.loc[df_hf_others.FacilityName.str.contains("Feeding|Nutrition|Rehabilitation"),'Type'] = "Feeding Center"

df_hf_others