import numpy as np 
import pandas as pd
loans = pd.read_csv("../input/kiva_loans.csv").set_index("id")
themes = pd.read_csv("../input/loan_theme_ids.csv").set_index("id")
locations = pd.read_csv("../input/loan_themes_by_region.csv",encoding = "ISO-8859-1").set_index(['Loan Theme ID','country','region'])

loans  = loans.join(themes['Loan Theme ID'],how='left').join(locations,on=['Loan Theme ID','country','region'],rsuffix = "_")
matched_pct = round(100*loans['geo'].count()/loans.shape[0])
message = "{}% of loans in kiva_loans.csv were successfully merged with loan_themes_by_region.csv"
print(message.format(matched_pct))
