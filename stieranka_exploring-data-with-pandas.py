import pandas as pd
import numpy as np
f500 = pd.read_csv("../input/f500.csv", index_col=0)
f500.index.name = None
f500.loc[f500["previous_rank"] == 0, "previous_rank"] == np.nan
f500
fifth_row = f500.iloc[4]
first_three_rows = f500.iloc[0:3]
first_seventh_row_slice = f500.iloc[[0,6],0:5]
print(fifth_row)
print(first_three_rows)
print(first_seventh_row_slice)
print(f500.iloc[:5, :3])
f500 = pd.read_csv('../input/f500.csv')
f500.loc[f500["previous_rank"] == 0, "previous_rank"] = np.nan
sorted_emp = f500.sort_values("employees", ascending=False)
sorted_emp
top5_emp = sorted_emp.iloc[:5]
top5_emp
previously_ranked = f500[f500["previous_rank"].notnull()]
rank_change = previously_ranked["rank"] - previously_ranked["previous_rank"]
rank_change
cols = ["company", "revenues", "country"]
f500_sel = f500[cols].head()
f500_sel
over_265 = f500_sel["revenues"] > 265000
china = f500_sel["country"] == "China"
combined = over_265 & china
final_cols = ["company", "revenues"]
result = f500_sel.loc[combined, final_cols]
result
big_rev_neg_profit = (f500["revenues"] > 100000) & (f500["profits"] < 0)
big_rev_neg_profit = f500[big_rev_neg_profit]
print(big_rev_neg_profit)
tech_outside_usa = (f500["sector"] == "Technology") & ~(f500["country"] == "USA")
tech_outside_usa = f500[tech_outside_usa].head()
print(tech_outside_usa)
f500["rank_change"] = rank_change
f500
top_employer_by_country = {}

countries = f500["country"].unique()
for c in countries:
    selected_rows = f500[f500["country"] == c]
    sorted_rows = selected_rows.sort_values("employees", ascending=False)
    top_employer = sorted_rows.iloc[0]
    employer_name = top_employer["company"]
    top_employer_by_country[c] = employer_name

top_employer_by_country
f500["roa"] = f500["profits"] / f500["assets"]

top_roa_by_sector = {}
for sector in f500["sector"].unique():
    is_sector = f500["sector"] == sector
    sector_companies = f500.loc[is_sector]
    top_company = sector_companies.sort_values("roa",ascending=False).iloc[0]
    company_name = top_company["company"]
    top_roa_by_sector[sector] = company_name
top_roa_by_sector
