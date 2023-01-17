import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
 
inspections = pd.read_csv("../input/restaurant-and-market-health-inspections.csv")
violations = pd.read_csv("../input/restaurant-and-market-health-violations.csv")
inspections.info()
inspections.head(5)
violations.info()
violations.head(5)
pd.DataFrame(round(violations['pe_description'].value_counts(normalize=True)*100,2))
pd.DataFrame(round(violations['program_status'].value_counts(normalize=True)*100,2))
pd.DataFrame(round(violations['service_description'].value_counts(normalize=True)*100,2))
pd.DataFrame(round(violations['violation_status'].value_counts(normalize=True)*100,4))
violations['Net Effective Score'] = violations['score'] - violations['points']
violations['Year'] = violations['activity_date'].str[:4]
violations_group = pd.pivot_table(violations, values='Net Effective Score', index=[ 'violation_status'],
                    columns=['Year'], aggfunc=np.mean)
pd.DataFrame(violations_group.to_records())
vio_Housing = violations[violations['violation_status'] == "HOUSING NON-CRITICAL"]
vio_Compliance = violations[violations['violation_status'] == "OUT OF COMPLIANCE"]
vio_Violation = violations[violations['violation_status'] == "VIOLATION"]
vio_Housing_group = pd.pivot_table(vio_Housing, values='Net Effective Score', index=['pe_description','violation_description','facility_name','facility_address','owner_name','grade','program_status','service_description','score','points'],
                    columns=['Year'], aggfunc=np.mean)
pd.DataFrame(vio_Housing_group.to_records())
vio_Violation_group = pd.pivot_table(vio_Violation, values='Net Effective Score', index=['pe_description','violation_description','facility_name','facility_address','owner_name','grade','program_status','service_description','score','points'],
                    columns=['Year'], aggfunc=np.mean)
pd.DataFrame(vio_Violation_group.to_records())
pd.DataFrame(round(violations['pe_description'].value_counts(normalize=True)*100,2)).head(4)
box1 = vio_Compliance[(vio_Compliance['pe_description'] == "RESTAURANT (0-30) SEATS HIGH RISK")|
                      (vio_Compliance['pe_description'] == "RESTAURANT (31-60) SEATS HIGH RISK") |
                      (vio_Compliance['pe_description'] == "RESTAURANT (0-30) SEATS MODERATE RISK") |
                      (vio_Compliance['pe_description'] == "RESTAURANT (61-150) SEATS HIGH RISK")]
sns.set()
fig, ax = plt.subplots()
fig.set_size_inches(15, 7)

ax = sns.boxplot(x='pe_description', y='Net Effective Score',hue="Year", data=box1)
ax.set_xlabel("pe_description",fontsize=15)
ax.set_ylabel("Net Effective Score",fontsize=15)
ax.tick_params(labelsize=13)
plt.xticks(rotation=9)


sns.set()
fig, ax = plt.subplots()
fig.set_size_inches(15, 7)

ax = sns.violinplot(x='pe_description', y='Net Effective Score',hue="grade", data=box1)
ax.set_xlabel("pe_description",fontsize=15)
ax.set_ylabel("Net Effective Score",fontsize=15)
ax.tick_params(labelsize=13)
plt.xticks(rotation=9)


pd.set_option("display.max_rows", 20)
vio_Compliance_peDes_group = pd.pivot_table(vio_Compliance, values='Net Effective Score', index=['pe_description'],columns=['Year'], aggfunc=np.mean)
vio_Compliance_peDes = pd.DataFrame(vio_Compliance_peDes_group.to_records())
vio_Compliance_peDes = pd.melt(vio_Compliance_peDes, id_vars='pe_description', value_vars=vio_Compliance_peDes.columns.drop('pe_description')).rename(columns={"variable":"Year", "value":"Avg. Net Effective Score"})
vio_Compliance_peDes
sns.set_style("ticks")
fig, ax = plt.subplots()
fig.set_size_inches(15, 6)

ax = sns.lineplot(x="pe_description", y="Avg. Net Effective Score",hue="Year",palette=["r", "b","g","y"], data=vio_Compliance_peDes)
ax.set_xlabel("pe_description",fontsize=15)
ax.set_ylabel("Avg. Net Effective Score",fontsize=15)
ax.grid(True)
ax.tick_params(labelsize=13)
plt.xticks(rotation=-270)
sns.set()

g = sns.FacetGrid(vio_Compliance, col='pe_description',col_wrap=3,height=4, aspect=1.5)
g = g.map(sns.boxplot, "Year", "Net Effective Score")

sns.set()

g = sns.FacetGrid(vio_Compliance, col='grade',col_wrap=3,height=8, aspect=0.6)
g = g.map(sns.violinplot, "Year", "Net Effective Score")
pd.set_option("display.max_rows", 20)
vio_Compliance_VioDes_group = pd.pivot_table(vio_Compliance, values='Net Effective Score', index=['violation_description'],columns=['Year'], aggfunc=np.mean)
vio_Compliance_VioDes = pd.DataFrame(vio_Compliance_VioDes_group.to_records())
vio_Compliance_VioDes
max_score =vio_Compliance.loc[vio_Compliance.groupby(['Year','violation_description'])['Net Effective Score'].idxmax()]
pd.set_option("display.max_rows", 40)
pd.pivot_table(max_score, values='Net Effective Score', index=['violation_description','Year','grade','facility_name'])
min_score =vio_Compliance.loc[vio_Compliance.groupby(['Year','violation_description'])['Net Effective Score'].idxmin()]
pd.set_option("display.max_rows", 40)
pd.pivot_table(min_score, values='Net Effective Score', index=['violation_description','Year','grade','facility_name'])









