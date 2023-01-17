# Import data analysis tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Matplotlib Style
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

# Matplotlib axis formats
import matplotlib.ticker as mtick
dfmt = '${x:,.0f}'
dtick = mtick.StrMethodFormatter(dfmt)
fmt = '{x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
# Import the data
st_summaries = pd.read_csv('../input/elsect_summary.csv')
# Preview the data
# (Some NaN's expected for 1992)
st_summaries.tail()
# Idaho Data
idaho_data = st_summaries.loc[st_summaries['STATE'] == 'Idaho']
idaho_data.tail()
# National Median Data

# Determine median values for all columns, for a given year
def mdn_by_year(year):
    year_data = st_summaries.loc[st_summaries['YEAR'] == year]
    year_mdn = {}
    for key in year_data.columns.values:
        if 'STATE' != key:
            year_mdn[key] = year_data[key].median()
    year_mdn['STATE'] = 'Median'
    return year_mdn

# Build the "median state"
years = range(1992,2016)
mdn_state = []
for year in years:
    year_mdn = mdn_by_year(year)
    mdn_state.append(year_mdn)

mdn_data = pd.DataFrame(mdn_state, columns=idaho_data.columns.values)
mdn_data.tail()
# Big picture finances; revenue vs. expenditure
def plot_rev_and_expend(data):
    plt.plot(data['YEAR'], data['TOTAL_REVENUE'], color='k')
    plt.plot(data['YEAR'], data['TOTAL_EXPENDITURE'], color='r')
    plt.gca().yaxis.set_major_formatter(dtick)
    
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Revenue vs. Expenditure')
plot_rev_and_expend(idaho_data)
plt.subplot(1, 2, 2)
plot_rev_and_expend(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()
# Revenue breakdown
def plot_revenue_breakdown(data):
    plt.plot(data['YEAR'], data['TOTAL_REVENUE'], color='k')
    plt.plot(data['YEAR'], data['STATE_REVENUE'])
    plt.plot(data['YEAR'], data['FEDERAL_REVENUE'])
    plt.plot(data['YEAR'], data['LOCAL_REVENUE'])
    plt.gca().yaxis.set_major_formatter(dtick)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Revenue Breakdown')
plot_revenue_breakdown(idaho_data)
plt.subplot(1, 2, 2)
plot_revenue_breakdown(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()
# Revenue Percentages
def plot_revenue_percent(data):
    data['STATE_PERCENTAGE'] = data['STATE_REVENUE'] / data['TOTAL_REVENUE']
    data['FEDERAL_PERCENTAGE'] = data['FEDERAL_REVENUE'] / data['TOTAL_REVENUE']
    data['LOCAL_PERCENTAGE'] = data['LOCAL_REVENUE'] / data['TOTAL_REVENUE']
    plt.plot(data['YEAR'], data['STATE_PERCENTAGE'])
    plt.plot(data['YEAR'], data['FEDERAL_PERCENTAGE'])
    plt.plot(data['YEAR'], data['LOCAL_PERCENTAGE'])

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Revenue Percentage Breakdown')
plot_revenue_percent(idaho_data)
plt.subplot(1, 2, 2)
plot_revenue_percent(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()
# Expenditure breakdown
def plot_expenditure_breakdown(data):
    plt.plot(data['YEAR'], data['TOTAL_EXPENDITURE'], color='r')
    plt.plot(data['YEAR'], data['INSTRUCTION_EXPENDITURE'])
    plt.plot(data['YEAR'], data['SUPPORT_SERVICES_EXPENDITURE'])
    plt.plot(data['YEAR'], data['CAPITAL_OUTLAY_EXPENDITURE'])
    plt.plot(data['YEAR'], data['OTHER_EXPENDITURE'])
    plt.gca().yaxis.set_major_formatter(dtick)


plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Expenditure Breakdown')
plot_expenditure_breakdown(idaho_data)
plt.subplot(1, 2, 2)
plot_expenditure_breakdown(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()
# Expenditure Percentages
def plot_expenditure_percent(data):
    data['INSTRUCTION_PERCENTAGE'] = data['INSTRUCTION_EXPENDITURE'] / data['TOTAL_EXPENDITURE']
    data['SUPPORT_PERCENTAGE'] = data['SUPPORT_SERVICES_EXPENDITURE'] / data['TOTAL_EXPENDITURE']
    data['OUTLAY_PERCENTAGE'] = data['CAPITAL_OUTLAY_EXPENDITURE'] / data['TOTAL_EXPENDITURE']
    data['OTHER_PERCENTAGE'] = data['OTHER_EXPENDITURE'] / data['TOTAL_EXPENDITURE']
    plt.plot(data['YEAR'], data['INSTRUCTION_PERCENTAGE'])
    plt.plot(data['YEAR'], data['SUPPORT_PERCENTAGE'])
    plt.plot(data['YEAR'], data['OUTLAY_PERCENTAGE'])
    plt.plot(data['YEAR'], data['OTHER_PERCENTAGE'])

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Expenditure Percentage Breakdown')
plot_expenditure_percent(idaho_data)
plt.subplot(1, 2, 2)
plot_expenditure_percent(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()
# State Enrollment
def plot_enroll(data):
    plt.plot(data['YEAR'], data['ENROLL'])
    plt.gca().yaxis.set_major_formatter(tick)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Enrollment')
plot_enroll(idaho_data)
plt.subplot(1, 2, 2)
plot_enroll(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()
# Expenditure per Student
def plot_expenditure_enroll(data):
    data['EXPENDITURE_PER_STUDENT'] = data['TOTAL_EXPENDITURE'] / data['ENROLL']
    plt.plot(data['YEAR'], data['EXPENDITURE_PER_STUDENT'])
    
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title('Expenditure per Student')
plot_expenditure_enroll(idaho_data)
plt.subplot(1, 2, 2)
plot_expenditure_enroll(mdn_data)
plt.tight_layout()
plt.legend()
plt.show()
# State enrollment by year
plt.figure(figsize=(10,5))
plt.title('State Enrollment by Year')
plt.scatter(st_summaries['YEAR'], st_summaries['ENROLL'], alpha=0.5)
plt.show()
# Zoom and enchance
plt.figure(figsize=(10,5))
plt.title('State Enrollment by Year (Zoomed-In)')
plt.scatter(st_summaries['YEAR'], st_summaries['ENROLL'], alpha=0.5)
plt.ylim((0,2000000))
plt.show()
data_1995 = st_summaries.loc[st_summaries['YEAR'] == 1995]
data_2005 = st_summaries.loc[st_summaries['YEAR'] == 2005]
data_2015 = st_summaries.loc[st_summaries['YEAR'] == 2015]

plt.figure(figsize=(10,5))
plt.subplot(1, 3, 1)
plt.violinplot(data_1995['ENROLL'])
plt.ylim((0,5000000))
plt.subplot(1, 3, 2)
plt.violinplot(data_2005['ENROLL'])
plt.ylim((0,5000000))
plt.gca().yaxis.set_ticks([])
plt.subplot(1, 3, 3)
plt.violinplot(data_2015['ENROLL'])
plt.ylim((0,5000000))
plt.gca().yaxis.set_ticks([])
plt.show()
