import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from textwrap import wrap



data = pd.read_csv('../input/Indicators.csv')



countries = ['BGD', 'BTN', 'IND', 'MDV', 'NPL', 'PAK', 'LKA']

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
mort_data = data.query("IndicatorCode == 'SH.DYN.MORT'").sort_values(by = 'Year', ascending = True)

plt.figure(figsize = (15, 8))



for i in range(len(countries)):

    count_mort = mort_data[mort_data.CountryCode == countries[i]]

    plt.plot(count_mort['Year'], count_mort['Value'], color = colors[i], label = count_mort['CountryName'].unique()[0])



plt.ylabel('Under-5 mortality rate, per 1000', fontsize = 14)

plt.title("Comparing the under-5 mortality rates of seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()
mort_data_ma = data.query("IndicatorCode == 'SH.STA.MMRT'").sort_values(by = 'Year', ascending = True)

plt.figure(figsize = (15, 8))



for i in range(len(countries)):

    count_mort_ma = mort_data_ma[mort_data_ma.CountryCode == countries[i]]

    plt.plot(count_mort_ma['Year'], count_mort_ma['Value'], color = colors[i], label = count_mort_ma['CountryName'].unique()[0])



plt.ylabel('Maternal mortality ratio, per 100,000 births', fontsize = 14)

plt.title("Comparing the maternal mortality rates of seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()
gdp_percapita = data.query("IndicatorCode == 'NY.GDP.PCAP.CD'").sort_values(by = 'Year', ascending = True)

plt.figure(figsize = (15, 8))



for i in range(len(countries)):

    count_gdp = gdp_percapita[gdp_percapita.CountryCode == countries[i]]

    plt.plot(count_gdp['Year'], count_gdp['Value'], color = colors[i], label = count_gdp['CountryName'].unique()[0])



plt.ylabel('GDP per capita (current USdollar)', fontsize = 14)

plt.title("Comparing the per capita GDP of seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()
gdp_health = data.query("IndicatorCode == 'SH.XPD.TOTL.ZS'").sort_values(by = 'Year', ascending = True)

plt.figure(figsize = (15, 8))



for i in range(len(countries)):

    count_health = gdp_health[gdp_health.CountryCode == countries[i]]

    plt.plot(count_health['Year'], count_health['Value'], color = colors[i], label = count_health['CountryName'].unique()[0])



plt.xticks(range(min(count_health['Year']), max(count_health['Year'])+1, 4))

plt.ylabel('Health expenditure, total (% of GDP)', fontsize = 14)

plt.title("Comparing the total health expenditure of seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()
anemia_wmn = data.query("IndicatorCode == 'SH.PRG.ANEM'").sort_values(by = 'Year', ascending = True)

anemia_chld = data.query("IndicatorCode == 'SH.ANM.CHLD.ZS'").sort_values(by = 'Year', ascending = True)



plt.figure(figsize = (15, 8))



for i in range(len(countries)):

    count_aw = anemia_wmn[anemia_wmn.CountryCode == countries[i]]

    plt.plot(count_aw['Year'], count_aw['Value'], color = colors[i], label = count_aw['CountryName'].unique()[0])



plt.ylabel('Prevalence of anemia among pregnant women (%)', fontsize = 14)

plt.title("Comparing the prevelance of anemia among pregnant women in seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()



plt.figure(figsize = (15, 8))



for i in range(len(countries)):

    count_ac = anemia_chld[anemia_chld.CountryCode == countries[i]]

    plt.plot(count_ac['Year'], count_ac['Value'], color = colors[i], label = count_ac['CountryName'].unique()[0])



plt.ylabel('Prevalence of anemia among children (% of children under 5)', fontsize = 14)

plt.title("Comparing the prevelance anemia among children in seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()
plt.figure(figsize = (15, 8))



impr_san = data.query("IndicatorCode == 'SH.STA.ACSN'").sort_values(by = 'Year', ascending = True)



for i in range(len(countries)):

    count_san = impr_san[impr_san.CountryCode == countries[i]]

    plt.plot(count_san['Year'], count_san['Value'], color = colors[i], label = count_san['CountryName'].unique()[0])



plt.ylabel('Improved sanitation facilities (% of population with access)', fontsize = 14)

plt.title("Comparing the sanitation facilities of seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()
cause_dth = data.query("IndicatorCode == 'SH.DTH.COMM.ZS'").sort_values(by = 'Year', ascending = True)

plt.figure(figsize = (15, 8))



for i in range(len(countries)):

    count_cause = cause_dth[cause_dth.CountryCode == countries[i]]

    plt.plot(count_cause['Year'], count_cause['Value'], color = colors[i], label = count_cause['CountryName'].unique()[0])



plt.ylabel("\n".join(wrap("Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total)", 60)), fontsize = 14)

plt.title("Comparing the cause of death of seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()
adl_fertl = data.query("IndicatorCode == 'SP.ADO.TFRT'").sort_values(by = 'Year', ascending = True)

plt.figure(figsize = (15, 8))



for i in range(len(countries)):

    count_fert = adl_fertl[adl_fertl.CountryCode == countries[i]]

    plt.plot(count_fert['Year'], count_fert['Value'], color = colors[i], label = count_fert['CountryName'].unique()[0])



plt.ylabel("Adolescent fertility rate (births per 1,000 women ages 15-19)", fontsize = 14)

plt.title("Comparing the adolescent fertility rates of seven South Asian countries", fontsize = 16)

plt.legend(fontsize = 14)

plt.show()