% matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

sns.set_style("white")



# read dataset

country = pd.read_csv('../input/Country.csv')

indicators = pd.read_csv('../input/Indicators.csv')
# Primary Gross Enrollment Rate (total)

ind_pri_enrr = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # India

chn_pri_enrr = indicators[(indicators.CountryCode=="CHN") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # China

jpn_pri_enrr = indicators[(indicators.CountryCode=="JPN") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # Japan

rus_pri_enrr = indicators[(indicators.CountryCode=="RUS") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # Russia

idn_pri_enrr = indicators[(indicators.CountryCode=="IDN") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # Indonesia



# Secondary Gross Enrollment Rate (total)

ind_sec_enrr = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # India

chn_sec_enrr = indicators[(indicators.CountryCode=="CHN") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # China

jpn_sec_enrr = indicators[(indicators.CountryCode=="JPN") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # Japan

rus_sec_enrr = indicators[(indicators.CountryCode=="RUS") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # Russia

idn_sec_enrr = indicators[(indicators.CountryCode=="IDN") & (indicators.IndicatorCode=='SE.SEC.ENRR')] # Indonesia



sns.set_palette("husl")



fig = plt.figure()

plt.title('Primary, Gross Enrollment Ratio (GER)')

plt.plot(ind_pri_enrr.Year, ind_pri_enrr.Value,  label='India')

plt.plot(chn_pri_enrr.Year, chn_pri_enrr.Value,  label='China')

plt.plot(jpn_pri_enrr.Year, jpn_pri_enrr.Value,  label='Japan')

plt.plot(rus_pri_enrr.Year, rus_pri_enrr.Value,  label='Russia')

plt.plot(idn_pri_enrr.Year, idn_pri_enrr.Value,  label='Indonesia')

plt.ylabel('Gross Enrollment Ratio')

plt.xlabel('Year')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)



fig = plt.figure()

plt.title('Secondary, Gross Enrollment Ratio (GER)')

plt.plot(ind_sec_enrr.Year, ind_sec_enrr.Value,  label='India')

plt.plot(chn_sec_enrr.Year, chn_sec_enrr.Value,  label='China')

plt.plot(jpn_sec_enrr.Year, jpn_sec_enrr.Value,  label='Japan')

plt.plot(rus_sec_enrr.Year, rus_sec_enrr.Value,  label='Russia')

plt.plot(idn_sec_enrr.Year, idn_sec_enrr.Value,  label='Indonesia')

plt.ylabel('Gross Enrollment Ratio')

plt.xlabel('Year')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)
ind_edu_exp_US = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # India

chn_edu_exp_US = indicators[(indicators.CountryCode=="CHN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # China

jpn_edu_exp_US = indicators[(indicators.CountryCode=="JPN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # Japan

rus_edu_exp_US = indicators[(indicators.CountryCode=="RUS") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # Russia

idn_edu_exp_US = indicators[(indicators.CountryCode=="IDN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.CD')]  # Indonesia



# Adjusted savings: education expenditure (% of GNI) 

ind_edu_exp_GNI = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # India

chn_edu_exp_GNI = indicators[(indicators.CountryCode=="CHN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # China

jpn_edu_exp_GNI = indicators[(indicators.CountryCode=="JPN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # Japan

rus_edu_exp_GNI = indicators[(indicators.CountryCode=="RUS") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # Russia

idn_edu_exp_GNI = indicators[(indicators.CountryCode=="IDN") & (indicators.IndicatorCode=='NY.ADJ.AEDU.GN.ZS')]  # Indonesia



fig = plt.figure()

plt.title('Adjusted savings education expenditure (Current US Dollar)')

plt.plot(ind_edu_exp_US.Year, ind_edu_exp_US.Value, 'o-', label='India')

plt.plot(chn_edu_exp_US.Year, chn_edu_exp_US.Value, 'o-', label='China')

plt.plot(jpn_edu_exp_US.Year, jpn_edu_exp_US.Value, 'o-', label='Japan')

plt.plot(rus_edu_exp_US.Year, rus_edu_exp_US.Value, 'o-', label='Russia')

plt.plot(idn_edu_exp_US.Year, idn_edu_exp_US.Value, 'o-', label='Indonesia')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)



fig = plt.figure()

plt.title('Adjusted savings education expenditure as % of Gross National Income')

plt.plot(ind_edu_exp_GNI.Year, ind_edu_exp_GNI.Value, 'o-', label='India')

plt.plot(chn_edu_exp_GNI.Year, chn_edu_exp_GNI.Value, 'o-', label='China')

plt.plot(jpn_edu_exp_GNI.Year, jpn_edu_exp_GNI.Value, 'o-', label='Japan')

plt.plot(rus_edu_exp_GNI.Year, rus_edu_exp_GNI.Value, 'o-', label='Russia')

plt.plot(idn_edu_exp_GNI.Year, idn_edu_exp_GNI.Value, 'o-', label='Indonesia')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)
# Primary completion rate 

pri_comp_fm = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.CMPT.FE.ZS')]  # female

pri_comp_ma = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.CMPT.MA.ZS')]  # male 

pri_comp_both = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.CMPT.ZS')]  # both sexes 



# Lower Secondary completion rate

lo_sec_comp_both = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.CMPT.LO.ZS')]# both

lo_sec_comp_fe = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.CMPT.LO.FE.ZS')]# female

lo_sec_comp_ma = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.CMPT.LO.MA.ZS')]# male



# Primary Gross enrollment ratio

pri_enrr_both = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.ENRR')] # both SE.PRM.ENRR

pri_enrr_fm = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.ENRR.FE')] # female

pri_enrr_ma = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.PRM.ENRR.MA')] # male



# Secondary gross enrollment ratio

sec_enrr_both = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.ENRR')]# both

sec_enrr_fe = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.ENRR.FE')]# female

sec_enrr_ma = indicators[(indicators.CountryCode=="IND") & (indicators.IndicatorCode=='SE.SEC.ENRR.MA')]# male



fig = plt.figure()

plt.title('Gross Enrollment Ratio (GER)')

plt.plot(pri_enrr_both.Year, pri_enrr_both.Value, 'bo-', label='Primary education, all')

plt.plot(pri_enrr_fm.Year, pri_enrr_fm.Value, 'go-', label='Primary education, female')

plt.plot(pri_enrr_ma.Year, pri_enrr_ma.Value, 'ro-', label='Primary education, male')



plt.plot(sec_enrr_both.Year, sec_enrr_both.Value, 'b--', label='Secondary education, all')

plt.plot(sec_enrr_fe.Year, sec_enrr_fe.Value, 'g--', label='Secondary education, female')

plt.plot(sec_enrr_ma.Year, sec_enrr_ma.Value, 'r--', label='Secondary education, male')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)



fig = plt.figure()

plt.title('Completion Rate')

plt.plot(pri_comp_both.Year, pri_comp_both.Value, 'bo-', label='Primary, all')

plt.plot(pri_comp_fm.Year, pri_comp_fm.Value, 'go-', label='Primary, female')

plt.plot(pri_comp_ma.Year, pri_comp_ma.Value, 'ro-', label='Primary, male')



plt.plot(lo_sec_comp_both.Year, lo_sec_comp_both.Value, 'ko-', label='Lower Secondary, all')

plt.plot(lo_sec_comp_fe.Year, lo_sec_comp_fe.Value, 'co-', label='Lower secondary, female')

plt.plot(lo_sec_comp_ma.Year, lo_sec_comp_ma.Value, 'yo-', label='Lower secondary, male')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2,  borderaxespad=1.)
