# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt

%matplotlib inline

import urllib.request

import re

from scipy.integrate import odeint

#get latest files containing case hospital deaths, by_sex and by_age data

#ATTN : Timeout error when hitting github, so use uploaded data as of 4/22

nyc_file_url = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/case-hosp-death.csv"

download_file_chd = "./case-hosp-death.csv"

nyc_by_sex_url = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/by-sex.csv"

download_file_bs = "./by_sex.csv"

nyc_by_age_url="https://raw.githubusercontent.com/nychealth/coronavirus-data/master/by-age.csv"

download_file_ba = "./by_age.csv"
#get pdf file with race data on covid cases

#ATTN : Timeout error when hitting github, so use uploaded data as of 4/22

nyc_by_race_url="https://www1.nyc.gov/assets/doh/downloads/pdf/imm/covid-19-deaths-race-ethnicity-04082020-1.pdf"

download_file_race="./covid-19-deaths-race-ethnicity-04082020-1.pdf"

download_file_chd = "/kaggle/input/nyc-covid-data-422/" + download_file_chd.split("/")[1]

download_file_bs = "/kaggle/input/nyc-covid-data-422/" + download_file_bs.split("/")[1]

download_file_ba = "/kaggle/input/nyc-covid-data-422/" + download_file_ba.split("/")[1]

download_file_race = "/kaggle/input/nyc-covid-data-422/" + download_file_race.split("/")[1]
def get_number_dead(race,line):

    search_substring = line.find(race)

    if search_substring!= -1:

        return float(re.findall(r'\d+\n',line[search_substring:])[0])
def get_text_race_table(fn):

    pdfFileObj = open(fn, 'rb')

    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

    # our file has just 1 page

    pageObj = pdfReader.getPage(0)

    pagecontent = pageObj.extractText()

    #table data after line 5

    lines = pagecontent.split(".",maxsplit=5)

    return lines[5]
#setup dataframe for covid rates by age,sex and the general case_hospitalization_death rate

by_age_df = pd.read_csv(download_file_ba)

by_sex_df = pd.read_csv(download_file_bs)

chd_df = pd.read_csv(download_file_chd)
def drop_last_row(df):

    df.drop(df.tail(1).index, inplace = True)
def define_cumulative_case_sum(df, case_col):

    cum_case_col = "CUMULATIVE " + case_col

    df[cum_case_col] = df[case_col].cumsum()
def plot_rate(df, xcol,ycol,plot_title="NYC rate"):

    fig = px.bar(df, x=xcol, y=ycol, labels={'x':'DATE'})

    fig.update_layout(title=plot_title)

    fig.show()
def plot_pie(df, values_col, names_col, plot_title="NYC Age/Sex Covid Rate"):

    fig = px.pie(df, values = values_col, names = names_col, title=plot_title)

    fig.show()
#Preprocessing

#Fill unavailable data with 0

by_age_df = by_age_df.fillna(0)

by_sex_df = by_sex_df.fillna(0)

chd_df = chd_df.fillna(0)

define_cumulative_case_sum(chd_df,"NEW_COVID_CASE_COUNT" )

define_cumulative_case_sum(chd_df,"HOSPITALIZED_CASE_COUNT" )

define_cumulative_case_sum(chd_df,"DEATH_COUNT" )

# Needed to access DATE OF INTEREST as column

mod_bs = by_sex_df.reset_index()

mod_ba = by_age_df.reset_index()

mod_chd = chd_df.reset_index()

# Needed to remove citywide totals before pie plot generation

drop_last_row(mod_bs)

drop_last_row(mod_ba)
plot_rate(mod_chd,"DATE_OF_INTEREST","NEW_COVID_CASE_COUNT", "NYC Covid New Case Rate")
plot_rate(mod_chd,"DATE_OF_INTEREST","CUMULATIVE NEW_COVID_CASE_COUNT", "NYC Cumulative Covid Case Rate")
plot_rate(mod_chd,"DATE_OF_INTEREST","HOSPITALIZED_CASE_COUNT", "NYC Covid Hospitalized Case Rate")
plot_rate(mod_chd,"DATE_OF_INTEREST","CUMULATIVE HOSPITALIZED_CASE_COUNT", "NYC Cumulative Covid Hospitalisation Rate")
plot_rate(mod_chd,"DATE_OF_INTEREST","DEATH_COUNT", "NYC Covid Death Rate")
plot_rate(mod_chd,"DATE_OF_INTEREST","CUMULATIVE DEATH_COUNT", "NYC Cumulative Covid Death Rate")
plot_pie(mod_bs, "COVID_CASE_RATE","SEX_GROUP","NYC Covid Case Rate By Gender")
plot_pie(mod_bs, "HOSPITALIZED_CASE_RATE","SEX_GROUP","NYC Covid Hopitalized Case Rate By Gender")
plot_pie(mod_bs, "DEATH_RATE","SEX_GROUP", "NYC Covid Death Rate by Sex")
plot_pie(mod_ba, "COVID_CASE_RATE","AGE_GROUP", "NYC Covid Case Rate by Age")
plot_pie(mod_ba, "HOSPITALIZED_CASE_RATE","AGE_GROUP", "NYC Hospitalization Case Rate by Age")
plot_pie(mod_ba, "DEATH_RATE","AGE_GROUP", "NYC Covid Death Rate by Age")
def plotCaseRate(plotS = False, plotI = False, plotR = False, plotD=False, start="03-02-2020t00:00"):

    datetime_col = pd.date_range(start, periods=len(t), freq="D")

    fig = plt.figure(facecolor = 'w')

    ax = fig.add_subplot(111, axisbelow = True)

    # replacing t with datetime_col.date in ax.plot

    if plotS:

        ax.plot(datetime_col.date, S, 'b', alpha=0.5, lw=2, label = "Susceptible")

    if plotI:

        ax.plot(datetime_col.date, I, 'r', alpha=0.5, lw=2, label = "Infected")

    if plotR:

        ax.plot(datetime_col.date, R, 'g', alpha=0.5, lw=2, label = "Recovered")

    if plotD:

        ax.plot(datetime_col.date, D, 'o', alpha=0.5, lw=2, label = "Dead")

    ax.set_xlabel('Time/days')

    ax.set_ylabel('Number of people')

    ax.yaxis.set_tick_params(length=0)

    ax.xaxis.set_tick_params(length=0)

    ax.grid(b=True, which = 'major', c='w', lw=2, ls='-')

    legend = ax.legend()

    legend.get_frame().set_alpha(0.5)

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    for spine in ('top', 'right', 'bottom', 'left'):

        ax.spines[spine].set_visible(False)

    plt.show()   
N = 8.7 * 1000000 # population of NYC 

I0, R0 = 1,0

S0 = N - I0 - R0

# contact rate - guesstimate start at 0.2,1.0, 0.6

# recovery rate - 2 weeks for 80 % and 3 to 6 weeks for 20 % approx 17 days

beta, gamma = 0.2, 1./17

# Time points in days

t = np.linspace(0,60, 60)
def derivSIR(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt
# We see from the cumulative infections graph above that in about 45 days total infected is about 120K

# With beta at 0.35 we see more than 100000 infections after 40 days BUT the rate of growth here is higher than seen

I0, R0 = 1,0

S0 = N - I0 - R0

t = np.linspace(0,60, 60)

beta, gamma = 0.35, 1./17

y0 = S0, I0, R0

ret = odeint(derivSIR, y0, t, args =(N, beta, gamma))

S, I, R = ret.T

plotCaseRate(False, True, False, False, start="03-02-2020t00:00")
# With stats on 3/21, cumulative infected were 20595

# number of recovered guesstimated using 10% for closed and further 50% of that recovered

I0, R0 = 20595,1250

S0 = N - I0 - R0

t = np.linspace(0,45, 45)

beta, gamma = 0.12, 1./17

y0 = S0, I0, R0

ret = odeint(derivSIR, y0, t, args =(N, beta, gamma))

S, I, R = ret.T

plotCaseRate(False, True, False, False, start="03-21-2020t00:00")
def derivSIRD(y, t, N, beta, gamma, mu):

    S, I, R, D = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I - mu * I

    dRdt = gamma * I

    dDdt = mu * I

    return dSdt, dIdt, dRdt, dDdt
# Adding death numbers to the first model we tried with SIR starting beginning March

# mu estimated at average mrtality after 5 weeks

I0, R0, D0 = 1,0, 0

S0 = N - I0 - R0 - D0

t = np.linspace(0,60, 60)

beta, gamma = 0.35, 1./17

mu = 1./35

y0 = S0, I0, R0, D0

ret = odeint(derivSIRD, y0, t, args =(N, beta, gamma, mu))

S, I, R, D = ret.T

plotCaseRate(False, True, False, True)
# Adding death numbers to the second model for SIR starting about 3/21

I0, R0, D0 = 20595,1250,158

S0 = N - I0 - R0 - D0

t = np.linspace(0,45, 45)

beta, gamma = 0.12, 1./17

mu = 1./35

y0 = S0, I0, R0, D0

ret = odeint(derivSIRD, y0, t, args =(N, beta, gamma, mu))

S, I, R, D = ret.T

plotCaseRate(False, True, False, True,start="03-21-2020t00:00")
# beta, gamma, mu from reference

I0, R0, D0 = 20595,1250,158

S0 = N - I0 - R0 - D0

t = np.linspace(0,60, 60)

beta, gamma = 0.319, 0.16

mu = 0.0005

y0 = S0, I0, R0, D0

ret = odeint(derivSIRD, y0, t, args =(N, beta, gamma, mu))

S, I, R, D = ret.T

plotCaseRate(False, True, False, False,start="03-21-2020t00:00")

plotCaseRate(False, False, False, True,start="03-21-2020t00:00")
# beta, gamma, mu from reference

I0, R0, D0 = 20595,1250,158

S0 = N - I0 - R0 - D0

t = np.linspace(0,365, 365)

beta, gamma = 0.319,0.16

mu = 0.0005

y0 = S0, I0, R0, D0

ret = odeint(derivSIRD, y0, t, args =(N, beta, gamma, mu))

S, I, R, D = ret.T

plotCaseRate(True, True, True, True,start="03-21-2020t00:00")