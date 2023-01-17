import pandas as pd

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt



sns.set_style("ticks")

sns.set_context("paper")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_cdrs = pd.DataFrame({})

for i in range(1,8):

    df = pd.read_csv('../input/sms-call-internet-mi-2013-11-0{}.csv'.format(i), parse_dates=['datetime'])

    df_cdrs = df_cdrs.append(df)

    

df_cdrs=df_cdrs.fillna(0)

df_cdrs['sms'] = df_cdrs['smsin'] + df_cdrs['smsout']

df_cdrs['calls'] = df_cdrs['callin'] + df_cdrs['callout']

df_cdrs.head()
df_cdrs_internet = df_cdrs[['datetime', 'CellID', 'internet', 'calls', 'sms']].groupby(['datetime', 'CellID'], as_index=False).sum()

df_cdrs_internet['hour'] = df_cdrs_internet.datetime.dt.hour+24*(df_cdrs_internet.datetime.dt.day-1)

df_cdrs_internet = df_cdrs_internet.set_index(['hour']).sort_index()
f = plt.figure()



ax = df_cdrs_internet[df_cdrs_internet.CellID==5060]['internet'].plot(label='Duomo')

df_cdrs_internet[df_cdrs_internet.CellID==4259]['internet'].plot(ax=ax, label='Bocconi')

df_cdrs_internet[df_cdrs_internet.CellID==4456]['internet'].plot(ax=ax, label='Navigli')

plt.xlabel("Weekly hour")

plt.ylabel("Number of connections")

sns.despine()



# Shrink current axis's height by 10% on the bottom

box = ax.get_position()

ax.set_position([box.x0, box.y0 + box.height * 0.1,

                 box.width, box.height * 0.9])

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),

          fancybox=True, shadow=True, ncol=5)
boxplots = {

    'calls': "Calls",

    'sms': "SMS",

    "internet": "Internet CDRs"

}



df_cdrs_internet['weekday'] = df_cdrs_internet.datetime.dt.weekday



f, axs = plt.subplots(len(boxplots.keys()), sharex=True, sharey=False)

f.subplots_adjust(hspace=.35,wspace=0.1)

i = 0

plt.suptitle("")

for k,v in boxplots.items():

    ax = df_cdrs_internet.reset_index().boxplot(column=k, by='weekday', grid=False, sym='', ax =axs[i])

    axs[i].set_title(v)

    axs[i].set_xlabel("")

    sns.despine()

    i += 1

    

plt.xlabel("Weekday (0=Monday, 6=Sunday)")

f.text(0, 0.5, "Number of events", rotation="vertical", va="center")