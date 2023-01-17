import pandas as pd

from datetime import datetime

from matplotlib import pyplot as plt

%matplotlib inline



train_df = pd.read_csv('../input/train.csv')
epoch_datetime = pd.datetime(1900, 1, 1)

trf_var_68_s = (train_df['var_68']*10000 - 7000 + epoch_datetime.toordinal()).astype(int)

date_s = trf_var_68_s.map(datetime.fromordinal)

train_df['date'] = date_s
years = [2017, 2018, 2019]

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

suspicious = ['var_12','var_37','var_40','var_53', 'var_81','var_88','var_89','var_92','var_95', 'var_99', 'var_126','var_153','var_158','var_164', 'var_177','var_180', 'var_188','var_194']



for year in years:

    for month in months:

        for feature in suspicious:

            

            if year == 2017:

                if month < 7:

                    continue

            if year == 2019:

                if month > 1:

                    continue

            

            tmp_df = train_df.loc[lambda df: (df.date.dt.year == year) & (df.date.dt.month == month)]

            

            negData = tmp_df.loc[tmp_df['target'] == 0][feature]

            posData = tmp_df.loc[tmp_df['target'] == 1][feature]



            fig, ax = plt.subplots(ncols = 1, figsize=(20,5))



            fig.suptitle(str(year) + 'y ' + str(month) + 'm ' + feature)

            outs1, outs2, outs3 = ax.hist([negData, posData], 

                                          bins=30, 

                                          density = True, 

                                          histtype='step', 

                                          linewidth=3)

            ax.set_xticks(outs2)

            ax.xaxis.grid(True)



            fig.show()

            plt.show()